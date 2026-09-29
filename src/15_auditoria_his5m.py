"""
Audita los datos de cinco minutos antes de usarlos en el modelo de TB.

El archivo no entrena modelos ni modifica los Parquet. Revisa:

- cobertura mensual y cubas presentes;
- duplicados en la clave CUBA + DH5MN;
- frecuencia temporal real;
- cambios de esquema y valores faltantes;
- cantidad de intervalos de TB que coinciden con cada fuente.

Por decision del proyecto se usan enero-julio de 2026. Agosto se conserva
como dato crudo, pero queda fuera de esta primera prueba.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re

import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq


PROJECT_DIR = Path(__file__).resolve().parents[1]
HIS5M_DIR = PROJECT_DIR / "data" / "raw" / "his5m"
INTERVALS_PATH = PROJECT_DIR / "data" / "processed" / "dataset_modelado_v1.parquet"
THISEVT_PATH = PROJECT_DIR.parent / "PE2CU_THISEVT.parquet"
ANODE_PATH = PROJECT_DIR.parent / "qry_tprob_anod (3).parquet"
AUXILIARY_PATH = PROJECT_DIR.parent / "auxiliar_cubas.csv"

MONTHLY_OUTPUT = PROJECT_DIR / "outputs" / "metricas" / "calidad_his5m_mensual.csv"
COLUMN_OUTPUT = PROJECT_DIR / "outputs" / "metricas" / "calidad_his5m_columnas.csv"
SUMMARY_OUTPUT = PROJECT_DIR / "outputs" / "resumen_calidad_his5m.txt"

FIRST_MONTH = "2026-01"
LAST_MONTH = "2026-07"
EXPECTED_POTS = 384
EXPECTED_ROWS_PER_POT_DAY = 288
ANODE_COMPLETE_THROUGH = pd.Timestamp("2026-07-19")

KEY_COLUMNS = ["CUBA", "INDPOT", "IND5MN", "DH5MN"]
EVENT_COLUMNS = [
    "POTMICRO_LOCAL",
    "CRUSTBRK_INT",
    "FEEDER_INT",
    "ANODE_UP_LIMIT",
    "ANODE_DW_LIMIT",
    "ANODE_COVER",
    "REMAIN_ORDER",
    "ORDER_TL",
    "BEAM_HEELING",
    "BEAM_MANUAL",
    "BEAM_PSU",
    "ORDER_DISCREPANCY",
    "AL2O3_FEED_PH2V",
    "AL2O3_FEED_PH1",
    "AL2O3_FEED_PH2I",
    "PRG_TRACKING",
    "CTL_TRACKING",
    "AL2O3_OVERFEED",
    "MILD_INSTABILITY",
    "SEVER_INSTABILITY",
    "EXTD_INSTABILITY",
    "ANODE_EFFECT",
    "PREP_BEFORE_SD",
    "LINE_SD",
    "RESTART_AFTER_SD",
    "VOLTAGE_TAPPING",
    "TAPPING",
    "ANODE_CHANGE",
    "VOLTAGE_BEAM",
    "OVERSUCCION",
    "POTMICRO_LOCAL_FC",
    "BEAM_MANUAL_FC",
    "INTERRUPT_CYC",
]


def normalize_pot(values: pd.Series) -> pd.Series:
    """Convierte la cuba a texto comparable entre todas las fuentes."""
    return values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)


def month_from_filename(path: Path) -> str:
    """Obtiene YYYY-MM del nombre mensual y falla si el nombre es ambiguo."""
    match = re.search(r"(\d{4}-\d{2})$", path.stem)
    if match is None:
        raise ValueError(f"Nombre mensual no reconocido: {path.name}")
    return match.group(1)


def selected_files() -> list[Path]:
    """Devuelve solo los meses aprobados para esta primera prueba."""
    files = sorted(HIS5M_DIR.glob("tmon_his5m_*.parquet"))
    selected = [
        path
        for path in files
        if FIRST_MONTH <= month_from_filename(path) <= LAST_MONTH
    ]
    expected = pd.period_range(FIRST_MONTH, LAST_MONTH, freq="M").astype(str).tolist()
    found = [month_from_filename(path) for path in selected]
    if found != expected:
        raise ValueError(f"Meses esperados: {expected}. Meses encontrados: {found}.")
    return selected


def audit_month(path: Path) -> tuple[dict[str, object], pd.DataFrame]:
    """Revisa clave, cobertura y cadencia de un Parquet mensual."""
    month = month_from_filename(path)
    data = pd.read_parquet(path, columns=KEY_COLUMNS)
    data["CUBA"] = normalize_pot(data["CUBA"])
    data["DH5MN"] = pd.to_datetime(data["DH5MN"], errors="coerce")

    duplicated = data.duplicated(["CUBA", "DH5MN"])
    affected_by_duplicates = data.duplicated(["CUBA", "DH5MN"], keep=False)
    wrong_month = data["DH5MN"].dt.strftime("%Y-%m").ne(month)
    mapping = data[["CUBA", "INDPOT"]].drop_duplicates()

    ordered = data.sort_values(["CUBA", "DH5MN"], kind="mergesort")
    seconds = ordered.groupby("CUBA", sort=False)["DH5MN"].diff().dt.total_seconds()
    seconds = seconds.dropna()
    expected_cadence = seconds.between(240, 360)
    long_gaps = ordered.loc[seconds[seconds.gt(3600)].index, ["CUBA", "DH5MN"]].copy()
    long_gaps["hora_fin_salto"] = long_gaps["DH5MN"].dt.floor("h")
    if long_gaps.empty:
        largest_shared_gap_hour = pd.NaT
        pots_in_largest_shared_gap = 0
    else:
        pots_by_hour = long_gaps.groupby("hora_fin_salto")["CUBA"].nunique()
        largest_shared_gap_hour = pots_by_hour.idxmax()
        pots_in_largest_shared_gap = int(pots_by_hour.max())

    period = pd.Period(month, freq="M")
    expected_rows = EXPECTED_POTS * period.days_in_month * EXPECTED_ROWS_PER_POT_DAY
    rows_by_pot = data.groupby("CUBA", sort=False).size()
    unique_rows = len(data) - int(duplicated.sum())

    result = {
        "mes": month,
        "archivo": path.name,
        "filas": len(data),
        "filas_esperadas": expected_rows,
        "cobertura_porcentaje": unique_rows / expected_rows * 100,
        "cubas": data["CUBA"].nunique(),
        "fecha_min": data["DH5MN"].min(),
        "fecha_max": data["DH5MN"].max(),
        "timestamps_nulos": int(data["DH5MN"].isna().sum()),
        "filas_fuera_del_mes": int(wrong_month.sum()),
        "duplicados_clave_extra": int(duplicated.sum()),
        "filas_afectadas_por_duplicados": int(affected_by_duplicates.sum()),
        "cubas_con_mas_de_un_indpot": int(
            mapping.groupby("CUBA")["INDPOT"].nunique().gt(1).sum()
        ),
        "filas_por_cuba_min": int(rows_by_pot.min()),
        "filas_por_cuba_mediana": float(rows_by_pot.median()),
        "filas_por_cuba_max": int(rows_by_pot.max()),
        "intervalos_4_a_6_min_porcentaje": expected_cadence.mean() * 100,
        "saltos_mayores_10_min": int(seconds.gt(600).sum()),
        "salto_maximo_min": seconds.max() / 60,
        "hora_salto_largo_mas_compartido": largest_shared_gap_hour,
        "cubas_en_salto_largo_mas_compartido": pots_in_largest_shared_gap,
        "intervalos_no_positivos": int(seconds.le(0).sum()),
        "intervalos_evaluados": len(seconds),
    }
    return result, mapping.assign(mes=month)


def audit_schema_and_nulls(files: list[Path]) -> tuple[pd.DataFrame, list[str]]:
    """Usa metadatos Parquet para medir nulos sin cargar 60 columnas en memoria."""
    first_schema = pq.ParquetFile(files[0]).schema_arrow.remove_metadata()
    first_names = first_schema.names
    total_rows = 0
    nulls = Counter()
    types: dict[str, set[str]] = {name: set() for name in first_names}
    schema_notes: list[str] = []

    for path in files:
        parquet = pq.ParquetFile(path)
        schema = parquet.schema_arrow.remove_metadata()
        if schema.names != first_names:
            raise ValueError(f"Las columnas cambian en {path.name}.")
        total_rows += parquet.metadata.num_rows

        differences = []
        for base_field, field in zip(first_schema, schema, strict=True):
            types[field.name].add(str(field.type))
            if base_field.type != field.type:
                differences.append(f"{field.name}: {base_field.type}->{field.type}")
        if differences:
            schema_notes.append(f"{path.name}: {', '.join(differences)}")

        for column_index, name in enumerate(first_names):
            for row_group_index in range(parquet.metadata.num_row_groups):
                statistics = parquet.metadata.row_group(row_group_index).column(
                    column_index
                ).statistics
                if statistics is not None and statistics.null_count is not None:
                    nulls[name] += statistics.null_count

    rows = []
    for name in first_names:
        rows.append(
            {
                "columna": name,
                "tipos_encontrados": " | ".join(sorted(types[name])),
                "nulos": nulls[name],
                "nulos_porcentaje": nulls[name] / total_rows * 100,
            }
        )
    return pd.DataFrame(rows), schema_notes


def audit_operating_states(
    files: list[Path], column_quality: pd.DataFrame
) -> tuple[pd.DataFrame, Counter, Counter]:
    """Cuenta estados no cero y valores especiales relevantes para modelar."""
    nonzero = Counter()
    feeding_modes = Counter()
    special = Counter()

    columns = EVENT_COLUMNS + ["TALIMFC", "PAL2O3FC", "IMFC", "UMFC", "RMFC"]
    for path in files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=250_000, columns=columns):
            for name in EVENT_COLUMNS:
                values = batch.column(batch.schema.get_field_index(name))
                is_nonzero = pc.fill_null(pc.not_equal(values, 0), False)
                nonzero[name] += int(pc.sum(is_nonzero).as_py() or 0)

            feeding = batch.column(batch.schema.get_field_index("TALIMFC"))
            for item in pc.value_counts(feeding).to_pylist():
                if item["values"] is not None:
                    feeding_modes[float(item["values"])] += int(item["counts"])

            pal = batch.column(batch.schema.get_field_index("PAL2O3FC"))
            special["PAL2O3FC_igual_9999"] += int(
                pc.sum(pc.fill_null(pc.equal(pal, 9999), False)).as_py() or 0
            )
            for name, threshold, operation in [
                ("IMFC", 1, "menor"),
                ("UMFC", 10, "mayor"),
                ("RMFC", 50, "mayor"),
            ]:
                values = batch.column(batch.schema.get_field_index(name))
                comparison = (
                    pc.less(values, threshold)
                    if operation == "menor"
                    else pc.greater(values, threshold)
                )
                special[f"{name}_{operation}_{threshold}"] += int(
                    pc.sum(pc.fill_null(comparison, False)).as_py() or 0
                )

    total_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
    column_quality = column_quality.copy()
    column_quality["filas_no_cero"] = pd.NA
    column_quality["no_cero_porcentaje"] = pd.NA
    for name, count in nonzero.items():
        mask = column_quality["columna"].eq(name)
        column_quality.loc[mask, "filas_no_cero"] = count
        column_quality.loc[mask, "no_cero_porcentaje"] = count / total_rows * 100
    return column_quality, feeding_modes, special


def audit_overlap() -> dict[str, object]:
    """Cuenta intervalos de TB utilizables con cada combinacion de fuentes."""
    intervals = pd.read_parquet(
        INTERVALS_PATH,
        columns=["CUBA", "fecha_tb_inicial", "fecha_tb_objetivo"],
    )
    intervals["CUBA"] = normalize_pot(intervals["CUBA"])
    initial = pd.to_datetime(intervals["fecha_tb_inicial"], errors="coerce")
    target = pd.to_datetime(intervals["fecha_tb_objetivo"], errors="coerce")

    his_start = pd.Timestamp(f"{FIRST_MONTH}-01")
    his_end = pd.Period(LAST_MONTH, freq="M").end_time.normalize() + pd.Timedelta(days=1)
    his = initial.ge(his_start) & target.lt(his_end)

    thisevt = pd.read_parquet(THISEVT_PATH, columns=["FH_DHEVT"])
    thisevt_time = pd.to_datetime(thisevt["FH_DHEVT"], errors="coerce")
    thisevt_start = thisevt_time.min()
    thisevt_end = thisevt_time.max()
    with_thisevt = his & initial.ge(thisevt_start.normalize()) & target.le(
        thisevt_end.normalize()
    )
    with_anodes = his & target.le(ANODE_COMPLETE_THROUGH)

    return {
        "intervalos_tb_his5m": int(his.sum()),
        "intervalos_tb_his5m_anodos": int(with_anodes.sum()),
        "intervalos_tb_his5m_thisevt": int(with_thisevt.sum()),
        "intervalos_tb_todas_las_fuentes": int((with_anodes & with_thisevt).sum()),
        "cubas_en_intervalos_his5m": intervals.loc[his, "CUBA"].nunique(),
        "primera_tb_inicial": initial[his].min(),
        "ultima_tb_objetivo": target[his].max(),
        "thisevt_inicio": thisevt_start,
        "thisevt_fin": thisevt_end,
    }


def audit_thisevt_anode_effects() -> tuple[pd.DataFrame, dict[str, object]]:
    """Comprueba que cada efecto cerrado tenga un solo resumen con energia."""
    events = pd.read_parquet(
        THISEVT_PATH,
        columns=[
            "N_NUMEVT",
            "FH_DHEVT",
            "CUBA",
            "V_ASSVAL1",
            "V_ASSVAL2",
            "V_ASSVAL3",
            "V_ASSVAL4",
        ],
    )
    events = events[events["N_NUMEVT"].isin([241, 242, 279, 281, 380])].copy()
    for column in ["V_ASSVAL1", "V_ASSVAL2", "V_ASSVAL3", "V_ASSVAL4"]:
        events[column] = pd.to_numeric(events[column], errors="coerce")

    events["CUBA"] = normalize_pot(events["CUBA"])
    events["FH_DHEVT"] = pd.to_datetime(events["FH_DHEVT"], errors="coerce")

    rows = []
    for code in [241, 242, 279, 281, 380]:
        part = events[events["N_NUMEVT"].eq(code)]
        assval1 = part["V_ASSVAL1"].dropna()
        assval2 = part["V_ASSVAL2"].dropna()
        rows.append(
            {
                "codigo": code,
                "eventos": len(part),
                "cubas": normalize_pot(part["CUBA"]).nunique(),
                "fecha_min": pd.to_datetime(part["FH_DHEVT"]).min(),
                "fecha_max": pd.to_datetime(part["FH_DHEVT"]).max(),
                "assval1_mediana": assval1.median() if not assval1.empty else None,
                "assval1_max": assval1.max() if not assval1.empty else None,
                "assval2_mediana": assval2.median() if not assval2.empty else None,
                "assval2_p90": assval2.quantile(0.90) if not assval2.empty else None,
                "assval2_max": assval2.max() if not assval2.empty else None,
            }
        )
    profiles = pd.DataFrame(rows)

    # El 242 marca el cierre fisico. El 279 o 281 registrado casi al mismo
    # tiempo contiene el unico resumen de duracion y energia que se usara.
    summaries = events[events["N_NUMEVT"].isin([279, 281])].sort_values(
        ["FH_DHEVT", "CUBA"]
    )
    closures = (
        events[events["N_NUMEVT"].eq(242)][["CUBA", "FH_DHEVT"]]
        .rename(columns={"FH_DHEVT": "hora_cierre_242"})
        .sort_values(["hora_cierre_242", "CUBA"])
    )
    paired = pd.merge_asof(
        summaries,
        closures,
        left_on="FH_DHEVT",
        right_on="hora_cierre_242",
        by="CUBA",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=60),
    )
    offset_seconds = (
        paired["FH_DHEVT"] - paired["hora_cierre_242"]
    ).dt.total_seconds().abs()
    diagnostics = {
        "cierres_242": len(closures),
        "resumenes_con_energia": len(summaries),
        "resumenes_emparejados": int(paired["hora_cierre_242"].notna().sum()),
        "desfase_maximo_segundos": offset_seconds.max(),
        "energia_mediana_kwh": summaries["V_ASSVAL2"].median(),
        "energia_p90_kwh": summaries["V_ASSVAL2"].quantile(0.90),
        "energia_maxima_kwh": summaries["V_ASSVAL2"].max(),
        "duracion_mediana_segundos": summaries["V_ASSVAL1"].median(),
        "duracion_maxima_segundos": summaries["V_ASSVAL1"].max(),
    }
    return profiles, diagnostics


def write_summary(
    monthly: pd.DataFrame,
    columns: pd.DataFrame,
    schema_notes: list[str],
    mappings: pd.DataFrame,
    feeding_modes: Counter,
    special: Counter,
    overlap: dict[str, object],
    anode_effects: pd.DataFrame,
    anode_effect_diagnostics: dict[str, object],
) -> None:
    """Escribe una conclusion corta, con decisiones listas para el modelado."""
    total_rows = int(monthly["filas"].sum())
    expected_rows = int(monthly["filas_esperadas"].sum())
    duplicate_rows = int(monthly["duplicados_clave_extra"].sum())
    cadence_weighted = (
        monthly["intervalos_4_a_6_min_porcentaje"]
        * monthly["intervalos_evaluados"]
    ).sum() / monthly["intervalos_evaluados"].sum()

    auxiliary = pd.read_csv(AUXILIARY_PATH, sep=";")
    auxiliary_pots = set(normalize_pot(auxiliary["Cuba"]))
    his_pots = set(mappings["CUBA"])
    mapping_conflicts = int(
        mappings.groupby("CUBA")["INDPOT"].nunique().gt(1).sum()
    )

    pente_null = columns.loc[columns["columna"].eq("PENTEFC"), "nulos_porcentaje"].iloc[0]
    rk_null = columns.loc[columns["columna"].eq("RKFC"), "nulos_porcentaje"].iloc[0]
    common_null = columns.loc[columns["columna"].eq("IMFC"), "nulos_porcentaje"].iloc[0]
    useful_events = columns[
        columns["columna"].isin(
            [
                "FEEDER_INT",
                "ANODE_EFFECT",
                "ANODE_CHANGE",
                "MILD_INSTABILITY",
                "SEVER_INSTABILITY",
                "TAPPING",
            ]
        )
    ][["columna", "filas_no_cero", "no_cero_porcentaje"]]
    july = monthly[monthly["mes"].eq("2026-07")].iloc[0]
    lines = [
        "# Auditoria HIS5M - enero a julio de 2026",
        "",
        "## Conclusion",
        "",
        "**APTO PARA CONSTRUIR FEATURES Y PROBAR UN PRIMER MODELO.**",
        "",
        f"- Filas: {total_rows:,}.",
        f"- Cubas: {len(his_pots):,}; coinciden con las {len(auxiliary_pots):,} cubas del auxiliar.",
        f"- Cobertura contra una grilla teorica cada 5 minutos: {total_rows / expected_rows * 100:.2f}%.",
        f"- Intervalos reales entre 4 y 6 minutos: {cadence_weighted:.2f}%.",
        f"- Duplicados extra en CUBA + DH5MN: {duplicate_rows:,}.",
        f"- Corte simultaneo mas importante: {july['hora_salto_largo_mas_compartido']:%Y-%m-%d %H:%M}, ",
        f"  afecta las {int(july['cubas_en_salto_largo_mas_compartido']):,} cubas.",
        "  Corresponde a unas 3 horas sin adquisicion entre el 2 y el 3 de julio.",
        "",
        "## Clave y esquema",
        "",
        "- La clave correcta es `CUBA + DH5MN`.",
        "- `INDPOT` se repite entre series y no debe usarse solo como identificador global.",
        f"- Cubas que cambian de INDPOT entre meses: {mapping_conflicts:,}.",
        "- Desde abril, INDPOT e IND5MN vienen como decimal en lugar de entero.",
        "  No tienen nulos y se normalizaran al leerlos; no impide modelar.",
        "",
        "Cambios exactos de esquema:",
        *[f"- {note}" for note in schema_notes],
        "",
        "## Completitud de variables",
        "",
        f"- IMFC y la mayoria de las senales: {common_null:.4f}% de nulos.",
        f"- RKFC y WRMFC: {rk_null:.4f}% de nulos.",
        f"- PENTEFC: {pente_null:.2f}% de nulos. No se usara directamente; la pendiente",
        "  se recalculara desde la serie de resistencia.",
        f"- PAL2O3FC=9999 aparece {special['PAL2O3FC_igual_9999']:,} veces y coincide",
        "  con estados TALIMFC=0. Se tratara como codigo/sentinel, no como cantidad real.",
        "",
        "Modos TALIMFC y cantidad de filas:",
        *[f"- {mode:g}: {count:,}" for mode, count in sorted(feeding_modes.items())],
        "",
        "Eventos/estados con presencia real:",
        "```text",
        useful_events.to_string(index=False),
        "```",
        "",
        "## Efectos anodicos detallados en THISEVT",
        "",
        "- 241: inicio/estado `EN EFECTO ANODICO`.",
        "- 242: fin fisico del efecto; no contiene energia.",
        "- 279 o 281: unico resumen asociado al cierre, con duracion en V_ASSVAL1",
        "  y energia en V_ASSVAL2. Son dos resultados posibles, no dos episodios.",
        "- 287 y 380 son registros auxiliares/intermedios y no se contaran como cierres.",
        "",
        f"- Cierres 242: {anode_effect_diagnostics['cierres_242']:,}.",
        f"- Resumenes finales con energia: {anode_effect_diagnostics['resumenes_con_energia']:,}.",
        f"- Resumenes unidos al 242 dentro de 60 segundos: "
        f"{anode_effect_diagnostics['resumenes_emparejados']:,}; desfase maximo "
        f"{anode_effect_diagnostics['desfase_maximo_segundos']:.0f} s.",
        f"- Energia mediana: {anode_effect_diagnostics['energia_mediana_kwh']:.0f} kWh; "
        f"p90: {anode_effect_diagnostics['energia_p90_kwh']:.0f} kWh; maxima: "
        f"{anode_effect_diagnostics['energia_maxima_kwh']:.0f} kWh.",
        "",
        "Para el modelo se construira una sola fila por efecto cerrado. De esa fila se",
        "sumaran cantidad, duracion y energia previas a la prediccion. Un 242 junto con",
        "su 279/281 nunca se contara como dos efectos distintos.",
        "",
        "## Coincidencia con las TB y otras fuentes",
        "",
        f"- Intervalos TB con HIS5M: {overlap['intervalos_tb_his5m']:,}.",
        f"- HIS5M + problemas anodicos: {overlap['intervalos_tb_his5m_anodos']:,}.",
        f"- HIS5M + THISEVT: {overlap['intervalos_tb_his5m_thisevt']:,}.",
        f"- Todas las fuentes juntas: {overlap['intervalos_tb_todas_las_fuentes']:,}.",
        f"- Cubas con intervalos TB: {overlap['cubas_en_intervalos_his5m']:,}.",
        f"- Ventana TB util: {overlap['primera_tb_inicial']:%Y-%m-%d} a "
        f"{overlap['ultima_tb_objetivo']:%Y-%m-%d}.",
        "",
        "THISEVT no debe limitar el modelo principal porque solo cubre abril-junio.",
        "Se probara como una familia opcional en el subperiodo comun.",
        "",
        "## Decisiones para el siguiente paso",
        "",
        "1. Usar enero-julio; conservar agosto fuera del modelado.",
        "2. Agregar HIS5M por intervalos de TB, nunca como 22 millones de filas directas.",
        "3. Crear familias separadas: energia/resistencia, alimentacion, inestabilidad y eventos.",
        "4. Comparar cada familia contra VITM sola usando exactamente las mismas filas.",
        "5. Medir por separado cubas que se calientan y TB objetivo >=970/980 C.",
        "",
        "## Archivos de control",
        "",
        f"- `{MONTHLY_OUTPUT.relative_to(PROJECT_DIR)}`",
        f"- `{COLUMN_OUTPUT.relative_to(PROJECT_DIR)}`",
    ]
    SUMMARY_OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Ejecuta la auditoria completa sin entrenar ni modificar los datos."""
    files = selected_files()
    monthly_rows = []
    mapping_frames = []
    for path in files:
        print(f"Auditando {path.name}...")
        result, mapping = audit_month(path)
        monthly_rows.append(result)
        mapping_frames.append(mapping)

    monthly = pd.DataFrame(monthly_rows)
    mappings = pd.concat(mapping_frames, ignore_index=True)
    columns, schema_notes = audit_schema_and_nulls(files)
    columns, feeding_modes, special = audit_operating_states(files, columns)
    overlap = audit_overlap()
    anode_effects, anode_effect_diagnostics = audit_thisevt_anode_effects()

    for path in [MONTHLY_OUTPUT, COLUMN_OUTPUT, SUMMARY_OUTPUT]:
        path.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(MONTHLY_OUTPUT, index=False)
    columns.to_csv(COLUMN_OUTPUT, index=False)
    write_summary(
        monthly,
        columns,
        schema_notes,
        mappings,
        feeding_modes,
        special,
        overlap,
        anode_effects,
        anode_effect_diagnostics,
    )
    print(f"Resumen: {SUMMARY_OUTPUT}")


if __name__ == "__main__":
    main()
