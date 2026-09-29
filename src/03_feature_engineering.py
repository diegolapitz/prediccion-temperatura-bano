"""
OBJETIVO DEL ARCHIVO
--------------------
Crear el primer dataset de modelado a partir de los intervalos validados.

ENTRADAS
--------
- `data/processed/intervalos_tb_8st.parquet`
- Parquet VITM original.
- CSV auxiliar de cubas.
- Archivo `config/parametros.yaml`.

SALIDAS
-------
- `data/processed/dataset_modelado_v1.parquet`
- `data/sample/muestra_dataset_modelado_v1.csv`
- `outputs/resumen_features_v1.txt`
- `docs/VALIDACION_FEATURES.md`

POR QUE EXISTE
--------------
El modelo no debe mirar la base cruda directamente. Necesita una tabla donde
cada fila sea un caso historico y cada columna sea una informacion disponible
antes de la TB objetivo.

Este archivo arma esas columnas iniciales de forma explicita y revisa que no
incluyamos el semiturno objetivo dentro de los agregados.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


# ============================================================
# FASE 0 - RUTAS DEL PROYECTO
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_DIR / "config" / "parametros.yaml"
DATA_PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
DATA_SAMPLE_DIR = PROJECT_DIR / "data" / "sample"
DOCS_DIR = PROJECT_DIR / "docs"
OUTPUTS_DIR = PROJECT_DIR / "outputs"


def print_title(title: str) -> None:
    """Imprime un titulo visible para seguir la ejecucion."""
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def load_config() -> dict[str, Any]:
    """Carga la configuracion central del proyecto."""
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_project_path(path_text: str) -> Path:
    """Convierte rutas relativas del YAML en rutas absolutas."""
    return (PROJECT_DIR / path_text).resolve()


def convert_decimal_comma_to_number(series: pd.Series) -> pd.Series:
    """Convierte textos con coma decimal a numeros."""
    return pd.to_numeric(
        series.astype("string").str.replace(",", ".", regex=False),
        errors="coerce",
    )


def load_auxiliary_cubas(path: Path) -> pd.DataFrame:
    """Carga el auxiliar Cuba -> Grupo -> Sala."""
    auxiliary = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    return auxiliary.rename(
        columns={
            "Cuba": "CUBA",
            "Grupo": "GRUPO",
            "Sala": "SALA",
        }
    )


def load_and_prepare_base(config: dict[str, Any]) -> pd.DataFrame:
    """
    Carga VITM, convierte variables numericas y aplica filtros acordados.

    Esta funcion repite pasos de fases anteriores para que este script pueda
    ejecutarse solo. Mas adelante se puede mover a un helper comun.
    """
    parquet_path = resolve_project_path(config["rutas"]["parquet_vitm"])
    auxiliary_path = resolve_project_path(config["rutas"]["auxiliar_cubas"])

    key_columns = config["columnas_clave"]
    date_column = key_columns["fecha"]
    semiturn_column = key_columns["semiturno"]
    potstate_column = key_columns["estado_operativo"]
    mother_pot_column = key_columns["cuba_madre"]

    expected_potstate = config["filtros_iniciales"]["potstate_operacion_normal"]
    expected_mother_value = config["filtros_iniciales"][
        "valor_cuba_madre_a_conservar"
    ]

    df = pd.read_parquet(parquet_path)
    auxiliary_cubas = load_auxiliary_cubas(auxiliary_path)

    df["FECHA_OPERATIVA"] = pd.to_datetime(
        df[date_column],
        dayfirst=True,
        errors="coerce",
    )
    df["ORDEN_SEMITURNO"] = (
        (df["FECHA_OPERATIVA"] - df["FECHA_OPERATIVA"].min()).dt.days * 6
        + df[semiturn_column].astype("Int64")
    )

    numeric_columns = [
        "TB",
        "TBD",
        "TBC",
        "RTH",
        "HB",
        "HBD",
        "HBC",
        "HM",
        "HMD",
        "HMC",
        "HMC",
        "ALF3",
        "ALF3D",
        "RC",
        "NDALF3",
        "AMNA2CO3",
        "NDAL2O3",
        "NTEA",
        "DTEA",
        "SEA",
        "WRMI",
        "SMRWFC",
        "V_ACD",
        "IMM",
        "RKM",
        "RRM",
        "POTAGE",
        "MBLTHE",
        "MBLC",
        "AGEBSQ",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = convert_decimal_comma_to_number(df[column])

    df["POTSTATE_NUM"] = convert_decimal_comma_to_number(df[potstate_column])
    df = df.merge(auxiliary_cubas, on=key_columns["cuba"], how="left")

    filtered = df[
        (df["POTSTATE_NUM"] == expected_potstate)
        & (df[mother_pot_column] == expected_mother_value)
    ].copy()

    filtered = filtered.sort_values(
        [key_columns["cuba"], "ORDEN_SEMITURNO"],
        kind="mergesort",
    ).reset_index(drop=True)

    return filtered


def create_expected_window_rows(
    intervals: pd.DataFrame,
    expected_semiturns: int,
) -> pd.DataFrame:
    """
    Crea una tabla con las filas esperadas de la ventana operativa.

    Si el intervalo tiene 8 semiturnos, los semiturnos intermedios son:
    inicio + 1, inicio + 2, ..., inicio + 7.

    No incluimos inicio porque ya esta representado por `tb_inicial`.
    No incluimos objetivo porque seria informacion simultanea al target.
    """
    offsets = list(range(1, expected_semiturns))

    repeated_intervals = intervals[
        ["id_intervalo", "CUBA", "orden_tb_inicial", "orden_tb_objetivo"]
    ].loc[intervals.index.repeat(len(offsets))].copy()

    repeated_intervals["offset_desde_tb_inicial"] = offsets * len(intervals)
    repeated_intervals["ORDEN_SEMITURNO"] = (
        repeated_intervals["orden_tb_inicial"]
        + repeated_intervals["offset_desde_tb_inicial"]
    )

    return repeated_intervals


def build_window_table(
    intervals: pd.DataFrame,
    base: pd.DataFrame,
    expected_semiturns: int,
) -> pd.DataFrame:
    """
    Une cada intervalo con sus semiturnos intermedios.

    El resultado tiene hasta 7 filas por intervalo. Si falta un semiturno en la
    base filtrada, esa fila queda con nulos y luego se cuenta como faltante.
    """
    expected_window = create_expected_window_rows(intervals, expected_semiturns)

    columns_to_keep = [
        "CUBA",
        "ORDEN_SEMITURNO",
        "FECHA_OPERATIVA",
        "TU_SEMI_TURNO",
        "TB",
        "TBD",
        "TBC",
        "RTH",
        "HB",
        "HBD",
        "HBC",
        "HM",
        "HMD",
        "HMC",
        "ALF3",
        "ALF3D",
        "RC",
        "NDALF3",
        "AMNA2CO3",
        "NDAL2O3",
        "NTEA",
        "DTEA",
        "SEA",
        "WRMI",
        "SMRWFC",
        "V_ACD",
        "IMM",
        "RKM",
        "RRM",
        "MBLTHE",
        "MBLC",
        "AGEBSQ",
    ]
    columns_to_keep = [column for column in columns_to_keep if column in base.columns]

    window = expected_window.merge(
        base[columns_to_keep],
        on=["CUBA", "ORDEN_SEMITURNO"],
        how="left",
        indicator=True,
    )
    window["fila_operativa_encontrada"] = window["_merge"] == "both"
    window = window.drop(columns=["_merge"])

    return window


def aggregate_window_features(window: pd.DataFrame) -> pd.DataFrame:
    """
    Resume los semiturnos intermedios en features por intervalo.

    Ejemplo: `rth_sum` es la suma de `RTH` entre la TB inicial y antes de la TB
    objetivo.
    """
    window_for_aggregation = window.copy()

    # Separamos potencia/regulacion positiva y negativa. Una suma neta puede
    # ocultar que hubo momentos de enfriamiento y calentamiento dentro de la
    # misma ventana.
    if "RTH" in window_for_aggregation.columns:
        window_for_aggregation["RTH_POSITIVO"] = window_for_aggregation["RTH"].clip(
            lower=0
        )
        window_for_aggregation["RTH_NEGATIVO"] = window_for_aggregation["RTH"].clip(
            upper=0
        )
    if "RC" in window_for_aggregation.columns:
        window_for_aggregation["RC_POSITIVO"] = window_for_aggregation["RC"].clip(
            lower=0
        )
        window_for_aggregation["RC_NEGATIVO"] = window_for_aggregation["RC"].clip(
            upper=0
        )
    if {"IMM", "V_ACD"}.issubset(window_for_aggregation.columns):
        # Proxy electrico por semiturno. No lo llamamos MW/MWh porque las
        # unidades deben validarse con instrumentacion/proceso.
        window_for_aggregation["POTENCIA_PROXY"] = (
            window_for_aggregation["IMM"] * window_for_aggregation["V_ACD"]
        )
    if "NTEA" in window_for_aggregation.columns:
        window_for_aggregation["SEMITURNO_CON_EA"] = (
            window_for_aggregation["NTEA"].fillna(0) > 0
        ).astype(int)
    if {"HBD", "HBC"}.issubset(window_for_aggregation.columns):
        # Altura de bano respecto de su objetivo. Un valor negativo significa
        # que el bano esta por debajo del objetivo operativo.
        window_for_aggregation["DESVIO_ALTURA_BANO"] = (
            window_for_aggregation["HBD"] - window_for_aggregation["HBC"]
        )
        window_for_aggregation["SEMITURNO_BANO_BAJO"] = (
            window_for_aggregation["DESVIO_ALTURA_BANO"] < 0
        ).astype("Int64")
    if {"HMD", "HMC"}.issubset(window_for_aggregation.columns):
        window_for_aggregation["DESVIO_ALTURA_METAL"] = (
            window_for_aggregation["HMD"] - window_for_aggregation["HMC"]
        )

    aggregations: dict[str, list[str]] = {
        "fila_operativa_encontrada": ["sum"],
        "TB": ["count"],
        "TBD": ["last"],
        "TBC": ["last"],
        "RTH": ["sum", "mean", "max", "min", "last"],
        "RTH_POSITIVO": ["sum"],
        "RTH_NEGATIVO": ["sum"],
        "HB": ["count"],
        "HBD": ["last"],
        "HBC": ["last"],
        "HM": ["count"],
        "HMD": ["last"],
        "HMC": ["last"],
        "DESVIO_ALTURA_BANO": ["mean", "min", "max", "last"],
        "SEMITURNO_BANO_BAJO": ["sum"],
        "DESVIO_ALTURA_METAL": ["mean", "min", "max", "last"],
        "ALF3": ["count", "last"],
        "ALF3D": ["last"],
        "RC": ["sum", "mean", "max", "min", "last"],
        "RC_POSITIVO": ["sum"],
        "RC_NEGATIVO": ["sum"],
        "NDALF3": ["sum", "max", "last"],
        "AMNA2CO3": ["sum", "max"],
        "NDAL2O3": ["sum", "mean", "max", "last"],
        "NTEA": ["sum", "max", "last"],
        "DTEA": ["sum", "max"],
        "SEA": ["sum", "max", "last"],
        "SEMITURNO_CON_EA": ["sum"],
        "WRMI": ["mean", "max", "min", "last"],
        "SMRWFC": ["mean", "max", "last"],
        "V_ACD": ["mean", "max", "min", "std", "last", "sum"],
        "IMM": ["mean", "max", "min", "std", "last", "sum"],
        "POTENCIA_PROXY": ["mean", "max", "min", "std", "last", "sum"],
        "RKM": ["mean", "last"],
        "RRM": ["mean", "last"],
        "MBLTHE": ["sum", "max", "min"],
        "MBLC": ["sum", "max", "min"],
        "AGEBSQ": ["last"],
    }

    existing_aggregations = {
        column: functions
        for column, functions in aggregations.items()
        if column in window_for_aggregation.columns
    }

    features = window_for_aggregation.groupby("id_intervalo").agg(
        existing_aggregations
    )

    # Aplanamos nombres como ('RTH', 'sum') -> 'rth_sum'.
    features.columns = [
        f"{column.lower()}_{function}"
        for column, function in features.columns.to_flat_index()
    ]
    features = features.reset_index()

    features = features.rename(
        columns={
            "fila_operativa_encontrada_sum": "filas_operativas_encontradas",
            "tb_count": "tb_reales_en_ventana",
            "hb_count": "hb_reales_en_ventana",
            "hm_count": "hm_reales_en_ventana",
            "alf3_count": "alf3_reales_en_ventana",
            "semiturno_con_ea_sum": "semiturnos_con_ea",
            "semiturno_bano_bajo_sum": "semiturnos_bano_bajo",
        }
    )

    return features


def add_alf3_history_features(
    intervals: pd.DataFrame,
    base: pd.DataFrame,
) -> pd.DataFrame:
    """
    Agrega historia de AlF3 real antes de la prediccion.

    `ALF3D` es el ultimo valor arrastrado y ya entra como snapshot. Aca buscamos
    mediciones reales de `ALF3`, que son mas espaciadas que las de TB.

    Usamos mediciones con orden <= TB inicial. No usamos AlF3 posterior a la TB
    inicial porque seria informacion futura.
    """
    result = intervals.copy()
    new_columns = [
        "alf3_ultima_real",
        "orden_alf3_ultima_real",
        "semiturnos_desde_alf3_ultima",
        "alf3_anterior_2",
        "alf3_anterior_3",
        "alf3_cambio_ultima_vs_anterior",
        "alf3_cambio_anterior_vs_tercera",
        "alf3_promedio_ultimas_2",
        "alf3_promedio_ultimas_3",
    ]
    for column in new_columns:
        result[column] = pd.NA

    for cuba, interval_group in result.groupby("CUBA", sort=False):
        alf3_measurements = base[
            (base["CUBA"] == cuba) & (base["ALF3"].notna())
        ].sort_values("ORDEN_SEMITURNO")

        if alf3_measurements.empty:
            continue

        alf3_orders = alf3_measurements["ORDEN_SEMITURNO"].astype("int64").to_numpy()
        alf3_values = alf3_measurements["ALF3"].astype(float).to_numpy()
        initial_orders = interval_group["orden_tb_inicial"].astype("int64").to_numpy()

        # Posicion de la ultima medicion de AlF3 con orden <= TB inicial.
        positions = np.searchsorted(alf3_orders, initial_orders, side="right") - 1

        for row_index, position, initial_order in zip(
            interval_group.index,
            positions,
            initial_orders,
        ):
            if position < 0:
                continue

            last_value = float(alf3_values[position])
            result.loc[row_index, "alf3_ultima_real"] = last_value
            result.loc[row_index, "orden_alf3_ultima_real"] = int(alf3_orders[position])
            result.loc[row_index, "semiturnos_desde_alf3_ultima"] = int(
                initial_order - alf3_orders[position]
            )

            if position >= 1:
                previous_value = float(alf3_values[position - 1])
                result.loc[row_index, "alf3_anterior_2"] = previous_value
                result.loc[row_index, "alf3_cambio_ultima_vs_anterior"] = (
                    last_value - previous_value
                )
                result.loc[row_index, "alf3_promedio_ultimas_2"] = (
                    last_value + previous_value
                ) / 2

            if position >= 2:
                previous_value = float(alf3_values[position - 1])
                third_value = float(alf3_values[position - 2])
                result.loc[row_index, "alf3_anterior_3"] = third_value
                result.loc[row_index, "alf3_cambio_anterior_vs_tercera"] = (
                    previous_value - third_value
                )
                result.loc[row_index, "alf3_promedio_ultimas_3"] = (
                    last_value + previous_value + third_value
                ) / 3

    return result


def add_thermal_history_features(intervals: pd.DataFrame) -> pd.DataFrame:
    """
    Crea features de historia termica reciente.

    Estas columnas usan solo mediciones reales anteriores o iguales a la TB
    inicial. No usan la TB objetivo para explicar la prediccion.
    """
    result = intervals.copy()

    result["tb_cambio_ultima_vs_anterior"] = (
        result["tb_inicial"] - result["tb_anterior_2"]
    )
    result["tb_cambio_anterior_vs_tercera"] = (
        result["tb_anterior_2"] - result["tb_anterior_3"]
    )
    result["tb_anterior_1"] = result["tb_inicial"]
    result["ultimo_cambio_tb"] = result["tb_cambio_ultima_vs_anterior"]
    result["cambio_anterior_al_ultimo"] = result["tb_cambio_anterior_vs_tercera"]
    result["tb_promedio_ultimas_2"] = result[["tb_inicial", "tb_anterior_2"]].mean(
        axis=1
    )
    result["tb_promedio_ultimas_3"] = result[
        ["tb_inicial", "tb_anterior_2", "tb_anterior_3"]
    ].mean(axis=1)
    result["tb_max_ultimas_3"] = result[
        ["tb_inicial", "tb_anterior_2", "tb_anterior_3"]
    ].max(axis=1)
    result["tb_min_ultimas_3"] = result[
        ["tb_inicial", "tb_anterior_2", "tb_anterior_3"]
    ].min(axis=1)
    result["tb_rango_ultimas_3"] = (
        result["tb_max_ultimas_3"] - result["tb_min_ultimas_3"]
    )
    result["tb_std_ultimas_3"] = result[
        ["tb_inicial", "tb_anterior_2", "tb_anterior_3"]
    ].std(axis=1)
    # Pendiente de una recta sobre las ultimas 3 TB, ordenadas en el tiempo:
    # tercera anterior -> anterior -> inicial. Como estan igualmente espaciadas
    # en mediciones, la pendiente es una tendencia por medicion.
    result["tb_pendiente_lineal_ultimas_3"] = (
        result["tb_inicial"] - result["tb_anterior_3"]
    ) / 2
    result["tb_mediciones_previas_disponibles"] = result[
        ["tb_anterior_2", "tb_anterior_3"]
    ].notna().sum(axis=1)

    return result


def build_modeling_dataset(
    intervals: pd.DataFrame,
    window_features: pd.DataFrame,
    base: pd.DataFrame,
) -> pd.DataFrame:
    """
    Une historia termica, contexto y agregados operativos.

    Mantenemos `tb_objetivo` y `delta_tb_objetivo` como targets, pero no deben
    usarse como features cuando entrenemos.
    """
    intervals_with_history = add_thermal_history_features(intervals)
    intervals_with_history = add_alf3_history_features(intervals_with_history, base)
    dataset = intervals_with_history.merge(
        window_features,
        on="id_intervalo",
        how="left",
    )

    # Variables derivadas simples, faciles de interpretar.
    dataset["rth_activo_en_ventana"] = (dataset["rth_sum"].fillna(0) != 0).astype(int)
    dataset["tuvo_efecto_anodico"] = (dataset["ntea_sum"].fillna(0) > 0).astype(int)
    dataset["hubo_ea_intervalo"] = dataset["tuvo_efecto_anodico"]
    dataset["tuvo_soda_manual"] = (dataset["amna2co3_sum"].fillna(0) > 0).astype(int)
    dataset["tuvo_movimiento_bano_real"] = (dataset["mblc_sum"].fillna(0) != 0).astype(
        int
    )
    # No imponemos una regla termica: dejamos que el modelo aprenda si un bano
    # bajo durante varios semiturnos anticipa un cambio de TB.
    dataset["bano_bajo_ultimo_semiturno"] = (
        dataset["desvio_altura_bano_last"] < 0
    ).astype("Int64")
    dataset["proporcion_semiturnos_bano_bajo"] = (
        dataset["semiturnos_bano_bajo"]
        / dataset["filas_operativas_encontradas"].replace(0, np.nan)
    )
    # El paper de referencia separa cubas nuevas, estables y cercanas al final
    # de vida. Conservamos la edad continua y agregamos esta categoria para
    # que el modelo pueda aprender comportamientos distintos por etapa.
    dataset["FASE_VIDA"] = pd.cut(
        dataset["agebsq_last"],
        bins=[-np.inf, 100, 1200, np.inf],
        labels=["arranque_0_100", "estable_101_1200", "final_mas_1200"],
    ).astype("string")
    dataset["diferencia_rrm_rkm_mean"] = dataset["rrm_mean"] - dataset["rkm_mean"]
    dataset["energia_efecto_anodico_proxy"] = (
        dataset["dtea_sum"].fillna(0) * dataset["sea_sum"].fillna(0)
    )
    dataset["tbd_menos_tbc_last"] = dataset["tbd_last"] - dataset["tbc_last"]
    dataset["rth_por_hbd_last"] = dataset["rth_sum"] / dataset["hbd_last"]
    dataset["rth_por_masa_liquida_last"] = dataset["rth_sum"] / (
        dataset["hbd_last"] + dataset["hmd_last"]
    )
    dataset["rc_por_hbd_last"] = dataset["rc_sum"] / dataset["hbd_last"]
    dataset["potencia_manual_mas_auto_sum"] = dataset["rc_sum"] + dataset["rth_sum"]
    dataset["potencia_manual_mas_auto_positiva_sum"] = (
        dataset["rc_positivo_sum"] + dataset["rth_positivo_sum"]
    )
    dataset["potencia_manual_mas_auto_negativa_sum"] = (
        dataset["rc_negativo_sum"] + dataset["rth_negativo_sum"]
    )
    dataset["interaccion_tb_inicial_alf3d"] = (
        dataset["tb_inicial"] * dataset["alf3d_last"]
    )
    dataset["interaccion_tb_inicial_rth_sum"] = (
        dataset["tb_inicial"] * dataset["rth_sum"]
    )

    return dataset


def validate_features(dataset: pd.DataFrame, expected_semiturns: int) -> tuple[list[str], list[str]]:
    """
    Revisa problemas criticos y advertencias del dataset de features.
    """
    issues: list[str] = []
    warnings: list[str] = []

    if dataset["id_intervalo"].duplicated().any():
        issues.append("Hay `id_intervalo` duplicados en el dataset de modelado.")

    if dataset["tb_objetivo"].isna().any():
        issues.append("Hay targets `tb_objetivo` vacios.")

    if (dataset["semiturnos_entre_tb"] != expected_semiturns).any():
        issues.append("Hay intervalos con distancia distinta de la esperada.")

    if (dataset["tb_reales_en_ventana"] > 0).any():
        issues.append(
            "Hay TB reales dentro de la ventana de features. Esto podria ser leakage."
        )

    short_windows = dataset[dataset["filas_operativas_encontradas"] < 7]
    if len(short_windows) > 0:
        warnings.append(
            f"{len(short_windows):,} filas tienen menos de 7 semiturnos "
            "operativos intermedios por faltantes despues de filtrar."
        )

    missing_history = dataset[dataset["tb_mediciones_previas_disponibles"] < 2]
    if len(missing_history) > 0:
        warnings.append(
            f"{len(missing_history):,} filas no tienen dos TB anteriores a la "
            "TB inicial. Es esperable al comienzo de la historia de cada cuba."
        )

    return issues, warnings


def write_feature_summary(
    dataset: pd.DataFrame,
    issues: list[str],
    warnings: list[str],
    output_path: Path,
) -> None:
    """Guarda resumen textual de features."""
    feature_count = len(
        [
            column
            for column in dataset.columns
            if column not in {"tb_objetivo", "delta_tb_objetivo"}
        ]
    )

    lines = [
        "RESUMEN DE FEATURES - FASE 3",
        "=" * 60,
        "",
        f"Filas del dataset de modelado: {len(dataset):,}",
        f"Columnas totales: {len(dataset.columns):,}",
        f"Columnas candidatas a features: {feature_count:,}",
        f"Cubas incluidas: {dataset['CUBA'].nunique():,}",
        "",
        "Targets disponibles:",
        "- tb_objetivo: proxima TB real.",
        "- delta_tb_objetivo: tb_objetivo - tb_inicial.",
        "",
        "Grupos de features creados:",
        "- Historia termica: ultimas TB reales, cambios, promedios y rango.",
        "- Contexto: CUBA, SALA y GRUPO.",
        "- Ventana operativa: agregados entre TB inicial y antes de TB objetivo.",
        "- Snapshot: ultimos valores disponibles dentro de la ventana.",
        "- Indicadores simples: efectos anodicos, soda, movimiento de bano, RTH activo.",
        "",
        "Nulos principales:",
    ]

    null_percentages = dataset.isna().mean().sort_values(ascending=False).head(20) * 100
    for column, percentage in null_percentages.items():
        lines.append(f"- {column}: {percentage:.2f}%")

    lines.append("")
    lines.append("Controles:")
    if issues:
        for issue in issues:
            lines.append(f"- PROBLEMA: {issue}")
    else:
        lines.append("- OK: no se encontraron problemas criticos.")
    for warning in warnings:
        lines.append(f"- ADVERTENCIA: {warning}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_validation_doc(
    dataset: pd.DataFrame,
    issues: list[str],
    warnings: list[str],
    output_path: Path,
) -> None:
    """Guarda una explicacion legible de validacion de features."""
    lines = [
        "# Validacion de features - Fase 3",
        "",
        "Este documento revisa si el dataset de modelado inicial esta listo para "
        "probar baselines y modelos simples.",
        "",
        "## Regla anti-leakage aplicada",
        "",
        "Para cada intervalo se usaron solo los semiturnos posteriores a la TB "
        "inicial y anteriores a la TB objetivo. El semiturno objetivo queda "
        "excluido de los agregados.",
        "",
        "## Controles realizados",
        "",
        "- `id_intervalo` no debe repetirse.",
        "- `tb_objetivo` no debe estar vacia.",
        "- La distancia entre TB inicial y objetivo debe ser 8 semiturnos.",
        "- No debe existir otra TB real dentro de la ventana de features.",
        "- Se documentan filas con ventanas incompletas o poca historia termica.",
        "",
        "## Resultado",
        "",
        f"- Filas: {len(dataset):,}.",
        f"- Columnas: {len(dataset.columns):,}.",
        f"- Cubas: {dataset['CUBA'].nunique():,}.",
        f"- Rango objetivo: {dataset['fecha_tb_objetivo'].min().date()} a "
        f"{dataset['fecha_tb_objetivo'].max().date()}.",
        "",
        "## Problemas criticos",
        "",
    ]

    if issues:
        for issue in issues:
            lines.append(f"- {issue}")
    else:
        lines.append("No se encontraron problemas criticos.")

    lines.append("")
    lines.append("## Advertencias")
    lines.append("")
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("No se encontraron advertencias relevantes.")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Ejecuta la Fase 3 completa."""
    print_title("FASE 3 - FEATURE ENGINEERING INICIAL")
    config = load_config()
    expected_semiturns = config["modelado"]["semiturnos_entre_mediciones_tb"]

    intervals_path = DATA_PROCESSED_DIR / "intervalos_tb_8st.parquet"
    if not intervals_path.exists():
        raise FileNotFoundError(
            "No existe el archivo de intervalos. Ejecuta primero "
            "`python src/02_construccion_intervalos.py`."
        )

    print_title("1. Cargando intervalos y base filtrada")
    intervals = pd.read_parquet(intervals_path)
    base = load_and_prepare_base(config)
    print(f"Intervalos: {len(intervals):,}")
    print(f"Filas base filtrada: {len(base):,}")

    print_title("2. Armando ventanas operativas")
    window = build_window_table(intervals, base, expected_semiturns)
    print(f"Filas esperadas de ventana: {len(window):,}")

    print_title("3. Agregando features por intervalo")
    window_features = aggregate_window_features(window)
    dataset = build_modeling_dataset(intervals, window_features, base)
    dataset = dataset.sort_values(["fecha_tb_objetivo", "CUBA"]).reset_index(drop=True)
    print(f"Dataset de modelado: {dataset.shape[0]:,} filas x {dataset.shape[1]:,} columnas")

    print_title("4. Validando features")
    issues, warnings = validate_features(dataset, expected_semiturns)
    if issues:
        for issue in issues:
            print(f"PROBLEMA: {issue}")
    else:
        print("OK: no se encontraron problemas criticos.")
    for warning in warnings:
        print(f"ADVERTENCIA: {warning}")

    print_title("5. Guardando salidas")
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    DATA_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    dataset_path = DATA_PROCESSED_DIR / "dataset_modelado_v1.parquet"
    sample_path = DATA_SAMPLE_DIR / "muestra_dataset_modelado_v1.csv"
    summary_path = OUTPUTS_DIR / "resumen_features_v1.txt"
    validation_path = DOCS_DIR / "VALIDACION_FEATURES.md"

    dataset.to_parquet(dataset_path, index=False)
    dataset.head(500).to_csv(sample_path, index=False, encoding="utf-8-sig")
    write_feature_summary(dataset, issues, warnings, summary_path)
    write_validation_doc(dataset, issues, warnings, validation_path)

    print("Archivos generados:")
    print(f"- {dataset_path}")
    print(f"- {sample_path}")
    print(f"- {summary_path}")
    print(f"- {validation_path}")


if __name__ == "__main__":
    main()
