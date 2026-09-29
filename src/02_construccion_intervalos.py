"""
OBJETIVO DEL ARCHIVO
--------------------
Construir los intervalos historicos que se usaran como base del modelo.

ENTRADAS
--------
- Parquet VITM original.
- CSV auxiliar de cubas.
- Archivo `config/parametros.yaml`.

SALIDAS
-------
- `data/processed/intervalos_tb_8st.parquet`
- `data/sample/muestra_intervalos_tb_8st.csv`
- `outputs/resumen_intervalos_tb_8st.txt`
- `docs/VALIDACION_INTERVALOS.md`

POR QUE EXISTE
--------------
Un modelo de machine learning necesita una tabla donde cada fila tenga un
significado claro. En este proyecto, cada fila representa:

    TB real inicial
    -> 8 semiturnos de evolucion operativa
    -> siguiente TB real objetivo

Este archivo NO entrena modelos. Solo arma y valida esa tabla de intervalos.
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
    """Carga la configuracion central."""
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_project_path(path_text: str) -> Path:
    """Convierte rutas relativas del YAML en rutas absolutas."""
    return (PROJECT_DIR / path_text).resolve()


def convert_decimal_comma_to_number(series: pd.Series) -> pd.Series:
    """
    Convierte valores como '956,0' en numeros.

    La base viene de un entorno donde la coma se usa como separador decimal.
    Python necesita punto decimal para calcular.
    """
    return pd.to_numeric(
        series.astype("string").str.replace(",", ".", regex=False),
        errors="coerce",
    )


def load_auxiliary_cubas(path: Path) -> pd.DataFrame:
    """Carga el auxiliar con la relacion Cuba -> Grupo -> Sala."""
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
    Carga VITM, convierte columnas clave y aplica los filtros acordados.

    Esta funcion existe para que la construccion de intervalos empiece desde
    una base limpia y siempre igual.
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

    # Convertimos las variables que se usan para filtrar, ordenar y crear target.
    df["POTSTATE_NUM"] = convert_decimal_comma_to_number(df[potstate_column])
    df["TB_NUM"] = convert_decimal_comma_to_number(df[key_columns["tb_real"]])
    df["TBD_NUM"] = convert_decimal_comma_to_number(df[key_columns["tb_arrastrada"]])
    df["ALF3_NUM"] = convert_decimal_comma_to_number(df[key_columns["alf3_real"]])
    df["ALF3D_NUM"] = convert_decimal_comma_to_number(
        df[key_columns["alf3_arrastrada"]]
    )

    # Reemplazamos estas columnas por version numerica para no arrastrar textos.
    df[key_columns["tb_real"]] = df["TB_NUM"]
    df[key_columns["tb_arrastrada"]] = df["TBD_NUM"]
    df[key_columns["alf3_real"]] = df["ALF3_NUM"]
    df[key_columns["alf3_arrastrada"]] = df["ALF3D_NUM"]

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


def build_tb_intervals(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """
    Construye una fila por par de mediciones reales de TB separadas por 8 ST.

    Importante:
    - `tb_inicial` es una medicion real de TB.
    - `tb_objetivo` es la siguiente medicion real de TB.
    - En esta primera version solo aceptamos distancia exacta de 8 semiturnos.
    """
    key_columns = config["columnas_clave"]
    cuba_column = key_columns["cuba"]
    semiturn_column = key_columns["semiturno"]
    tb_column = key_columns["tb_real"]
    expected_semiturns = config["modelado"]["semiturnos_entre_mediciones_tb"]

    measured_tb = df[df[tb_column].notna()].copy()
    measured_tb = measured_tb.sort_values(
        [cuba_column, "ORDEN_SEMITURNO"],
        kind="mergesort",
    ).reset_index(drop=True)

    rows: list[dict[str, Any]] = []

    # Agrupamos por cuba porque cada cuba tiene su propia secuencia temporal.
    for cuba, group in measured_tb.groupby(cuba_column, sort=False):
        group = group.reset_index(drop=True)

        # Recorremos desde la segunda medicion porque necesitamos una anterior.
        for position in range(1, len(group)):
            initial = group.iloc[position - 1]
            target = group.iloc[position]

            semiturns_between = int(
                target["ORDEN_SEMITURNO"] - initial["ORDEN_SEMITURNO"]
            )

            if semiturns_between != expected_semiturns:
                continue

            # Lags anteriores a la TB inicial. Sirven luego para historia termica.
            previous_2 = group.iloc[position - 2] if position >= 2 else None
            previous_3 = group.iloc[position - 3] if position >= 3 else None

            rows.append(
                {
                    "id_intervalo": (
                        f"{int(cuba)}_"
                        f"{int(initial['ORDEN_SEMITURNO'])}_"
                        f"{int(target['ORDEN_SEMITURNO'])}"
                    ),
                    "CUBA": int(cuba),
                    "SALA": initial["SALA"],
                    "GRUPO": initial["GRUPO"],
                    "fecha_tb_inicial": initial["FECHA_OPERATIVA"],
                    "semiturno_tb_inicial": int(initial[semiturn_column]),
                    "orden_tb_inicial": int(initial["ORDEN_SEMITURNO"]),
                    "tb_inicial": float(initial[tb_column]),
                    "fecha_tb_objetivo": target["FECHA_OPERATIVA"],
                    "semiturno_tb_objetivo": int(target[semiturn_column]),
                    "orden_tb_objetivo": int(target["ORDEN_SEMITURNO"]),
                    "tb_objetivo": float(target[tb_column]),
                    "delta_tb_objetivo": float(target[tb_column] - initial[tb_column]),
                    "semiturnos_entre_tb": semiturns_between,
                    "tb_anterior_2": (
                        float(previous_2[tb_column]) if previous_2 is not None else pd.NA
                    ),
                    "orden_tb_anterior_2": (
                        int(previous_2["ORDEN_SEMITURNO"])
                        if previous_2 is not None
                        else pd.NA
                    ),
                    "tb_anterior_3": (
                        float(previous_3[tb_column]) if previous_3 is not None else pd.NA
                    ),
                    "orden_tb_anterior_3": (
                        int(previous_3["ORDEN_SEMITURNO"])
                        if previous_3 is not None
                        else pd.NA
                    ),
                }
            )

    intervals = pd.DataFrame(rows)

    if intervals.empty:
        raise ValueError(
            "No se construyo ningun intervalo. Revisa filtros o distancia de ST."
        )

    return intervals


def add_window_counts(
    intervals: pd.DataFrame,
    df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """
    Cuenta cuantas filas operativas hay entre la TB inicial y la TB objetivo.

    Para distancia 8, esperamos normalmente 7 filas intermedias:
    - despues de la medicion inicial
    - antes de la medicion objetivo

    No incluimos el semiturno objetivo para evitar fuga de informacion.
    """
    cuba_column = config["columnas_clave"]["cuba"]
    tb_column = config["columnas_clave"]["tb_real"]
    intervals_with_counts = intervals.copy()
    intervals_with_counts["filas_ventana_operativa"] = 0
    intervals_with_counts["tb_reales_dentro_ventana"] = 0

    # Hacemos el conteo por cuba. Dentro de cada cuba usamos `searchsorted`,
    # que encuentra posiciones en arreglos ordenados de forma muy rapida.
    for cuba, cuba_rows in df.groupby(cuba_column, sort=False):
        interval_mask = intervals_with_counts["CUBA"] == cuba
        if not interval_mask.any():
            continue

        base_orders = cuba_rows["ORDEN_SEMITURNO"].astype("int64").to_numpy()
        tb_orders = (
            cuba_rows.loc[cuba_rows[tb_column].notna(), "ORDEN_SEMITURNO"]
            .astype("int64")
            .to_numpy()
        )

        initial_orders = (
            intervals_with_counts.loc[interval_mask, "orden_tb_inicial"]
            .astype("int64")
            .to_numpy()
        )
        target_orders = (
            intervals_with_counts.loc[interval_mask, "orden_tb_objetivo"]
            .astype("int64")
            .to_numpy()
        )

        # Posiciones de filas con orden > inicio y orden < objetivo.
        start_positions = np.searchsorted(base_orders, initial_orders, side="right")
        end_positions = np.searchsorted(base_orders, target_orders, side="left")
        window_row_counts = end_positions - start_positions

        # Mismo conteo, pero solo para filas donde TB real no es nula.
        tb_start_positions = np.searchsorted(tb_orders, initial_orders, side="right")
        tb_end_positions = np.searchsorted(tb_orders, target_orders, side="left")
        tb_counts = tb_end_positions - tb_start_positions

        intervals_with_counts.loc[
            interval_mask,
            "filas_ventana_operativa",
        ] = window_row_counts
        intervals_with_counts.loc[
            interval_mask,
            "tb_reales_dentro_ventana",
        ] = tb_counts

    return intervals_with_counts


def validate_intervals(
    intervals: pd.DataFrame,
    expected_semiturns: int,
) -> tuple[list[str], list[str]]:
    """
    Devuelve problemas criticos y advertencias.

    Problema critico: algo que invalida la tabla.
    Advertencia: algo que no invalida, pero conviene mirar.
    """
    issues: list[str] = []
    warnings: list[str] = []

    if intervals["id_intervalo"].duplicated().any():
        issues.append("Hay id_intervalo duplicados.")

    if intervals["tb_inicial"].isna().any():
        issues.append("Hay intervalos sin TB inicial.")

    if intervals["tb_objetivo"].isna().any():
        issues.append("Hay intervalos sin TB objetivo.")

    wrong_distance = intervals[
        intervals["semiturnos_entre_tb"] != expected_semiturns
    ]
    if len(wrong_distance) > 0:
        issues.append(
            f"Hay {len(wrong_distance):,} intervalos con distancia distinta "
            f"de {expected_semiturns} semiturnos."
        )

    windows_with_tb = intervals[intervals["tb_reales_dentro_ventana"] > 0]
    if len(windows_with_tb) > 0:
        issues.append(
            f"Hay {len(windows_with_tb):,} intervalos con otra TB real dentro "
            "de la ventana operativa."
        )

    short_windows = intervals[intervals["filas_ventana_operativa"] < 7]
    if len(short_windows) > 0:
        warnings.append(
            f"Hay {len(short_windows):,} intervalos con menos de 7 filas "
            "operativas intermedias. La distancia entre TB es 8 ST, pero algun "
            "semiturno intermedio no quedo en la base filtrada."
        )

    return issues, warnings


def write_interval_summary(
    intervals: pd.DataFrame,
    issues: list[str],
    warnings: list[str],
    output_path: Path,
) -> None:
    """Guarda un resumen textual de los intervalos construidos."""
    lines: list[str] = []
    lines.append("RESUMEN DE INTERVALOS TB - FASE 2")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Intervalos construidos: {len(intervals):,}")
    lines.append(f"Cubas con intervalos: {intervals['CUBA'].nunique():,}")
    lines.append(
        "Rango objetivo: "
        f"{intervals['fecha_tb_objetivo'].min().date()} a "
        f"{intervals['fecha_tb_objetivo'].max().date()}"
    )
    lines.append("")
    lines.append("Distribucion de TB inicial:")
    lines.append(str(intervals["tb_inicial"].describe()))
    lines.append("")
    lines.append("Distribucion de TB objetivo:")
    lines.append(str(intervals["tb_objetivo"].describe()))
    lines.append("")
    lines.append("Distribucion de delta TB objetivo:")
    lines.append(str(intervals["delta_tb_objetivo"].describe()))
    lines.append("")
    lines.append("Filas operativas disponibles entre mediciones:")
    lines.append(str(intervals["filas_ventana_operativa"].value_counts().sort_index()))
    lines.append("")
    lines.append("Controles de calidad:")
    if issues:
        for issue in issues:
            lines.append(f"- PROBLEMA: {issue}")
    else:
        lines.append("- OK: no se encontraron problemas criticos.")
    if warnings:
        for warning in warnings:
            lines.append(f"- ADVERTENCIA: {warning}")
    lines.append("")
    lines.append("Nota metodologica:")
    lines.append(
        "La ventana operativa excluye el semiturno de la TB objetivo para evitar "
        "usar informacion simultanea o posterior a la medicion que queremos "
        "predecir."
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_validation_doc(
    intervals: pd.DataFrame,
    issues: list[str],
    warnings: list[str],
    output_path: Path,
) -> None:
    """Guarda una explicacion de calidad de los intervalos."""
    most_common_windows = intervals["filas_ventana_operativa"].value_counts()
    most_common_window_text = ", ".join(
        f"{index} filas: {value:,}"
        for index, value in most_common_windows.sort_index().items()
    )

    lines = [
        "# Validacion de intervalos TB - Fase 2",
        "",
        "Cada fila de `intervalos_tb_8st.parquet` representa una medicion real "
        "de TB inicial y la siguiente medicion real de TB exactamente 8 "
        "semiturnos despues.",
        "",
        "## Controles realizados",
        "",
        "- `tb_inicial` no debe estar vacia.",
        "- `tb_objetivo` no debe estar vacia.",
        "- `semiturnos_entre_tb` debe ser siempre 8.",
        "- No debe haber otra TB real dentro de la ventana operativa intermedia.",
        "- `id_intervalo` no debe repetirse.",
        "",
        "## Resultado",
        "",
        f"- Intervalos generados: {len(intervals):,}.",
        f"- Cubas incluidas: {intervals['CUBA'].nunique():,}.",
        f"- Filas operativas intermedias por intervalo: {most_common_window_text}.",
        "",
    ]

    if issues:
        lines.append("## Problemas detectados")
        lines.append("")
        for issue in issues:
            lines.append(f"- {issue}")
    else:
        lines.append("## Problemas detectados")
        lines.append("")
        lines.append("No se encontraron problemas criticos en los controles de Fase 2.")

    lines.append("")
    lines.append("## Advertencias")
    lines.append("")
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("No se encontraron advertencias relevantes.")

    lines.extend(
        [
            "",
            "## Decision documentada",
            "",
            "Los intervalos distintos de 8 semiturnos quedan fuera de esta primera "
            "version. Mas adelante se puede comparar si conviene incluir "
            "mediciones adelantadas o retrasadas con features que indiquen la "
            "duracion real del intervalo.",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Ejecuta la Fase 2 completa."""
    print_title("FASE 2 - CONSTRUCCION DE INTERVALOS TB")
    config = load_config()
    expected_semiturns = config["modelado"]["semiturnos_entre_mediciones_tb"]

    print_title("1. Cargando y filtrando base")
    df = load_and_prepare_base(config)
    print(f"Filas filtradas: {len(df):,}")
    print(f"Cubas filtradas: {df['CUBA'].nunique():,}")

    print_title("2. Construyendo intervalos de TB")
    intervals = build_tb_intervals(df, config)
    intervals = add_window_counts(intervals, df, config)
    intervals = intervals.sort_values(
        ["CUBA", "orden_tb_inicial"],
        kind="mergesort",
    ).reset_index(drop=True)

    print(f"Intervalos de {expected_semiturns} ST: {len(intervals):,}")

    print_title("3. Validando intervalos")
    issues, warnings = validate_intervals(intervals, expected_semiturns)
    if issues:
        for issue in issues:
            print(f"PROBLEMA: {issue}")
    else:
        print("OK: no se encontraron problemas criticos.")
    if warnings:
        for warning in warnings:
            print(f"ADVERTENCIA: {warning}")

    print_title("4. Guardando salidas")
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    DATA_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    intervals_path = DATA_PROCESSED_DIR / "intervalos_tb_8st.parquet"
    sample_path = DATA_SAMPLE_DIR / "muestra_intervalos_tb_8st.csv"
    summary_path = OUTPUTS_DIR / "resumen_intervalos_tb_8st.txt"
    validation_path = DOCS_DIR / "VALIDACION_INTERVALOS.md"

    intervals.to_parquet(intervals_path, index=False)
    intervals.head(500).to_csv(sample_path, index=False, encoding="utf-8-sig")
    write_interval_summary(intervals, issues, warnings, summary_path)
    write_validation_doc(intervals, issues, warnings, validation_path)

    print("Archivos generados:")
    print(f"- {intervals_path}")
    print(f"- {sample_path}")
    print(f"- {summary_path}")
    print(f"- {validation_path}")


if __name__ == "__main__":
    main()
