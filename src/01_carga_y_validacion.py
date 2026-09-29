"""
OBJETIVO DEL ARCHIVO
--------------------
Inspeccionar la fuente VITM y el auxiliar de cubas antes de construir el
dataset de machine learning.

ENTRADAS
--------
- Parquet VITM con datos por cuba y semiturno.
- CSV auxiliar con la relacion Cuba -> Grupo -> Sala.
- Archivo de configuracion `config/parametros.yaml`.

SALIDAS
-------
- `docs/DICCIONARIO_DATOS.md`
- `outputs/resumen_datos_inicial.txt`
- `outputs/muestra_datos_filtrados.csv`

POR QUE EXISTE
--------------
Antes de entrenar cualquier modelo necesitamos entender que columnas existen,
como vienen escritas, cuantos datos faltan y si las reglas operativas acordadas
se pueden aplicar sin romper la base.

Este script esta escrito de forma deliberadamente clara. La prioridad es que
Diego pueda leerlo, modificarlo y aprender que hace cada paso.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError as exc:  # pragma: no cover - solo se usa si falta PyYAML
    raise SystemExit(
        "Falta instalar PyYAML. Ejecuta: pip install -r requirements.txt"
    ) from exc


# ============================================================
# FASE 0 - RUTAS DEL PROYECTO
# ============================================================
#
# PROJECT_DIR es la carpeta `prediccion_proxima_tb`.
# Usamos rutas absolutas internamente para evitar errores cuando el script se
# ejecuta desde otra ubicacion.

PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_DIR / "config" / "parametros.yaml"
DOCS_DIR = PROJECT_DIR / "docs"
OUTPUTS_DIR = PROJECT_DIR / "outputs"


def print_title(title: str) -> None:
    """Imprime un titulo visible en consola."""
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def load_config() -> dict[str, Any]:
    """Carga la configuracion central del proyecto."""
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config


def resolve_project_path(path_text: str) -> Path:
    """
    Convierte una ruta de configuracion en ruta absoluta.

    Las rutas del YAML estan escritas relativas a `prediccion_proxima_tb`.
    """
    return (PROJECT_DIR / path_text).resolve()


def convert_decimal_comma_to_number(series: pd.Series) -> pd.Series:
    """
    Convierte textos numericos con coma decimal a numeros reales.

    En la base aparecen valores como "956,0". Pandas no los entiende como
    numeros hasta reemplazar la coma por punto.
    """
    return pd.to_numeric(
        series.astype("string").str.replace(",", ".", regex=False),
        errors="coerce",
    )


def is_mostly_numeric(series: pd.Series, minimum_ratio: float = 0.80) -> bool:
    """
    Decide si una columna object parece numerica.

    Esto sirve para crear un diccionario de datos inicial. No modifica la base
    original; solo ayuda a describirla.
    """
    non_null = series.notna().sum()
    if non_null == 0:
        return False

    converted = convert_decimal_comma_to_number(series)
    numeric_ratio = converted.notna().sum() / non_null
    return numeric_ratio >= minimum_ratio


def add_operational_time_columns(
    df: pd.DataFrame,
    date_column: str,
    semiturn_column: str,
) -> pd.DataFrame:
    """
    Agrega columnas temporales limpias.

    `FT_TURNO` viene como fecha y `TU_SEMI_TURNO` indica el orden dentro del
    dia. No inventamos una hora exacta; trabajamos con esa convencion.
    """
    result = df.copy()
    result["FECHA_OPERATIVA"] = pd.to_datetime(
        result[date_column],
        dayfirst=True,
        errors="coerce",
    )

    result["ORDEN_SEMITURNO"] = (
        (result["FECHA_OPERATIVA"] - result["FECHA_OPERATIVA"].min()).dt.days * 6
        + result[semiturn_column].astype("Int64")
    )
    return result


def load_auxiliary_cubas(path: Path) -> pd.DataFrame:
    """Carga el archivo auxiliar Cuba -> Grupo -> Sala."""
    auxiliary = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    expected_columns = {"Cuba", "Grupo", "Sala"}
    missing_columns = expected_columns - set(auxiliary.columns)

    if missing_columns:
        raise ValueError(
            "El auxiliar de cubas no tiene las columnas esperadas: "
            + ", ".join(sorted(missing_columns))
        )

    auxiliary = auxiliary.rename(
        columns={
            "Cuba": "CUBA",
            "Grupo": "GRUPO",
            "Sala": "SALA",
        }
    )
    return auxiliary


def create_initial_column_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea una tabla con informacion basica por columna.

    Este diccionario no reemplaza el conocimiento operativo, pero ayuda a ver
    que columnas existen, que tan completas estan y si parecen numericas.
    """
    rows: list[dict[str, Any]] = []

    for column in df.columns:
        series = df[column]
        non_null_count = int(series.notna().sum())
        null_count = int(series.isna().sum())
        null_percentage = float(series.isna().mean() * 100)

        if pd.api.types.is_numeric_dtype(series):
            interpreted_type = "numerica"
        elif column == "FECHA_OPERATIVA":
            interpreted_type = "fecha"
        elif is_mostly_numeric(series):
            interpreted_type = "texto numerico con coma decimal"
        else:
            interpreted_type = "categorica/texto"

        example_values = (
            series.dropna()
            .astype(str)
            .drop_duplicates()
            .head(5)
            .tolist()
        )

        rows.append(
            {
                "columna": column,
                "tipo_pandas": str(series.dtype),
                "tipo_interpretado": interpreted_type,
                "no_nulos": non_null_count,
                "nulos": null_count,
                "porcentaje_nulos": round(null_percentage, 2),
                "valores_unicos": int(series.nunique(dropna=True)),
                "ejemplos": ", ".join(example_values),
            }
        )

    return pd.DataFrame(rows)


def describe_numeric_column(series: pd.Series) -> dict[str, Any]:
    """Resume una columna numerica ya convertida."""
    clean = series.dropna()

    if clean.empty:
        return {
            "no_nulos": 0,
            "media": np.nan,
            "min": np.nan,
            "p01": np.nan,
            "p50": np.nan,
            "p99": np.nan,
            "max": np.nan,
        }

    return {
        "no_nulos": int(clean.shape[0]),
        "media": round(float(clean.mean()), 4),
        "min": round(float(clean.min()), 4),
        "p01": round(float(clean.quantile(0.01)), 4),
        "p50": round(float(clean.quantile(0.50)), 4),
        "p99": round(float(clean.quantile(0.99)), 4),
        "max": round(float(clean.max()), 4),
    }


def analyze_tb_intervals(
    df: pd.DataFrame,
    cuba_column: str,
    tb_column: str,
    expected_semiturns: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calcula la distancia entre mediciones reales consecutivas de TB.

    No construye todavia el dataset final. Solo mide si la regla de 8
    semiturnos tiene suficiente soporte en los datos.
    """
    measured_tb = df[df[tb_column].notna()].copy()
    measured_tb = measured_tb.sort_values(
        [cuba_column, "ORDEN_SEMITURNO"],
        kind="mergesort",
    )

    measured_tb["SEMITURNOS_DESDE_TB_ANTERIOR"] = measured_tb.groupby(cuba_column)[
        "ORDEN_SEMITURNO"
    ].diff()

    interval_counts = (
        measured_tb["SEMITURNOS_DESDE_TB_ANTERIOR"]
        .value_counts(dropna=False)
        .rename_axis("semiturnos")
        .reset_index(name="cantidad")
        .sort_values("cantidad", ascending=False)
    )

    expected_intervals = measured_tb[
        measured_tb["SEMITURNOS_DESDE_TB_ANTERIOR"] == expected_semiturns
    ].copy()

    return interval_counts, expected_intervals


def write_dictionary_markdown(dictionary: pd.DataFrame, output_path: Path) -> None:
    """Guarda el diccionario de datos en formato Markdown."""
    def clean_markdown_cell(value: Any) -> str:
        """Evita que caracteres especiales rompan la tabla Markdown."""
        text = "" if pd.isna(value) else str(value)
        return text.replace("|", "/").replace("\n", " ")

    headers = list(dictionary.columns)
    table_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for _, row in dictionary.iterrows():
        table_lines.append(
            "| "
            + " | ".join(clean_markdown_cell(row[column]) for column in headers)
            + " |"
        )

    lines: list[str] = []
    lines.append("# Diccionario de datos inicial\n")
    lines.append(
        "Este archivo fue generado automaticamente por "
        "`src/01_carga_y_validacion.py`.\n"
    )
    lines.append(
        "Todavia es un diccionario tecnico inicial: describe columnas, tipos, "
        "nulos y ejemplos. En fases siguientes se completara con significado "
        "operativo validado.\n"
    )
    lines.extend(table_lines)
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_initial_summary(
    output_path: Path,
    lines: list[str],
) -> None:
    """Guarda el resumen inicial en texto plano."""
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_quality_analysis(
    output_path: Path,
    *,
    raw_rows: int,
    raw_columns: int,
    filtered_rows: int,
    filtered_cubas: int,
    invalid_dates: int,
    key_duplicate_count: int,
    missing_auxiliary_rows_raw: int,
    missing_auxiliary_cubas_raw: int,
    missing_auxiliary_rows_filtered: int,
    missing_auxiliary_cubas_filtered: int,
    expected_semiturns: int,
    expected_interval_count: int,
    total_interval_count: int,
) -> None:
    """Guarda un analisis de calidad pensado para lectura humana."""
    eligible_ratio = (
        expected_interval_count / total_interval_count * 100
        if total_interval_count > 0
        else 0
    )

    lines = [
        "# Analisis de calidad de datos - Fase 1",
        "",
        "Este documento resume los controles hechos antes de construir features o "
        "entrenar modelos.",
        "",
        "## Resultado general",
        "",
        f"- Filas crudas inspeccionadas: {raw_rows:,}.",
        f"- Columnas crudas inspeccionadas: {raw_columns:,}.",
        f"- Filas luego de filtros iniciales: {filtered_rows:,}.",
        f"- Cubas luego de filtros iniciales: {filtered_cubas:,}.",
        "",
        "## Controles de estructura",
        "",
        f"- Fechas invalidas: {invalid_dates:,}.",
        f"- Duplicados por `CUBA + FT_TURNO + TU_SEMI_TURNO`: "
        f"{key_duplicate_count:,}.",
        f"- Filas crudas sin match en auxiliar de cubas: "
        f"{missing_auxiliary_rows_raw:,}.",
        f"- Cubas crudas sin match en auxiliar de cubas: "
        f"{missing_auxiliary_cubas_raw:,}.",
        f"- Filas filtradas sin match en auxiliar de cubas: "
        f"{missing_auxiliary_rows_filtered:,}.",
        f"- Cubas filtradas sin match en auxiliar de cubas: "
        f"{missing_auxiliary_cubas_filtered:,}.",
        "",
        "Interpretacion: las 96 cubas sin auxiliar aparecen en la base cruda, "
        "pero desaparecen al aplicar los filtros confirmados. Esto coincide con "
        "la explicacion de que son cubas de expansion o no operativas.",
        "",
        "## Control de intervalos de TB",
        "",
        f"- Regla inicial documentada: usar solo intervalos de "
        f"{expected_semiturns} semiturnos.",
        f"- Intervalos totales entre TB reales luego de filtros: "
        f"{total_interval_count:,}.",
        f"- Intervalos elegibles para V1: {expected_interval_count:,} "
        f"({eligible_ratio:.1f}%).",
        "",
        "Interpretacion: hay una cantidad suficiente de casos de 8 semiturnos "
        "para arrancar. Los intervalos adelantados o retrasados quedan fuera de "
        "la primera version para reducir ruido metodologico.",
        "",
        "## Diferencias contra la documentacion previa",
        "",
        "- El Parquet actual si contiene `TB`, `TBD`, `ALF3` y `ALF3D`.",
        "- `SALA` y `GRUPO` no vienen en el Parquet; se agregan desde "
        "`auxiliar_cubas.csv`.",
        "- No hay columna explicita de `MODELO` ni de `SERIE` en el Parquet.",
        "- Varias columnas numericas vienen como texto con coma decimal, por "
        "ejemplo `956,0`; el codigo debe convertirlas antes de calcular.",
        "- `FT_TURNO` debe parsearse con formato dia/mes/anio. Si se interpreta "
        "como mes/dia/anio, las fechas quedan mal.",
        "",
        "## Riesgos a vigilar en la Fase 2",
        "",
        "- No incluir el semiturno objetivo dentro de los acumulados si sus datos "
        "no estarian disponibles antes de predecir.",
        "- No usar `TBD` como target. El target debe ser siempre `TB` real.",
        "- No construir features con mediciones reales posteriores al inicio del "
        "intervalo.",
        "- Documentar cuantos casos se pierden por exigir exactamente 8 "
        "semiturnos.",
    ]

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Ejecuta la Fase 1 completa."""
    print_title("FASE 1 - CARGA Y VALIDACION INICIAL")

    config = load_config()

    parquet_path = resolve_project_path(config["rutas"]["parquet_vitm"])
    auxiliary_path = resolve_project_path(config["rutas"]["auxiliar_cubas"])

    key_columns = config["columnas_clave"]
    cuba_column = key_columns["cuba"]
    date_column = key_columns["fecha"]
    semiturn_column = key_columns["semiturno"]
    potstate_column = key_columns["estado_operativo"]
    mother_pot_column = key_columns["cuba_madre"]
    tb_column = key_columns["tb_real"]
    dragged_tb_column = key_columns["tb_arrastrada"]
    alf3_column = key_columns["alf3_real"]
    dragged_alf3_column = key_columns["alf3_arrastrada"]

    expected_potstate = config["filtros_iniciales"]["potstate_operacion_normal"]
    expected_mother_value = config["filtros_iniciales"][
        "valor_cuba_madre_a_conservar"
    ]
    expected_semiturns = config["modelado"]["semiturnos_entre_mediciones_tb"]

    if not parquet_path.exists():
        raise FileNotFoundError(f"No existe el parquet VITM: {parquet_path}")
    if not auxiliary_path.exists():
        raise FileNotFoundError(f"No existe el auxiliar de cubas: {auxiliary_path}")

    print(f"Parquet VITM: {parquet_path}")
    print(f"Auxiliar cubas: {auxiliary_path}")

    # ============================================================
    # FASE 1 - CARGAR DATOS CRUDOS
    # ============================================================
    #
    # En este punto no filtramos ni corregimos nada. Primero leemos la base tal
    # como viene para poder medir su calidad.

    print_title("1. Cargando datos crudos")
    df_raw = pd.read_parquet(parquet_path)
    auxiliary_cubas = load_auxiliary_cubas(auxiliary_path)

    print(f"Filas crudas VITM: {len(df_raw):,}")
    print(f"Columnas crudas VITM: {len(df_raw.columns):,}")
    print(f"Cubas en auxiliar: {len(auxiliary_cubas):,}")

    # ============================================================
    # FASE 2 - CONVERTIR TIPOS MINIMOS
    # ============================================================
    #
    # La base trae muchas columnas como texto con coma decimal. Creamos
    # versiones convertidas solo para las columnas que necesitamos medir y
    # filtrar ahora.

    print_title("2. Preparando fechas, numeros y auxiliar")
    df = add_operational_time_columns(df_raw, date_column, semiturn_column)

    df["POTSTATE_NUM"] = convert_decimal_comma_to_number(df[potstate_column])
    df["TB_NUM"] = convert_decimal_comma_to_number(df[tb_column])
    df["TBD_NUM"] = convert_decimal_comma_to_number(df[dragged_tb_column])
    df["ALF3_NUM"] = convert_decimal_comma_to_number(df[alf3_column])
    df["ALF3D_NUM"] = convert_decimal_comma_to_number(df[dragged_alf3_column])

    df = df.merge(auxiliary_cubas, on=cuba_column, how="left")

    # A partir de aca usamos las columnas numericas limpias para medir.
    # No borramos las columnas originales porque sirven para auditoria.
    df[tb_column] = df["TB_NUM"]
    df[dragged_tb_column] = df["TBD_NUM"]
    df[alf3_column] = df["ALF3_NUM"]
    df[dragged_alf3_column] = df["ALF3D_NUM"]

    # ============================================================
    # FASE 3 - APLICAR FILTROS ACORDADOS
    # ============================================================
    #
    # Estos filtros fueron confirmados antes de construir:
    # - POTSTATE = 3
    # - M_CUBA_MADRE = N

    print_title("3. Aplicando filtros acordados")
    mask_potstate = df["POTSTATE_NUM"] == expected_potstate
    mask_not_mother = df[mother_pot_column] == expected_mother_value
    df_filtered = df[mask_potstate & mask_not_mother].copy()
    df_filtered = df_filtered.sort_values(
        [cuba_column, "FECHA_OPERATIVA", semiturn_column],
        kind="mergesort",
    )

    print(f"Filas luego de POTSTATE = {expected_potstate}: {mask_potstate.sum():,}")
    print(
        f"Filas luego de {mother_pot_column} = {expected_mother_value}: "
        f"{mask_not_mother.sum():,}"
    )
    print(f"Filas luego de ambos filtros: {len(df_filtered):,}")
    print(f"Cubas luego de ambos filtros: {df_filtered[cuba_column].nunique():,}")

    # ============================================================
    # FASE 4 - REVISION DE CALIDAD
    # ============================================================
    #
    # Buscamos problemas antes de crear features: fechas invalidas, claves
    # duplicadas, cubas sin auxiliar, variables objetivo faltantes y frecuencias
    # reales de medicion.

    print_title("4. Analizando calidad inicial")

    key_duplicate_count = int(
        df.duplicated([cuba_column, date_column, semiturn_column]).sum()
    )
    invalid_dates = int(df["FECHA_OPERATIVA"].isna().sum())
    missing_auxiliary_rows = int(df["SALA"].isna().sum())
    missing_auxiliary_cubas = int(
        df.loc[df["SALA"].isna(), cuba_column].nunique(dropna=True)
    )
    missing_auxiliary_rows_filtered = int(df_filtered["SALA"].isna().sum())
    missing_auxiliary_cubas_filtered = int(
        df_filtered.loc[df_filtered["SALA"].isna(), cuba_column].nunique(dropna=True)
    )

    interval_counts, expected_intervals = analyze_tb_intervals(
        df_filtered,
        cuba_column=cuba_column,
        tb_column=tb_column,
        expected_semiturns=expected_semiturns,
    )

    tb_summary = describe_numeric_column(df_filtered[tb_column])
    dragged_tb_summary = describe_numeric_column(df_filtered[dragged_tb_column])
    alf3_summary = describe_numeric_column(df_filtered[alf3_column])
    dragged_alf3_summary = describe_numeric_column(df_filtered[dragged_alf3_column])

    # ============================================================
    # FASE 5 - GUARDAR SALIDAS
    # ============================================================
    #
    # Guardamos archivos que se puedan leer sin abrir Python.

    print_title("5. Guardando diccionario, resumen y muestra")
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    dictionary = create_initial_column_dictionary(df)
    write_dictionary_markdown(dictionary, DOCS_DIR / "DICCIONARIO_DATOS.md")

    sample_columns = [
        cuba_column,
        "FECHA_OPERATIVA",
        semiturn_column,
        "ORDEN_SEMITURNO",
        potstate_column,
        "POTSTATE_NUM",
        mother_pot_column,
        "SALA",
        "GRUPO",
        tb_column,
        dragged_tb_column,
        alf3_column,
        dragged_alf3_column,
    ]
    sample_columns = [column for column in sample_columns if column in df_filtered.columns]
    df_filtered[sample_columns].head(300).to_csv(
        OUTPUTS_DIR / "muestra_datos_filtrados.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_lines: list[str] = []
    summary_lines.append("RESUMEN DE DATOS INICIAL - FASE 1")
    summary_lines.append("=" * 60)
    summary_lines.append("")
    summary_lines.append("1) Archivos inspeccionados")
    summary_lines.append(f"- Parquet VITM: {parquet_path}")
    summary_lines.append(f"- Auxiliar cubas: {auxiliary_path}")
    summary_lines.append("")
    summary_lines.append("2) Tamano general")
    summary_lines.append(f"- Filas crudas: {len(df_raw):,}")
    summary_lines.append(f"- Columnas crudas originales: {len(df_raw.columns):,}")
    summary_lines.append(
        f"- Columnas disponibles luego de auxiliares y columnas de control: "
        f"{len(df.columns):,}"
    )
    summary_lines.append(f"- Cubas crudas: {df[cuba_column].nunique():,}")
    summary_lines.append(
        f"- Rango fecha: {df['FECHA_OPERATIVA'].min().date()} "
        f"a {df['FECHA_OPERATIVA'].max().date()}"
    )
    summary_lines.append("")
    summary_lines.append("3) Filtros confirmados")
    summary_lines.append(f"- POTSTATE = {expected_potstate}")
    summary_lines.append(f"- {mother_pot_column} = {expected_mother_value}")
    summary_lines.append("- Sin filtro de edad por ahora")
    summary_lines.append(f"- Filas filtradas: {len(df_filtered):,}")
    summary_lines.append(f"- Cubas filtradas: {df_filtered[cuba_column].nunique():,}")
    summary_lines.append("")
    summary_lines.append("4) Columnas principales identificadas")
    summary_lines.append(f"- TB real: {tb_column}")
    summary_lines.append(f"- TB arrastrada: {dragged_tb_column}")
    summary_lines.append(f"- AlF3 real: {alf3_column}")
    summary_lines.append(f"- AlF3 arrastrada: {dragged_alf3_column}")
    summary_lines.append(f"- Fecha: {date_column}")
    summary_lines.append(f"- Semiturno: {semiturn_column}")
    summary_lines.append(f"- Cuba: {cuba_column}")
    summary_lines.append("- Sala: SALA, desde auxiliar_cubas.csv")
    summary_lines.append("- Grupo: GRUPO, desde auxiliar_cubas.csv")
    summary_lines.append("- Modelo: no disponible en el Parquet")
    summary_lines.append("- Serie: no disponible en el Parquet")
    summary_lines.append("")
    summary_lines.append("5) Cobertura de variables clave luego de filtros")
    summary_lines.append(f"- TB real no nula: {df_filtered[tb_column].notna().sum():,}")
    summary_lines.append(f"- TBD no nula: {df_filtered[dragged_tb_column].notna().sum():,}")
    summary_lines.append(f"- AlF3 real no nula: {df_filtered[alf3_column].notna().sum():,}")
    summary_lines.append(
        f"- AlF3D no nula: {df_filtered[dragged_alf3_column].notna().sum():,}"
    )
    summary_lines.append("")
    summary_lines.append("6) Resumen numerico de variables clave luego de filtros")
    for name, stats in [
        ("TB", tb_summary),
        ("TBD", dragged_tb_summary),
        ("ALF3", alf3_summary),
        ("ALF3D", dragged_alf3_summary),
    ]:
        summary_lines.append(f"- {name}: {stats}")
    summary_lines.append("")
    summary_lines.append("7) Intervalos entre mediciones reales de TB")
    summary_lines.append(
        f"- Regla documentada para V1: usar solo {expected_semiturns} semiturnos"
    )
    summary_lines.append(
        f"- Intervalos elegibles de {expected_semiturns} semiturnos: "
        f"{len(expected_intervals):,}"
    )
    summary_lines.append("- Frecuencias principales:")
    for _, row in interval_counts.head(15).iterrows():
        summary_lines.append(f"  {row['semiturnos']}: {row['cantidad']:,}")
    summary_lines.append("")
    summary_lines.append("8) Analisis de calidad")
    summary_lines.append(f"- Fechas invalidas: {invalid_dates:,}")
    summary_lines.append(
        f"- Duplicados por CUBA + FT_TURNO + TU_SEMI_TURNO: {key_duplicate_count:,}"
    )
    summary_lines.append(
        f"- Filas crudas sin match en auxiliar: {missing_auxiliary_rows:,}"
    )
    summary_lines.append(
        f"- Cubas crudas sin match en auxiliar: {missing_auxiliary_cubas:,}"
    )
    summary_lines.append(
        f"- Filas filtradas sin match en auxiliar: {missing_auxiliary_rows_filtered:,}"
    )
    summary_lines.append(
        f"- Cubas filtradas sin match en auxiliar: {missing_auxiliary_cubas_filtered:,}"
    )
    summary_lines.append(
        "- Advertencia: hay cubas de expansion o no operativas en la base cruda. "
        "No deberian afectar si se filtra por POTSTATE=3 y por auxiliar cuando "
        "corresponda."
    )
    summary_lines.append(
        "- Advertencia: los intervalos distintos de 8 semiturnos quedan fuera de "
        "la primera version. Esto reduce ruido, pero tambien descarta casos "
        "adelantados o retrasados."
    )
    summary_lines.append(
        "- Advertencia: algunas variables vienen como texto con coma decimal. "
        "Toda fase posterior debe convertirlas antes de calcular."
    )
    summary_lines.append("")
    summary_lines.append("9) Proximo paso propuesto")
    summary_lines.append(
        "Construir intervalos historicos de modelado. Cada fila debera tener una "
        "TB inicial real, una TB objetivo real exactamente 8 semiturnos despues, "
        "y solo informacion disponible entre ambas mediciones sin incluir datos "
        "futuros."
    )

    write_initial_summary(OUTPUTS_DIR / "resumen_datos_inicial.txt", summary_lines)
    write_quality_analysis(
        DOCS_DIR / "ANALISIS_CALIDAD_DATOS.md",
        raw_rows=len(df_raw),
        raw_columns=len(df_raw.columns),
        filtered_rows=len(df_filtered),
        filtered_cubas=df_filtered[cuba_column].nunique(),
        invalid_dates=invalid_dates,
        key_duplicate_count=key_duplicate_count,
        missing_auxiliary_rows_raw=missing_auxiliary_rows,
        missing_auxiliary_cubas_raw=missing_auxiliary_cubas,
        missing_auxiliary_rows_filtered=missing_auxiliary_rows_filtered,
        missing_auxiliary_cubas_filtered=missing_auxiliary_cubas_filtered,
        expected_semiturns=expected_semiturns,
        expected_interval_count=len(expected_intervals),
        total_interval_count=int(
            interval_counts.loc[
                interval_counts["semiturnos"].notna(),
                "cantidad",
            ].sum()
        ),
    )

    print("Archivos generados:")
    print(f"- {DOCS_DIR / 'DICCIONARIO_DATOS.md'}")
    print(f"- {DOCS_DIR / 'ANALISIS_CALIDAD_DATOS.md'}")
    print(f"- {OUTPUTS_DIR / 'resumen_datos_inicial.txt'}")
    print(f"- {OUTPUTS_DIR / 'muestra_datos_filtrados.csv'}")


if __name__ == "__main__":
    main()
