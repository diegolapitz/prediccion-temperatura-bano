"""
OBJETIVO DEL ARCHIVO
--------------------
Calcular baselines antes de entrenar modelos de machine learning.

ENTRADAS
--------
- `data/processed/dataset_modelado_v1.parquet`
- Archivo `config/parametros.yaml`

SALIDAS
-------
- `outputs/metricas/baselines_metricas.csv`
- `outputs/predicciones/baselines_predicciones_test.csv`
- `outputs/resumen_baselines.txt`
- `docs/VALIDACION_BASELINES.md`

POR QUE EXISTE
--------------
Un modelo solo tiene valor si mejora reglas simples. Antes de entrenar Random
Forest, Gradient Boosting o cualquier otro algoritmo, necesitamos saber cuanto
error tienen reglas faciles de explicar.
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
DOCS_DIR = PROJECT_DIR / "docs"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metricas"
PREDICTIONS_DIR = OUTPUTS_DIR / "predicciones"


def print_title(title: str) -> None:
    """Imprime un titulo visible en consola."""
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def load_config() -> dict[str, Any]:
    """Carga la configuracion central."""
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def split_by_time(dataset: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """
    Asigna cada fila a train, validation o test segun fecha objetivo.

    No usamos split aleatorio porque en operacion real siempre usaremos datos
    pasados para predecir datos futuros.
    """
    result = dataset.copy()
    validation_start = pd.Timestamp(
        config["validacion_temporal"]["inicio_validacion"]
    )
    test_start = pd.Timestamp(config["validacion_temporal"]["inicio_test"])

    result["fecha_tb_objetivo"] = pd.to_datetime(result["fecha_tb_objetivo"])

    result["segmento_temporal"] = "train"
    result.loc[
        result["fecha_tb_objetivo"] >= validation_start,
        "segmento_temporal",
    ] = "validacion"
    result.loc[
        result["fecha_tb_objetivo"] >= test_start,
        "segmento_temporal",
    ] = "test"

    return result


def add_baseline_predictions(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega predicciones de baselines.

    Baseline 1:
        Proxima TB = ultima TB real.

    Baseline 2:
        Proxima TB = promedio de las ultimas dos TB reales disponibles.

    Baseline 3:
        Extrapolacion simple:
        proxima TB = ultima TB + (ultima TB - TB anterior).
    """
    result = dataset.copy()

    result["pred_baseline_ultima_tb"] = result["tb_inicial"]

    result["pred_baseline_promedio_2_tb"] = result[
        ["tb_inicial", "tb_anterior_2"]
    ].mean(axis=1)

    result["pred_baseline_extrapolacion_lineal"] = (
        result["tb_inicial"]
        + (result["tb_inicial"] - result["tb_anterior_2"])
    )

    # Si falta `tb_anterior_2`, volvemos al baseline mas simple.
    result["pred_baseline_extrapolacion_lineal"] = result[
        "pred_baseline_extrapolacion_lineal"
    ].fillna(result["pred_baseline_ultima_tb"])

    return result


def calculate_metrics(
    real: pd.Series,
    predicted: pd.Series,
) -> dict[str, float]:
    """
    Calcula metricas tecnicas de error.

    Error = prediccion - real.
    Sesgo positivo significa que el baseline tiende a predecir mas caliente que
    lo observado.
    """
    error = predicted - real
    absolute_error = error.abs()

    return {
        "mae": float(absolute_error.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "sesgo": float(error.mean()),
        "error_mediano": float(absolute_error.median()),
        "p75_error_abs": float(absolute_error.quantile(0.75)),
        "p90_error_abs": float(absolute_error.quantile(0.90)),
        "p95_error_abs": float(absolute_error.quantile(0.95)),
        "porcentaje_dentro_3c": float((absolute_error <= 3).mean() * 100),
        "porcentaje_dentro_5c": float((absolute_error <= 5).mean() * 100),
        "porcentaje_dentro_10c": float((absolute_error <= 10).mean() * 100),
    }


def evaluate_baselines(dataset: pd.DataFrame) -> pd.DataFrame:
    """Calcula metricas por baseline y segmento temporal."""
    baseline_columns = {
        "ultima_tb": "pred_baseline_ultima_tb",
        "promedio_2_tb": "pred_baseline_promedio_2_tb",
        "extrapolacion_lineal": "pred_baseline_extrapolacion_lineal",
    }

    rows: list[dict[str, Any]] = []

    for segment, segment_data in dataset.groupby("segmento_temporal", sort=False):
        for baseline_name, prediction_column in baseline_columns.items():
            metrics = calculate_metrics(
                real=segment_data["tb_objetivo"],
                predicted=segment_data[prediction_column],
            )
            rows.append(
                {
                    "segmento_temporal": segment,
                    "baseline": baseline_name,
                    "filas": len(segment_data),
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def write_summary(
    dataset: pd.DataFrame,
    metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Guarda un resumen legible de baselines."""
    test_metrics = metrics[metrics["segmento_temporal"] == "test"].copy()
    best_test = test_metrics.sort_values("mae").iloc[0]

    lines = [
        "RESUMEN DE BASELINES - FASE 4",
        "=" * 60,
        "",
        "Cortes temporales usados:",
    ]

    for segment in ["train", "validacion", "test"]:
        segment_data = dataset[dataset["segmento_temporal"] == segment]
        lines.append(
            f"- {segment}: {len(segment_data):,} filas, "
            f"{segment_data['fecha_tb_objetivo'].min().date()} a "
            f"{segment_data['fecha_tb_objetivo'].max().date()}"
        )

    lines.extend(
        [
            "",
            "Mejor baseline en test:",
            f"- {best_test['baseline']}",
            f"- MAE: {best_test['mae']:.3f} C",
            f"- RMSE: {best_test['rmse']:.3f} C",
            f"- Sesgo: {best_test['sesgo']:.3f} C",
            "",
            "Metricas completas por segmento:",
            metrics.round(4).to_string(index=False),
            "",
            "Interpretacion:",
            "El futuro modelo de ML debera mejorar estos numeros en test, no "
            "solo en train. Si mejora train pero empeora test, seria una senal "
            "de sobreajuste o leakage.",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_validation_doc(
    dataset: pd.DataFrame,
    metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Documenta como se validaron los baselines."""
    lines = [
        "# Validacion de baselines - Fase 4",
        "",
        "Los baselines se evaluaron con division temporal, no aleatoria.",
        "",
        "## Baselines calculados",
        "",
        "- `ultima_tb`: predice que la proxima TB sera igual a la TB inicial.",
        "- `promedio_2_tb`: usa el promedio entre la TB inicial y la TB anterior.",
        "- `extrapolacion_lineal`: extiende el ultimo cambio observado.",
        "",
        "## Segmentos temporales",
        "",
    ]

    for segment in ["train", "validacion", "test"]:
        segment_data = dataset[dataset["segmento_temporal"] == segment]
        lines.append(
            f"- `{segment}`: {len(segment_data):,} filas "
            f"({segment_data['fecha_tb_objetivo'].min().date()} a "
            f"{segment_data['fecha_tb_objetivo'].max().date()})."
        )

    lines.extend(
        [
            "",
            "## Control de calidad",
            "",
            "- No se entreno ningun modelo en este paso.",
            "- Todas las metricas se calcularon contra `tb_objetivo`, que es TB real.",
            "- El conjunto `test` representa el periodo futuro reservado para "
            "comparacion honesta.",
            "",
            "## Resultado sintetico en test",
            "",
        ]
    )

    test_metrics = metrics[metrics["segmento_temporal"] == "test"].copy()
    for _, row in test_metrics.sort_values("mae").iterrows():
        lines.append(
            f"- `{row['baseline']}`: MAE {row['mae']:.3f} C, "
            f"RMSE {row['rmse']:.3f} C, sesgo {row['sesgo']:.3f} C."
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Ejecuta la Fase 4 de baselines."""
    print_title("FASE 4 - BASELINES")

    config = load_config()
    dataset_path = DATA_PROCESSED_DIR / "dataset_modelado_v1.parquet"
    if not dataset_path.exists():
        raise FileNotFoundError(
            "No existe el dataset de modelado. Ejecuta primero "
            "`python src/03_feature_engineering.py`."
        )

    print_title("1. Cargando dataset de modelado")
    dataset = pd.read_parquet(dataset_path)
    print(f"Filas: {len(dataset):,}")
    print(f"Columnas: {len(dataset.columns):,}")

    print_title("2. Aplicando split temporal")
    dataset = split_by_time(dataset, config)
    print(dataset["segmento_temporal"].value_counts().to_string())

    print_title("3. Calculando predicciones simples")
    dataset = add_baseline_predictions(dataset)

    print_title("4. Evaluando metricas")
    metrics = evaluate_baselines(dataset)
    print(metrics.round(3).to_string(index=False))

    print_title("5. Guardando salidas")
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = METRICS_DIR / "baselines_metricas.csv"
    predictions_path = PREDICTIONS_DIR / "baselines_predicciones_test.csv"
    summary_path = OUTPUTS_DIR / "resumen_baselines.txt"
    validation_path = DOCS_DIR / "VALIDACION_BASELINES.md"

    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    prediction_columns = [
        "id_intervalo",
        "CUBA",
        "SALA",
        "GRUPO",
        "fecha_tb_inicial",
        "fecha_tb_objetivo",
        "tb_inicial",
        "tb_objetivo",
        "delta_tb_objetivo",
        "pred_baseline_ultima_tb",
        "pred_baseline_promedio_2_tb",
        "pred_baseline_extrapolacion_lineal",
    ]
    test_predictions = dataset[dataset["segmento_temporal"] == "test"][
        prediction_columns
    ].copy()
    test_predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    write_summary(dataset, metrics, summary_path)
    write_validation_doc(dataset, metrics, validation_path)

    print("Archivos generados:")
    print(f"- {metrics_path}")
    print(f"- {predictions_path}")
    print(f"- {summary_path}")
    print(f"- {validation_path}")


if __name__ == "__main__":
    main()
