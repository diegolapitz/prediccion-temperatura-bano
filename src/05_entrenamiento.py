"""
OBJETIVO DEL ARCHIVO
--------------------
Entrenar los primeros modelos simples para predecir la proxima TB real.

ENTRADAS
--------
- `data/processed/dataset_modelado_v1.parquet`
- `outputs/metricas/baselines_metricas.csv`
- Archivo `config/parametros.yaml`

SALIDAS
-------
- `outputs/metricas/modelos_metricas.csv`
- `outputs/predicciones/modelos_predicciones_test.csv`
- `outputs/resumen_modelos.txt`
- `models/modelo_inicial_mejor.joblib`
- `docs/VALIDACION_MODELOS.md`

POR QUE EXISTE
--------------
Despues de construir datos y baselines, probamos modelos simples. La regla
principal es que el modelo debe mejorar los baselines en datos futuros, no solo
en los datos usados para aprender.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# FASE 0 - RUTAS DEL PROYECTO
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_DIR / "config" / "parametros.yaml"
DATA_PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
DOCS_DIR = PROJECT_DIR / "docs"
MODELS_DIR = PROJECT_DIR / "models"
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
    """Asigna train, validacion y test usando la fecha de TB objetivo."""
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


def choose_feature_columns(
    dataset: pd.DataFrame,
    include_life_phase: bool = False,
) -> tuple[list[str], list[str], list[str]]:
    """
    Define que columnas entran al modelo.

    Excluimos:
    - targets;
    - fechas y ordenes absolutos;
    - identificadores;
    - CUBA, para evitar que el primer modelo memorice cubas.
    """
    excluded_columns = {
        "id_intervalo",
        "CUBA",
        "fecha_tb_inicial",
        "semiturno_tb_inicial",
        "orden_tb_inicial",
        "fecha_tb_objetivo",
        "semiturno_tb_objetivo",
        "orden_tb_objetivo",
        "orden_tb_anterior_2",
        "orden_tb_anterior_3",
        "tb_objetivo",
        "delta_tb_objetivo",
        "segmento_temporal",
        "tb_reales_en_ventana",
    }

    categorical_candidates = ["SALA", "GRUPO"]
    if include_life_phase:
        # La fase de vida se probo para regresion y no mejoro el MAE general.
        # El clasificador de TB alta puede activarla de forma explicita.
        categorical_candidates.append("FASE_VIDA")

    categorical_columns = [
        column
        for column in categorical_candidates
        if column in dataset.columns and column not in excluded_columns
    ]

    numeric_columns = [
        column
        for column in dataset.columns
        if column not in excluded_columns
        and column not in categorical_columns
        and pd.api.types.is_numeric_dtype(dataset[column])
    ]

    feature_columns = categorical_columns + numeric_columns
    return feature_columns, categorical_columns, numeric_columns


def build_preprocessor(
    categorical_columns: list[str],
    numeric_columns: list[str],
    scale_numeric: bool,
) -> ColumnTransformer:
    """
    Crea las transformaciones previas al modelo.

    - Numericas: imputacion por mediana.
    - Categoricas: imputacion por valor "faltante" y one-hot encoding.
    - Escalado: se usa para modelos lineales, no es necesario para arboles.
    """
    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(steps=numeric_steps)

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="faltante")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_columns),
            ("categorical", categorical_transformer, categorical_columns),
        ],
        remainder="drop",
    )


def create_model_pipelines(
    categorical_columns: list[str],
    numeric_columns: list[str],
) -> dict[str, Pipeline]:
    """Define los primeros modelos a probar."""
    linear_preprocessor = build_preprocessor(
        categorical_columns,
        numeric_columns,
        scale_numeric=True,
    )
    tree_preprocessor = build_preprocessor(
        categorical_columns,
        numeric_columns,
        scale_numeric=False,
    )

    return {
        "regresion_lineal": Pipeline(
            steps=[
                ("preproceso", linear_preprocessor),
                ("modelo", LinearRegression()),
            ]
        ),
        "elastic_net": Pipeline(
            steps=[
                ("preproceso", linear_preprocessor),
                (
                    "modelo",
                    ElasticNet(
                        alpha=0.05,
                        l1_ratio=0.20,
                        max_iter=5000,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "gradient_boosting_hist": Pipeline(
            steps=[
                ("preproceso", tree_preprocessor),
                (
                    "modelo",
                    HistGradientBoostingRegressor(
                        max_iter=250,
                        learning_rate=0.05,
                        max_leaf_nodes=31,
                        l2_regularization=0.05,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def calculate_metrics(real: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    """Calcula las mismas metricas principales que usamos en baselines."""
    error = predicted - real.to_numpy()
    absolute_error = np.abs(error)

    return {
        "mae": float(np.mean(absolute_error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "sesgo": float(np.mean(error)),
        "error_mediano": float(np.median(absolute_error)),
        "p75_error_abs": float(np.quantile(absolute_error, 0.75)),
        "p90_error_abs": float(np.quantile(absolute_error, 0.90)),
        "p95_error_abs": float(np.quantile(absolute_error, 0.95)),
        "porcentaje_dentro_3c": float(np.mean(absolute_error <= 3) * 100),
        "porcentaje_dentro_5c": float(np.mean(absolute_error <= 5) * 100),
        "porcentaje_dentro_10c": float(np.mean(absolute_error <= 10) * 100),
    }


def train_and_evaluate_models(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    numeric_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Entrena modelos y devuelve metricas, predicciones de test y mejor modelo.

    Probamos dos estrategias:
    - `tb_absoluta`: el modelo predice directamente `tb_objetivo`.
    - `delta_tb`: el modelo predice cambio de TB y despues reconstruimos:
      TB predicha = TB inicial + delta predicho.
    """
    train = dataset[dataset["segmento_temporal"] == "train"].copy()
    validation = dataset[dataset["segmento_temporal"] == "validacion"].copy()
    test = dataset[dataset["segmento_temporal"] == "test"].copy()

    model_templates = create_model_pipelines(categorical_columns, numeric_columns)
    target_strategies = {
        "tb_absoluta": "tb_objetivo",
        "delta_tb": "delta_tb_objetivo",
    }

    metric_rows: list[dict[str, Any]] = []
    test_prediction_frames: list[pd.DataFrame] = []
    trained_models: dict[str, Any] = {}

    for strategy_name, target_column in target_strategies.items():
        for model_name, pipeline in model_templates.items():
            full_model_name = f"{model_name}__{strategy_name}"
            print(f"Entrenando {full_model_name}...")

            pipeline.fit(train[feature_columns], train[target_column])
            trained_models[full_model_name] = {
                "pipeline": pipeline,
                "target_strategy": strategy_name,
                "feature_columns": feature_columns,
                "categorical_columns": categorical_columns,
                "numeric_columns": numeric_columns,
            }

            for segment_name, segment_data in [
                ("train", train),
                ("validacion", validation),
                ("test", test),
            ]:
                raw_prediction = pipeline.predict(segment_data[feature_columns])

                if strategy_name == "tb_absoluta":
                    tb_prediction = raw_prediction
                else:
                    tb_prediction = segment_data["tb_inicial"].to_numpy() + raw_prediction

                metrics = calculate_metrics(
                    real=segment_data["tb_objetivo"],
                    predicted=tb_prediction,
                )
                metric_rows.append(
                    {
                        "segmento_temporal": segment_name,
                        "modelo": model_name,
                        "estrategia_target": strategy_name,
                        "modelo_completo": full_model_name,
                        "filas": len(segment_data),
                        **metrics,
                    }
                )

                if segment_name == "test":
                    predictions = segment_data[
                        [
                            "id_intervalo",
                            "CUBA",
                            "SALA",
                            "GRUPO",
                            "fecha_tb_objetivo",
                            "tb_inicial",
                            "tb_objetivo",
                            "delta_tb_objetivo",
                        ]
                    ].copy()
                    predictions["modelo_completo"] = full_model_name
                    predictions["tb_predicha"] = tb_prediction
                    predictions["error"] = predictions["tb_predicha"] - predictions[
                        "tb_objetivo"
                    ]
                    predictions["error_abs"] = predictions["error"].abs()
                    test_prediction_frames.append(predictions)

    metrics = pd.DataFrame(metric_rows)
    test_predictions = pd.concat(test_prediction_frames, ignore_index=True)

    validation_metrics = metrics[metrics["segmento_temporal"] == "validacion"]
    best_row = validation_metrics.sort_values("mae").iloc[0]
    best_model_key = best_row["modelo_completo"]
    best_model_package = trained_models[best_model_key]
    best_model_package["selected_by"] = "menor_mae_validacion"
    best_model_package["validation_mae"] = float(best_row["mae"])

    return metrics, test_predictions, best_model_package


def write_summary(
    metrics: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Guarda resumen de modelos y comparacion con baseline."""
    validation_best = (
        metrics[metrics["segmento_temporal"] == "validacion"]
        .sort_values("mae")
        .iloc[0]
    )
    test_for_best = metrics[
        (metrics["segmento_temporal"] == "test")
        & (metrics["modelo_completo"] == validation_best["modelo_completo"])
    ].iloc[0]

    best_baseline_test = (
        baseline_metrics[baseline_metrics["segmento_temporal"] == "test"]
        .sort_values("mae")
        .iloc[0]
    )
    improvement = (
        (best_baseline_test["mae"] - test_for_best["mae"])
        / best_baseline_test["mae"]
        * 100
    )

    lines = [
        "RESUMEN DE MODELOS - FASE 5",
        "=" * 60,
        "",
        "Modelo elegido por menor MAE en validacion:",
        f"- {validation_best['modelo_completo']}",
        f"- MAE validacion: {validation_best['mae']:.3f} C",
        "",
        "Resultado del modelo elegido en test:",
        f"- MAE test: {test_for_best['mae']:.3f} C",
        f"- RMSE test: {test_for_best['rmse']:.3f} C",
        f"- Sesgo test: {test_for_best['sesgo']:.3f} C",
        "",
        "Mejor baseline en test:",
        f"- {best_baseline_test['baseline']}",
        f"- MAE test: {best_baseline_test['mae']:.3f} C",
        "",
        f"Mejora relativa vs mejor baseline de test: {improvement:.2f}%",
        "",
        "Metricas de modelos:",
        metrics.round(4).to_string(index=False),
        "",
        "Lectura prudente:",
        "Este es un primer modelo. Aunque mejore el baseline, todavia falta "
        "analizar errores por sala, rango de TB y casos calientes antes de usarlo "
        "para una recomendacion operativa.",
    ]

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_validation_doc(
    metrics: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Documenta la validacion del primer entrenamiento."""
    validation_best = (
        metrics[metrics["segmento_temporal"] == "validacion"]
        .sort_values("mae")
        .iloc[0]
    )
    test_for_best = metrics[
        (metrics["segmento_temporal"] == "test")
        & (metrics["modelo_completo"] == validation_best["modelo_completo"])
    ].iloc[0]
    best_baseline_test = (
        baseline_metrics[baseline_metrics["segmento_temporal"] == "test"]
        .sort_values("mae")
        .iloc[0]
    )

    lines = [
        "# Validacion de modelos - Fase 5",
        "",
        "Este documento resume el primer entrenamiento real del proyecto.",
        "",
        "## Decisiones",
        "",
        "- Se uso division temporal.",
        "- Se entreno solo con `train`.",
        "- Se eligio el modelo por MAE en `validacion`.",
        "- `test` se uso como periodo futuro de comparacion.",
        "- `CUBA` no se uso como feature para evitar memorizacion inicial.",
        "- Se probaron targets `tb_absoluta` y `delta_tb`.",
        "",
        "## Resultado",
        "",
        f"- Mejor modelo por validacion: `{validation_best['modelo_completo']}`.",
        f"- MAE validacion: {validation_best['mae']:.3f} C.",
        f"- MAE test del modelo elegido: {test_for_best['mae']:.3f} C.",
        f"- Mejor baseline test: `{best_baseline_test['baseline']}` con MAE "
        f"{best_baseline_test['mae']:.3f} C.",
        "",
        "## Advertencia",
        "",
        "Este resultado todavia no alcanza para decidir retiros operativos. La "
        "siguiente fase debe revisar errores en cubas calientes y medir el costo "
        "de falsos retiros.",
    ]

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Ejecuta el primer entrenamiento completo."""
    print_title("FASE 5 - PRIMEROS MODELOS")

    config = load_config()
    dataset_path = DATA_PROCESSED_DIR / "dataset_modelado_v1.parquet"
    baseline_metrics_path = METRICS_DIR / "baselines_metricas.csv"

    if not dataset_path.exists():
        raise FileNotFoundError("Falta dataset_modelado_v1.parquet.")
    if not baseline_metrics_path.exists():
        raise FileNotFoundError("Faltan metricas de baselines.")

    print_title("1. Cargando datos")
    dataset = pd.read_parquet(dataset_path)
    dataset = split_by_time(dataset, config)
    baseline_metrics = pd.read_csv(baseline_metrics_path)

    feature_columns, categorical_columns, numeric_columns = choose_feature_columns(
        dataset
    )
    print(f"Features totales: {len(feature_columns):,}")
    print(f"Features categoricas: {categorical_columns}")
    print(f"Features numericas: {len(numeric_columns):,}")

    print_title("2. Entrenando y evaluando")
    metrics, test_predictions, best_model_package = train_and_evaluate_models(
        dataset,
        feature_columns,
        categorical_columns,
        numeric_columns,
    )

    print_title("3. Resultados")
    print(metrics.round(3).to_string(index=False))

    print_title("4. Guardando salidas")
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = METRICS_DIR / "modelos_metricas.csv"
    predictions_path = PREDICTIONS_DIR / "modelos_predicciones_test.csv"
    model_path = MODELS_DIR / "modelo_inicial_mejor.joblib"
    summary_path = OUTPUTS_DIR / "resumen_modelos.txt"
    validation_path = DOCS_DIR / "VALIDACION_MODELOS.md"

    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    test_predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    joblib.dump(best_model_package, model_path)
    write_summary(metrics, baseline_metrics, summary_path)
    write_validation_doc(metrics, baseline_metrics, validation_path)

    print("Archivos generados:")
    print(f"- {metrics_path}")
    print(f"- {predictions_path}")
    print(f"- {model_path}")
    print(f"- {summary_path}")
    print(f"- {validation_path}")


if __name__ == "__main__":
    main()
