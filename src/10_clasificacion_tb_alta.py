"""
OBJETIVO DEL ARCHIVO
--------------------
Predecir si la proxima medicion real de TB sera mayor a 972 C.

Es un modelo distinto de la regresion de temperatura. La regresion responde
"cuantos grados esperamos"; este clasificador responde "que probabilidad hay
de terminar arriba de 972 C".

La separacion temporal es la misma del proyecto. El umbral de probabilidad se
elige solo con validacion y se informa luego en test, sin mirar test para elegirlo.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_DIR / "data" / "processed" / "dataset_modelado_v1.parquet"
METRICS_DIR = PROJECT_DIR / "outputs" / "metricas"
PREDICTIONS_DIR = PROJECT_DIR / "outputs" / "predicciones"
MODELS_DIR = PROJECT_DIR / "models"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
DOCS_DIR = PROJECT_DIR / "docs"
TARGET_THRESHOLD_C = 972.0


def print_title(title: str) -> None:
    """Imprime un titulo visible."""
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def load_training_helpers() -> Any:
    """Carga helpers del entrenamiento de regresion para mantener reglas comunes."""
    script_path = PROJECT_DIR / "src" / "05_entrenamiento.py"
    spec = importlib.util.spec_from_file_location("entrenamiento", script_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("No se pudo cargar 05_entrenamiento.py")
    spec.loader.exec_module(module)
    return module


def calculate_metrics(
    real: pd.Series,
    probability: np.ndarray,
    probability_threshold: float,
) -> dict[str, float | int]:
    """Calcula calidad de probabilidades y de la decision binaria."""
    actual = real.to_numpy(dtype=int)
    predicted = (probability >= probability_threshold).astype(int)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1],
    ).ravel()

    result: dict[str, float | int] = {
        "prevalencia_real_pct": float(actual.mean() * 100),
        "tasa_predicha_pct": float(predicted.mean() * 100),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "especificidad": float(true_negative / (true_negative + false_positive)),
        "brier": float(brier_score_loss(actual, probability)),
        "verdaderos_positivos": int(true_positive),
        "falsos_positivos": int(false_positive),
        "falsos_negativos": int(false_negative),
        "verdaderos_negativos": int(true_negative),
    }
    if len(np.unique(actual)) == 2:
        result["roc_auc"] = float(roc_auc_score(actual, probability))
        result["average_precision"] = float(average_precision_score(actual, probability))
    else:
        result["roc_auc"] = np.nan
        result["average_precision"] = np.nan
    return result


def choose_probability_threshold(real: pd.Series, probability: np.ndarray) -> float:
    """Elige en validacion el umbral que maximiza F1, sin usar el test."""
    candidates = np.arange(0.05, 0.96, 0.01)
    scored: list[tuple[float, float, float]] = []
    actual = real.to_numpy(dtype=int)

    for threshold in candidates:
        predicted = (probability >= threshold).astype(int)
        scored.append(
            (
                float(f1_score(actual, predicted, zero_division=0)),
                float(precision_score(actual, predicted, zero_division=0)),
                float(threshold),
            )
        )

    # Ante empate de F1, preferimos mayor precision y luego el umbral mas alto.
    return max(scored)[2]


def create_models(helpers: Any, categorical: list[str], numeric: list[str]) -> dict[str, Pipeline]:
    """Crea dos clasificadores simples y no lineales cuando corresponde."""
    logistic_preprocessor = helpers.build_preprocessor(categorical, numeric, scale_numeric=True)
    tree_preprocessor = helpers.build_preprocessor(categorical, numeric, scale_numeric=False)

    return {
        "regresion_logistica": Pipeline(
            [
                ("preproceso", logistic_preprocessor),
                ("modelo", LogisticRegression(max_iter=2000, random_state=42)),
            ]
        ),
        "gradient_boosting_hist_clasificador": Pipeline(
            [
                ("preproceso", tree_preprocessor),
                (
                    "modelo",
                    HistGradientBoostingClassifier(
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


def write_doc(metrics: pd.DataFrame, output_path: Path) -> None:
    """Guarda una explicacion breve y util para el proyecto."""
    validation = metrics[metrics["segmento"] == "validacion"]
    best = validation.sort_values("f1", ascending=False).iloc[0]
    test = metrics[
        (metrics["segmento"] == "test")
        & (metrics["modelo"] == best["modelo"])
    ].iloc[0]

    lines = [
        "# Clasificacion de TB alta",
        "",
        f"Objetivo: estimar si la proxima TB real sera mayor a {TARGET_THRESHOLD_C:.0f} C.",
        "",
        "## Modelo elegido",
        "",
        f"- Modelo: `{best['modelo']}`.",
        f"- Umbral de probabilidad elegido en validacion: {best['umbral_probabilidad']:.2f}.",
        f"- En test: precision {test['precision']:.3f}, recall {test['recall']:.3f}, F1 {test['f1']:.3f}.",
        f"- AUC ROC test: {test['roc_auc']:.3f}; precision media: {test['average_precision']:.3f}.",
        "",
        "## Como leerlo",
        "",
        "- Precision: de las cubas marcadas como TB alta, que proporcion realmente termino arriba de 972 C.",
        "- Recall: de todas las cubas que realmente terminaron arriba de 972 C, que proporcion encontro el modelo.",
        "- El umbral fue elegido para equilibrar precision y recall (F1). No es todavia un umbral operativo.",
        "",
        "## Limite",
        "",
        "El clasificador usa la misma informacion ITM disponible antes de la TB objetivo. Si faltan eventos que causan calentamientos, tambien faltaran para este modelo.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Entrena, selecciona y guarda el primer clasificador de TB alta."""
    print_title("CLASIFICACION - PROXIMA TB MAYOR A 972 C")
    if not DATASET_PATH.exists():
        raise FileNotFoundError("Falta data/processed/dataset_modelado_v1.parquet")

    helpers = load_training_helpers()
    dataset = pd.read_parquet(DATASET_PATH)
    dataset = helpers.split_by_time(dataset, helpers.load_config())
    dataset["tb_alta_objetivo"] = (dataset["tb_objetivo"] > TARGET_THRESHOLD_C).astype(int)

    features, categorical, numeric = helpers.choose_feature_columns(dataset)
    # La etiqueta de clasificacion es el resultado que queremos predecir; nunca
    # puede entrar como feature. Esta exclusion evita leakage de forma explicita.
    features = [column for column in features if column != "tb_alta_objetivo"]
    numeric = [column for column in numeric if column != "tb_alta_objetivo"]
    train = dataset[dataset["segmento_temporal"] == "train"].copy()
    validation = dataset[dataset["segmento_temporal"] == "validacion"].copy()
    test = dataset[dataset["segmento_temporal"] == "test"].copy()

    print(f"Features: {len(features):,}; categoricas: {categorical}")
    print(
        "Prevalencia TB > 972 C: "
        f"train {train['tb_alta_objetivo'].mean() * 100:.2f}% | "
        f"validacion {validation['tb_alta_objetivo'].mean() * 100:.2f}% | "
        f"test {test['tb_alta_objetivo'].mean() * 100:.2f}%"
    )

    rows: list[dict[str, Any]] = []
    trained: dict[str, dict[str, Any]] = {}
    models = create_models(helpers, categorical, numeric)

    # Regla muy simple para saber si ML realmente agrega valor.
    for segment_name, segment in [("train", train), ("validacion", validation), ("test", test)]:
        baseline_probability = (segment["tb_inicial"] > TARGET_THRESHOLD_C).astype(float).to_numpy()
        rows.append(
            {
                "modelo": "baseline_tb_inicial_mayor_972",
                "segmento": segment_name,
                "umbral_probabilidad": 0.5,
                **calculate_metrics(segment["tb_alta_objetivo"], baseline_probability, 0.5),
            }
        )

    for name, pipeline in models.items():
        print(f"Entrenando {name}...")
        pipeline.fit(train[features], train["tb_alta_objetivo"])
        validation_probability = pipeline.predict_proba(validation[features])[:, 1]
        threshold = choose_probability_threshold(
            validation["tb_alta_objetivo"],
            validation_probability,
        )
        trained[name] = {"pipeline": pipeline, "threshold": threshold}

        for segment_name, segment in [("train", train), ("validacion", validation), ("test", test)]:
            probability = pipeline.predict_proba(segment[features])[:, 1]
            rows.append(
                {
                    "modelo": name,
                    "segmento": segment_name,
                    "umbral_probabilidad": threshold,
                    **calculate_metrics(segment["tb_alta_objetivo"], probability, threshold),
                }
            )

    metrics = pd.DataFrame(rows)
    validation_metrics = metrics[
        (metrics["segmento"] == "validacion")
        & (metrics["modelo"] != "baseline_tb_inicial_mayor_972")
    ]
    best_name = validation_metrics.sort_values("f1", ascending=False).iloc[0]["modelo"]
    best = trained[best_name]
    test_probability = best["pipeline"].predict_proba(test[features])[:, 1]
    test_predictions = test[
        [
            "id_intervalo",
            "CUBA",
            "SALA",
            "GRUPO",
            "FASE_VIDA",
            "fecha_tb_objetivo",
            "tb_inicial",
            "tb_objetivo",
        ]
    ].copy()
    test_predictions["tb_alta_real"] = test["tb_alta_objetivo"].to_numpy()
    test_predictions["probabilidad_tb_mayor_972"] = test_probability
    test_predictions["tb_alta_predicha"] = (
        test_probability >= best["threshold"]
    ).astype(int)

    print_title("RESULTADOS EN TEST")
    print(metrics[metrics["segmento"] == "test"].round(3).to_string(index=False))

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = METRICS_DIR / "clasificacion_tb_alta_metricas.csv"
    predictions_path = PREDICTIONS_DIR / "clasificacion_tb_alta_test.csv"
    model_path = MODELS_DIR / "modelo_clasificacion_tb_alta.joblib"
    summary_path = OUTPUTS_DIR / "resumen_clasificacion_tb_alta.txt"
    doc_path = DOCS_DIR / "CLASIFICACION_TB_ALTA.md"

    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    test_predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    joblib.dump(
        {
            "pipeline": best["pipeline"],
            "feature_columns": features,
            "categorical_columns": categorical,
            "numeric_columns": numeric,
            "target_definition": f"tb_objetivo > {TARGET_THRESHOLD_C:.0f}",
            "probability_threshold": float(best["threshold"]),
            "selected_by": "mayor_f1_validacion",
        },
        model_path,
    )
    selected_test = metrics[
        (metrics["segmento"] == "test") & (metrics["modelo"] == best_name)
    ].iloc[0]
    summary_path.write_text(
        "CLASIFICACION TB ALTA\n"
        "=" * 60
        + "\n"
        + f"Objetivo: TB objetivo > {TARGET_THRESHOLD_C:.0f} C.\n"
        + f"Modelo elegido: {best_name}.\n"
        + f"Umbral elegido en validacion: {best['threshold']:.2f}.\n"
        + f"F1 test: {selected_test['f1']:.3f}.\n"
        + f"Precision test: {selected_test['precision']:.3f}.\n"
        + f"Recall test: {selected_test['recall']:.3f}.\n"
        + f"AUC ROC test: {selected_test['roc_auc']:.3f}.\n",
        encoding="utf-8",
    )
    write_doc(metrics, doc_path)

    print("Archivos generados:")
    print(f"- {metrics_path}")
    print(f"- {predictions_path}")
    print(f"- {model_path}")
    print(f"- {doc_path}")


if __name__ == "__main__":
    main()
