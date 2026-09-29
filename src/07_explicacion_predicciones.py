"""
OBJETIVO DEL ARCHIVO
--------------------
Generar explicaciones aproximadas para las predicciones del modelo elegido.

ENTRADAS
--------
- `models/modelo_inicial_mejor.joblib`
- `data/processed/dataset_modelado_v1.parquet`
- Archivo `config/parametros.yaml`

SALIDAS
-------
- `outputs/metricas/importancia_permutacion_modelo_actual.csv`
- `outputs/predicciones/explicaciones_predicciones_test.csv`
- `docs/EXPLICACION_PREDICCIONES.md`

POR QUE EXISTE
--------------
Diego necesita saber por que una prediccion se movio hacia arriba o hacia
abajo. Este script no prueba causalidad, pero da una explicacion practica:

    TB predicha = TB inicial + delta predicho por el modelo

Luego estima que variables empujaron ese delta comparando la prediccion real
contra una prediccion donde una variable se reemplaza por un valor normal de
train.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, mean_absolute_error


PROJECT_DIR = Path(__file__).resolve().parents[1]
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


def load_training_helpers() -> Any:
    """Carga funciones del script 05 aunque el nombre empiece con numero."""
    script_path = PROJECT_DIR / "src" / "05_entrenamiento.py"
    spec = importlib.util.spec_from_file_location("entrenamiento", script_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("No se pudo cargar 05_entrenamiento.py")
    spec.loader.exec_module(module)
    return module


def get_reference_values(train: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    """
    Calcula valores normales de referencia usando solo train.

    Numericas: mediana.
    Categoricas: moda.
    """
    reference_values: dict[str, Any] = {}

    for feature in features:
        if pd.api.types.is_numeric_dtype(train[feature]):
            reference_values[feature] = train[feature].median()
        else:
            mode = train[feature].mode(dropna=True)
            reference_values[feature] = mode.iloc[0] if len(mode) else "faltante"

    return reference_values


def calculate_global_importance(
    model: Any,
    validation: pd.DataFrame,
    features: list[str],
    target_strategy: str,
) -> pd.DataFrame:
    """
    Calcula importancia global por permutacion.

    Para el modelo `delta_tb`, la importancia se calcula contra el delta porque
    eso es lo que aprende internamente.
    """
    sample = validation.sample(n=min(8000, len(validation)), random_state=42)
    target_column = "delta_tb_objetivo" if target_strategy == "delta_tb" else "tb_objetivo"

    scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    result = permutation_importance(
        model,
        sample[features],
        sample[target_column],
        scoring=scorer,
        n_repeats=3,
        random_state=42,
        n_jobs=-1,
    )

    importance = pd.DataFrame(
        {
            "feature": features,
            "importance_mae_increase": result.importances_mean,
            "std": result.importances_std,
        }
    ).sort_values("importance_mae_increase", ascending=False)

    return importance


def explain_test_predictions(
    model_package: dict[str, Any],
    dataset: pd.DataFrame,
    top_features: list[str],
    reference_values: dict[str, Any],
) -> pd.DataFrame:
    """
    Genera explicaciones locales aproximadas para cada prediccion de test.

    Si `contribucion_delta` es positiva, esa variable empujo el delta predicho
    hacia arriba. Si es negativa, lo empujo hacia abajo.
    """
    model = model_package["pipeline"]
    features = model_package["feature_columns"]
    target_strategy = model_package["target_strategy"]

    test = dataset[dataset["segmento_temporal"] == "test"].copy()
    full_raw_prediction = model.predict(test[features])

    if target_strategy == "delta_tb":
        delta_predicho = full_raw_prediction
        tb_predicha = test["tb_inicial"].to_numpy() + delta_predicho
    else:
        tb_predicha = full_raw_prediction
        delta_predicho = tb_predicha - test["tb_inicial"].to_numpy()

    base_output = test[
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
    base_output["tb_predicha"] = tb_predicha
    base_output["delta_predicho"] = delta_predicho
    base_output["error"] = base_output["tb_predicha"] - base_output["tb_objetivo"]
    base_output["error_abs"] = base_output["error"].abs()

    contribution_columns: list[str] = []
    for feature in top_features:
        changed = test[features].copy()
        changed[feature] = reference_values[feature]
        changed_raw_prediction = model.predict(changed)

        # Explicamos el delta que aprende el modelo. La TB inicial queda como
        # ancla externa de la prediccion final.
        contribution = full_raw_prediction - changed_raw_prediction
        column_name = f"contrib_delta__{feature}"
        base_output[column_name] = contribution
        contribution_columns.append(column_name)

    top_rows: list[dict[str, Any]] = []
    for _, row in base_output.iterrows():
        contributions = {
            column.replace("contrib_delta__", ""): row[column]
            for column in contribution_columns
        }
        ordered = sorted(
            contributions.items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
        top_rows.append(
            {
                "id_intervalo": row["id_intervalo"],
                "top_1_variable": ordered[0][0],
                "top_1_contrib_delta": ordered[0][1],
                "top_2_variable": ordered[1][0],
                "top_2_contrib_delta": ordered[1][1],
                "top_3_variable": ordered[2][0],
                "top_3_contrib_delta": ordered[2][1],
            }
        )

    top_explanations = pd.DataFrame(top_rows)
    explained = base_output.merge(top_explanations, on="id_intervalo", how="left")
    return explained


def write_doc(top_features: list[str], output_path: Path) -> None:
    """Documenta como leer las explicaciones."""
    lines = [
        "# Explicacion de predicciones",
        "",
        "La prediccion del modelo elegido se lee asi:",
        "",
        "```text",
        "TB predicha = TB inicial + delta predicho por el modelo",
        "```",
        "",
        "Las columnas `contrib_delta__...` explican el delta, no la TB inicial. "
        "Una contribucion positiva empuja la prediccion hacia mas temperatura. "
        "Una contribucion negativa la empuja hacia menos temperatura.",
        "",
        "Estas contribuciones son una aproximacion por sensibilidad: se cambia "
        "una variable por un valor normal de train y se observa cuanto cambia "
        "la prediccion. No prueba causalidad.",
        "",
        "Variables usadas para explicacion local:",
        "",
    ]

    for feature in top_features:
        lines.append(f"- `{feature}`")

    lines.extend(
        [
            "",
            "Archivo generado:",
            "",
            "- `outputs/predicciones/explicaciones_predicciones_test.csv`",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Ejecuta la explicacion de predicciones."""
    print_title("FASE 7 - EXPLICACION DE PREDICCIONES")

    helpers = load_training_helpers()
    config = helpers.load_config()

    dataset = pd.read_parquet(DATA_PROCESSED_DIR / "dataset_modelado_v1.parquet")
    dataset = helpers.split_by_time(dataset, config)
    model_package = joblib.load(MODELS_DIR / "modelo_inicial_mejor.joblib")

    features = model_package["feature_columns"]
    train = dataset[dataset["segmento_temporal"] == "train"]
    validation = dataset[dataset["segmento_temporal"] == "validacion"]

    print_title("1. Calculando importancia global")
    importance = calculate_global_importance(
        model_package["pipeline"],
        validation,
        features,
        model_package["target_strategy"],
    )
    top_features = importance.head(15)["feature"].tolist()
    print(importance.head(15).round(4).to_string(index=False))

    print_title("2. Calculando explicaciones locales")
    reference_values = get_reference_values(train, features)
    explanations = explain_test_predictions(
        model_package,
        dataset,
        top_features,
        reference_values,
    )
    print(f"Predicciones explicadas: {len(explanations):,}")

    print_title("3. Guardando salidas")
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    importance_path = METRICS_DIR / "importancia_permutacion_modelo_actual.csv"
    explanations_path = PREDICTIONS_DIR / "explicaciones_predicciones_test.csv"
    doc_path = DOCS_DIR / "EXPLICACION_PREDICCIONES.md"

    importance.to_csv(importance_path, index=False, encoding="utf-8-sig")
    explanations.to_csv(explanations_path, index=False, encoding="utf-8-sig")
    write_doc(top_features, doc_path)

    print("Archivos generados:")
    print(f"- {importance_path}")
    print(f"- {explanations_path}")
    print(f"- {doc_path}")


if __name__ == "__main__":
    main()
