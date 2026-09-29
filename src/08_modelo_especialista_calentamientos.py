"""
OBJETIVO DEL ARCHIVO
--------------------
Probar si un modelo especialista mejora los casos donde la TB sube o termina
caliente.

ENTRADAS
--------
- `data/processed/dataset_modelado_v1.parquet`
- Archivo `config/parametros.yaml`

SALIDAS
-------
- `outputs/metricas/modelo_especialista_calentamientos.csv`
- `outputs/resumen_modelo_especialista_calentamientos.txt`
- `docs/MODELO_ESPECIALISTA_CALENTAMIENTOS.md`

POR QUE EXISTE
--------------
El modelo general mejora el promedio, pero falla en cubas que terminan calientes.
Este experimento responde una pregunta concreta:

    Con las variables actuales, un modelo que presta mas atencion a
    calentamientos aprende algo mejor?

Si no mejora, probablemente falta informacion relevante en los datos actuales.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
DOCS_DIR = PROJECT_DIR / "docs"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metricas"


def print_title(title: str) -> None:
    """Imprime un titulo visible."""
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def load_training_helpers() -> Any:
    """Carga funciones de `05_entrenamiento.py`."""
    script_path = PROJECT_DIR / "src" / "05_entrenamiento.py"
    spec = importlib.util.spec_from_file_location("entrenamiento", script_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("No se pudo cargar 05_entrenamiento.py")
    spec.loader.exec_module(module)
    return module


def create_sample_weights(train: pd.DataFrame, strategy: str) -> np.ndarray:
    """
    Crea pesos para que el modelo mire mas ciertos casos.

    Un peso mayor no agrega informacion nueva. Solo le dice al modelo que
    equivocarse en esos casos cuesta mas.
    """
    weights = np.ones(len(train), dtype=float)

    if strategy == "peso_delta_mayor_5":
        weights[train["delta_tb_objetivo"].to_numpy() > 5] = 4.0
    elif strategy == "peso_objetivo_mayor_970":
        weights[train["tb_objetivo"].to_numpy() >= 970] = 4.0
    elif strategy == "peso_ambos":
        weights[train["delta_tb_objetivo"].to_numpy() > 5] = 3.0
        weights[train["tb_objetivo"].to_numpy() >= 970] = 4.0
    elif strategy == "sin_pesos":
        pass
    else:
        raise ValueError(f"Estrategia desconocida: {strategy}")

    return weights


def evaluate_subgroups(real_data: pd.DataFrame, prediction: np.ndarray) -> list[dict[str, Any]]:
    """Calcula metricas globales y por subgrupos importantes."""
    data = real_data.copy()
    data["tb_predicha"] = prediction
    data["error"] = data["tb_predicha"] - data["tb_objetivo"]
    data["error_abs"] = data["error"].abs()

    masks = {
        "global": np.ones(len(data), dtype=bool),
        "delta_>5": data["delta_tb_objetivo"] > 5,
        "objetivo_>=970": data["tb_objetivo"] >= 970,
        "inicial_>=970": data["tb_inicial"] >= 970,
        "inicial_>=970_y_objetivo_<970": (
            (data["tb_inicial"] >= 970) & (data["tb_objetivo"] < 970)
        ),
    }

    rows: list[dict[str, Any]] = []
    for subgroup, mask in masks.items():
        subset = data[mask]
        if subset.empty:
            continue

        rows.append(
            {
                "subgrupo": subgroup,
                "filas": len(subset),
                "mae": subset["error_abs"].mean(),
                "rmse": float(np.sqrt(np.mean(subset["error"] ** 2))),
                "sesgo": subset["error"].mean(),
                "p90_error_abs": subset["error_abs"].quantile(0.90),
                "p95_error_abs": subset["error_abs"].quantile(0.95),
            }
        )

    return rows


def run_experiment() -> pd.DataFrame:
    """Entrena variantes con distintos pesos y devuelve metricas."""
    helpers = load_training_helpers()
    config = helpers.load_config()

    dataset = pd.read_parquet(DATA_PROCESSED_DIR / "dataset_modelado_v1.parquet")
    dataset = helpers.split_by_time(dataset, config)
    features, categorical_columns, numeric_columns = helpers.choose_feature_columns(
        dataset
    )

    train = dataset[dataset["segmento_temporal"] == "train"].copy()
    validation = dataset[dataset["segmento_temporal"] == "validacion"].copy()
    test = dataset[dataset["segmento_temporal"] == "test"].copy()

    strategies = [
        "sin_pesos",
        "peso_delta_mayor_5",
        "peso_objetivo_mayor_970",
        "peso_ambos",
    ]

    rows: list[dict[str, Any]] = []

    for strategy in strategies:
        print(f"Entrenando estrategia: {strategy}")

        preprocessor = helpers.build_preprocessor(
            categorical_columns,
            numeric_columns,
            scale_numeric=False,
        )
        model = HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=42,
        )
        pipeline = Pipeline(
            steps=[
                ("preproceso", preprocessor),
                ("modelo", model),
            ]
        )

        weights = create_sample_weights(train, strategy)
        pipeline.fit(
            train[features],
            train["delta_tb_objetivo"],
            modelo__sample_weight=weights,
        )

        for segment_name, segment_data in [
            ("validacion", validation),
            ("test", test),
        ]:
            delta_prediction = pipeline.predict(segment_data[features])
            tb_prediction = segment_data["tb_inicial"].to_numpy() + delta_prediction

            subgroup_rows = evaluate_subgroups(segment_data, tb_prediction)
            for row in subgroup_rows:
                rows.append(
                    {
                        "estrategia": strategy,
                        "segmento": segment_name,
                        **row,
                    }
                )

    return pd.DataFrame(rows)


def write_summary(metrics: pd.DataFrame, output_path: Path) -> None:
    """Guarda lectura humana del experimento."""
    test_metrics = metrics[metrics["segmento"] == "test"].copy()
    global_metrics = test_metrics[test_metrics["subgrupo"] == "global"].sort_values(
        "mae"
    )
    hot_metrics = test_metrics[
        test_metrics["subgrupo"] == "objetivo_>=970"
    ].sort_values("mae")
    rising_metrics = test_metrics[test_metrics["subgrupo"] == "delta_>5"].sort_values(
        "mae"
    )

    lines = [
        "RESUMEN MODELO ESPECIALISTA EN CALENTAMIENTOS",
        "=" * 60,
        "",
        "Mejor estrategia global en test:",
        global_metrics.head(1).round(3).to_string(index=False),
        "",
        "Mejor estrategia para TB objetivo >= 970:",
        hot_metrics.head(1).round(3).to_string(index=False),
        "",
        "Mejor estrategia para delta TB > 5:",
        rising_metrics.head(1).round(3).to_string(index=False),
        "",
        "Todas las metricas test:",
        test_metrics.round(3).to_string(index=False),
        "",
        "Lectura:",
        "Si ponderar calentamientos no mejora mucho los subgrupos calientes, "
        "la limitacion probablemente no es solo el algoritmo. Puede faltar una "
        "senal operacional que explique por que esas cubas suben.",
    ]

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_doc(metrics: pd.DataFrame, output_path: Path) -> None:
    """Documenta el experimento."""
    test_metrics = metrics[metrics["segmento"] == "test"].copy()
    hot = test_metrics[test_metrics["subgrupo"] == "objetivo_>=970"].sort_values(
        "mae"
    )

    lines = [
        "# Modelo especialista en calentamientos",
        "",
        "Este experimento probo si el modelo mejora cuando damos mas peso a casos "
        "que suben o terminan calientes.",
        "",
        "## Punto importante",
        "",
        "Poner mas peso no crea informacion nueva. Si no hay variables que "
        "expliquen el calentamiento, el modelo no puede adivinarlo.",
        "",
        "## Mejor resultado en `TB objetivo >= 970`",
        "",
        hot.head(1).round(3).to_string(index=False),
        "",
        "## Interpretacion",
        "",
        "Si la mejora es chica, el siguiente avance real deberia venir de nuevas "
        "fuentes: alta frecuencia electrica, eventos precisos, problemas anodicos "
        "retirados, incrementos con duracion real o algun registro operativo que "
        "capture la causa del calentamiento.",
    ]

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Ejecuta el experimento de especialista."""
    print_title("FASE 8 - MODELO ESPECIALISTA EN CALENTAMIENTOS")

    metrics = run_experiment()

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = METRICS_DIR / "modelo_especialista_calentamientos.csv"
    summary_path = OUTPUTS_DIR / "resumen_modelo_especialista_calentamientos.txt"
    doc_path = DOCS_DIR / "MODELO_ESPECIALISTA_CALENTAMIENTOS.md"

    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    write_summary(metrics, summary_path)
    write_doc(metrics, doc_path)

    print("Archivos generados:")
    print(f"- {metrics_path}")
    print(f"- {summary_path}")
    print(f"- {doc_path}")


if __name__ == "__main__":
    main()
