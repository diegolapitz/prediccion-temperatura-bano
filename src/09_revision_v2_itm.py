"""
OBJETIVO DEL ARCHIVO
--------------------
Revisar cuanto valor predictivo real aporta ITM/semiturno en una V2
metodologica, sin usar datos de 5 minutos ni cambiar el split temporal.

ENTRADAS
--------
- `data/processed/dataset_modelado_v1.parquet`
- Archivo `config/parametros.yaml`

SALIDAS
-------
- `outputs/metricas/v2_baselines_autorregresivos.csv`
- `outputs/metricas/v2_ablation_study.csv`
- `outputs/metricas/v2_regimen_termico.csv`
- `outputs/metricas/v2_revision_efectos_anodicos.csv`
- `outputs/resumen_v2_itm.txt`
- `docs/REVISION_V2_ITM.md`

POR QUE EXISTE
--------------
El modelo general ya mejora el baseline, pero hay dudas metodologicas:

- cuanto mejora contra una regresion autorregresiva simple;
- que familia de variables aporta senal;
- si IMM/V_ACD agregan valor desde ITM;
- si los efectos anodicos resumidos ayudan en calentamientos;
- si el MAE cercano a 5 C parece limite de algoritmo o de informacion.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_DIR / "config" / "parametros.yaml"
DATA_PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
DOCS_DIR = PROJECT_DIR / "docs"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metricas"


def print_title(title: str) -> None:
    """Imprime un titulo visible."""
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def load_config() -> dict[str, Any]:
    """Carga configuracion central."""
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_training_helpers() -> Any:
    """Carga funciones del script 05 para reutilizar split y metricas."""
    script_path = PROJECT_DIR / "src" / "05_entrenamiento.py"
    spec = importlib.util.spec_from_file_location("entrenamiento", script_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("No se pudo cargar 05_entrenamiento.py")
    spec.loader.exec_module(module)
    return module


def build_preprocessor(
    categorical_columns: list[str],
    numeric_columns: list[str],
    scale_numeric: bool,
) -> ColumnTransformer:
    """Preprocesamiento simple para modelos lineales y arboles."""
    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    transformers: list[tuple[str, Any, list[str]]] = [
        ("numeric", Pipeline(numeric_steps), numeric_columns),
    ]

    if categorical_columns:
        categorical_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value="faltante")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("categorical", categorical_transformer, categorical_columns))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def calculate_metrics(real: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    """Calcula metricas pedidas para comparacion."""
    error = predicted - real.to_numpy()
    absolute_error = np.abs(error)
    return {
        "mae": float(np.mean(absolute_error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
        "p90_error_abs": float(np.quantile(absolute_error, 0.90)),
        "p95_error_abs": float(np.quantile(absolute_error, 0.95)),
        "porcentaje_dentro_5c": float(np.mean(absolute_error <= 5) * 100),
        "porcentaje_dentro_10c": float(np.mean(absolute_error <= 10) * 100),
    }


def split_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """Aplica exactamente el split temporal del proyecto."""
    helpers = load_training_helpers()
    return helpers.split_by_time(dataset, load_config())


def thermal_feature_columns(dataset: pd.DataFrame) -> list[str]:
    """Features termicas historicas, siempre anteriores o iguales a TB inicial."""
    candidates = [
        "tb_inicial",
        "tb_anterior_1",
        "tb_anterior_2",
        "tb_anterior_3",
        "tb_promedio_ultimas_2",
        "tb_promedio_ultimas_3",
        "tb_std_ultimas_3",
        "tb_max_ultimas_3",
        "tb_min_ultimas_3",
        "tb_rango_ultimas_3",
        "ultimo_cambio_tb",
        "cambio_anterior_al_ultimo",
        "tb_cambio_ultima_vs_anterior",
        "tb_cambio_anterior_vs_tercera",
        "tb_pendiente_lineal_ultimas_3",
        "tb_mediciones_previas_disponibles",
        "semiturnos_entre_tb",
    ]
    return [column for column in candidates if column in dataset.columns]


def add_baseline_predictions(dataset: pd.DataFrame) -> pd.DataFrame:
    """Agrega baselines simples que no requieren entrenamiento."""
    result = dataset.copy()
    result["pred_naive_ultima_tb"] = result["tb_inicial"]
    result["pred_promedio_2_tb"] = result[["tb_inicial", "tb_anterior_2"]].mean(axis=1)
    result["pred_extrapolacion_lineal"] = (
        result["tb_inicial"] + (result["tb_inicial"] - result["tb_anterior_2"])
    ).fillna(result["tb_inicial"])
    return result


def evaluate_autoregressive_baselines(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Compara baselines existentes contra regresiones autorregresivas simples.

    Todos usan el mismo split train/validation/test.
    """
    dataset = add_baseline_predictions(dataset)
    train = dataset[dataset["segmento_temporal"] == "train"].copy()
    rows: list[dict[str, Any]] = []

    simple_predictions = {
        "naive_ultima_tb": "pred_naive_ultima_tb",
        "promedio_2_tb": "pred_promedio_2_tb",
        "extrapolacion_lineal": "pred_extrapolacion_lineal",
    }

    for segment, segment_data in dataset.groupby("segmento_temporal", sort=False):
        for name, prediction_column in simple_predictions.items():
            rows.append(
                {
                    "segmento": segment,
                    "baseline": name,
                    "filas": len(segment_data),
                    **calculate_metrics(
                        segment_data["tb_objetivo"],
                        segment_data[prediction_column].to_numpy(),
                    ),
                }
            )

    model_specs = [
        (
            "lineal_ar2_tb",
            ["tb_inicial", "tb_anterior_2"],
            "tb_objetivo",
            LinearRegression(),
            "tb_absoluta",
        ),
        (
            "lineal_ar3_tb",
            ["tb_inicial", "tb_anterior_2", "tb_anterior_3"],
            "tb_objetivo",
            LinearRegression(),
            "tb_absoluta",
        ),
        (
            "ridge_ar3_tb",
            ["tb_inicial", "tb_anterior_2", "tb_anterior_3"],
            "tb_objetivo",
            Ridge(alpha=1.0),
            "tb_absoluta",
        ),
        (
            "lineal_delta_historia_tb",
            thermal_feature_columns(dataset),
            "delta_tb_objetivo",
            LinearRegression(),
            "delta_tb",
        ),
    ]

    for name, features, target_column, model, strategy in model_specs:
        features = [column for column in features if column in dataset.columns]
        preprocessor = build_preprocessor([], features, scale_numeric=True)
        pipeline = Pipeline([("preproceso", preprocessor), ("modelo", model)])
        pipeline.fit(train[features], train[target_column])

        for segment, segment_data in dataset.groupby("segmento_temporal", sort=False):
            raw_prediction = pipeline.predict(segment_data[features])
            if strategy == "delta_tb":
                tb_prediction = segment_data["tb_inicial"].to_numpy() + raw_prediction
            else:
                tb_prediction = raw_prediction

            rows.append(
                {
                    "segmento": segment,
                    "baseline": name,
                    "filas": len(segment_data),
                    **calculate_metrics(segment_data["tb_objetivo"], tb_prediction),
                }
            )

    return pd.DataFrame(rows)


def subgroup_masks(data: pd.DataFrame) -> dict[str, pd.Series]:
    """Subgrupos pedidos para comparar calentamientos."""
    return {
        "global": pd.Series(True, index=data.index),
        "delta_tb_>=5": data["delta_tb_objetivo"] >= 5,
        "delta_tb_>=10": data["delta_tb_objetivo"] >= 10,
        "tb_objetivo_>=970": data["tb_objetivo"] >= 970,
        "tb_inicial_<970_y_objetivo_>=970": (
            (data["tb_inicial"] < 970) & (data["tb_objetivo"] >= 970)
        ),
    }


def evaluate_prediction_by_subgroup(
    data: pd.DataFrame,
    prediction: np.ndarray,
) -> dict[str, float]:
    """Metricas especificas de subgrupos para una prediccion."""
    result = data.copy()
    result["pred"] = prediction
    result["error"] = result["pred"] - result["tb_objetivo"]
    result["error_abs"] = result["error"].abs()

    output: dict[str, float] = {}
    for name, mask in subgroup_masks(result).items():
        subset = result[mask]
        if subset.empty:
            output[f"n_{name}"] = 0
            output[f"mae_{name}"] = np.nan
            output[f"bias_{name}"] = np.nan
            continue
        output[f"n_{name}"] = int(len(subset))
        output[f"mae_{name}"] = float(subset["error_abs"].mean())
        output[f"bias_{name}"] = float(subset["error"].mean())
    return output


def all_non_target_features(dataset: pd.DataFrame) -> list[str]:
    """Columnas candidatas a features sin IDs, fechas ni targets."""
    excluded = {
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
        "orden_alf3_ultima_real",
        "tb_objetivo",
        "delta_tb_objetivo",
        "segmento_temporal",
        "tb_reales_en_ventana",
    }
    return [
        column
        for column in dataset.columns
        if column not in excluded
        and (
            pd.api.types.is_numeric_dtype(dataset[column])
            or column in {"SALA", "GRUPO", "FASE_VIDA"}
        )
    ]


def columns_containing(dataset: pd.DataFrame, fragments: list[str]) -> list[str]:
    """Busca columnas por fragmentos de nombre."""
    result: list[str] = []
    for column in dataset.columns:
        lower = column.lower()
        if any(fragment in lower for fragment in fragments):
            result.append(column)
    return result


def ablation_feature_sets(dataset: pd.DataFrame) -> dict[str, list[str]]:
    """Define familias acumulativas A-F."""
    thermal = thermal_feature_columns(dataset)

    state = columns_containing(
        dataset,
        [
            "hbd",
            "hbc",
            "hmd",
            "hmc",
            "hb_",
            "hm_",
            "altura_bano",
            "altura_metal",
            "bano_bajo",
            "semiturnos_bano_bajo",
            "agebsq",
            "potage",
            "sala",
            "grupo",
            "fase_vida",
        ],
    )
    chemistry = columns_containing(
        dataset,
        ["alf3", "ndalf3", "ndal2o3", "amna2co3", "soda"],
    )
    electric = columns_containing(dataset, ["imm", "v_acd", "potencia_proxy"])
    anodic = columns_containing(
        dataset,
        ["ntea", "dtea", "sea", "ea_", "efecto_anodico", "semiturnos_con_ea"],
    )
    control = columns_containing(
        dataset,
        [
            "wrmi",
            "smrwfc",
            "rth",
            "rc_",
            "rrm",
            "rkm",
            "mbl",
            "tbd",
            "tbc",
            "diferencia_rrm_rkm",
        ],
    )

    def unique(columns: list[str]) -> list[str]:
        valid = [column for column in columns if column in all_non_target_features(dataset)]
        return list(dict.fromkeys(valid))

    a = unique(thermal)
    b = unique(a + state)
    c = unique(b + chemistry)
    d = unique(c + electric)
    e = unique(d + anodic)
    f = unique(e + control)

    return {
        "A_historia_termica": a,
        "B_A_mas_estado_cuba": b,
        "C_B_mas_quimica": c,
        "D_C_mas_electricas": d,
        "E_D_mas_efectos_anodicos": e,
        "F_E_mas_control_operacion": f,
    }


def train_hgb_delta(
    train: pd.DataFrame,
    features: list[str],
    categorical_columns: list[str],
    numeric_columns: list[str],
) -> Pipeline:
    """Entrena el mismo tipo de modelo usado actualmente, prediciendo delta TB."""
    preprocessor = build_preprocessor(categorical_columns, numeric_columns, scale_numeric=False)
    model = HistGradientBoostingRegressor(
        max_iter=250,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=0.05,
        random_state=42,
    )
    pipeline = Pipeline([("preproceso", preprocessor), ("modelo", model)])
    pipeline.fit(train[features], train["delta_tb_objetivo"])
    return pipeline


def run_ablation_study(dataset: pd.DataFrame) -> pd.DataFrame:
    """Entrena ablations acumulativas y evalua en test."""
    train = dataset[dataset["segmento_temporal"] == "train"].copy()
    test = dataset[dataset["segmento_temporal"] == "test"].copy()
    rows: list[dict[str, Any]] = []

    for stage, features in ablation_feature_sets(dataset).items():
        categorical = [
            column for column in features if column in {"SALA", "GRUPO", "FASE_VIDA"}
        ]
        numeric = [column for column in features if column not in categorical]
        print(f"Entrenando ablation {stage}: {len(features)} features")
        pipeline = train_hgb_delta(train, features, categorical, numeric)
        delta_prediction = pipeline.predict(test[features])
        tb_prediction = test["tb_inicial"].to_numpy() + delta_prediction
        global_metrics = calculate_metrics(test["tb_objetivo"], tb_prediction)
        subgroup_metrics = evaluate_prediction_by_subgroup(test, tb_prediction)

        rows.append(
            {
                "etapa": stage,
                "cantidad_features": len(features),
                **global_metrics,
                **subgroup_metrics,
            }
        )

    return pd.DataFrame(rows)


def evaluate_regimes(
    data: pd.DataFrame,
    prediction: np.ndarray,
    model_name: str,
) -> pd.DataFrame:
    """Tabla unica por regimen termico."""
    result = data.copy()
    result["pred"] = prediction
    result["error"] = result["pred"] - result["tb_objetivo"]
    result["error_abs"] = result["error"].abs()

    regimes = {
        "delta_tb <= -10": result["delta_tb_objetivo"] <= -10,
        "-10 < delta_tb <= -5": (result["delta_tb_objetivo"] > -10)
        & (result["delta_tb_objetivo"] <= -5),
        "-5 < delta_tb < 5": (result["delta_tb_objetivo"] > -5)
        & (result["delta_tb_objetivo"] < 5),
        "5 <= delta_tb < 10": (result["delta_tb_objetivo"] >= 5)
        & (result["delta_tb_objetivo"] < 10),
        "delta_tb >= 10": result["delta_tb_objetivo"] >= 10,
        "TB_inicial >= 970": result["tb_inicial"] >= 970,
        "TB_objetivo >= 970": result["tb_objetivo"] >= 970,
        "TB_inicial < 970 y TB_objetivo >= 970": (result["tb_inicial"] < 970)
        & (result["tb_objetivo"] >= 970),
    }

    rows: list[dict[str, Any]] = []
    for regime, mask in regimes.items():
        subset = result[mask]
        if subset.empty:
            continue
        rows.append(
            {
                "modelo": model_name,
                "regimen": regime,
                "n": len(subset),
                "mae": subset["error_abs"].mean(),
                "rmse": float(np.sqrt(np.mean(subset["error"] ** 2))),
                "bias": subset["error"].mean(),
            }
        )

    return pd.DataFrame(rows)


def review_anodic_encoding(dataset: pd.DataFrame) -> pd.DataFrame:
    """Revisa ceros/nulos de NTEA, DTEA y SEA en las features agregadas."""
    columns = [
        column
        for column in ["ntea_sum", "ntea_max", "dtea_sum", "dtea_max", "sea_sum", "sea_max"]
        if column in dataset.columns
    ]
    rows: list[dict[str, Any]] = []
    for column in columns:
        rows.append(
            {
                "columna": column,
                "n": len(dataset),
                "nulos": int(dataset[column].isna().sum()),
                "ceros": int((dataset[column].fillna(np.nan) == 0).sum()),
                "positivos": int((dataset[column].fillna(0) > 0).sum()),
                "min": float(dataset[column].min(skipna=True)),
                "max": float(dataset[column].max(skipna=True)),
            }
        )
    return pd.DataFrame(rows)


def write_summary(
    baselines: pd.DataFrame,
    ablation: pd.DataFrame,
    regimes: pd.DataFrame,
    ea_review: pd.DataFrame,
    output_path: Path,
) -> None:
    """Guarda respuestas concretas de V2."""
    test_baselines = baselines[baselines["segmento"] == "test"].sort_values("mae")
    best_ar = test_baselines.head(1).iloc[0]
    ablation_sorted = ablation.sort_values("mae")
    best_ablation = ablation_sorted.head(1).iloc[0]
    stage_a = ablation[ablation["etapa"] == "A_historia_termica"].iloc[0]
    stage_c = ablation[ablation["etapa"] == "C_B_mas_quimica"].iloc[0]
    stage_d = ablation[ablation["etapa"] == "D_C_mas_electricas"].iloc[0]
    stage_e = ablation[ablation["etapa"] == "E_D_mas_efectos_anodicos"].iloc[0]
    stage_f = ablation[ablation["etapa"] == "F_E_mas_control_operacion"].iloc[0]

    lines = [
        "RESUMEN V2 ITM/SEMITURNO",
        "=" * 60,
        "",
        "1) Baselines autorregresivos",
        f"- Mejor baseline/modelo AR en test: {best_ar['baseline']} con MAE {best_ar['mae']:.3f} C.",
        "- Comparar contra el modelo general actual: MAE ~4.99 C.",
        "",
        "2) Ablation study",
        f"- A historia termica sola: MAE {stage_a['mae']:.3f} C.",
        f"- C con quimica: MAE {stage_c['mae']:.3f} C.",
        f"- D con electricas IMM/V_ACD/proxy: MAE {stage_d['mae']:.3f} C.",
        f"- E agregando EA: MAE {stage_e['mae']:.3f} C.",
        f"- F completo control/operacion: MAE {stage_f['mae']:.3f} C.",
        f"- Mejor etapa: {best_ablation['etapa']} con MAE {best_ablation['mae']:.3f} C.",
        "",
        "3) Respuestas concretas",
        f"- AR(2)/AR(3) reducen la ventaja si quedan cerca de 4.99 C; si quedan arriba, el proceso agrega senal real.",
        f"- La mejora de A a F mide cuanto aporta ITM extra sobre historia termica: {stage_a['mae'] - stage_f['mae']:.3f} C de MAE.",
        f"- Electricas aportan de C a D: {stage_c['mae'] - stage_d['mae']:.3f} C de MAE.",
        f"- EA aporta de D a E: {stage_d['mae'] - stage_e['mae']:.3f} C de MAE global.",
        f"- Control/operacion aporta de E a F: {stage_e['mae'] - stage_f['mae']:.3f} C de MAE.",
        "",
        "4) Codificacion EA",
        ea_review.to_string(index=False),
        "",
        "5) Baselines test",
        test_baselines.round(4).to_string(index=False),
        "",
        "6) Ablation completo",
        ablation.round(4).to_string(index=False),
        "",
        "7) Regimen termico",
        regimes.round(4).to_string(index=False),
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_doc(
    baselines: pd.DataFrame,
    ablation: pd.DataFrame,
    regimes: pd.DataFrame,
    output_path: Path,
) -> None:
    """Guarda documento V2 corto y legible."""
    test_baselines = baselines[baselines["segmento"] == "test"].sort_values("mae")
    best_ar = test_baselines.iloc[0]
    stage_a = ablation[ablation["etapa"] == "A_historia_termica"].iloc[0]
    stage_f = ablation[ablation["etapa"] == "F_E_mas_control_operacion"].iloc[0]
    hot_target = regimes[regimes["regimen"] == "TB_objetivo >= 970"].iloc[0]
    strong_heating = regimes[regimes["regimen"] == "delta_tb >= 10"].iloc[0]

    lines = [
        "# Revision V2 ITM",
        "",
        "Esta revision usa solo ITM/semiturno y mantiene el mismo split temporal.",
        "",
        "## Resultado",
        "",
        f"- Mejor modelo simple de historia de TB: {best_ar['baseline']} con MAE {best_ar['mae']:.3f} C.",
        f"- Historia termica sola: MAE {stage_a['mae']:.3f} C.",
        f"- Modelo ITM completo: MAE {stage_f['mae']:.3f} C.",
        "",
        "ITM aporta senal real sobre la historia de TB: mejora el MAE en "
        f"{stage_a['mae'] - stage_f['mae']:.3f} C.",
        "",
        "## Alturas y movimientos",
        "",
        "- Se usan altura de bano y metal medidas/arrastradas: `HB`/`HBD` y `HM`/`HMD`.",
        "- Se agregaron los objetivos `HBC` y `HMC`, junto con el desvio de altura.",
        "- El modelo ve cuantos semiturnos el bano estuvo por debajo de su objetivo.",
        "- `MBLC` positivo significa colada de bano; negativo, agregado de bano.",
        "- No existe `MBLM` ni `MMLC` en el ITM entregado; no hay masa de metal disponible.",
        "",
        "## Experimento del paper",
        "",
        "- Se probo una categoria de etapa de vida (arranque, estable y final).",
        "- No mejoro el modelo general de temperatura, por lo que no se usa en el modelo final.",
        "- Sala y grupo ya quedan disponibles como contexto de las cubas.",
        "",
        "## Limite actual",
        "",
        f"- Cuando la TB objetivo termina en 970 C o mas: MAE {hot_target['mae']:.3f} C.",
        f"- En calentamientos fuertes (delta TB >= 10 C): MAE {strong_heating['mae']:.3f} C.",
        "- Sigue faltando informacion para explicar por que algunas cubas se calientan fuerte.",
        "",
        "Los CSV de `outputs/metricas` conservan el detalle tecnico de cada prueba.",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Ejecuta todos los experimentos V2."""
    print_title("REVISION V2 ITM/SEMITURNO")

    dataset_path = DATA_PROCESSED_DIR / "dataset_modelado_v1.parquet"
    if not dataset_path.exists():
        raise FileNotFoundError("Falta data/processed/dataset_modelado_v1.parquet")

    dataset = pd.read_parquet(dataset_path)
    dataset = split_dataset(dataset)

    print_title("1. Baselines autorregresivos")
    baselines = evaluate_autoregressive_baselines(dataset)
    print(baselines[baselines["segmento"] == "test"].sort_values("mae").round(3).to_string(index=False))

    print_title("2. Ablation study")
    ablation = run_ablation_study(dataset)
    print(ablation.round(3).to_string(index=False))

    print_title("3. Regimen termico")
    test = dataset[dataset["segmento_temporal"] == "test"].copy()
    feature_sets = ablation_feature_sets(dataset)
    best_features = feature_sets["F_E_mas_control_operacion"]
    categorical = [
        column for column in best_features if column in {"SALA", "GRUPO", "FASE_VIDA"}
    ]
    numeric = [column for column in best_features if column not in categorical]
    train = dataset[dataset["segmento_temporal"] == "train"].copy()
    model = train_hgb_delta(train, best_features, categorical, numeric)
    prediction = test["tb_inicial"].to_numpy() + model.predict(test[best_features])
    regimes = evaluate_regimes(test, prediction, "F_E_mas_control_operacion")
    print(regimes.round(3).to_string(index=False))

    print_title("4. Revision EA nulos/ceros")
    ea_review = review_anodic_encoding(dataset)
    print(ea_review.to_string(index=False))

    print_title("5. Guardando salidas")
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    baselines_path = METRICS_DIR / "v2_baselines_autorregresivos.csv"
    ablation_path = METRICS_DIR / "v2_ablation_study.csv"
    regimes_path = METRICS_DIR / "v2_regimen_termico.csv"
    ea_path = METRICS_DIR / "v2_revision_efectos_anodicos.csv"
    summary_path = OUTPUTS_DIR / "resumen_v2_itm.txt"
    doc_path = DOCS_DIR / "REVISION_V2_ITM.md"

    baselines.to_csv(baselines_path, index=False, encoding="utf-8-sig")
    ablation.to_csv(ablation_path, index=False, encoding="utf-8-sig")
    regimes.to_csv(regimes_path, index=False, encoding="utf-8-sig")
    ea_review.to_csv(ea_path, index=False, encoding="utf-8-sig")
    write_summary(baselines, ablation, regimes, ea_review, summary_path)
    write_doc(baselines, ablation, regimes, doc_path)

    print("Archivos generados:")
    print(f"- {baselines_path}")
    print(f"- {ablation_path}")
    print(f"- {regimes_path}")
    print(f"- {ea_path}")
    print(f"- {summary_path}")
    print(f"- {doc_path}")


if __name__ == "__main__":
    main()
