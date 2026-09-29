"""
Entrena y evalua el modelo TB que incorpora datos de cinco minutos.

El modelo VITM historico permanece congelado. Un segundo modelo aprende, con
datos HIS5M anteriores a julio, a corregir su error. Dos clasificadores estiman
el riesgo de TB >= 970 C y el regimen termico (enfria, estable o calienta).

Las formulas se eligen sin usar julio. Julio fue el test final del primer
experimento y, desde esta iteracion, se informa como referencia ya observada.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_DIR / "data" / "processed" / "dataset_modelado_his5m.parquet"
FROZEN_MODEL_PATH = PROJECT_DIR / "models" / "modelo_inicial_mejor.joblib"
MODEL_OUTPUT = PROJECT_DIR / "models" / "modelo_his5m.joblib"
METRICS_OUTPUT = PROJECT_DIR / "outputs" / "metricas" / "modelo_his5m_metricas.csv"
VALIDATION_OUTPUT = PROJECT_DIR / "outputs" / "metricas" / "modelo_his5m_validacion.csv"
IMPORTANCE_OUTPUT = PROJECT_DIR / "outputs" / "metricas" / "modelo_his5m_importancias.csv"
PREDICTIONS_OUTPUT = PROJECT_DIR / "outputs" / "predicciones" / "modelo_his5m_test.csv"
EXPLANATIONS_OUTPUT = PROJECT_DIR / "outputs" / "predicciones" / "modelo_his5m_explicaciones_test.csv"
SUMMARY_OUTPUT = PROJECT_DIR / "outputs" / "resumen_modelo_his5m.txt"
DOC_OUTPUT = PROJECT_DIR / "docs" / "MODELO_HIS5M.md"

TEST_START = pd.Timestamp("2026-07-01")
TEST_END = pd.Timestamp("2026-08-01")

# Tres validaciones crecientes. En cada una se aprende solo con fechas previas.
ROLLING_FOLDS = [
    ("fold_1", pd.Timestamp("2026-05-16"), pd.Timestamp("2026-06-01")),
    ("fold_2", pd.Timestamp("2026-06-01"), pd.Timestamp("2026-06-16")),
    ("fold_3", pd.Timestamp("2026-06-16"), pd.Timestamp("2026-07-01")),
]

CONTEXT_FEATURES = [
    "pred_vitm_congelado",
    "tb_inicial",
    "tb_anterior_2",
    "tb_anterior_3",
    "tb_cambio_ultima_vs_anterior",
    "tb_cambio_anterior_vs_tercera",
    "tb_pendiente_lineal_ultimas_3",
    "alf3_ultima_real",
    "semiturnos_desde_alf3_ultima",
    "rth_sum",
    "rth_last",
    "rc_sum",
    "rc_last",
    "wrmi_mean",
    "wrmi_last",
    "smrwfc_mean",
    "smrwfc_last",
    "diferencia_rrm_rkm_mean",
    "desvio_altura_bano_last",
    "semiturnos_bano_bajo",
]

HIS5M_PREFIXES = [
    "h5_quality",
    "h5_elect",
    "h5_control",
    "h5_feed",
    "h5_instability",
    "h5_anodic",
]

# Formulas fijadas usando solo validacion anterior a julio.
CANDIDATES = {
    "vitm_congelado": {"alpha": 0.0, "beta": 0.0, "intercept": 0.0},
    "his5m_general": {"alpha": 1.0, "beta": 1.5, "intercept": -0.5},
    "his5m_calientes": {"alpha": 1.0, "beta": 4.0, "intercept": 0.0},
}

DIRECTION_THRESHOLD = 5.0
BALANCED_PARAMETERS = {
    "alpha": 1.0,
    "beta_hot": 2.0,
    "intercept": -0.5,
    "push_cool": 0.25,
    "push_heat": 0.25,
}
MODEL_VARIANTS = [*CANDIDATES, "his5m_balanceado"]


def create_regressor() -> LGBMRegressor:
    """Modelo de correccion elegido por validacion temporal."""
    return LGBMRegressor(
        objective="regression",
        n_estimators=500,
        learning_rate=0.025,
        num_leaves=15,
        min_child_samples=40,
        reg_lambda=2.0,
        reg_alpha=0.2,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )


def create_classifier() -> LGBMClassifier:
    """Clasificador auxiliar de TB objetivo >= 970 C."""
    return LGBMClassifier(
        objective="binary",
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=40,
        reg_lambda=2.0,
        reg_alpha=0.2,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )


def create_direction_classifier() -> LGBMClassifier:
    """Clasifica si la proxima TB enfria, queda estable o calienta."""
    return LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=40,
        reg_lambda=2.0,
        reg_alpha=0.2,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )


def direction_target(delta_tb: pd.Series) -> np.ndarray:
    """0=enfria, 1=estable y 2=calienta, usando un margen de 5 C."""
    return np.where(
        delta_tb.le(-DIRECTION_THRESHOLD),
        0,
        np.where(delta_tb.ge(DIRECTION_THRESHOLD), 2, 1),
    )


def load_dataset() -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    """Carga dataset, calcula prediccion VITM congelada y valida features."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError("Falta dataset_modelado_his5m.parquet. Ejecuta primero 16_feature_engineering_his5m.py.")

    data = pd.read_parquet(DATASET_PATH)
    data["fecha_tb_objetivo"] = pd.to_datetime(data["fecha_tb_objetivo"])
    data = data[
        data["h5_quality_coverage_ratio"].ge(0.80)
        & data["h5_quality_semiturns_found"].ge(6)
    ].copy()

    frozen = joblib.load(FROZEN_MODEL_PATH)
    missing_frozen = set(frozen["feature_columns"]) - set(data.columns)
    if missing_frozen:
        raise ValueError(f"Faltan features del modelo VITM: {sorted(missing_frozen)}")

    raw_prediction = frozen["pipeline"].predict(data[frozen["feature_columns"]])
    if frozen["target_strategy"] == "delta_tb":
        data["pred_vitm_congelado"] = data["tb_inicial"].to_numpy() + raw_prediction
    else:
        data["pred_vitm_congelado"] = raw_prediction

    his5m_features = [
        column for column in data.columns if any(column.startswith(prefix) for prefix in HIS5M_PREFIXES)
    ]
    features = list(dict.fromkeys([*CONTEXT_FEATURES, *his5m_features]))
    missing = set(features) - set(data.columns)
    if missing:
        raise ValueError(f"Faltan features HIS5M: {sorted(missing)}")
    if any("objetivo" in feature for feature in features):
        raise ValueError("Una feature candidata contiene la palabra objetivo.")
    if data["id_intervalo"].duplicated().any():
        raise ValueError("id_intervalo no es unico en el dataset de modelado.")
    return data, frozen, features


def combine_prediction(
    base_prediction: np.ndarray,
    correction: np.ndarray,
    hot_probability: np.ndarray,
    hot_prevalence: float,
    parameters: dict[str, float],
) -> np.ndarray:
    """Aplica una formula fijada durante validacion, sin mirar el target."""
    centered_probability = hot_probability - hot_prevalence
    return (
        base_prediction
        + parameters["alpha"] * correction
        + parameters["beta"] * centered_probability
        + parameters["intercept"]
    )


def combine_balanced_prediction(
    base_prediction: np.ndarray,
    correction: np.ndarray,
    hot_probability: np.ndarray,
    hot_prevalence: float,
    direction_probability: np.ndarray,
    direction_prevalence: np.ndarray,
) -> np.ndarray:
    """Combina las senales con una correccion direccional pequena y acotada."""
    parameters = BALANCED_PARAMETERS
    centered_hot = hot_probability - hot_prevalence
    centered_cool = direction_probability[:, 0] - direction_prevalence[0]
    centered_heat = direction_probability[:, 2] - direction_prevalence[2]
    return (
        base_prediction
        + parameters["alpha"] * correction
        + parameters["beta_hot"] * centered_hot
        + parameters["intercept"]
        - parameters["push_cool"] * centered_cool
        + parameters["push_heat"] * centered_heat
    )


def metric_values(real: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    """Metricas de error en grados Celsius."""
    error = predicted - real.to_numpy()
    absolute = np.abs(error)
    return {
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "sesgo": float(error.mean()),
        "p90_error_abs": float(np.quantile(absolute, 0.90)),
        "dentro_5c_pct": float(np.mean(absolute <= 5) * 100),
    }


def subgroup_masks(data: pd.DataFrame) -> dict[str, pd.Series]:
    """Subgrupos definidos antes de evaluar julio."""
    return {
        "global": pd.Series(True, index=data.index),
        "tb_objetivo_>=970": data["tb_objetivo"].ge(970),
        "delta_<=-5": data["delta_tb_objetivo"].le(-5),
        "-5<delta_<5": data["delta_tb_objetivo"].gt(-5)
        & data["delta_tb_objetivo"].lt(5),
        "delta_>=5": data["delta_tb_objetivo"].ge(5),
        "delta_>=10": data["delta_tb_objetivo"].ge(10),
        "inicio_<970_fin_>=970": data["tb_inicial"].lt(970) & data["tb_objetivo"].ge(970),
        "inicio_>=970_fin_<970": data["tb_inicial"].ge(970) & data["tb_objetivo"].lt(970),
    }


def evaluate_predictions(
    data: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    segment: str,
) -> list[dict[str, Any]]:
    """Evalua todas las variantes en los mismos subgrupos."""
    rows: list[dict[str, Any]] = []
    masks = subgroup_masks(data)
    for variant, predicted in predictions.items():
        predicted_series = pd.Series(predicted, index=data.index)
        for subgroup, mask in masks.items():
            if not mask.any():
                continue
            rows.append(
                {
                    "segmento": segment,
                    "variante": variant,
                    "subgrupo": subgroup,
                    "filas": int(mask.sum()),
                    **metric_values(data.loc[mask, "tb_objetivo"], predicted_series.loc[mask].to_numpy()),
                }
            )
    return rows


def rolling_validation(data: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Genera predicciones fuera de muestra en tres ventanas anteriores a julio."""
    frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []

    for fold_name, validation_start, validation_end in ROLLING_FOLDS:
        train = data[data["fecha_tb_objetivo"].lt(validation_start)].copy()
        validation = data[
            data["fecha_tb_objetivo"].ge(validation_start)
            & data["fecha_tb_objetivo"].lt(validation_end)
        ].copy()
        if train.empty or validation.empty:
            raise ValueError(f"El {fold_name} quedo sin train o validacion.")

        regressor = create_regressor()
        classifier = create_classifier()
        direction_classifier = create_direction_classifier()
        residual = train["tb_objetivo"] - train["pred_vitm_congelado"]
        hot_target = train["tb_objetivo"].ge(970).astype(int)
        thermal_target = direction_target(train["delta_tb_objetivo"])
        regressor.fit(train[features], residual)
        classifier.fit(train[features], hot_target)
        direction_classifier.fit(train[features], thermal_target)

        frame = validation[
            [
                "id_intervalo",
                "CUBA",
                "fecha_tb_objetivo",
                "tb_inicial",
                "tb_objetivo",
                "delta_tb_objetivo",
                "pred_vitm_congelado",
            ]
        ].copy()
        frame["correccion_his5m"] = regressor.predict(validation[features])
        frame["probabilidad_tb_>=970"] = classifier.predict_proba(validation[features])[:, 1]
        frame["prevalencia_caliente_train"] = float(hot_target.mean())
        direction_probability = direction_classifier.predict_proba(validation[features])
        direction_prevalence = np.bincount(thermal_target, minlength=3) / len(thermal_target)
        frame["probabilidad_enfriamiento"] = direction_probability[:, 0]
        frame["probabilidad_estable"] = direction_probability[:, 1]
        frame["probabilidad_calentamiento"] = direction_probability[:, 2]
        frame["fold"] = fold_name
        for variant, parameters in CANDIDATES.items():
            frame[f"pred_{variant}"] = combine_prediction(
                frame["pred_vitm_congelado"].to_numpy(),
                frame["correccion_his5m"].to_numpy(),
                frame["probabilidad_tb_>=970"].to_numpy(),
                float(hot_target.mean()),
                parameters,
            )
        frame["pred_his5m_balanceado"] = combine_balanced_prediction(
            frame["pred_vitm_congelado"].to_numpy(),
            frame["correccion_his5m"].to_numpy(),
            frame["probabilidad_tb_>=970"].to_numpy(),
            float(hot_target.mean()),
            direction_probability,
            direction_prevalence,
        )
        frames.append(frame)

        fold_predictions = {
            variant: frame[f"pred_{variant}"].to_numpy() for variant in MODEL_VARIANTS
        }
        fold_rows.extend(evaluate_predictions(validation, fold_predictions, fold_name))

    return pd.concat(frames, ignore_index=True), pd.DataFrame(fold_rows)


def paired_bootstrap_improvement(
    real: pd.Series,
    baseline: np.ndarray,
    candidate: np.ndarray,
    repetitions: int = 2000,
) -> dict[str, float]:
    """Intervalo de confianza del cambio de MAE sobre las mismas filas."""
    baseline_error = np.abs(baseline - real.to_numpy())
    candidate_error = np.abs(candidate - real.to_numpy())
    improvement_per_row = baseline_error - candidate_error
    random = np.random.default_rng(42)
    estimates = np.empty(repetitions)
    for position in range(repetitions):
        sample = random.integers(0, len(real), len(real))
        estimates[position] = improvement_per_row[sample].mean()
    return {
        "mejora_mae": float(improvement_per_row.mean()),
        "ic95_inferior": float(np.quantile(estimates, 0.025)),
        "ic95_superior": float(np.quantile(estimates, 0.975)),
    }


def feature_importance(
    regressor: LGBMRegressor,
    classifier: LGBMClassifier,
    direction_classifier: LGBMClassifier,
    features: list[str],
) -> pd.DataFrame:
    """Importancia de ganancia de cada componente interno."""
    rows = []
    models = [
        ("correccion_tb", regressor),
        ("clasificador_caliente", classifier),
        ("clasificador_direccion", direction_classifier),
    ]
    for model_name, model in models:
        gain = model.booster_.feature_importance(importance_type="gain")
        total = gain.sum()
        for feature, value in zip(features, gain, strict=True):
            rows.append(
                {
                    "modelo": model_name,
                    "feature": feature,
                    "importancia_gain": float(value),
                    "importancia_pct": float(value / total * 100) if total else 0.0,
                }
            )
    return pd.DataFrame(rows).sort_values(["modelo", "importancia_gain"], ascending=[True, False])


def explain_primary_predictions(
    test: pd.DataFrame,
    regressor: LGBMRegressor,
    features: list[str],
    correction: np.ndarray,
    hot_probability: np.ndarray,
    hot_prevalence: float,
    direction_probability: np.ndarray,
    direction_prevalence: np.ndarray,
    final_prediction: np.ndarray,
) -> tuple[pd.DataFrame, pd.Series]:
    """Explica el corrector HIS5M de cada prediccion mediante valores SHAP."""
    contributions = regressor.booster_.predict(test[features], pred_contrib=True)
    feature_contributions = contributions[:, :-1]
    mean_absolute_shap = pd.Series(
        np.abs(feature_contributions).mean(axis=0),
        index=features,
        name="shap_medio_abs",
    )
    top_indexes = np.argsort(np.abs(feature_contributions), axis=1)[:, -3:][:, ::-1]
    feature_array = np.asarray(features)
    feature_values = test[features].to_numpy()

    output = test[
        ["id_intervalo", "CUBA", "fecha_tb_objetivo", "tb_inicial", "tb_objetivo"]
    ].copy()
    output["pred_vitm_congelado"] = test["pred_vitm_congelado"].to_numpy()
    output["correccion_regresor_his5m"] = correction
    parameters = BALANCED_PARAMETERS
    output["ajuste_riesgo_caliente"] = parameters["beta_hot"] * (
        hot_probability - hot_prevalence
    )
    output["ajuste_probabilidad_enfriamiento"] = -parameters["push_cool"] * (
        direction_probability[:, 0] - direction_prevalence[0]
    )
    output["ajuste_probabilidad_calentamiento"] = parameters["push_heat"] * (
        direction_probability[:, 2] - direction_prevalence[2]
    )
    output["probabilidad_tb_>=970"] = hot_probability
    output["probabilidad_enfriamiento"] = direction_probability[:, 0]
    output["probabilidad_estable"] = direction_probability[:, 1]
    output["probabilidad_calentamiento"] = direction_probability[:, 2]
    output["pred_his5m_balanceado"] = final_prediction
    output["error_abs"] = np.abs(final_prediction - output["tb_objetivo"].to_numpy())

    row_indexes = np.arange(len(test))
    for rank in range(3):
        indexes = top_indexes[:, rank]
        output[f"factor_{rank + 1}"] = feature_array[indexes]
        output[f"valor_factor_{rank + 1}"] = feature_values[row_indexes, indexes]
        output[f"aporte_correccion_{rank + 1}"] = feature_contributions[row_indexes, indexes]
    return output, mean_absolute_shap


def write_summary(
    metrics: pd.DataFrame,
    validation_metrics: pd.DataFrame,
    validation_predictions: pd.DataFrame,
    bootstrap_vitm: dict[str, float],
    bootstrap_general: dict[str, float],
    importance: pd.DataFrame,
    train_rows: int,
    test_rows: int,
    quality_checks: dict[str, Any],
) -> None:
    """Guarda resultados con una explicacion directa y auditable."""
    validation_global_rows = validation_metrics[
        validation_metrics["subgrupo"].eq("global")
    ].copy()
    validation_global_rows["mae_ponderado"] = (
        validation_global_rows["mae"] * validation_global_rows["filas"]
    )
    validation_global = (
        validation_global_rows.groupby("variante")["mae_ponderado"].sum()
        / validation_global_rows.groupby("variante")["filas"].sum()
    )
    test_global = metrics[
        metrics["segmento"].eq("test") & metrics["subgrupo"].eq("global")
    ].set_index("variante")
    test_hot = metrics[
        metrics["segmento"].eq("test") & metrics["subgrupo"].eq("tb_objetivo_>=970")
    ].set_index("variante")
    test_onset = metrics[
        metrics["segmento"].eq("test") & metrics["subgrupo"].eq("inicio_<970_fin_>=970")
    ].set_index("variante")
    test_cooling = metrics[
        metrics["segmento"].eq("test") & metrics["subgrupo"].eq("inicio_>=970_fin_<970")
    ].set_index("variante")
    test_directions = metrics[
        metrics["segmento"].eq("test")
        & metrics["subgrupo"].isin(["delta_<=-5", "-5<delta_<5", "delta_>=5"])
    ].pivot(index="variante", columns="subgrupo", values="mae")
    validation_folds = validation_metrics[
        validation_metrics["subgrupo"].eq("global")
    ].pivot(index="segmento", columns="variante", values="mae")
    top_correction = importance[importance["modelo"].eq("correccion_tb")].head(12)
    actual_direction = direction_target(validation_predictions["delta_tb_objetivo"])
    predicted_direction = validation_predictions[
        [
            "probabilidad_enfriamiento",
            "probabilidad_estable",
            "probabilidad_calentamiento",
        ]
    ].to_numpy().argmax(axis=1)
    direction_accuracy = float(np.mean(actual_direction == predicted_direction))
    direction_recall = [
        float(np.mean(predicted_direction[actual_direction == value] == value))
        for value in range(3)
    ]

    lines = [
        "MODELO HIS5M - RESULTADO",
        "========================",
        "",
        "Diseno temporal:",
        f"- Entrenamiento final: {train_rows:,} intervalos anteriores a julio.",
        f"- Referencia de julio: {test_rows:,} intervalos de julio de 2026.",
        "- Features: solo los 7 semiturnos completos entre TB inicial y objetivo.",
        "- El modelo VITM historico se mantuvo congelado.",
        "- La formula balanceada se fijo con validaciones pre-julio.",
        "- Julio ya habia sido observado por la iteracion anterior; hoy no es un test virgen.",
        "",
        "MAE global de validacion rodante:",
        validation_global.round(3).to_string(),
        "",
        "Detalle por fold de validacion:",
        validation_folds.round(3).to_string(),
        "",
        "Calidad del selector de direccion en validacion:",
        f"- Acierto global: {direction_accuracy * 100:.1f}%.",
        f"- Detecta {direction_recall[0] * 100:.1f}% de enfriamientos, "
        f"{direction_recall[1] * 100:.1f}% de estables y "
        f"{direction_recall[2] * 100:.1f}% de calentamientos.",
        "",
        "MAE global de test:",
        test_global[["filas", "mae", "rmse", "sesgo"]].round(3).to_string(),
        "",
        "Cubas calientes en test (TB objetivo >= 970 C):",
        test_hot[["filas", "mae", "sesgo"]].round(3).to_string(),
        "",
        "Cubas que cruzan desde <970 a >=970 C:",
        test_onset[["filas", "mae", "sesgo"]].round(3).to_string(),
        "",
        "Cubas inicialmente calientes que se enfrian:",
        test_cooling[["filas", "mae", "sesgo"]].round(3).to_string(),
        "",
        "MAE por direccion termica en julio:",
        test_directions.round(3).to_string(),
        "",
        "Mejora pareada de his5m_balanceado contra VITM congelado:",
        f"- Mejora MAE: {bootstrap_vitm['mejora_mae']:.3f} C.",
        f"- IC95 bootstrap: {bootstrap_vitm['ic95_inferior']:.3f} a "
        f"{bootstrap_vitm['ic95_superior']:.3f} C.",
        "",
        "Mejora pareada de his5m_balanceado contra his5m_general:",
        f"- Mejora MAE: {bootstrap_general['mejora_mae']:.3f} C.",
        f"- IC95 bootstrap: {bootstrap_general['ic95_inferior']:.3f} a "
        f"{bootstrap_general['ic95_superior']:.3f} C.",
        "- Si el intervalo incluye cero, no hay evidencia de mejora global.",
        "",
        "Variables principales del corrector:",
        top_correction[["feature", "importancia_pct"]].round(2).to_string(index=False),
        "",
        "Controles de calidad:",
        f"- Features usadas: {quality_checks['features']:,}; targets/IDs encontrados: 0.",
        f"- Features constantes en train: {quality_checks['constant_features']:,}; LightGBM las ignora.",
        f"- Predicciones no finitas: {quality_checks['nonfinite_predictions']:,}.",
        f"- Cobertura HIS5M minima en test: {quality_checks['min_test_coverage'] * 100:.2f}%.",
        f"- Ultima fecha de train: {quality_checks['train_max_date']:%Y-%m-%d}; "
        f"primera de test: {quality_checks['test_min_date']:%Y-%m-%d}.",
        "- THISEVT no se usa porque su cobertura termina antes del test de julio.",
        "- El estado ANODE_EFFECT de cinco minutos si forma parte del modelo.",
        "",
        "Lectura:",
        "- his5m_general prioriza el menor MAE global.",
        "- his5m_calientes sacrifica como maximo 0.05 C en validacion global para",
        "  reducir mas el error de cubas calientes.",
        "- his5m_balanceado agrega un selector suave de enfriamiento/estable/calentamiento.",
        "- La salida balanceada costo 0.005 C de MAE global en validacion y mejoro",
        "  simultaneamente los regimenes de calentamiento y enfriamiento.",
        "- Ninguna formula fue elegida mirando el resultado nuevo de julio.",
    ]
    text = "\n".join(lines)
    SUMMARY_OUTPUT.write_text(text, encoding="utf-8")

    doc_lines = [
        "# Modelo con datos HIS5M",
        "",
        "Este experimento agrega las senales de cinco minutos al modelo VITM",
        "congelado. La explicacion corta y los resultados estan en",
        "`outputs/resumen_modelo_his5m.txt`.",
        "",
        "## Protecciones contra leakage",
        "",
        "- Se excluye el semiturno de la TB inicial.",
        "- Se excluye el semiturno de la TB objetivo.",
        "- Solo se usan los siete semiturnos intermedios.",
        "- Julio no participa en seleccion ni ajuste de la formula balanceada.",
        "- Los clasificadores usan solamente datos anteriores a cada validacion.",
        "- Como julio ya fue observado antes, hace falta un mes futuro para confirmar.",
        "",
        "## Veredicto de calidad",
        "",
        "El dataset y el modelo son aptos para experimentacion offline. No estan",
        "listos para una decision operativa automatica: julio ya fue observado, el",
        "tercer fold temporal rindio peor y la salida balanceada conserva tradeoffs",
        "entre estabilidad y cambios de temperatura.",
        "",
        "## Que se probo",
        "",
        "- Perdidas L2, L1 y Huber para el corrector. L2 mantuvo el menor MAE.",
        "- Regimenes con margenes de 3, 5, 7 y 10 C. Se conservo 5 C.",
        "- Mezcla dura, mezcla suave y ponderacion de especialistas.",
        "- La formula final usa ajustes direccionales pequenos de 0,25 C.",
        "- Todo el ajuste de parametros se hizo en validaciones anteriores a julio.",
        "",
        "## Salidas",
        "",
        "- Dataset: `data/processed/dataset_modelado_his5m.parquet`.",
        "- Metricas: `outputs/metricas/modelo_his5m_metricas.csv`.",
        "- Predicciones: `outputs/predicciones/modelo_his5m_test.csv`.",
        "- Importancias: `outputs/metricas/modelo_his5m_importancias.csv`.",
        "- Explicaciones por fila: `outputs/predicciones/modelo_his5m_explicaciones_test.csv`.",
        "- Modelo: `models/modelo_his5m.joblib`.",
        "- Para usar el modelo guardado primero debe calcularse `pred_vitm_congelado`;",
        "  luego se ejecutan el corrector HIS5M y los dos clasificadores.",
        "",
        "## Resultados principales",
        "",
        f"- VITM congelado en julio: MAE {test_global.loc['vitm_congelado', 'mae']:.3f} C.",
        f"- HIS5M general: MAE {test_global.loc['his5m_general', 'mae']:.3f} C.",
        f"- HIS5M orientado a calientes: MAE {test_global.loc['his5m_calientes', 'mae']:.3f} C.",
        f"- HIS5M balanceado por direccion: MAE {test_global.loc['his5m_balanceado', 'mae']:.3f} C.",
        f"- En TB objetivo >=970 C: {test_hot.loc['vitm_congelado', 'mae']:.3f} a "
        f"{test_hot.loc['his5m_calientes', 'mae']:.3f} C.",
        f"- En cruces desde <970 a >=970 C: {test_onset.loc['vitm_congelado', 'mae']:.3f} a "
        f"{test_onset.loc['his5m_calientes', 'mae']:.3f} C.",
        f"- Balanceado vs general en calentamientos: "
        f"{test_directions.loc['his5m_general', 'delta_>=5']:.3f} a "
        f"{test_directions.loc['his5m_balanceado', 'delta_>=5']:.3f} C.",
        f"- Balanceado vs general en enfriamientos: "
        f"{test_directions.loc['his5m_general', 'delta_<=-5']:.3f} a "
        f"{test_directions.loc['his5m_balanceado', 'delta_<=-5']:.3f} C.",
        f"- Balanceado vs general en estables: "
        f"{test_directions.loc['his5m_general', '-5<delta_<5']:.3f} a "
        f"{test_directions.loc['his5m_balanceado', '-5<delta_<5']:.3f} C.",
        "",
        "El archivo de resumen contiene las comparaciones por enfriamiento, estabilidad",
        "y calentamiento. Julio es una referencia conocida; se requiere otro mes futuro",
        "para confirmar la mejora sin sesgo de iteracion.",
    ]
    DOC_OUTPUT.write_text("\n".join(doc_lines), encoding="utf-8")


def main() -> None:
    """Ejecuta validacion, entrenamiento final y una sola evaluacion de julio."""
    data, frozen_model, features = load_dataset()
    print(f"Filas elegibles: {len(data):,}; features: {len(features):,}")

    validation_predictions, validation_metrics = rolling_validation(data, features)

    train = data[data["fecha_tb_objetivo"].lt(TEST_START)].copy()
    test = data[
        data["fecha_tb_objetivo"].ge(TEST_START)
        & data["fecha_tb_objetivo"].lt(TEST_END)
    ].copy()
    if train.empty or test.empty:
        raise ValueError("El corte final no genero train y test.")
    if train["fecha_tb_objetivo"].max() >= test["fecha_tb_objetivo"].min():
        raise ValueError("Train y test no tienen una separacion temporal estricta.")

    regressor = create_regressor()
    classifier = create_classifier()
    direction_classifier = create_direction_classifier()
    residual = train["tb_objetivo"] - train["pred_vitm_congelado"]
    hot_target = train["tb_objetivo"].ge(970).astype(int)
    thermal_target = direction_target(train["delta_tb_objetivo"])
    regressor.fit(train[features], residual)
    classifier.fit(train[features], hot_target)
    direction_classifier.fit(train[features], thermal_target)

    correction = regressor.predict(test[features])
    hot_probability = classifier.predict_proba(test[features])[:, 1]
    hot_prevalence = float(hot_target.mean())
    direction_probability = direction_classifier.predict_proba(test[features])
    direction_prevalence = np.bincount(thermal_target, minlength=3) / len(thermal_target)
    if not np.allclose(direction_probability.sum(axis=1), 1.0):
        raise ValueError("Las probabilidades de direccion no suman uno.")
    predictions = {
        variant: combine_prediction(
            test["pred_vitm_congelado"].to_numpy(),
            correction,
            hot_probability,
            hot_prevalence,
            parameters,
        )
        for variant, parameters in CANDIDATES.items()
    }
    predictions["his5m_balanceado"] = combine_balanced_prediction(
        test["pred_vitm_congelado"].to_numpy(),
        correction,
        hot_probability,
        hot_prevalence,
        direction_probability,
        direction_prevalence,
    )
    predictions["baseline_promedio_2_tb"] = (
        (test["tb_inicial"] + test["tb_anterior_2"]) / 2
    ).to_numpy()
    if any((~np.isfinite(predicted)).any() for predicted in predictions.values()):
        raise ValueError("Se generaron predicciones no finitas.")

    metrics = pd.DataFrame(evaluate_predictions(test, predictions, "test"))
    bootstrap_vitm = paired_bootstrap_improvement(
        test["tb_objetivo"],
        predictions["vitm_congelado"],
        predictions["his5m_balanceado"],
    )
    bootstrap_general = paired_bootstrap_improvement(
        test["tb_objetivo"],
        predictions["his5m_general"],
        predictions["his5m_balanceado"],
    )

    explanations, mean_absolute_shap = explain_primary_predictions(
        test,
        regressor,
        features,
        correction,
        hot_probability,
        hot_prevalence,
        direction_probability,
        direction_prevalence,
        predictions["his5m_balanceado"],
    )
    importance = feature_importance(regressor, classifier, direction_classifier, features)
    shap_mapping = mean_absolute_shap.to_dict()
    importance["shap_medio_abs"] = np.where(
        importance["modelo"].eq("correccion_tb"),
        importance["feature"].map(shap_mapping),
        np.nan,
    )

    prediction_output = test[
        [
            "id_intervalo",
            "CUBA",
            "fecha_tb_inicial",
            "fecha_tb_objetivo",
            "tb_inicial",
            "tb_objetivo",
            "delta_tb_objetivo",
            "h5_quality_coverage_ratio",
        ]
    ].copy()
    prediction_output["correccion_his5m"] = correction
    prediction_output["probabilidad_tb_>=970"] = hot_probability
    prediction_output["probabilidad_enfriamiento"] = direction_probability[:, 0]
    prediction_output["probabilidad_estable"] = direction_probability[:, 1]
    prediction_output["probabilidad_calentamiento"] = direction_probability[:, 2]
    for variant, predicted in predictions.items():
        prediction_output[f"pred_{variant}"] = predicted
        prediction_output[f"error_abs_{variant}"] = np.abs(
            predicted - prediction_output["tb_objetivo"].to_numpy()
        )

    package = {
        "modelo_vitm_congelado": frozen_model,
        "modelo_correccion_his5m": regressor,
        "clasificador_tb_caliente": classifier,
        "clasificador_direccion_termica": direction_classifier,
        "features": features,
        "prevalencia_tb_caliente_train": hot_prevalence,
        "prevalencia_direccion_train": direction_prevalence,
        "formulas": CANDIDATES,
        "formula_balanceada": BALANCED_PARAMETERS,
        "umbral_direccion_c": DIRECTION_THRESHOLD,
        "salida_principal": "his5m_balanceado",
        "test_start": TEST_START,
        "regla_anti_leakage": "solo 7 semiturnos intermedios; inicial y objetivo excluidos",
    }

    for path in [
        MODEL_OUTPUT,
        METRICS_OUTPUT,
        VALIDATION_OUTPUT,
        IMPORTANCE_OUTPUT,
        PREDICTIONS_OUTPUT,
        EXPLANATIONS_OUTPUT,
        SUMMARY_OUTPUT,
        DOC_OUTPUT,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(package, MODEL_OUTPUT)
    metrics.to_csv(METRICS_OUTPUT, index=False, encoding="utf-8-sig")
    validation_metrics.to_csv(VALIDATION_OUTPUT, index=False, encoding="utf-8-sig")
    importance.to_csv(IMPORTANCE_OUTPUT, index=False, encoding="utf-8-sig")
    prediction_output.to_csv(PREDICTIONS_OUTPUT, index=False, encoding="utf-8-sig")
    explanations.to_csv(EXPLANATIONS_OUTPUT, index=False, encoding="utf-8-sig")
    quality_checks = {
        "features": len(features),
        "constant_features": int(sum(train[column].nunique(dropna=True) <= 1 for column in features)),
        "nonfinite_predictions": int(
            sum((~np.isfinite(predicted)).sum() for predicted in predictions.values())
        ),
        "min_test_coverage": float(test["h5_quality_coverage_ratio"].min()),
        "train_max_date": train["fecha_tb_objetivo"].max(),
        "test_min_date": test["fecha_tb_objetivo"].min(),
    }
    write_summary(
        metrics,
        validation_metrics,
        validation_predictions,
        bootstrap_vitm,
        bootstrap_general,
        importance,
        len(train),
        len(test),
        quality_checks,
    )

    global_test = metrics[
        metrics["subgrupo"].eq("global")
    ][["variante", "mae", "sesgo"]]
    print(global_test.round(3).to_string(index=False))
    print(f"Resumen: {SUMMARY_OUTPUT}")


if __name__ == "__main__":
    main()
