"""
Entrena el modelo TB con libro mayor termico HIS5M.

La seleccion de features, hiperparametros y formulas usa tres validaciones
anteriores a julio. Julio se informa como referencia ya observada, no como una
prueba final nueva.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_DIR / "data" / "processed" / "dataset_modelado_his5m_ledger.parquet"
PREVIOUS_MODEL_PATH = PROJECT_DIR / "models" / "modelo_his5m.joblib"
MODEL_OUTPUT = PROJECT_DIR / "models" / "modelo_his5m_ledger.joblib"
METRICS_OUTPUT = PROJECT_DIR / "outputs" / "metricas" / "modelo_his5m_ledger_metricas.csv"
VALIDATION_OUTPUT = PROJECT_DIR / "outputs" / "metricas" / "modelo_his5m_ledger_validacion.csv"
IMPORTANCE_OUTPUT = PROJECT_DIR / "outputs" / "metricas" / "modelo_his5m_ledger_importancias.csv"
PREDICTIONS_OUTPUT = PROJECT_DIR / "outputs" / "predicciones" / "modelo_his5m_ledger_test.csv"
EXPLANATIONS_OUTPUT = (
    PROJECT_DIR / "outputs" / "predicciones" / "modelo_his5m_ledger_explicaciones_test.csv"
)
SUMMARY_OUTPUT = PROJECT_DIR / "outputs" / "resumen_modelo_his5m_ledger.txt"
DOC_OUTPUT = PROJECT_DIR / "docs" / "MODELO_HIS5M_LEDGER.md"

TEST_START = pd.Timestamp("2026-07-01")
TEST_END = pd.Timestamp("2026-08-01")
ROLLING_FOLDS = [
    ("fold_1", pd.Timestamp("2026-05-16"), pd.Timestamp("2026-06-01")),
    ("fold_2", pd.Timestamp("2026-06-01"), pd.Timestamp("2026-06-16")),
    ("fold_3", pd.Timestamp("2026-06-16"), pd.Timestamp("2026-07-01")),
]

# Correccion exclusiva de la salida para cubas calientes. Los parametros se
# eligieron con los tres folds anteriores a julio. La regularizacion evita que
# pocas mediciones de una cuba produzcan una correccion grande.
POT_CALIBRATION_SHRINKAGE = 20.0
POT_CALIBRATION_STRENGTH = 0.80

# Estas 25 posiciones temporales se eligieron con datos anteriores al primer
# fold (antes del 16 de mayo). Conservan el orden sin sumar toda la secuencia.
SEQUENCE_FEATURES = [
    "h5_ledger_sequence_feed_scdocc_lag3_mean",
    "h5_ledger_sequence_control_wrmfc_lag4_mean",
    "h5_ledger_sequence_feed_talim_0_lag5_fraction",
    "h5_ledger_sequence_feed_talim_0_lag6_fraction",
    "h5_ledger_sequence_feed_cdordfc_lag3_mean",
    "h5_ledger_sequence_feed_talim_6_lag5_fraction",
    "h5_ledger_sequence_control_wrmfc_lag3_mean",
    "h5_ledger_sequence_feed_talim_5_lag2_fraction",
    "h5_ledger_sequence_feed_talim_1_lag4_fraction",
    "h5_ledger_sequence_feed_cdordfc_lag5_mean",
    "h5_ledger_sequence_control_wrmfc_lag5_mean",
    "h5_ledger_sequence_feed_scdocc_lag4_mean",
    "h5_ledger_sequence_feed_talim_4_lag4_fraction",
    "h5_ledger_sequence_feed_pal2o3fc_limpio_lag4_mean",
    "h5_ledger_sequence_control_wrmfc_lag7_mean",
    "h5_ledger_sequence_feed_pentefc_lag5_mean",
    "h5_ledger_sequence_feed_cdordfc_lag6_mean",
    "h5_ledger_sequence_feed_pentefc_lag7_mean",
    "h5_ledger_sequence_feed_talim_4_lag3_fraction",
    "h5_ledger_sequence_feed_cdordfc_lag4_mean",
    "h5_ledger_sequence_feed_pentefc_lag4_mean",
    "h5_ledger_sequence_elect_imfc_lag5_mean",
    "h5_ledger_sequence_feed_talim_1_lag7_fraction",
    "h5_ledger_sequence_elect_dif_rm_rk_lag1_mean",
    "h5_ledger_sequence_elect_imfc_lag1_mean",
]

# Formulas fijadas solo con validacion anterior a julio.
FORMULAS = {
    "ledger_general": {
        "gamma_weighted": 0.0,
        "beta_hot": 1.0,
        "intercept": -0.25,
        "push_cool": 0.0,
        "push_heat": 0.0,
    },
    "ledger_balanceado": {
        "gamma_weighted": 0.0,
        "beta_hot": 1.0,
        "intercept": 0.0,
        "push_cool": 0.50,
        "push_heat": 0.25,
    },
    "ledger_calientes": {
        "gamma_weighted": 0.50,
        "beta_hot": 2.0,
        "intercept": 0.0,
        "push_cool": 0.0,
        "push_heat": 0.0,
    },
}

# Elegida por validacion pre-julio con MAE global limitado a 5 C. Acepta
# mayor error en enfriamientos para reducir calentamientos y cruces a >=970 C.
PRIORITY_HOT_FORMULA = {
    "gamma_weighted": 0.50,
    "beta_hot": 0.0,
    "intercept": 1.50,
    "push_cool": 0.50,
    "push_heat": 2.0,
}


def create_regressor() -> LGBMRegressor:
    """Regresor elegido por validacion temporal."""
    return LGBMRegressor(
        objective="regression",
        n_estimators=600,
        learning_rate=0.02,
        num_leaves=31,
        min_child_samples=80,
        reg_lambda=5.0,
        reg_alpha=0.5,
        colsample_bytree=0.75,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )


def create_hot_classifier() -> LGBMClassifier:
    """Clasificador auxiliar de TB objetivo mayor o igual a 970 C."""
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
    """Clasifica enfriamiento, estabilidad o calentamiento."""
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
    """0=enfria al menos 5 C, 1=estable y 2=calienta al menos 5 C."""
    return np.where(delta_tb.le(-5), 0, np.where(delta_tb.ge(5), 2, 1))


def load_dataset() -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    """Carga datos, reconstruye VITM y selecciona las familias congeladas."""
    data = pd.read_parquet(DATASET_PATH)
    data["fecha_tb_objetivo"] = pd.to_datetime(data["fecha_tb_objetivo"])
    data = data[
        data["h5_quality_coverage_ratio"].ge(0.80)
        & data["h5_quality_semiturns_found"].ge(6)
    ].copy()
    previous = joblib.load(PREVIOUS_MODEL_PATH)
    frozen = previous["modelo_vitm_congelado"]
    raw = frozen["pipeline"].predict(data[frozen["feature_columns"]])
    data["pred_vitm_congelado"] = (
        data["tb_inicial"].to_numpy() + raw
        if frozen["target_strategy"] == "delta_tb"
        else raw
    )

    legacy = previous["features"]
    quality = [
        column
        for column in data
        if column.startswith("h5_ledger_quality_")
        and not column.startswith("h5_ledger_band_")
    ]
    cumulative_feed = [column for column in data if column.startswith("h5_ledger_feed_")]
    feed_bands = [column for column in data if column.startswith("h5_ledger_band_")]
    control = [column for column in data if column.startswith("h5_ledger_control_")]
    features = list(
        dict.fromkeys(
            [
                *legacy,
                *quality,
                *cumulative_feed,
                *feed_bands,
                *control,
                *SEQUENCE_FEATURES,
            ]
        )
    )

    missing = set(features) - set(data.columns)
    if missing:
        raise ValueError(f"Faltan features ledger: {sorted(missing)}")
    forbidden = [
        feature
        for feature in features
        if "objetivo" in feature.lower() or feature.lower() in {"cuba", "id_intervalo"}
    ]
    if forbidden:
        raise ValueError(f"Features prohibidas: {forbidden}")
    if data["id_intervalo"].duplicated().any():
        raise ValueError("id_intervalo no es unico.")
    return data, frozen, features


def combine_predictions(
    base_prediction: np.ndarray,
    weighted_prediction: np.ndarray,
    hot_probability: np.ndarray,
    hot_prevalence: float,
    direction_probability: np.ndarray,
    direction_prevalence: np.ndarray,
    parameters: dict[str, float],
) -> np.ndarray:
    """Combina componentes internos con una formula congelada."""
    centered_hot = hot_probability - hot_prevalence
    centered_cool = direction_probability[:, 0] - direction_prevalence[0]
    centered_heat = direction_probability[:, 2] - direction_prevalence[2]
    return (
        base_prediction
        + parameters["gamma_weighted"] * (weighted_prediction - base_prediction)
        + parameters["beta_hot"] * centered_hot
        + parameters["intercept"]
        - parameters["push_cool"] * centered_cool
        + parameters["push_heat"] * centered_heat
    )


def subgroup_masks(data: pd.DataFrame) -> dict[str, pd.Series]:
    """Subgrupos necesarios para no esconder errores direccionales."""
    delta = data["delta_tb_objetivo"]
    return {
        "global": pd.Series(True, index=data.index),
        "delta_<=-5": delta.le(-5),
        "-5<delta_<5": delta.gt(-5) & delta.lt(5),
        "delta_>=5": delta.ge(5),
        "delta_>=10": delta.ge(10),
        "tb_objetivo_>=970": data["tb_objetivo"].ge(970),
        "inicio_<970_fin_>=970": data["tb_inicial"].lt(970) & data["tb_objetivo"].ge(970),
        "inicio_>=970_fin_<970": data["tb_inicial"].ge(970) & data["tb_objetivo"].lt(970),
    }


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


def evaluate_predictions(
    data: pd.DataFrame, predictions: dict[str, np.ndarray], segment: str
) -> list[dict[str, Any]]:
    """Evalua variantes en las mismas filas y subgrupos."""
    rows: list[dict[str, Any]] = []
    for variant, predicted in predictions.items():
        predicted_series = pd.Series(predicted, index=data.index)
        for subgroup, mask in subgroup_masks(data).items():
            if mask.any():
                rows.append(
                    {
                        "segmento": segment,
                        "variante": variant,
                        "subgrupo": subgroup,
                        "filas": int(mask.sum()),
                        **metric_values(
                            data.loc[mask, "tb_objetivo"],
                            predicted_series.loc[mask].to_numpy(),
                        ),
                    }
                )
    return rows


def fit_components(
    train: pd.DataFrame, features: list[str]
) -> tuple[
    LGBMRegressor,
    LGBMRegressor,
    LGBMRegressor,
    LGBMClassifier,
    LGBMClassifier,
    float,
    np.ndarray,
]:
    """Entrena los cinco componentes internos usando solo train."""
    residual = train["tb_objetivo"] - train["pred_vitm_congelado"]
    base_regressor = create_regressor()
    base_regressor.fit(train[features], residual)

    hot_weight = np.where(train["tb_objetivo"].ge(970), 3.0, 1.0)
    hot_regressor = create_regressor()
    hot_regressor.fit(train[features], residual, sample_weight=hot_weight)

    priority_weight = np.where(train["tb_objetivo"].ge(970), 8.0, 1.0)
    priority_regressor = create_regressor()
    priority_regressor.fit(train[features], residual, sample_weight=priority_weight)

    hot_target = train["tb_objetivo"].ge(970).astype(int)
    hot_classifier = create_hot_classifier()
    hot_classifier.fit(train[features], hot_target)

    thermal_target = direction_target(train["delta_tb_objetivo"])
    direction_classifier = create_direction_classifier()
    direction_classifier.fit(train[features], thermal_target)
    direction_prevalence = np.bincount(thermal_target, minlength=3) / len(thermal_target)
    return (
        base_regressor,
        hot_regressor,
        priority_regressor,
        hot_classifier,
        direction_classifier,
        float(hot_target.mean()),
        direction_prevalence,
    )


def predict_components(
    data: pd.DataFrame,
    features: list[str],
    components: tuple[
        LGBMRegressor,
        LGBMRegressor,
        LGBMRegressor,
        LGBMClassifier,
        LGBMClassifier,
        float,
        np.ndarray,
    ],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Calcula componentes y las cuatro salidas finales."""
    (
        base_regressor,
        hot_regressor,
        priority_regressor,
        hot_classifier,
        direction_classifier,
        hot_prev,
        dir_prev,
    ) = components
    vitm = data["pred_vitm_congelado"].to_numpy()
    base_prediction = vitm + base_regressor.predict(data[features])
    weighted_prediction = vitm + hot_regressor.predict(data[features])
    priority_prediction = vitm + priority_regressor.predict(data[features])
    hot_probability = hot_classifier.predict_proba(data[features])[:, 1]
    direction_probability = direction_classifier.predict_proba(data[features])
    predictions = {
        name: combine_predictions(
            base_prediction,
            weighted_prediction,
            hot_probability,
            hot_prev,
            direction_probability,
            dir_prev,
            parameters,
        )
        for name, parameters in FORMULAS.items()
    }
    predictions["ledger_prioridad_caliente"] = combine_predictions(
        base_prediction,
        priority_prediction,
        hot_probability,
        hot_prev,
        direction_probability,
        dir_prev,
        PRIORITY_HOT_FORMULA,
    )
    internals = {
        "pred_regresor_base": base_prediction,
        "pred_regresor_caliente_peso3": weighted_prediction,
        "pred_regresor_prioridad_peso8": priority_prediction,
        "probabilidad_tb_>=970": hot_probability,
        "probabilidad_enfriamiento": direction_probability[:, 0],
        "probabilidad_estable": direction_probability[:, 1],
        "probabilidad_calentamiento": direction_probability[:, 2],
    }
    return predictions, internals


def fit_pot_calibration(train: pd.DataFrame) -> dict[str, Any]:
    """Estima el desvio historico de cada cuba usando solo el pasado."""
    history = train[["CUBA", "tb_objetivo", "pred_vitm_congelado"]].copy()
    history["residuo_vitm"] = (
        history["tb_objetivo"] - history["pred_vitm_congelado"]
    )
    global_residual = float(history["residuo_vitm"].mean())
    stats = history.groupby("CUBA")["residuo_vitm"].agg(["mean", "count"])
    deviation = (
        (stats["mean"] - global_residual)
        * stats["count"]
        / (stats["count"] + POT_CALIBRATION_SHRINKAGE)
    )
    return {
        "desvio_por_cuba": deviation.to_dict(),
        "residuo_global_vitm": global_residual,
        "regularizacion": POT_CALIBRATION_SHRINKAGE,
        "intensidad": POT_CALIBRATION_STRENGTH,
    }


def predict_pot_calibration(
    data: pd.DataFrame, calibration: dict[str, Any]
) -> np.ndarray:
    """Devuelve cero para una cuba sin historia y no usa la fila objetivo."""
    deviation = data["CUBA"].map(calibration["desvio_por_cuba"]).fillna(0.0)
    return deviation.to_numpy() * calibration["intensidad"]


def rolling_validation(data: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Genera predicciones fuera de muestra anteriores a julio."""
    frames: list[pd.DataFrame] = []
    rows: list[dict[str, Any]] = []
    for fold, start, end in ROLLING_FOLDS:
        train = data[data["fecha_tb_objetivo"].lt(start)].copy()
        validation = data[
            data["fecha_tb_objetivo"].ge(start) & data["fecha_tb_objetivo"].lt(end)
        ].copy()
        components = fit_components(train, features)
        predictions, internals = predict_components(validation, features, components)
        pot_calibration = fit_pot_calibration(train)
        pot_correction = predict_pot_calibration(validation, pot_calibration)
        predictions["ledger_calientes"] = predictions["ledger_calientes"] + pot_correction
        predictions["ledger_prioridad_caliente"] = (
            predictions["ledger_prioridad_caliente"] + pot_correction
        )
        internals["correccion_historica_cuba_calientes"] = pot_correction
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
        frame["fold"] = fold
        for name, values in internals.items():
            frame[name] = values
        for name, values in predictions.items():
            frame[f"pred_{name}"] = values
        frames.append(frame)
        rows.extend(evaluate_predictions(validation, predictions, fold))
    return pd.concat(frames, ignore_index=True), pd.DataFrame(rows)


def paired_bootstrap(
    real: pd.Series, baseline: np.ndarray, candidate: np.ndarray, repetitions: int = 3000
) -> dict[str, float]:
    """Intervalo de confianza de la mejora pareada de MAE."""
    difference = np.abs(baseline - real.to_numpy()) - np.abs(candidate - real.to_numpy())
    random = np.random.default_rng(42)
    estimates = np.array(
        [
            difference[random.integers(0, len(difference), len(difference))].mean()
            for _ in range(repetitions)
        ]
    )
    return {
        "mejora_mae": float(difference.mean()),
        "ic95_inferior": float(np.quantile(estimates, 0.025)),
        "ic95_superior": float(np.quantile(estimates, 0.975)),
    }


def feature_importance(
    components: tuple[
        LGBMRegressor,
        LGBMRegressor,
        LGBMRegressor,
        LGBMClassifier,
        LGBMClassifier,
        float,
        np.ndarray,
    ],
    features: list[str],
) -> pd.DataFrame:
    """Importancia de ganancia separada por componente."""
    names = [
        "regresor_base",
        "regresor_caliente",
        "regresor_prioridad_caliente",
        "clasificador_caliente",
        "selector_direccion",
    ]
    rows = []
    for name, model in zip(names, components[:5], strict=True):
        gain = model.booster_.feature_importance(importance_type="gain")
        total = gain.sum()
        for feature, value in zip(features, gain, strict=True):
            rows.append(
                {
                    "modelo": name,
                    "feature": feature,
                    "importancia_gain": float(value),
                    "importancia_pct": float(value / total * 100) if total else 0.0,
                }
            )
    return pd.DataFrame(rows).sort_values(["modelo", "importancia_gain"], ascending=[True, False])


def explain_predictions(
    test: pd.DataFrame,
    features: list[str],
    components: tuple[
        LGBMRegressor,
        LGBMRegressor,
        LGBMRegressor,
        LGBMClassifier,
        LGBMClassifier,
        float,
        np.ndarray,
    ],
    predictions: dict[str, np.ndarray],
    internals: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Explica por fila el corrector base y los ajustes finales."""
    base_regressor, _, _, _, _, hot_prev, dir_prev = components
    shap = base_regressor.booster_.predict(test[features], pred_contrib=True)[:, :-1]
    top_indexes = np.argsort(np.abs(shap), axis=1)[:, -3:][:, ::-1]
    feature_array = np.asarray(features)
    feature_values = test[features].to_numpy()
    rows = np.arange(len(test))
    output = test[
        ["id_intervalo", "CUBA", "fecha_tb_objetivo", "tb_inicial", "tb_objetivo"]
    ].copy()
    for name, values in internals.items():
        output[name] = values
    for name, values in predictions.items():
        output[f"pred_{name}"] = values
        output[f"error_abs_{name}"] = np.abs(values - output["tb_objetivo"].to_numpy())
    output["prevalencia_tb_>=970_train"] = hot_prev
    output["prevalencia_enfriamiento_train"] = dir_prev[0]
    output["prevalencia_calentamiento_train"] = dir_prev[2]
    for rank in range(3):
        indexes = top_indexes[:, rank]
        output[f"factor_{rank + 1}"] = feature_array[indexes]
        output[f"valor_factor_{rank + 1}"] = feature_values[rows, indexes]
        output[f"aporte_correccion_{rank + 1}"] = shap[rows, indexes]
    return output


def write_reports(
    metrics: pd.DataFrame,
    validation_metrics: pd.DataFrame,
    importance: pd.DataFrame,
    bootstrap: dict[str, dict[str, float]],
    quality: dict[str, Any],
) -> None:
    """Escribe un resumen simple y un documento tecnico corto."""
    global_test = metrics[
        metrics["segmento"].eq("julio_referencia") & metrics["subgrupo"].eq("global")
    ].set_index("variante")
    hot_test = metrics[
        metrics["segmento"].eq("julio_referencia")
        & metrics["subgrupo"].eq("tb_objetivo_>=970")
    ].set_index("variante")
    onset_test = metrics[
        metrics["segmento"].eq("julio_referencia")
        & metrics["subgrupo"].eq("inicio_<970_fin_>=970")
    ].set_index("variante")
    directions = metrics[
        metrics["segmento"].eq("julio_referencia")
        & metrics["subgrupo"].isin(["delta_<=-5", "-5<delta_<5", "delta_>=5"])
    ].pivot(index="variante", columns="subgrupo", values="mae")
    validation_global_rows = validation_metrics[validation_metrics["subgrupo"].eq("global")].copy()
    validation_global_rows["weighted"] = (
        validation_global_rows["mae"] * validation_global_rows["filas"]
    )
    validation_global = (
        validation_global_rows.groupby("variante")["weighted"].sum()
        / validation_global_rows.groupby("variante")["filas"].sum()
    )
    top = importance[importance["modelo"].eq("regresor_base")].head(15)
    lines = [
        "MODELO HIS5M LEDGER - RESULTADO",
        "===============================",
        "",
        f"Features finales: {quality['features']:,}.",
        f"Train final: {quality['train_rows']:,}; julio: {quality['test_rows']:,}.",
        "Horizonte: 8 semiturnos, aproximadamente 32 horas.",
        "Ventanas causales: 1, 2, 3, 6, 12 y 21 ST; bandas no superpuestas.",
        "Secuencia compacta: 25 posiciones elegidas antes del primer fold.",
        "Julio ya fue observado anteriormente y se informa solo como referencia.",
        "",
        "MAE global de validacion pre-julio:",
        validation_global.round(3).to_string(),
        "",
        "MAE global de julio:",
        global_test[["filas", "mae", "rmse", "sesgo"]].round(3).to_string(),
        "",
        "MAE por direccion en julio:",
        directions.round(3).to_string(),
        "",
        "TB objetivo >=970 C en julio:",
        hot_test[["filas", "mae", "sesgo"]].round(3).to_string(),
        "",
        "Cruces desde <970 a >=970 C en julio:",
        onset_test[["filas", "mae", "sesgo"]].round(3).to_string(),
        "",
        "Mejoras pareadas contra el modelo HIS5M anterior:",
    ]
    for name, result in bootstrap.items():
        lines.append(
            f"- {name}: {result['mejora_mae']:.3f} C "
            f"(IC95 {result['ic95_inferior']:.3f} a {result['ic95_superior']:.3f})."
        )
    lines.extend(
        [
            "",
            "Variables principales del regresor base:",
            top[["feature", "importancia_pct"]].round(2).to_string(index=False),
            "",
            "Calidad:",
            f"- IDs duplicados: {quality['duplicate_ids']:,}.",
            f"- Predicciones no finitas: {quality['nonfinite_predictions']:,}.",
            f"- Features prohibidas: {quality['forbidden_features']:,}.",
            f"- Cobertura minima original en julio: {quality['min_test_coverage'] * 100:.2f}%.",
            f"- Ultima fecha train: {quality['train_max_date']:%Y-%m-%d}; "
            f"primera fecha julio: {quality['test_min_date']:%Y-%m-%d}.",
            "",
            "Lectura:",
            "- ledger_general prioriza MAE global.",
            "- ledger_balanceado reduce simultaneamente enfriamiento y calentamiento",
            "  en validacion respecto del regresor sin ajuste.",
            "- ledger_calientes prioriza TB >=970 y fria->caliente, con mayor costo",
            "  en enfriamientos y casos estables. Incluye una correccion historica",
            "  regularizada por cuba que se calcula solamente con datos anteriores.",
            "- ledger_prioridad_caliente acepta un MAE global cercano a 5 C para",
            "  penalizar con mas fuerza calentamientos y subestimaciones calientes.",
            "- Hace falta un mes futuro no observado para confirmar cualquier ganancia.",
        ]
    )
    SUMMARY_OUTPUT.write_text("\n".join(lines), encoding="utf-8")

    doc = [
        "# Modelo HIS5M con ledger termico",
        "",
        "Este modelo agrega alimentacion y control en ventanas de aproximadamente",
        "4 a 84 horas y conserva 25 posiciones de semiturno. La prediccion",
        "sigue ocurriendo antes del semiturno objetivo.",
        "",
        "## Resultado",
        "",
        f"- General en julio: {global_test.loc['ledger_general', 'mae']:.3f} C.",
        f"- Balanceado en julio: {global_test.loc['ledger_balanceado', 'mae']:.3f} C.",
        f"- Calientes en julio: {global_test.loc['ledger_calientes', 'mae']:.3f} C global.",
        f"- Calientes, TB >=970: {hot_test.loc['ledger_calientes', 'mae']:.3f} C.",
        f"- Calientes, fria->caliente: {onset_test.loc['ledger_calientes', 'mae']:.3f} C.",
        f"- Prioridad caliente en julio: "
        f"{global_test.loc['ledger_prioridad_caliente', 'mae']:.3f} C global.",
        f"- Prioridad caliente, TB >=970: "
        f"{hot_test.loc['ledger_prioridad_caliente', 'mae']:.3f} C.",
        f"- Prioridad caliente, fria->caliente: "
        f"{onset_test.loc['ledger_prioridad_caliente', 'mae']:.3f} C.",
        f"- Prioridad caliente, calentamientos/enfriamientos: "
        f"{directions.loc['ledger_prioridad_caliente', 'delta_>=5']:.3f} / "
        f"{directions.loc['ledger_prioridad_caliente', 'delta_<=-5']:.3f} C.",
        f"- Prioridad caliente, sesgo en TB >=970: "
        f"{hot_test.loc['ledger_prioridad_caliente', 'sesgo']:.3f} C.",
        "",
        "## Uso",
        "",
        "El archivo `models/modelo_his5m_ledger.joblib` contiene VITM congelado,",
        "tres regresores y dos clasificadores. `ledger_balanceado` es la estimacion",
        "neutral; `ledger_calientes` es el especialista moderado y",
        "`ledger_prioridad_caliente` acepta mas error en cubas frias.",
        "Las dos salidas calientes agregan una correccion historica por cuba.",
        "",
        "## Advertencia",
        "",
        "Julio no es un test virgen porque fue observado en iteraciones anteriores.",
        "El resultado es exploratorio hasta evaluarlo en un mes futuro completo.",
    ]
    DOC_OUTPUT.write_text("\n".join(doc), encoding="utf-8")


def main() -> None:
    """Ejecuta validacion, entrenamiento final y referencia de julio."""
    data, frozen, features = load_dataset()
    print(f"Filas elegibles: {len(data):,}; features: {len(features):,}")
    _, validation_metrics = rolling_validation(data, features)

    train = data[data["fecha_tb_objetivo"].lt(TEST_START)].copy()
    test = data[
        data["fecha_tb_objetivo"].ge(TEST_START) & data["fecha_tb_objetivo"].lt(TEST_END)
    ].copy()
    if train.empty or test.empty:
        raise ValueError("El corte train/julio quedo vacio.")
    if train["fecha_tb_objetivo"].max() >= test["fecha_tb_objetivo"].min():
        raise ValueError("Train y julio no tienen separacion temporal estricta.")

    components = fit_components(train, features)
    predictions, internals = predict_components(test, features, components)
    pot_calibration = fit_pot_calibration(train)
    pot_correction = predict_pot_calibration(test, pot_calibration)
    predictions["ledger_calientes"] = predictions["ledger_calientes"] + pot_correction
    predictions["ledger_prioridad_caliente"] = (
        predictions["ledger_prioridad_caliente"] + pot_correction
    )
    internals["correccion_historica_cuba_calientes"] = pot_correction
    predictions["vitm_congelado"] = test["pred_vitm_congelado"].to_numpy()
    predictions["baseline_promedio_2_tb"] = (
        (test["tb_inicial"] + test["tb_anterior_2"]) / 2
    ).to_numpy()
    if any((~np.isfinite(values)).any() for values in predictions.values()):
        raise ValueError("Hay predicciones no finitas.")

    metrics = pd.DataFrame(evaluate_predictions(test, predictions, "julio_referencia"))
    importance = feature_importance(components, features)
    explanations = explain_predictions(test, features, components, predictions, internals)

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
            "h5_ledger_quality_21st_coverage",
        ]
    ].copy()
    for name, values in internals.items():
        prediction_output[name] = values
    for name, values in predictions.items():
        prediction_output[f"pred_{name}"] = values
        prediction_output[f"error_abs_{name}"] = np.abs(
            values - prediction_output["tb_objetivo"].to_numpy()
        )

    previous_predictions = pd.read_csv(
        PROJECT_DIR / "outputs" / "predicciones" / "modelo_his5m_test.csv"
    )
    previous_predictions = previous_predictions.set_index("id_intervalo").reindex(
        test["id_intervalo"]
    )
    bootstrap = {
        "general": paired_bootstrap(
            test["tb_objetivo"],
            previous_predictions["pred_his5m_general"].to_numpy(),
            predictions["ledger_general"],
        ),
        "balanceado": paired_bootstrap(
            test["tb_objetivo"],
            previous_predictions["pred_his5m_balanceado"].to_numpy(),
            predictions["ledger_balanceado"],
        ),
        "calientes": paired_bootstrap(
            test["tb_objetivo"],
            previous_predictions["pred_his5m_calientes"].to_numpy(),
            predictions["ledger_calientes"],
        ),
        "prioridad_caliente": paired_bootstrap(
            test["tb_objetivo"],
            previous_predictions["pred_his5m_calientes"].to_numpy(),
            predictions["ledger_prioridad_caliente"],
        ),
    }

    (
        base_regressor,
        hot_regressor,
        priority_regressor,
        hot_classifier,
        direction_classifier,
        hot_prev,
        dir_prev,
    ) = components
    package = {
        "modelo_vitm_congelado": frozen,
        "modelo_regresor_base": base_regressor,
        "modelo_regresor_caliente_peso3": hot_regressor,
        "modelo_regresor_prioridad_peso8": priority_regressor,
        "clasificador_tb_caliente": hot_classifier,
        "clasificador_direccion": direction_classifier,
        "prevalencia_tb_caliente_train": hot_prev,
        "prevalencia_direccion_train": dir_prev,
        "features": features,
        "formulas": FORMULAS,
        "formula_prioridad_caliente": PRIORITY_HOT_FORMULA,
        "calibracion_historica_cuba_calientes": pot_calibration,
        "salida_principal": "ledger_balanceado",
        "salida_para_priorizar_calientes": "ledger_prioridad_caliente",
        "test_start": TEST_START,
        "regla_anti_leakage": "todas las ventanas terminan antes del semiturno objetivo",
    }
    quality = {
        "features": len(features),
        "train_rows": len(train),
        "test_rows": len(test),
        "duplicate_ids": int(prediction_output["id_intervalo"].duplicated().sum()),
        "nonfinite_predictions": int(
            sum((~np.isfinite(values)).sum() for values in predictions.values())
        ),
        "forbidden_features": int(
            sum(
                "objetivo" in feature.lower()
                or feature.lower() in {"cuba", "id_intervalo"}
                for feature in features
            )
        ),
        "min_test_coverage": float(test["h5_quality_coverage_ratio"].min()),
        "train_max_date": train["fecha_tb_objetivo"].max(),
        "test_min_date": test["fecha_tb_objetivo"].min(),
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
    write_reports(metrics, validation_metrics, importance, bootstrap, quality)

    global_result = metrics[metrics["subgrupo"].eq("global")][
        ["variante", "mae", "sesgo"]
    ]
    print(global_result.round(3).to_string(index=False))
    print(f"Resumen: {SUMMARY_OUTPUT}")


if __name__ == "__main__":
    main()
