"""Auditoria reproducible del modelo HIS5M ledger.

No entrena ni elige parametros. Verifica el paquete final, recalcula las
metricas de julio y estima la incertidumbre agrupando por cuba.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_SCRIPT = PROJECT_DIR / "src" / "19_modelo_ledger.py"
MODEL_PATH = PROJECT_DIR / "models" / "modelo_his5m_ledger.joblib"
PREDICTIONS_PATH = (
    PROJECT_DIR / "outputs" / "predicciones" / "modelo_his5m_ledger_test.csv"
)
PREVIOUS_PREDICTIONS_PATH = (
    PROJECT_DIR / "outputs" / "predicciones" / "modelo_his5m_test.csv"
)
METRICS_PATH = PROJECT_DIR / "outputs" / "metricas" / "modelo_his5m_ledger_metricas.csv"
COMPARISON_OUTPUT = (
    PROJECT_DIR / "outputs" / "metricas" / "modelo_his5m_ledger_comparacion_cluster.csv"
)
REPORT_OUTPUT = PROJECT_DIR / "outputs" / "calidad_modelo_his5m_ledger.txt"


def load_model_code() -> Any:
    """Carga funciones del modelo sin duplicar su logica."""
    spec = importlib.util.spec_from_file_location("modelo_ledger", MODEL_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("No se pudo cargar 19_modelo_ledger.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cluster_bootstrap(
    frame: pd.DataFrame,
    baseline_column: str,
    candidate_column: str,
    repetitions: int = 3000,
) -> dict[str, float]:
    """IC pareado: remuestrea cubas completas, no filas independientes."""
    work = frame[["CUBA", "tb_objetivo", baseline_column, candidate_column]].copy()
    work["mejora_abs"] = (
        (work[baseline_column] - work["tb_objetivo"]).abs()
        - (work[candidate_column] - work["tb_objetivo"]).abs()
    )
    by_pot = work.groupby("CUBA")["mejora_abs"].agg(["sum", "count"])
    sums = by_pot["sum"].to_numpy()
    counts = by_pot["count"].to_numpy()
    random = np.random.default_rng(42)
    estimates = np.empty(repetitions)
    for iteration in range(repetitions):
        sample = random.integers(0, len(by_pot), len(by_pot))
        estimates[iteration] = sums[sample].sum() / counts[sample].sum()
    return {
        "mejora_mae": float(work["mejora_abs"].mean()),
        "ic95_inferior": float(np.quantile(estimates, 0.025)),
        "ic95_superior": float(np.quantile(estimates, 0.975)),
        "cubas": int(len(by_pot)),
        "filas": int(len(work)),
    }


def main() -> None:
    """Ejecuta controles y escribe un informe legible."""
    model_code = load_model_code()
    data, _, features = model_code.load_dataset()
    train = data[data["fecha_tb_objetivo"].lt(model_code.TEST_START)].copy()
    test = data[
        data["fecha_tb_objetivo"].ge(model_code.TEST_START)
        & data["fecha_tb_objetivo"].lt(model_code.TEST_END)
    ].copy()
    package = joblib.load(MODEL_PATH)

    components = (
        package["modelo_regresor_base"],
        package["modelo_regresor_caliente_peso3"],
        package["modelo_regresor_prioridad_peso8"],
        package["clasificador_tb_caliente"],
        package["clasificador_direccion"],
        package["prevalencia_tb_caliente_train"],
        package["prevalencia_direccion_train"],
    )
    reproduced, internals = model_code.predict_components(test, features, components)
    pot_correction = model_code.predict_pot_calibration(
        test, package["calibracion_historica_cuba_calientes"]
    )
    reproduced["ledger_calientes"] += pot_correction
    reproduced["ledger_prioridad_caliente"] += pot_correction
    reproduced["vitm_congelado"] = test["pred_vitm_congelado"].to_numpy()
    reproduced["baseline_promedio_2_tb"] = (
        (test["tb_inicial"] + test["tb_anterior_2"]) / 2
    ).to_numpy()

    published = pd.read_csv(PREDICTIONS_PATH).set_index("id_intervalo").reindex(
        test["id_intervalo"]
    )
    max_prediction_difference = max(
        float(
            np.max(
                np.abs(
                    published[f"pred_{name}"].to_numpy()
                    - reproduced[name]
                )
            )
        )
        for name in reproduced
    )

    recalculated = pd.DataFrame(
        model_code.evaluate_predictions(test, reproduced, "julio_referencia")
    )
    stored = pd.read_csv(METRICS_PATH)
    metric_check = recalculated.merge(
        stored,
        on=["segmento", "variante", "subgrupo", "filas"],
        suffixes=("_nuevo", "_guardado"),
    )
    max_metric_difference = float(
        max(
            (metric_check[f"{metric}_nuevo"] - metric_check[f"{metric}_guardado"])
            .abs()
            .max()
            for metric in ["mae", "rmse", "sesgo", "p90_error_abs", "dentro_5c_pct"]
        )
    )

    direction_probability = np.column_stack(
        [
            internals["probabilidad_enfriamiento"],
            internals["probabilidad_estable"],
            internals["probabilidad_calentamiento"],
        ]
    )
    actual_direction = model_code.direction_target(test["delta_tb_objetivo"])
    predicted_direction = direction_probability.argmax(axis=1)
    direction_matrix = confusion_matrix(actual_direction, predicted_direction, labels=[0, 1, 2])
    direction_recall = recall_score(
        actual_direction, predicted_direction, labels=[0, 1, 2], average=None
    )
    actual_hot = test["tb_objetivo"].ge(970).astype(int)
    predicted_hot = (internals["probabilidad_tb_>=970"] >= 0.50).astype(int)

    old = pd.read_csv(PREVIOUS_PREDICTIONS_PATH).set_index("id_intervalo").reindex(
        test["id_intervalo"]
    )
    comparison = test[
        ["id_intervalo", "CUBA", "tb_inicial", "tb_objetivo", "delta_tb_objetivo"]
    ].copy()
    comparison["old_general"] = old["pred_his5m_general"].to_numpy()
    comparison["old_balanceado"] = old["pred_his5m_balanceado"].to_numpy()
    comparison["old_calientes"] = old["pred_his5m_calientes"].to_numpy()
    for variant in [
        "ledger_general",
        "ledger_balanceado",
        "ledger_calientes",
        "ledger_prioridad_caliente",
    ]:
        comparison[variant] = reproduced[variant]

    masks = {
        "global": pd.Series(True, index=comparison.index),
        "enfria_5c_o_mas": comparison["delta_tb_objetivo"].le(-5),
        "estable": comparison["delta_tb_objetivo"].between(-5, 5, inclusive="neither"),
        "calienta_5c_o_mas": comparison["delta_tb_objetivo"].ge(5),
        "tb_futura_>=970": comparison["tb_objetivo"].ge(970),
        "cruza_a_>=970": comparison["tb_inicial"].lt(970)
        & comparison["tb_objetivo"].ge(970),
    }
    pairs = {
        "general": ("old_general", "ledger_general"),
        "balanceado": ("old_balanceado", "ledger_balanceado"),
        "calientes": ("old_calientes", "ledger_calientes"),
        "prioridad_caliente": ("old_calientes", "ledger_prioridad_caliente"),
    }
    comparison_rows = []
    for output_name, (old_column, new_column) in pairs.items():
        for subgroup, mask in masks.items():
            result = cluster_bootstrap(comparison.loc[mask], old_column, new_column)
            comparison_rows.append(
                {"salida": output_name, "subgrupo": subgroup, **result}
            )
    comparison_result = pd.DataFrame(comparison_rows)

    null_rate = train[features].isna().mean()
    constant_features = sum(train[feature].nunique(dropna=True) <= 1 for feature in features)
    pots_without_july = sorted(set(train["CUBA"]) - set(test["CUBA"]))
    calibration = package["calibracion_historica_cuba_calientes"]
    hot_mask = test["tb_objetivo"].ge(970).to_numpy()
    hot_behavior_rows = []
    for variant in [
        "ledger_balanceado",
        "ledger_calientes",
        "ledger_prioridad_caliente",
    ]:
        error = reproduced[variant][hot_mask] - test.loc[hot_mask, "tb_objetivo"].to_numpy()
        hot_behavior_rows.append(
            {
                "salida": variant,
                "mae": np.abs(error).mean(),
                "sesgo": error.mean(),
                "subestima_pct": np.mean(error < 0) * 100,
                "subestima_5c_pct": np.mean(error <= -5) * 100,
                "p90_error_abs": np.quantile(np.abs(error), 0.90),
            }
        )
    hot_behavior = pd.DataFrame(hot_behavior_rows)
    report = [
        "AUDITORIA FINAL - MODELO HIS5M LEDGER",
        "====================================",
        "",
        "Veredicto: apto para experimentacion offline; falta un mes futuro virgen",
        "antes de usarlo para decisiones automaticas.",
        "",
        "Reproduccion y leakage:",
        f"- Diferencia maxima paquete vs CSV: {max_prediction_difference:.12f} C.",
        f"- Diferencia maxima al recalcular metricas: {max_metric_difference:.12f}.",
        f"- IDs duplicados en julio: {published.index.duplicated().sum():,}.",
        f"- Features prohibidas: {sum('objetivo' in x.lower() or x.lower() in {'cuba', 'id_intervalo'} for x in features):,}.",
        f"- Train termina {train['fecha_tb_objetivo'].max():%Y-%m-%d}; julio empieza {test['fecha_tb_objetivo'].min():%Y-%m-%d}.",
        f"- Maximo error en suma de probabilidades: {np.abs(direction_probability.sum(axis=1) - 1).max():.12f}.",
        "",
        "Datos y features:",
        f"- Filas train/julio: {len(train):,} / {len(test):,}.",
        f"- Cubas train/julio: {train['CUBA'].nunique():,} / {test['CUBA'].nunique():,}.",
        f"- Cubas sin intervalo elegible en julio: {pots_without_july}.",
        f"- Features: {len(features):,}; constantes: {constant_features:,}.",
        f"- Features con mas de 50% de nulos: {(null_rate > 0.50).sum():,}.",
        f"- Mayor porcentaje de nulos: {null_rate.max() * 100:.2f}%.",
        f"- Cobertura HIS5M minima en julio: {test['h5_quality_coverage_ratio'].min() * 100:.2f}%.",
        "",
        "Clasificadores auxiliares en julio:",
        f"- Direccion, exactitud: {(actual_direction == predicted_direction).mean() * 100:.1f}%.",
        f"- Recall enfriamiento/estable/calentamiento: {direction_recall[0] * 100:.1f}% / {direction_recall[1] * 100:.1f}% / {direction_recall[2] * 100:.1f}%.",
        f"- TB >=970 a umbral 50%, precision: {precision_score(actual_hot, predicted_hot, zero_division=0) * 100:.1f}%; recall: {recall_score(actual_hot, predicted_hot, zero_division=0) * 100:.1f}%.",
        f"- Matriz direccion real x predicha: {direction_matrix.tolist()}.",
        "",
        "Comportamiento cuando la TB futura termina >=970 C:",
        hot_behavior.round(3).to_string(index=False),
        "",
        "Calibracion exclusiva para calientes:",
        f"- Cubas con historia: {len(calibration['desvio_por_cuba']):,}.",
        f"- Regularizacion/intensidad: {calibration['regularizacion']:.0f} / {calibration['intensidad']:.2f}.",
        f"- Correccion minima/maxima en julio: {pot_correction.min():.3f} / {pot_correction.max():.3f} C.",
        "",
        "Intervalos por cuba:",
        "- Una mejora es convincente cuando el IC95 queda completamente por encima de cero.",
        comparison_result.round(3).to_string(index=False),
        "",
        "Caveat principal: julio fue consultado durante el desarrollo. Sus metricas",
        "sirven como referencia consistente, pero no como confirmacion independiente.",
    ]

    COMPARISON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    comparison_result.to_csv(COMPARISON_OUTPUT, index=False, encoding="utf-8-sig")
    REPORT_OUTPUT.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
