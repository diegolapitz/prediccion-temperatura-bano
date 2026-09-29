"""
OBJETIVO DEL ARCHIVO
--------------------
Analizar errores del primer modelo en el periodo de test.

ENTRADAS
--------
- `outputs/metricas/modelos_metricas.csv`
- `outputs/predicciones/modelos_predicciones_test.csv`
- `outputs/metricas/baselines_metricas.csv`

SALIDAS
-------
- `outputs/metricas/errores_modelo_por_subgrupo.csv`
- `outputs/resumen_analisis_errores.txt`
- `docs/ANALISIS_ERRORES_MODELO.md`

POR QUE EXISTE
--------------
Un MAE promedio no alcanza para decidir si el modelo sirve. Necesitamos saber
donde se equivoca mas: por sala, por grupo, por rango de TB inicial y en cubas
que arrancan calientes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_DIR / "docs"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metricas"
PREDICTIONS_DIR = OUTPUTS_DIR / "predicciones"


def print_title(title: str) -> None:
    """Imprime un titulo visible en consola."""
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def calculate_group_metrics(data: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """Calcula metricas de error por un subgrupo."""
    rows = []

    for value, group in data.groupby(group_column, dropna=False, observed=False):
        error = group["error"]
        absolute_error = group["error_abs"]
        rows.append(
            {
                "grupo_analisis": group_column,
                "valor": value,
                "filas": len(group),
                "mae": absolute_error.mean(),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "sesgo": error.mean(),
                "p90_error_abs": absolute_error.quantile(0.90),
                "p95_error_abs": absolute_error.quantile(0.95),
                "porcentaje_dentro_5c": (absolute_error <= 5).mean() * 100,
                "porcentaje_dentro_10c": (absolute_error <= 10).mean() * 100,
            }
        )

    return pd.DataFrame(rows)


def add_analysis_groups(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega rangos exploratorios para analizar errores.

    Estos rangos NO son todavia reglas operativas de retiro. Solo sirven para
    mirar donde el modelo se equivoca mas.
    """
    result = predictions.copy()

    result["rango_tb_inicial"] = pd.cut(
        result["tb_inicial"],
        bins=[-np.inf, 955, 965, 975, 985, np.inf],
        labels=["<=955", "956-965", "966-975", "976-985", ">985"],
    )
    result["rango_tb_objetivo"] = pd.cut(
        result["tb_objetivo"],
        bins=[-np.inf, 955, 965, 975, 985, np.inf],
        labels=["<=955", "956-965", "966-975", "976-985", ">985"],
    )
    result["cuba_inicial_caliente_exploratorio"] = np.where(
        result["tb_inicial"] >= 970,
        "tb_inicial_>=970",
        "tb_inicial_<970",
    )
    result["objetivo_caliente_exploratorio"] = np.where(
        result["tb_objetivo"] >= 970,
        "tb_objetivo_>=970",
        "tb_objetivo_<970",
    )

    return result


def write_summary(
    best_model_name: str,
    best_predictions: pd.DataFrame,
    subgroup_metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Guarda resumen textual del analisis de errores."""
    worst_groups = subgroup_metrics[
        subgroup_metrics["filas"] >= 100
    ].sort_values("mae", ascending=False).head(10)

    error = best_predictions["error"]
    absolute_error = best_predictions["error_abs"]

    lines = [
        "RESUMEN DE ANALISIS DE ERRORES - FASE 6",
        "=" * 60,
        "",
        f"Modelo analizado: {best_model_name}",
        f"Filas test: {len(best_predictions):,}",
        "",
        "Metricas globales en test:",
        f"- MAE: {absolute_error.mean():.3f} C",
        f"- RMSE: {np.sqrt(np.mean(error**2)):.3f} C",
        f"- Sesgo: {error.mean():.3f} C",
        f"- P90 error absoluto: {absolute_error.quantile(0.90):.3f} C",
        f"- P95 error absoluto: {absolute_error.quantile(0.95):.3f} C",
        "",
        "Peores subgrupos con al menos 100 casos:",
        worst_groups.round(3).to_string(index=False),
        "",
        "Lectura de calidad:",
        "- El modelo mejora el baseline promedio, pero mantiene sesgo negativo en test.",
        "- Sesgo negativo significa que tiende a predecir algo mas frio que lo real.",
        "- Eso es importante operativamente porque podria ser riesgoso para retiros.",
        "- La proxima fase debe mirar falsos retiros con umbrales operativos definidos.",
    ]

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dataframe_to_markdown_table(df: pd.DataFrame) -> str:
    """Convierte un DataFrame chico en tabla Markdown sin depender de tabulate."""
    if df.empty:
        return "_Sin filas para mostrar._"

    text_df = df.copy()
    for column in text_df.columns:
        text_df[column] = text_df[column].astype(str).str.replace("|", "/", regex=False)

    headers = list(text_df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in text_df.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in headers) + " |")
    return "\n".join(lines)


def write_markdown_report(
    best_model_name: str,
    best_predictions: pd.DataFrame,
    subgroup_metrics: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    model_metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Guarda un informe Markdown de errores."""
    best_model_test = model_metrics[
        (model_metrics["segmento_temporal"] == "test")
        & (model_metrics["modelo_completo"] == best_model_name)
    ].iloc[0]
    best_baseline_test = (
        baseline_metrics[baseline_metrics["segmento_temporal"] == "test"]
        .sort_values("mae")
        .iloc[0]
    )
    improvement = (
        (best_baseline_test["mae"] - best_model_test["mae"])
        / best_baseline_test["mae"]
        * 100
    )

    hot_initial = subgroup_metrics[
        (subgroup_metrics["grupo_analisis"] == "cuba_inicial_caliente_exploratorio")
    ].copy()
    by_sala = subgroup_metrics[subgroup_metrics["grupo_analisis"] == "SALA"].copy()

    lines = [
        "# Analisis de errores del primer modelo",
        "",
        f"Modelo analizado: `{best_model_name}`.",
        "",
        "## Comparacion contra baseline",
        "",
        f"- Mejor baseline test: `{best_baseline_test['baseline']}` con MAE "
        f"{best_baseline_test['mae']:.3f} C.",
        f"- Modelo test: MAE {best_model_test['mae']:.3f} C.",
        f"- Mejora relativa: {improvement:.2f}%.",
        "",
        "## Calidad global",
        "",
        f"- Sesgo test: {best_model_test['sesgo']:.3f} C.",
        f"- P90 error absoluto: {best_model_test['p90_error_abs']:.3f} C.",
        f"- P95 error absoluto: {best_model_test['p95_error_abs']:.3f} C.",
        "",
        "El sesgo negativo indica que el modelo tiende a estimar temperaturas "
        "algo menores que las reales. Para una decision conservadora de retiro, "
        "esto debe revisarse con mucho cuidado.",
        "",
        "## Error por sala",
        "",
        dataframe_to_markdown_table(by_sala.round(3)),
        "",
        "## Cubas inicialmente calientes - rango exploratorio",
        "",
        "Este corte usa `TB inicial >= 970` solo como analisis exploratorio. No es "
        "todavia una regla operativa.",
        "",
        dataframe_to_markdown_table(hot_initial.round(3)),
        "",
        "## Siguiente control recomendado",
        "",
        "Definir limite operativo de cuba caliente y umbral de retiro. Con eso se "
        "pueden medir falsos retiros, retiros correctos y cubas calientes no "
        "detectadas.",
    ]

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Ejecuta el analisis de errores."""
    print_title("FASE 6 - ANALISIS DE ERRORES")

    model_metrics_path = METRICS_DIR / "modelos_metricas.csv"
    model_predictions_path = PREDICTIONS_DIR / "modelos_predicciones_test.csv"
    baseline_metrics_path = METRICS_DIR / "baselines_metricas.csv"

    model_metrics = pd.read_csv(model_metrics_path)
    predictions = pd.read_csv(model_predictions_path)
    baseline_metrics = pd.read_csv(baseline_metrics_path)

    best_validation_row = (
        model_metrics[model_metrics["segmento_temporal"] == "validacion"]
        .sort_values("mae")
        .iloc[0]
    )
    best_model_name = best_validation_row["modelo_completo"]

    print(f"Modelo elegido por validacion: {best_model_name}")

    best_predictions = predictions[
        predictions["modelo_completo"] == best_model_name
    ].copy()
    best_predictions = add_analysis_groups(best_predictions)

    subgroup_frames = []
    for group_column in [
        "SALA",
        "GRUPO",
        "rango_tb_inicial",
        "rango_tb_objetivo",
        "cuba_inicial_caliente_exploratorio",
        "objetivo_caliente_exploratorio",
    ]:
        subgroup_frames.append(calculate_group_metrics(best_predictions, group_column))

    subgroup_metrics = pd.concat(subgroup_frames, ignore_index=True)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    subgroup_path = METRICS_DIR / "errores_modelo_por_subgrupo.csv"
    summary_path = OUTPUTS_DIR / "resumen_analisis_errores.txt"
    report_path = DOCS_DIR / "ANALISIS_ERRORES_MODELO.md"

    subgroup_metrics.to_csv(subgroup_path, index=False, encoding="utf-8-sig")
    write_summary(best_model_name, best_predictions, subgroup_metrics, summary_path)
    write_markdown_report(
        best_model_name,
        best_predictions,
        subgroup_metrics,
        baseline_metrics,
        model_metrics,
        report_path,
    )

    print("Archivos generados:")
    print(f"- {subgroup_path}")
    print(f"- {summary_path}")
    print(f"- {report_path}")


if __name__ == "__main__":
    main()
