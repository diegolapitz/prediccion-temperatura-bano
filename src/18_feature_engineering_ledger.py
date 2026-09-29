"""
Construye un libro mayor termico causal a partir del cache HIS5M.

Cada feature termina antes del semiturno de la TB objetivo. Las ventanas usan
1, 2, 3, 6, 12 y 21 semiturnos, aproximadamente 4, 8, 12, 24, 48 y 84 horas.
Las ventanas largas incluyen historia anterior a la TB inicial, pero nunca el
semiturno objetivo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
BASE_DATASET_PATH = PROJECT_DIR / "data" / "processed" / "dataset_modelado_his5m.parquet"
SEMITURN_PATH = PROJECT_DIR / "data" / "processed" / "his5m_por_semiturno.parquet"
OUTPUT_PATH = PROJECT_DIR / "data" / "processed" / "dataset_modelado_his5m_ledger.parquet"
QUALITY_OUTPUT = PROJECT_DIR / "outputs" / "metricas" / "calidad_ledger_intervalos.csv"
SUMMARY_OUTPUT = PROJECT_DIR / "outputs" / "resumen_features_ledger.txt"

ORDER_ORIGIN = pd.Timestamp("2000-01-01")
EXPECTED_RECORDS_BY_SEMITURN = {1: 48, 2: 48, 3: 48, 4: 48, 5: 36, 6: 60}
WINDOWS_ST = [1, 2, 3, 6, 12, 21]
FEED_BANDS_ST = [
    ("lag1", 1, 1),
    ("lag2_3", 2, 3),
    ("lag4_6", 4, 6),
    ("lag7_12", 7, 12),
    ("lag13_21", 13, 21),
]

CONTINUOUS_PREFIXES = [
    "h5_elect_potencia_mw_proxy",
    "h5_elect_i2r_mw_proxy",
    "h5_elect_i2_dif_rm_rk_mw_proxy",
    "h5_elect_imfc",
    "h5_elect_rmfc",
    "h5_elect_rkfc",
    "h5_elect_dif_rm_rk",
    "h5_control_rthfc",
    "h5_control_rcmfc",
    "h5_control_rcfc",
    "h5_control_wrmfc",
    "h5_control_smrwfc",
    "h5_feed_pal2o3fc_limpio",
    "h5_feed_pentefc",
    "h5_feed_cdordfc",
    "h5_feed_scdocc",
]

INTEGRAL_PREFIXES = {
    "h5_elect_potencia_mw_proxy",
    "h5_elect_i2r_mw_proxy",
    "h5_elect_i2_dif_rm_rk_mw_proxy",
    "h5_control_rthfc",
    "h5_control_rcmfc",
    "h5_control_rcfc",
    "h5_feed_pentefc",
    "h5_feed_cdordfc",
    "h5_feed_scdocc",
}

SIGNED_INTEGRAL_PREFIXES = {
    "h5_elect_i2_dif_rm_rk_mw_proxy",
    "h5_control_rthfc",
    "h5_control_rcmfc",
    "h5_control_rcfc",
    "h5_feed_cdordfc",
    "h5_feed_scdocc",
}

EXTREME_PREFIXES = {
    "h5_elect_potencia_mw_proxy",
    "h5_elect_i2r_mw_proxy",
    "h5_elect_i2_dif_rm_rk_mw_proxy",
    "h5_elect_rmfc",
    "h5_control_rthfc",
    "h5_control_rcfc",
}

STATE_PREFIXES = [
    "h5_feed_crustbrk_int",
    "h5_feed_feeder_int",
    "h5_feed_al2o3_feed_ph2v",
    "h5_feed_al2o3_feed_ph1",
    "h5_feed_al2o3_feed_ph2i",
    "h5_feed_al2o3_overfeed",
    "h5_instability_mild_instability",
    "h5_instability_sever_instability",
    "h5_instability_extd_instability",
    "h5_anodic_anode_effect",
    "h5_operation_tapping",
    "h5_operation_voltage_tapping",
    "h5_operation_anode_change",
    "h5_operation_beam_manual",
    "h5_operation_potmicro_local",
    "h5_operation_line_sd",
    "h5_operation_restart_after_sd",
]

MODE_SLUGS = [str(code) for code in range(18)] + ["otro"]
FEED_CONTINUOUS_PREFIXES = [
    "h5_feed_pal2o3fc_limpio",
    "h5_feed_pentefc",
    "h5_feed_cdordfc",
    "h5_feed_scdocc",
]
FEED_STATE_PREFIXES = STATE_PREFIXES[:6]
SEQUENCE_LAGS_ST = list(range(1, 8))


def absolute_order(dates: pd.Series, semiturns: pd.Series) -> pd.Series:
    """Convierte fecha operativa y semiturno a un orden entero creciente."""
    normalized = pd.to_datetime(dates, errors="coerce").dt.normalize()
    return ((normalized - ORDER_ORIGIN).dt.days * 6 + semiturns.astype(int)).astype("int64")


def target_semiturn_start(dates: pd.Series, semiturns: pd.Series) -> pd.Series:
    """Devuelve el inicio calendario del semiturno objetivo."""
    dates = pd.to_datetime(dates).dt.normalize()
    semiturns = semiturns.astype(int)
    hour = semiturns.map({1: 6, 2: 10, 3: 14, 4: 18, 5: 22, 6: 1})
    next_day = semiturns.eq(6).astype(int)
    return dates + pd.to_timedelta(next_day, unit="D") + pd.to_timedelta(hour, unit="h")


def required_semiturn_columns() -> list[str]:
    """Lista minima de columnas necesarias del cache por semiturno."""
    columns = [
        "CUBA",
        "ORDEN_SEMITURNO_ABS",
        "h5_records",
        "h5_ultima_hora",
        "semiturno",
        "h5_feed_talim_cambios__count",
    ]
    for prefix in CONTINUOUS_PREFIXES:
        columns.extend(
            [
                f"{prefix}__mean",
                f"{prefix}__count",
                f"{prefix}__min",
                f"{prefix}__max",
            ]
        )
    for prefix in STATE_PREFIXES:
        columns.extend([f"{prefix}__active_records", f"{prefix}__any"])
    for slug in MODE_SLUGS:
        columns.append(f"h5_feed_talim_{slug}__records")
    return list(dict.fromkeys(columns))


def build_window_rows(dataset: pd.DataFrame, semiturns: pd.DataFrame) -> pd.DataFrame:
    """Une a cada intervalo los 21 semiturnos previos al objetivo."""
    intervals = dataset[
        ["id_intervalo", "CUBA", "fecha_tb_objetivo", "semiturno_tb_objetivo"]
    ].copy()
    intervals["orden_objetivo_abs"] = absolute_order(
        intervals["fecha_tb_objetivo"], intervals["semiturno_tb_objetivo"]
    )

    lags = np.arange(1, max(WINDOWS_ST) + 1, dtype="int8")
    window = intervals[["id_intervalo", "CUBA", "orden_objetivo_abs"]].loc[
        intervals.index.repeat(len(lags))
    ].copy()
    window["lag_st"] = np.tile(lags, len(intervals))
    window["ORDEN_SEMITURNO_ABS"] = window["orden_objetivo_abs"] - window["lag_st"]
    window["semiturno_esperado"] = ((window["ORDEN_SEMITURNO_ABS"] - 1) % 6) + 1
    window["records_esperados"] = window["semiturno_esperado"].map(
        EXPECTED_RECORDS_BY_SEMITURN
    )

    result = window.merge(
        semiturns,
        on=["CUBA", "ORDEN_SEMITURNO_ABS"],
        how="left",
        validate="many_to_one",
    ).sort_values(["id_intervalo", "lag_st"], kind="mergesort")
    if not result["ORDEN_SEMITURNO_ABS"].lt(result["orden_objetivo_abs"]).all():
        raise ValueError("El ledger incluyo el semiturno objetivo o uno posterior.")
    return result


def add_window_features(window: pd.DataFrame, interval_ids: pd.Index) -> pd.DataFrame:
    """Calcula acumulados, niveles y estados para cada escala temporal."""
    features: dict[str, pd.Series] = {}
    group_key = window["id_intervalo"]

    for size in WINDOWS_ST:
        selected = window["lag_st"].le(size)
        records = window["h5_records"].where(selected, 0).fillna(0)
        expected = window["records_esperados"].where(selected, 0).fillna(0)
        records_total = records.groupby(group_key).sum().reindex(interval_ids)
        expected_total = expected.groupby(group_key).sum().reindex(interval_ids)
        coverage = records_total / expected_total.replace(0, np.nan)
        complete = coverage.ge(0.80)

        features[f"h5_ledger_quality_{size}st_coverage"] = coverage
        features[f"h5_ledger_quality_{size}st_found"] = (
            window["h5_records"].where(selected).notna().groupby(group_key).sum().reindex(interval_ids)
        )

        for prefix in CONTINUOUS_PREFIXES:
            count = window[f"{prefix}__count"].where(selected, 0).fillna(0)
            mean = window[f"{prefix}__mean"].where(selected)
            weighted = mean.fillna(0) * count
            count_total = count.groupby(group_key).sum().reindex(interval_ids)
            value_total = weighted.groupby(group_key).sum().reindex(interval_ids)
            short_name = prefix.removeprefix("h5_")
            features[f"h5_ledger_{short_name}_{size}st_mean"] = (
                value_total / count_total.replace(0, np.nan)
            )

            if prefix in INTEGRAL_PREFIXES:
                integral = value_total * (5.0 / 60.0)
                features[f"h5_ledger_{short_name}_{size}st_integral"] = integral.where(complete)

            if prefix in SIGNED_INTEGRAL_PREFIXES:
                positive = mean.clip(lower=0).fillna(0) * count
                negative = mean.clip(upper=0).fillna(0) * count
                positive_total = positive.groupby(group_key).sum().reindex(interval_ids) * (5.0 / 60.0)
                negative_total = negative.groupby(group_key).sum().reindex(interval_ids) * (5.0 / 60.0)
                features[f"h5_ledger_{short_name}_{size}st_positive_integral"] = (
                    positive_total.where(complete)
                )
                features[f"h5_ledger_{short_name}_{size}st_negative_integral"] = (
                    negative_total.where(complete)
                )

            if prefix in EXTREME_PREFIXES:
                features[f"h5_ledger_{short_name}_{size}st_max"] = (
                    window[f"{prefix}__max"].where(selected).groupby(group_key).max().reindex(interval_ids)
                )
                features[f"h5_ledger_{short_name}_{size}st_min"] = (
                    window[f"{prefix}__min"].where(selected).groupby(group_key).min().reindex(interval_ids)
                )

        for prefix in STATE_PREFIXES:
            active = window[f"{prefix}__active_records"].where(selected, 0).fillna(0)
            active_total = active.groupby(group_key).sum().reindex(interval_ids)
            short_name = prefix.removeprefix("h5_")
            features[f"h5_ledger_{short_name}_{size}st_minutes"] = (
                active_total * 5.0
            ).where(complete)

        for slug in MODE_SLUGS:
            active = window[f"h5_feed_talim_{slug}__records"].where(selected, 0).fillna(0)
            active_total = active.groupby(group_key).sum().reindex(interval_ids)
            features[f"h5_ledger_feed_talim_{slug}_{size}st_fraction"] = (
                active_total / records_total.replace(0, np.nan)
            )

        changes = (
            window["h5_feed_talim_cambios__count"]
            .where(selected, 0)
            .fillna(0)
            .groupby(group_key)
            .sum()
            .reindex(interval_ids)
        )
        features[f"h5_ledger_feed_talim_cambios_{size}st"] = changes.where(complete)

    result = pd.DataFrame(features, index=interval_ids)
    for prefix in CONTINUOUS_PREFIXES:
        short_name = prefix.removeprefix("h5_")
        result[f"h5_ledger_{short_name}_1st_vs_6st"] = (
            result[f"h5_ledger_{short_name}_1st_mean"]
            - result[f"h5_ledger_{short_name}_6st_mean"]
        )
        result[f"h5_ledger_{short_name}_3st_vs_12st"] = (
            result[f"h5_ledger_{short_name}_3st_mean"]
            - result[f"h5_ledger_{short_name}_12st_mean"]
        )
    return result


def add_feed_band_features(window: pd.DataFrame, interval_ids: pd.Index) -> pd.DataFrame:
    """Separa alimentacion reciente y antigua en cinco bandas no superpuestas."""
    features: dict[str, pd.Series] = {}
    group_key = window["id_intervalo"]

    for band_name, first_lag, last_lag in FEED_BANDS_ST:
        selected = window["lag_st"].between(first_lag, last_lag)
        records = window["h5_records"].where(selected, 0).fillna(0)
        expected = window["records_esperados"].where(selected, 0).fillna(0)
        records_total = records.groupby(group_key).sum().reindex(interval_ids)
        expected_total = expected.groupby(group_key).sum().reindex(interval_ids)
        coverage = records_total / expected_total.replace(0, np.nan)
        complete = coverage.ge(0.80)
        features[f"h5_ledger_band_quality_{band_name}_coverage"] = coverage

        for prefix in FEED_CONTINUOUS_PREFIXES:
            count = window[f"{prefix}__count"].where(selected, 0).fillna(0)
            mean = window[f"{prefix}__mean"].where(selected)
            weighted = mean.fillna(0) * count
            count_total = count.groupby(group_key).sum().reindex(interval_ids)
            value_total = weighted.groupby(group_key).sum().reindex(interval_ids)
            short_name = prefix.removeprefix("h5_feed_")
            features[f"h5_ledger_band_feed_{short_name}_{band_name}_mean"] = (
                value_total / count_total.replace(0, np.nan)
            )
            features[f"h5_ledger_band_feed_{short_name}_{band_name}_integral"] = (
                value_total * (5.0 / 60.0)
            ).where(complete)
            if prefix in SIGNED_INTEGRAL_PREFIXES:
                positive = mean.clip(lower=0).fillna(0) * count
                negative = mean.clip(upper=0).fillna(0) * count
                features[
                    f"h5_ledger_band_feed_{short_name}_{band_name}_positive_integral"
                ] = (
                    positive.groupby(group_key).sum().reindex(interval_ids) * (5.0 / 60.0)
                ).where(complete)
                features[
                    f"h5_ledger_band_feed_{short_name}_{band_name}_negative_integral"
                ] = (
                    negative.groupby(group_key).sum().reindex(interval_ids) * (5.0 / 60.0)
                ).where(complete)

        for prefix in FEED_STATE_PREFIXES:
            active = window[f"{prefix}__active_records"].where(selected, 0).fillna(0)
            active_total = active.groupby(group_key).sum().reindex(interval_ids)
            short_name = prefix.removeprefix("h5_feed_")
            features[f"h5_ledger_band_feed_{short_name}_{band_name}_minutes"] = (
                active_total * 5.0
            ).where(complete)

        for slug in MODE_SLUGS:
            active = window[f"h5_feed_talim_{slug}__records"].where(selected, 0).fillna(0)
            active_total = active.groupby(group_key).sum().reindex(interval_ids)
            features[f"h5_ledger_band_feed_talim_{slug}_{band_name}_fraction"] = (
                active_total / records_total.replace(0, np.nan)
            )

        changes = (
            window["h5_feed_talim_cambios__count"]
            .where(selected, 0)
            .fillna(0)
            .groupby(group_key)
            .sum()
            .reindex(interval_ids)
        )
        features[f"h5_ledger_band_feed_talim_cambios_{band_name}"] = changes.where(complete)

    return pd.DataFrame(features, index=interval_ids)


def add_semiturn_sequence_features(
    window: pd.DataFrame, interval_ids: pd.Index
) -> pd.DataFrame:
    """Conserva el orden de los 7 semiturnos entre ambas mediciones de TB."""
    features: dict[str, pd.Series] = {}
    for lag in SEQUENCE_LAGS_ST:
        selected = window[window["lag_st"].eq(lag)].set_index("id_intervalo")
        records = selected["h5_records"].reindex(interval_ids)
        expected = selected["records_esperados"].reindex(interval_ids)
        features[f"h5_ledger_sequence_quality_lag{lag}_coverage"] = (
            records / expected.replace(0, np.nan)
        )

        for prefix in CONTINUOUS_PREFIXES:
            short_name = prefix.removeprefix("h5_")
            features[f"h5_ledger_sequence_{short_name}_lag{lag}_mean"] = selected[
                f"{prefix}__mean"
            ].reindex(interval_ids)

        for prefix in STATE_PREFIXES:
            short_name = prefix.removeprefix("h5_")
            active = selected[f"{prefix}__active_records"].reindex(interval_ids)
            features[f"h5_ledger_sequence_{short_name}_lag{lag}_minutes"] = active * 5.0

        for slug in MODE_SLUGS:
            active = selected[f"h5_feed_talim_{slug}__records"].reindex(interval_ids)
            features[f"h5_ledger_sequence_feed_talim_{slug}_lag{lag}_fraction"] = (
                active / records.replace(0, np.nan)
            )

        features[f"h5_ledger_sequence_feed_talim_cambios_lag{lag}"] = selected[
            "h5_feed_talim_cambios__count"
        ].reindex(interval_ids)

    return pd.DataFrame(features, index=interval_ids)


def add_height_interactions(dataset: pd.DataFrame) -> pd.DataFrame:
    """Relaciona energia acumulada con alturas de bano y metal."""
    result = dataset.copy()
    liquid_height = (result["hbd_last"] + result["hmd_last"]).replace(0, np.nan)
    bath_height = result["hbd_last"].replace(0, np.nan)
    for size in [6, 12, 21]:
        i2r = result[f"h5_ledger_elect_i2r_mw_proxy_{size}st_integral"]
        excess = result[
            f"h5_ledger_elect_i2_dif_rm_rk_mw_proxy_{size}st_integral"
        ]
        rth = result[f"h5_ledger_control_rthfc_{size}st_integral"]
        feed = result[f"h5_ledger_feed_pentefc_{size}st_integral"]
        result[f"h5_ledger_i2r_{size}st_por_altura_liquida"] = i2r / liquid_height
        result[f"h5_ledger_exceso_i2r_{size}st_por_altura_bano"] = excess / bath_height
        result[f"h5_ledger_rth_{size}st_por_altura_bano"] = rth / bath_height
        result[f"h5_ledger_feed_{size}st_por_altura_bano"] = feed / bath_height
        result[f"h5_ledger_exceso_i2r_{size}st_x_bano_bajo"] = (
            excess * result["proporcion_semiturnos_bano_bajo"]
        )
    numeric_columns = result.select_dtypes(include=[np.number]).columns
    result[numeric_columns] = result[numeric_columns].replace(
        [np.inf, -np.inf], np.nan
    )
    return result


def write_summary(dataset: pd.DataFrame, quality: pd.DataFrame, leakage_rows: int) -> None:
    """Documenta alcance, cobertura y controles del ledger."""
    ledger_columns = [column for column in dataset if column.startswith("h5_ledger_")]
    monthly = (
        quality.assign(mes=quality["fecha_tb_objetivo"].dt.to_period("M").astype(str))
        .groupby("mes")
        .agg(
            intervalos=("id_intervalo", "size"),
            cobertura_6st=("h5_ledger_quality_6st_coverage", "mean"),
            cobertura_12st=("h5_ledger_quality_12st_coverage", "mean"),
            cobertura_21st=("h5_ledger_quality_21st_coverage", "mean"),
        )
    )
    lines = [
        "FEATURE ENGINEERING - LEDGER TERMICO",
        "====================================",
        "",
        f"Intervalos: {len(dataset):,}",
        f"Features ledger: {len(ledger_columns):,}",
        "Ventanas: 1, 2, 3, 6, 12 y 21 semiturnos (~4 a 84 horas).",
        "Todas las ventanas terminan antes del semiturno objetivo.",
        f"Filas con timestamp prohibido: {leakage_rows:,}.",
        "",
        "Cobertura media por mes:",
        monthly.round(4).to_string(),
        "",
        "Familias:",
        "- Energia total y resistiva I2R; exceso I2(RM-RK).",
        "- RTH, RC, WRMI, alimentacion y cambios de modo.",
        "- Minutos de inestabilidad, efecto anodico y operaciones.",
        "- Secuencia por cada uno de los 7 semiturnos intermedios.",
        "- Interacciones con alturas de bano y metal.",
    ]
    SUMMARY_OUTPUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Construye y guarda el dataset HIS5M con ledger termico."""
    dataset = pd.read_parquet(BASE_DATASET_PATH)
    required = required_semiturn_columns()
    available = set(pd.read_parquet(SEMITURN_PATH, columns=[]).columns)
    # PyArrow/Pandas no siempre devuelve schema al pedir cero columnas.
    if not available:
        import pyarrow.parquet as pq

        available = set(pq.ParquetFile(SEMITURN_PATH).schema_arrow.names)
    missing = set(required) - available
    if missing:
        raise ValueError(f"Faltan columnas en el cache HIS5M: {sorted(missing)}")
    semiturns = pd.read_parquet(SEMITURN_PATH, columns=required)

    window = build_window_rows(dataset, semiturns)
    interval_ids = pd.Index(dataset["id_intervalo"], name="id_intervalo")
    ledger = pd.concat(
        [
            add_window_features(window, interval_ids),
            add_feed_band_features(window, interval_ids),
            add_semiturn_sequence_features(window, interval_ids),
        ],
        axis=1,
    ).reset_index()
    final = dataset.merge(ledger, on="id_intervalo", how="left", validate="one_to_one")
    final = add_height_interactions(final)

    target_start = target_semiturn_start(
        dataset["fecha_tb_objetivo"], dataset["semiturno_tb_objetivo"]
    )
    last_timestamp = window.groupby("id_intervalo")["h5_ultima_hora"].max().reindex(interval_ids)
    leakage_rows = int(
        (
            last_timestamp.notna()
            & last_timestamp.ge(pd.Series(target_start.to_numpy(), index=interval_ids))
        ).sum()
    )
    if leakage_rows:
        raise ValueError(f"Hay {leakage_rows:,} intervalos con datos del semiturno objetivo.")
    if not final["id_intervalo"].is_unique:
        raise ValueError("id_intervalo dejo de ser unico al agregar el ledger.")

    quality_columns = [
        "id_intervalo",
        "fecha_tb_objetivo",
        *[f"h5_ledger_quality_{size}st_coverage" for size in WINDOWS_ST],
    ]
    quality = final[quality_columns].copy()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUALITY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    final.to_parquet(OUTPUT_PATH, index=False)
    quality.to_csv(QUALITY_OUTPUT, index=False, encoding="utf-8-sig")
    write_summary(final, quality, leakage_rows)
    print(f"Dataset: {OUTPUT_PATH}")
    print(f"Filas: {len(final):,}; columnas: {len(final.columns):,}")
    print(f"Resumen: {SUMMARY_OUTPUT}")


if __name__ == "__main__":
    main()
