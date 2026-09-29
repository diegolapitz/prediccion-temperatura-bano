"""
Construye features de cinco minutos para predecir la proxima TB.

Regla anti-leakage
------------------
Cada intervalo tiene una TB inicial y una TB objetivo separadas por 8
semiturnos. Solo se usan los 7 semiturnos completos que quedan entre ambas:

    TB inicial | semiturnos 1..7 permitidos | TB objetivo

Se excluyen por completo tanto el semiturno inicial como el objetivo. Esto es
conservador, pero necesario porque la base de TB no trae la hora exacta de la
medicion.

Salidas
-------
- data/processed/his5m_por_semiturno.parquet
- data/processed/dataset_modelado_his5m.parquet
- outputs/metricas/calidad_his5m_intervalos.csv
- outputs/resumen_features_his5m.txt
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_DIR / "config" / "parametros.yaml"
HIS5M_DIR = PROJECT_DIR / "data" / "raw" / "his5m"
BASE_DATASET_PATH = PROJECT_DIR / "data" / "processed" / "dataset_modelado_v1.parquet"
SEMITURN_OUTPUT = PROJECT_DIR / "data" / "processed" / "his5m_por_semiturno.parquet"
DATASET_OUTPUT = PROJECT_DIR / "data" / "processed" / "dataset_modelado_his5m.parquet"
QUALITY_OUTPUT = PROJECT_DIR / "outputs" / "metricas" / "calidad_his5m_intervalos.csv"
SUMMARY_OUTPUT = PROJECT_DIR / "outputs" / "resumen_features_his5m.txt"

# Usamos un origen fijo. Solo importa que VITM e HIS5M usen el mismo calculo.
ORDER_ORIGIN = pd.Timestamp("2000-01-01")

# Cantidad teorica de registros de 5 minutos por semiturno. Los limites fueron
# comprobados contra todos los timestamps/N_FECHA_SMTU disponibles en THISEVT.
EXPECTED_RECORDS_BY_SEMITURN = {1: 48, 2: 48, 3: 48, 4: 48, 5: 36, 6: 60}

SIGNALS_BY_FAMILY = {
    "elect": [
        "IMFC",
        "UMFC",
        "RMFC",
        "RKFC",
        "POTENCIA_MW_PROXY",
        "I2R_MW_PROXY",
        "I2_DIF_RM_RK_MW_PROXY",
        "DIF_RM_RK",
        "RATIO_RM_RK",
    ],
    "control": [
        "RTHFC",
        "SMRWFC",
        "WRMFC",
        "RCMFC",
        "RDEMFC",
        "RCFC",
        "RCACFC",
        "RUCCFC",
        "RCDLFC",
        "RABBFC",
        "RCDAFC",
        "RCOCFC",
        "RWTBFFC",
        "RMIARFC",
    ],
    "feed": ["PAL2O3FC_LIMPIO", "PENTEFC", "CDORDFC", "SCDOCC"],
}

STATE_BY_FAMILY = {
    "feed": [
        "CRUSTBRK_INT",
        "FEEDER_INT",
        "AL2O3_FEED_PH2V",
        "AL2O3_FEED_PH1",
        "AL2O3_FEED_PH2I",
        "AL2O3_OVERFEED",
        "PRG_TRACKING",
        "CTL_TRACKING",
    ],
    "instability": ["MILD_INSTABILITY", "SEVER_INSTABILITY", "EXTD_INSTABILITY"],
    "anodic": ["ANODE_EFFECT"],
    "operation": [
        "POTMICRO_LOCAL",
        "ANODE_UP_LIMIT",
        "ANODE_DW_LIMIT",
        "ANODE_COVER",
        "REMAIN_ORDER",
        "ORDER_TL",
        "BEAM_HEELING",
        "BEAM_MANUAL",
        "BEAM_PSU",
        "ORDER_DISCREPANCY",
        "PREP_BEFORE_SD",
        "LINE_SD",
        "RESTART_AFTER_SD",
        "VOLTAGE_TAPPING",
        "TAPPING",
        "ANODE_CHANGE",
        "VOLTAGE_BEAM",
        "OVERSUCCION",
        "POTMICRO_LOCAL_FC",
        "BEAM_MANUAL_FC",
        "INTERRUPT_CYC",
    ],
}

# Los codigos se conservan como codigos. Todavia no se les asignan nombres
# operativos (lenta/rapida/etc.) sin confirmacion del especialista.
FEEDING_MODE_CODES = list(range(18))


def load_config() -> dict[str, Any]:
    """Carga la configuracion central del proyecto."""
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def month_range(first_month: str, last_month: str) -> list[pd.Period]:
    """Lista inclusiva de meses a procesar."""
    return list(pd.period_range(first_month, last_month, freq="M"))


def signal_prefixes() -> dict[str, str]:
    """Devuelve nombre crudo -> prefijo legible de feature."""
    prefixes: dict[str, str] = {}
    for family, columns in SIGNALS_BY_FAMILY.items():
        for column in columns:
            prefixes[column] = f"h5_{family}_{column.lower()}"
    return prefixes


def state_prefixes() -> dict[str, str]:
    """Devuelve estado crudo -> prefijo legible de feature."""
    prefixes: dict[str, str] = {}
    for family, columns in STATE_BY_FAMILY.items():
        for column in columns:
            prefixes[column] = f"h5_{family}_{column.lower()}"
    return prefixes


def add_operating_time(frame: pd.DataFrame) -> pd.DataFrame:
    """Convierte timestamp calendario a fecha y semiturno operativos."""
    result = frame.copy()
    timestamp = pd.to_datetime(result["DH5MN"], errors="coerce")
    hour = timestamp.dt.hour

    semiturn = np.select(
        [
            hour.between(6, 9),
            hour.between(10, 13),
            hour.between(14, 17),
            hour.between(18, 21),
            (hour >= 22) | (hour < 1),
            hour.between(1, 5),
        ],
        [1, 2, 3, 4, 5, 6],
        default=0,
    )
    operating_date = timestamp.dt.normalize() - pd.to_timedelta(
        (hour < 6).astype(int), unit="D"
    )

    result["FECHA_OPERATIVA"] = operating_date
    result["SEMITURNO"] = semiturn.astype("int8")
    result["ORDEN_SEMITURNO_ABS"] = (
        (operating_date - ORDER_ORIGIN).dt.days * 6 + semiturn
    ).astype("int64")
    return result


def read_next_day_early_hours(path: Path, next_month_start: pd.Timestamp) -> pd.DataFrame:
    """Lee del archivo siguiente solo las horas que pertenecen al mes anterior."""
    columns = required_raw_columns()
    upper = next_month_start + pd.Timedelta(hours=6)
    filters = [
        ("DH5MN", ">=", next_month_start.to_pydatetime()),
        ("DH5MN", "<", upper.to_pydatetime()),
    ]
    try:
        return pd.read_parquet(path, columns=columns, filters=filters)
    except (TypeError, ValueError):
        frame = pd.read_parquet(path, columns=columns)
        timestamp = pd.to_datetime(frame["DH5MN"], errors="coerce")
        return frame[timestamp.between(next_month_start, upper, inclusive="left")]


def required_raw_columns() -> list[str]:
    """Columnas HIS5M necesarias para las features elegidas."""
    signals = [
        column
        for columns in SIGNALS_BY_FAMILY.values()
        for column in columns
        if column
        not in {
            "POTENCIA_MW_PROXY",
            "I2R_MW_PROXY",
            "I2_DIF_RM_RK_MW_PROXY",
            "DIF_RM_RK",
            "RATIO_RM_RK",
            "PAL2O3FC_LIMPIO",
        }
    ]
    states = [column for columns in STATE_BY_FAMILY.values() for column in columns]
    return list(dict.fromkeys(["CUBA", "DH5MN", "TALIMFC", "PAL2O3FC", *signals, *states]))


def prepare_five_minute_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Limpia tipos y crea variables fisicas simples antes de resumir."""
    result = add_operating_time(frame)
    result["CUBA"] = pd.to_numeric(result["CUBA"], errors="coerce").astype("Int64")

    raw_numeric = set(required_raw_columns()) - {"CUBA", "DH5MN"}
    for column in raw_numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    result["PAL2O3FC_LIMPIO"] = result["PAL2O3FC"].mask(result["PAL2O3FC"].eq(9999))
    result["POTENCIA_MW_PROXY"] = result["IMFC"] * result["UMFC"] / 1000.0
    result["I2R_MW_PROXY"] = result["IMFC"] ** 2 * result["RMFC"] / 1000.0
    result["I2_DIF_RM_RK_MW_PROXY"] = (
        result["IMFC"] ** 2 * (result["RMFC"] - result["RKFC"]) / 1000.0
    )
    result["DIF_RM_RK"] = result["RMFC"] - result["RKFC"]
    result["RATIO_RM_RK"] = result["RMFC"] / result["RKFC"].replace(0, np.nan)

    for column in state_prefixes():
        result[column] = result[column].fillna(0).gt(0).astype("int8")

    for code in FEEDING_MODE_CODES:
        result[f"TALIM_MODO_{code}"] = result["TALIMFC"].eq(code).astype("int8")
    result["TALIM_MODO_OTRO"] = (
        result["TALIMFC"].notna() & ~result["TALIMFC"].isin(FEEDING_MODE_CODES)
    ).astype("int8")

    result = result.sort_values(["CUBA", "DH5MN"], kind="mergesort")
    previous_mode = result.groupby("CUBA", sort=False)["TALIMFC"].shift()
    result["TALIM_CAMBIO"] = (
        previous_mode.notna()
        & result["TALIMFC"].notna()
        & result["TALIMFC"].ne(previous_mode)
    ).astype("int8")
    return result


def summarize_prepared_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Resume filas HIS5M ya preparadas en una fila por cuba y semiturno."""
    if frame.empty:
        raise ValueError("No hay filas HIS5M para resumir.")

    keys = ["CUBA", "ORDEN_SEMITURNO_ABS"]
    grouped = frame.groupby(keys, sort=False, observed=True)
    output = grouped.agg(
        fecha_operativa=("FECHA_OPERATIVA", "first"),
        semiturno=("SEMITURNO", "first"),
        h5_primera_hora=("DH5MN", "min"),
        h5_ultima_hora=("DH5MN", "max"),
        h5_records=("DH5MN", "size"),
    )
    output["h5_records_esperados"] = output["semiturno"].map(
        EXPECTED_RECORDS_BY_SEMITURN
    )
    output["h5_coverage_st"] = output["h5_records"] / output["h5_records_esperados"]

    for column, prefix in signal_prefixes().items():
        stats = grouped[column].agg(["mean", "std", "count", "min", "max", "first", "last"])
        stats.columns = [f"{prefix}__{stat}" for stat in stats.columns]
        output = output.join(stats)

    for column, prefix in state_prefixes().items():
        state = grouped[column].agg(["sum", "mean", "max"])
        state.columns = [
            f"{prefix}__active_records",
            f"{prefix}__fraction",
            f"{prefix}__any",
        ]
        output = output.join(state)

    mode_columns = [f"TALIM_MODO_{code}" for code in FEEDING_MODE_CODES] + [
        "TALIM_MODO_OTRO"
    ]
    for column in mode_columns:
        slug = column.removeprefix("TALIM_MODO_").lower()
        mode = grouped[column].agg(["sum", "mean"])
        mode.columns = [
            f"h5_feed_talim_{slug}__records",
            f"h5_feed_talim_{slug}__fraction",
        ]
        output = output.join(mode)

    output = output.join(
        grouped["TALIM_CAMBIO"].sum().rename("h5_feed_talim_cambios__count")
    )
    return output.reset_index()


def summarize_operating_month(month: pd.Period, next_month_exists: bool) -> pd.DataFrame:
    """Resume un mes operativo en una fila por cuba y semiturno."""
    current_path = HIS5M_DIR / f"tmon_his5m_{month}.parquet"
    if not current_path.exists():
        raise FileNotFoundError(f"Falta {current_path.name}.")

    print(f"Procesando mes operativo {month}...")
    frame = pd.read_parquet(current_path, columns=required_raw_columns())

    if next_month_exists:
        next_month = month + 1
        next_path = HIS5M_DIR / f"tmon_his5m_{next_month}.parquet"
        early_next_day = read_next_day_early_hours(next_path, next_month.start_time)
        frame = pd.concat([frame, early_next_day], ignore_index=True)

    frame = prepare_five_minute_rows(frame)
    frame = frame[frame["FECHA_OPERATIVA"].dt.to_period("M").eq(month)].copy()
    if frame.empty:
        raise ValueError(f"El mes operativo {month} quedo vacio.")
    return summarize_prepared_rows(frame)


def build_semiturn_cache(months: list[pd.Period]) -> pd.DataFrame:
    """Procesa todos los meses y guarda el cache por semiturno."""
    pieces = []
    for month in months:
        next_path = HIS5M_DIR / f"tmon_his5m_{month + 1}.parquet"
        pieces.append(summarize_operating_month(month, next_path.exists()))

    semiturns = pd.concat(pieces, ignore_index=True)
    duplicates = semiturns.duplicated(["CUBA", "ORDEN_SEMITURNO_ABS"]).sum()
    if duplicates:
        raise ValueError(f"Hay {duplicates:,} semiturnos duplicados en el cache HIS5M.")

    SEMITURN_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    semiturns.to_parquet(SEMITURN_OUTPUT, index=False)
    return semiturns


def absolute_order(dates: pd.Series, semiturns: pd.Series) -> pd.Series:
    """Convierte fecha operativa y semiturno a un orden absoluto."""
    normalized_dates = pd.to_datetime(dates, errors="coerce").dt.normalize()
    return ((normalized_dates - ORDER_ORIGIN).dt.days * 6 + semiturns.astype(int)).astype(
        "int64"
    )


def target_semiturn_start(dates: pd.Series, semiturns: pd.Series) -> pd.Series:
    """Calcula el inicio calendario del semiturno objetivo."""
    dates = pd.to_datetime(dates).dt.normalize()
    semiturns = semiturns.astype(int)
    hour = semiturns.map({1: 6, 2: 10, 3: 14, 4: 18, 5: 22, 6: 1})
    next_day = semiturns.eq(6).astype(int)
    return dates + pd.to_timedelta(next_day, unit="D") + pd.to_timedelta(hour, unit="h")


def create_window_rows(intervals: pd.DataFrame, semiturns: pd.DataFrame) -> pd.DataFrame:
    """Une a cada intervalo exactamente sus siete semiturnos permitidos."""
    offsets = np.arange(1, 8, dtype="int8")
    window = intervals[["id_intervalo", "CUBA", "orden_inicial_abs"]].loc[
        intervals.index.repeat(len(offsets))
    ].copy()
    window["offset"] = np.tile(offsets, len(intervals))
    window["ORDEN_SEMITURNO_ABS"] = window["orden_inicial_abs"] + window["offset"]
    window["semiturno_esperado"] = ((window["ORDEN_SEMITURNO_ABS"] - 1) % 6) + 1
    window["records_esperados_ventana"] = window["semiturno_esperado"].map(
        EXPECTED_RECORDS_BY_SEMITURN
    )

    return window.merge(
        semiturns,
        on=["CUBA", "ORDEN_SEMITURNO_ABS"],
        how="left",
        validate="many_to_one",
    ).sort_values(["id_intervalo", "offset"], kind="mergesort")


def add_continuous_interval_features(
    output: pd.DataFrame,
    window: pd.DataFrame,
) -> pd.DataFrame:
    """Agrega nivel, dispersion, extremos y tendencia de cada senal."""
    group_key = window["id_intervalo"]
    last_three = window["offset"].ge(5)
    features: dict[str, pd.Series] = {}

    for prefix in signal_prefixes().values():
        count = window[f"{prefix}__count"].fillna(0)
        mean = window[f"{prefix}__mean"]
        std = window[f"{prefix}__std"].fillna(0)
        total = mean.fillna(0) * count
        total_sq = (std**2 * (count - 1).clip(lower=0)) + mean.fillna(0) ** 2 * count

        pooled_count = count.groupby(group_key).sum().reindex(output.index)
        pooled_sum = total.groupby(group_key).sum().reindex(output.index)
        pooled_sq = total_sq.groupby(group_key).sum().reindex(output.index)
        pooled_mean = pooled_sum / pooled_count.replace(0, np.nan)
        numerator = pooled_sq - (pooled_sum**2 / pooled_count.replace(0, np.nan))
        pooled_std = np.sqrt((numerator / (pooled_count - 1).replace(0, np.nan)).clip(lower=0))

        features[f"{prefix}_mean"] = pooled_mean
        features[f"{prefix}_std"] = pooled_std
        features[f"{prefix}_min"] = (
            window[f"{prefix}__min"].groupby(group_key).min().reindex(output.index)
        )
        features[f"{prefix}_max"] = (
            window[f"{prefix}__max"].groupby(group_key).max().reindex(output.index)
        )

        first = (
            window.loc[window["offset"].eq(1)]
            .set_index("id_intervalo")[f"{prefix}__first"]
            .reindex(output.index)
        )
        last = (
            window.loc[window["offset"].eq(7)]
            .set_index("id_intervalo")[f"{prefix}__last"]
            .reindex(output.index)
        )
        features[f"{prefix}_first"] = first
        features[f"{prefix}_last"] = last
        features[f"{prefix}_change"] = last - first

        count_last_three = count.where(last_three, 0)
        total_last_three = total.where(last_three, 0)
        features[f"{prefix}_last3st_mean"] = (
            total_last_three.groupby(group_key).sum().reindex(output.index)
            / count_last_three.groupby(group_key).sum().reindex(output.index).replace(0, np.nan)
        )
    return pd.concat([output, pd.DataFrame(features, index=output.index)], axis=1)


def add_state_interval_features(output: pd.DataFrame, window: pd.DataFrame) -> pd.DataFrame:
    """Agrega minutos y porcentajes en cada estado operativo."""
    group_key = window["id_intervalo"]
    records = window["h5_records"].fillna(0)
    records_total = records.groupby(group_key).sum().reindex(output.index).replace(0, np.nan)
    last_three = window["offset"].ge(5)
    features: dict[str, pd.Series] = {}

    for prefix in state_prefixes().values():
        active = window[f"{prefix}__active_records"].fillna(0)
        active_total = active.groupby(group_key).sum().reindex(output.index)
        active_last_three = active.where(last_three, 0).groupby(group_key).sum().reindex(output.index)
        records_last_three = records.where(last_three, 0).groupby(group_key).sum().reindex(output.index)

        features[f"{prefix}_minutes"] = active_total * 5.0
        features[f"{prefix}_fraction"] = active_total / records_total
        features[f"{prefix}_semiturns"] = (
            window[f"{prefix}__any"].fillna(0).groupby(group_key).sum().reindex(output.index)
        )
        features[f"{prefix}_last3st_fraction"] = active_last_three / records_last_three.replace(
            0, np.nan
        )
    return pd.concat([output, pd.DataFrame(features, index=output.index)], axis=1)


def add_feeding_mode_features(output: pd.DataFrame, window: pd.DataFrame) -> pd.DataFrame:
    """Agrega porcentaje y minutos por codigo de modo de alimentacion."""
    group_key = window["id_intervalo"]
    records_total = (
        window["h5_records"].fillna(0).groupby(group_key).sum().reindex(output.index)
    )
    mode_slugs = [str(code) for code in FEEDING_MODE_CODES] + ["otro"]
    features: dict[str, pd.Series] = {}
    for slug in mode_slugs:
        column = f"h5_feed_talim_{slug}__records"
        active = window[column].fillna(0).groupby(group_key).sum().reindex(output.index)
        features[f"h5_feed_talim_{slug}_minutes"] = active * 5.0
        features[f"h5_feed_talim_{slug}_fraction"] = active / records_total.replace(0, np.nan)

    features["h5_feed_talim_cambios"] = (
        window["h5_feed_talim_cambios__count"]
        .fillna(0)
        .groupby(group_key)
        .sum()
        .reindex(output.index)
    )
    return pd.concat([output, pd.DataFrame(features, index=output.index)], axis=1)


def build_interval_dataset(
    base_dataset: pd.DataFrame,
    semiturns: pd.DataFrame,
    first_month: str,
    last_month: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construye una fila final por intervalo de TB."""
    first_date = pd.Period(first_month, freq="M").start_time
    end_date = (pd.Period(last_month, freq="M") + 1).start_time

    intervals = base_dataset.copy()
    intervals["fecha_tb_inicial"] = pd.to_datetime(intervals["fecha_tb_inicial"])
    intervals["fecha_tb_objetivo"] = pd.to_datetime(intervals["fecha_tb_objetivo"])
    intervals = intervals[
        intervals["fecha_tb_inicial"].ge(first_date)
        & intervals["fecha_tb_objetivo"].lt(end_date)
    ].copy()
    intervals["orden_inicial_abs"] = absolute_order(
        intervals["fecha_tb_inicial"], intervals["semiturno_tb_inicial"]
    )
    intervals["orden_objetivo_abs"] = absolute_order(
        intervals["fecha_tb_objetivo"], intervals["semiturno_tb_objetivo"]
    )
    if not intervals["orden_objetivo_abs"].sub(intervals["orden_inicial_abs"]).eq(8).all():
        raise ValueError("Hay intervalos que no respetan los 8 semiturnos.")

    window = create_window_rows(intervals, semiturns)
    if not (
        window["ORDEN_SEMITURNO_ABS"].gt(window["orden_inicial_abs"])
        & window["ORDEN_SEMITURNO_ABS"].lt(
            window["id_intervalo"].map(intervals.set_index("id_intervalo")["orden_objetivo_abs"])
        )
    ).all():
        raise ValueError("La ventana HIS5M incluyo un semiturno prohibido.")

    output = pd.DataFrame(index=intervals["id_intervalo"])
    group_key = window["id_intervalo"]
    output["h5_quality_records"] = (
        window["h5_records"].fillna(0).groupby(group_key).sum().reindex(output.index)
    )
    output["h5_quality_records_expected"] = (
        window["records_esperados_ventana"].groupby(group_key).sum().reindex(output.index)
    )
    output["h5_quality_coverage_ratio"] = (
        output["h5_quality_records"] / output["h5_quality_records_expected"]
    )
    output["h5_quality_semiturns_found"] = (
        window["h5_records"].notna().groupby(group_key).sum().reindex(output.index)
    )
    output["h5_quality_min_st_coverage"] = (
        window["h5_coverage_st"].fillna(0).groupby(group_key).min().reindex(output.index)
    )
    output["h5_quality_low_coverage_st"] = (
        window["h5_coverage_st"].fillna(0).lt(0.80).groupby(group_key).sum().reindex(output.index)
    )

    output = add_continuous_interval_features(output, window)
    output = add_state_interval_features(output, window)
    output = add_feeding_mode_features(output, window)

    last_timestamp = window.groupby(group_key)["h5_ultima_hora"].max().reindex(output.index)
    target_start = target_semiturn_start(
        intervals.set_index("id_intervalo")["fecha_tb_objetivo"],
        intervals.set_index("id_intervalo")["semiturno_tb_objetivo"],
    ).reindex(output.index)
    leakage_rows = int((last_timestamp.notna() & last_timestamp.ge(target_start)).sum())
    if leakage_rows:
        raise ValueError(f"Hay {leakage_rows:,} intervalos con datos HIS5M del target.")

    output = output.reset_index().rename(columns={"index": "id_intervalo"})
    final = intervals.drop(columns=["orden_inicial_abs", "orden_objetivo_abs"]).merge(
        output,
        on="id_intervalo",
        how="left",
        validate="one_to_one",
    )
    quality = final[
        [
            "id_intervalo",
            "CUBA",
            "fecha_tb_inicial",
            "fecha_tb_objetivo",
            "tb_inicial",
            "tb_objetivo",
            "delta_tb_objetivo",
            "h5_quality_records",
            "h5_quality_records_expected",
            "h5_quality_coverage_ratio",
            "h5_quality_semiturns_found",
            "h5_quality_min_st_coverage",
            "h5_quality_low_coverage_st",
        ]
    ].copy()
    return final, quality


def write_summary(dataset: pd.DataFrame, quality: pd.DataFrame) -> None:
    """Escribe un resumen corto de construccion y controles de leakage."""
    his_columns = [column for column in dataset if column.startswith("h5_")]
    eligible = quality["h5_quality_coverage_ratio"].ge(0.80) & quality[
        "h5_quality_semiturns_found"
    ].ge(6)
    by_month = (
        quality.assign(mes=quality["fecha_tb_objetivo"].dt.to_period("M").astype(str))
        .groupby("mes")
        .agg(
            intervalos=("id_intervalo", "size"),
            cobertura_media=("h5_quality_coverage_ratio", "mean"),
            cobertura_minima=("h5_quality_coverage_ratio", "min"),
        )
    )

    lines = [
        "FEATURE ENGINEERING HIS5M",
        "=========================",
        "",
        "Regla anti-leakage:",
        "- Se usan solo los 7 semiturnos completos entre TB inicial y TB objetivo.",
        "- No se usa ningun registro del semiturno inicial ni del objetivo.",
        "- La conversion horaria fue validada contra N_FECHA_SMTU de THISEVT.",
        "",
        f"Intervalos construidos: {len(dataset):,}",
        f"Cubas: {dataset['CUBA'].nunique():,}",
        f"Features HIS5M: {len(his_columns):,}",
        f"Intervalos con cobertura >=80% y al menos 6 ST: {int(eligible.sum()):,}",
        f"Cobertura media por intervalo: {quality['h5_quality_coverage_ratio'].mean() * 100:.2f}%",
        "",
        "Cobertura por mes objetivo:",
        by_month.round(4).to_string(),
        "",
        "Familias creadas:",
        "- elect: corriente, tension, resistencia, referencia y potencia proxy.",
        "- control: RTH, WRMI, SMRWFC y componentes de control/resistencia.",
        "- feed: modos TALIM, alimentadores, rompe-costra y consignas.",
        "- instability: minutos de inestabilidad leve, severa y extendida.",
        "- anodic: minutos con estado ANODE_EFFECT.",
        "- operation: colada, cambio de anodo, movimientos de viga y paradas.",
        "",
        "Los codigos TALIM se mantienen numericos hasta confirmar su significado operativo.",
    ]
    SUMMARY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUTPUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Ejecuta la construccion completa del dataset HIS5M."""
    config = load_config()
    first_month = config["modelado"]["primer_mes_his5m"]
    last_month = config["modelado"]["ultimo_mes_his5m"]
    months = month_range(first_month, last_month)

    semiturns = build_semiturn_cache(months)
    base_dataset = pd.read_parquet(BASE_DATASET_PATH)
    dataset, quality = build_interval_dataset(
        base_dataset,
        semiturns,
        first_month,
        last_month,
    )

    DATASET_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    QUALITY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(DATASET_OUTPUT, index=False)
    quality.to_csv(QUALITY_OUTPUT, index=False, encoding="utf-8-sig")
    write_summary(dataset, quality)

    print(f"Dataset: {DATASET_OUTPUT}")
    print(f"Filas: {len(dataset):,}; columnas: {len(dataset.columns):,}")
    print(f"Resumen: {SUMMARY_OUTPUT}")


if __name__ == "__main__":
    main()
