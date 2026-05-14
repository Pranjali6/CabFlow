"""Feature engineering for hourly per-zone taxi demand.

Mirrors the original (M5) pipeline but operates on hours instead of days:
    - Lag features at 1h, 2h, 3h, 24h (day), 48h, 168h (week)
    - Rolling means/std/min/max at 3h, 6h, 24h, 168h windows
    - Calendar features: hour_of_day, day_of_week, weekend/rush flags
    - Fourier terms for daily (24h) and weekly (168h) seasonality
    - Categorical label encodings for zone/borough
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def add_lag_features(
    df: pd.DataFrame,
    target: str,
    lags: Sequence[int],
    group_col: str = "PULocationID",
) -> pd.DataFrame:
    """Add lagged values of the target."""
    print(f"Adding lag features: {list(lags)}")
    g = df.groupby(group_col)[target]
    for lag in lags:
        df[f"lag_{lag}h"] = g.shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    target: str,
    windows: Sequence[int],
    stats: Sequence[str],
    group_col: str = "PULocationID",
    min_periods: int = 1,
) -> pd.DataFrame:
    """Add rolling-window statistics. Shifted by 1 to avoid target leak."""
    print(f"Adding rolling features: windows={list(windows)} stats={list(stats)}")
    for window in windows:
        shifted = df.groupby(group_col)[target].transform(lambda x: x.shift(1))
        helper = shifted.groupby(df[group_col])
        if "mean" in stats:
            df[f"rolling_mean_{window}h"] = helper.transform(
                lambda x: x.rolling(window, min_periods=min_periods).mean()
            )
        if "std" in stats:
            df[f"rolling_std_{window}h"] = helper.transform(
                lambda x: x.rolling(window, min_periods=min_periods).std()
            )
        if "min" in stats:
            df[f"rolling_min_{window}h"] = helper.transform(
                lambda x: x.rolling(window, min_periods=min_periods).min()
            )
        if "max" in stats:
            df[f"rolling_max_{window}h"] = helper.transform(
                lambda x: x.rolling(window, min_periods=min_periods).max()
            )
    return df


def add_calendar_features(df: pd.DataFrame, date_col: str = "hour") -> pd.DataFrame:
    """Extract hour/day/month and weekend / rush-hour flags."""
    print("Adding calendar features...")
    dt = pd.to_datetime(df[date_col])
    df["hour_of_day"] = dt.dt.hour.astype(np.int8)
    df["day_of_week"] = dt.dt.dayofweek.astype(np.int8)
    df["day_of_month"] = dt.dt.day.astype(np.int8)
    df["month"] = dt.dt.month.astype(np.int8)
    df["year"] = dt.dt.year.astype(np.int16)
    df["quarter"] = dt.dt.quarter.astype(np.int8)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(np.int8)
    morning = df["hour_of_day"].between(7, 9)
    evening = df["hour_of_day"].between(17, 19)
    df["is_rush_hour"] = ((morning | evening) & (df["is_weekend"] == 0)).astype(np.int8)
    df["is_late_night"] = (
        df["hour_of_day"].between(22, 23) | (df["hour_of_day"] <= 2)
    ).astype(np.int8)
    return df


def add_fourier_features(
    df: pd.DataFrame,
    date_col: str = "hour",
    daily_period: float = 24.0,
    weekly_period: float = 168.0,
    order: int = 3,
) -> pd.DataFrame:
    """Fourier terms for daily and weekly seasonality."""
    print(
        f"Adding Fourier features: daily={daily_period}, weekly={weekly_period}, order={order}"
    )
    dt = pd.to_datetime(df[date_col])
    hour_num = (dt - dt.min()).dt.total_seconds() / 3600.0
    for k in range(1, order + 1):
        df[f"fourier_day_sin_{k}"] = np.sin(2 * np.pi * k * hour_num / daily_period)
        df[f"fourier_day_cos_{k}"] = np.cos(2 * np.pi * k * hour_num / daily_period)
        df[f"fourier_week_sin_{k}"] = np.sin(2 * np.pi * k * hour_num / weekly_period)
        df[f"fourier_week_cos_{k}"] = np.cos(2 * np.pi * k * hour_num / weekly_period)
    return df


def add_encoding_features(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode zone metadata so tree models can use it."""
    print("Adding encoding features...")
    if "PULocationID" in df.columns:
        df["zone_enc"] = df["PULocationID"].astype("category").cat.codes
    if "Borough" in df.columns:
        df["borough_enc"] = df["Borough"].astype("category").cat.codes
    if "service_zone" in df.columns:
        df["service_zone_enc"] = df["service_zone"].astype("category").cat.codes
    return df


def add_target_encoding(
    df: pd.DataFrame,
    target: str = "pickup_count",
    group_cols: Sequence[str] = ("PULocationID", "Borough"),
    smoothing: float = 10.0,
) -> pd.DataFrame:
    """Smoothed target mean per zone / borough."""
    print(f"Adding target encodings for: {list(group_cols)}")
    global_mean = df[target].mean()
    for col in group_cols:
        if col not in df.columns:
            continue
        stats = df.groupby(col)[target].agg(["mean", "count"])
        smooth = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
            stats["count"] + smoothing
        )
        df[f"{col}_target_enc"] = df[col].map(smooth).astype(np.float32)
    return df


def build_features(
    df: pd.DataFrame,
    config: dict | None = None,
    include_target_encoding: bool = True,
) -> pd.DataFrame:
    """Run the full feature engineering pipeline."""
    if config is None:
        config = load_config()

    feat_cfg = config["features"]
    target = config["data"]["target_col"]
    date_col = config["data"]["date_col"]

    df = df.sort_values(["PULocationID", date_col]).reset_index(drop=True)

    df = add_lag_features(df, target, feat_cfg["lag_hours"])
    df = add_rolling_features(
        df, target, feat_cfg["rolling_windows"], feat_cfg["rolling_stats"]
    )
    df = add_calendar_features(df, date_col)
    df = add_fourier_features(
        df,
        date_col,
        daily_period=feat_cfg["fourier_period_daily"],
        weekly_period=feat_cfg["fourier_period_weekly"],
        order=feat_cfg["fourier_order"],
    )
    df = add_encoding_features(df)

    if include_target_encoding:
        df = add_target_encoding(df, target)

    initial = len(df)
    longest_lag = max(feat_cfg["lag_hours"])
    df = df.dropna(subset=[f"lag_{longest_lag}h"])
    print(f"Dropped {initial - len(df):,} rows with NaN lags ({len(df):,} remaining)")

    return df


if __name__ == "__main__":
    from pathlib import Path

    cfg = load_config()
    pdir = Path(cfg["data"]["processed_dir"])
    df = pd.read_parquet(pdir / "trips_hourly.parquet")
    df = build_features(df, cfg)
    out = pdir / "trips_featured.parquet"
    df.to_parquet(out, index=False)
    print(f"Saved featured data: {out} ({out.stat().st_size / 1024**2:.1f} MB)")
