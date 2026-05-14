"""Aggregate raw NYC TLC trip records into hourly per-zone demand series.

Each pickup is a demand event. We bucket them by (pickup_zone, hour) to
get a panel of time series where each zone is one series and each row is
one hour. Output schema:

    PULocationID  | int      zone id (1..264)
    hour          | datetime hourly bucket (floor of pickup time)
    pickup_count  | int      number of pickups in that zone-hour
    + zone metadata columns: Borough, zone_name, service_zone
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_raw_trips(config: dict | None = None) -> pd.DataFrame:
    """Load every monthly Parquet file and concatenate.

    Only the columns we need are read (pickup datetime + zone) to keep
    memory reasonable - a single month can be 300+ MB on disk.
    """
    if config is None:
        config = load_config()

    raw_dir = Path(config["data"]["raw_dir"]) / "trips"
    taxi_type = config["data"]["taxi_type"]
    files = sorted(raw_dir.glob(f"{taxi_type}_tripdata_*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No trip files found in {raw_dir}. Run downloader first."
        )

    pickup_col = "tpep_pickup_datetime" if taxi_type == "yellow" else "lpep_pickup_datetime"
    cols = [pickup_col, "PULocationID"]

    frames = []
    print(f"Reading {len(files)} trip file(s)...")
    for f in files:
        df = pd.read_parquet(f, columns=cols)
        df = df.rename(columns={pickup_col: "pickup_dt"})
        frames.append(df)
        print(f"  {f.name}: {len(df):,} trips")
    return pd.concat(frames, ignore_index=True)


def load_zone_lookup(config: dict | None = None) -> pd.DataFrame:
    if config is None:
        config = load_config()
    path = Path(config["data"]["raw_dir"]) / "taxi_zone_lookup.csv"
    if not path.exists():
        raise FileNotFoundError(f"Zone lookup not found at {path}. Run downloader first.")
    return pd.read_csv(path)


def filter_outliers(trips: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Drop rows with bad pickup datetimes or invalid zones."""
    if config is None:
        config = load_config()

    months = list(config["data"]["months"])
    start = pd.Timestamp(months[0] + "-01")
    end = (pd.Timestamp(months[-1] + "-01") + pd.offsets.MonthEnd(1)).replace(
        hour=23, minute=59, second=59
    )

    n0 = len(trips)
    trips = trips.dropna(subset=["pickup_dt", "PULocationID"]).copy()
    trips["pickup_dt"] = pd.to_datetime(trips["pickup_dt"], errors="coerce")
    trips = trips.dropna(subset=["pickup_dt"])

    in_window = trips["pickup_dt"].between(start, end)
    trips = trips.loc[in_window].copy()

    valid_zones = trips["PULocationID"].between(1, 263)
    trips = trips.loc[valid_zones].copy()
    trips["PULocationID"] = trips["PULocationID"].astype(np.int16)

    print(f"Filtered {n0 - len(trips):,} bad rows ({len(trips):,} remaining)")
    return trips


def aggregate_hourly(trips: pd.DataFrame) -> pd.DataFrame:
    """Bucket trips into (zone, hour) pickup counts on a dense grid."""
    print("Aggregating to (zone, hour)...")
    trips["hour"] = trips["pickup_dt"].dt.floor("h")
    grouped = (
        trips.groupby(["PULocationID", "hour"], observed=True)
        .size()
        .reset_index(name="pickup_count")
    )

    all_zones = trips["PULocationID"].unique()
    all_hours = pd.date_range(trips["hour"].min(), trips["hour"].max(), freq="h")
    full = pd.MultiIndex.from_product(
        [all_zones, all_hours], names=["PULocationID", "hour"]
    )
    grouped = (
        grouped.set_index(["PULocationID", "hour"])
        .reindex(full, fill_value=0)
        .reset_index()
    )
    grouped["pickup_count"] = grouped["pickup_count"].astype(np.int32)
    grouped["PULocationID"] = grouped["PULocationID"].astype(np.int16)

    print(
        f"  {grouped['PULocationID'].nunique()} zones x {len(all_hours)} hours = {len(grouped):,} rows"
    )
    return grouped


def attach_zone_metadata(
    hourly: pd.DataFrame, zone_lookup: pd.DataFrame
) -> pd.DataFrame:
    """Left-join borough/zone/service_zone onto the hourly panel."""
    lookup = zone_lookup.rename(
        columns={"LocationID": "PULocationID", "Zone": "zone_name"}
    )[["PULocationID", "Borough", "zone_name", "service_zone"]]
    hourly = hourly.merge(lookup, on="PULocationID", how="left")
    hourly["Borough"] = hourly["Borough"].fillna("Unknown")
    hourly["zone_name"] = hourly["zone_name"].fillna("Unknown")
    hourly["service_zone"] = hourly["service_zone"].fillna("Unknown")
    return hourly


def preprocess_pipeline(config: dict | None = None, save: bool = True) -> pd.DataFrame:
    """Run the full preprocessing pipeline.

    Reads raw monthly trip Parquet files, filters bad rows, aggregates to
    hourly per-zone pickup counts, joins zone metadata, and optionally
    saves to ``data/processed/trips_hourly.parquet``.
    """
    if config is None:
        config = load_config()

    trips = load_raw_trips(config)
    trips = filter_outliers(trips, config)
    hourly = aggregate_hourly(trips)

    zone_lookup = load_zone_lookup(config)
    hourly = attach_zone_metadata(hourly, zone_lookup)

    if save:
        out_dir = Path(config["data"]["processed_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "trips_hourly.parquet"
        hourly.to_parquet(out, index=False)
        size = out.stat().st_size / 1024**2
        print(f"Saved to {out} ({size:.1f} MB)")

    return hourly


if __name__ == "__main__":
    preprocess_pipeline()
