"""Open-Meteo hourly weather for NYC, cached to Parquet.

Open-Meteo's Historical Weather API is free, requires no API key, and
returns hourly temperature, precipitation, wind speed, and snowfall for
any latitude/longitude. We pull a single NYC-wide series (Manhattan
center) and merge it onto every zone-hour - weather is roughly uniform
across NYC for demand-forecasting purposes.

Endpoint:
    https://archive-api.open-meteo.com/v1/archive?latitude=40.71&longitude=-74.01&
        start_date=YYYY-MM-DD&end_date=YYYY-MM-DD&
        hourly=temperature_2m,precipitation,snowfall,wind_speed_10m,relative_humidity_2m
"""

from __future__ import annotations

import json
import ssl
import urllib.parse
import urllib.request
from pathlib import Path

import certifi
import numpy as np
import pandas as pd

_SSL_CTX = ssl.create_default_context(cafile=certifi.where())

# Manhattan-central. NYC is small enough that one weather series is fine for
# demand forecasting; the differences between LGA/JFK and Manhattan rarely
# move pickup counts more than a couple of percent.
NYC_LAT = 40.7128
NYC_LON = -74.0060

HOURLY_VARS = [
    "temperature_2m",
    "precipitation",
    "snowfall",
    "wind_speed_10m",
    "relative_humidity_2m",
]


def fetch_weather(
    start_date: str,
    end_date: str,
    lat: float = NYC_LAT,
    lon: float = NYC_LON,
    cache_dir: Path = Path("data/raw/weather"),
) -> pd.DataFrame:
    """Fetch hourly NYC weather between ``start_date`` and ``end_date`` (YYYY-MM-DD).

    Result is cached to a Parquet file keyed by date range so subsequent
    runs don't re-hit the API.

    Returns
    -------
    DataFrame with columns ``hour`` + the variables in ``HOURLY_VARS``.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"nyc_{start_date}_to_{end_date}.parquet"
    if cache_path.exists():
        print(f"  Loading cached weather from {cache_path}")
        return pd.read_parquet(cache_path)

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "America/New_York",
    }
    url = "https://archive-api.open-meteo.com/v1/archive?" + urllib.parse.urlencode(params)
    print(f"  Fetching weather: {url}")

    with urllib.request.urlopen(url, context=_SSL_CTX) as resp:
        payload = json.loads(resp.read())

    hourly = payload.get("hourly", {})
    if "time" not in hourly:
        raise RuntimeError(f"Open-Meteo response missing 'time' field: {payload}")

    df = pd.DataFrame(
        {
            "hour": pd.to_datetime(hourly["time"]),
            **{v: hourly.get(v, []) for v in HOURLY_VARS},
        }
    )
    df.to_parquet(cache_path, index=False)
    print(f"  Cached {len(df):,} hourly rows to {cache_path}")
    return df


def derive_weather_features(weather: pd.DataFrame) -> pd.DataFrame:
    """Engineer features from raw weather.

    Adds:
        - is_raining (precip > 0.1mm in last hour)
        - is_heavy_rain (precip > 5mm)
        - is_snowing (snowfall > 0)
        - is_freezing (temp < 0 C)
        - wind_chill_proxy (low-temp + high-wind interaction, normalized)
        - rolling_precip_3h (recent rainfall lingers in demand effect)
    """
    w = weather.copy().sort_values("hour").reset_index(drop=True)

    w["is_raining"] = (w["precipitation"] > 0.1).astype(np.int8)
    w["is_heavy_rain"] = (w["precipitation"] > 5.0).astype(np.int8)
    w["is_snowing"] = (w["snowfall"] > 0).astype(np.int8)
    w["is_freezing"] = (w["temperature_2m"] < 0).astype(np.int8)

    w["rolling_precip_3h"] = (
        w["precipitation"].rolling(3, min_periods=1).sum().astype(np.float32)
    )
    # Wind chill proxy: high wind + low temp = much lower bike/walk willingness
    # which means more taxi demand. Normalized to roughly [0, 1].
    w["wind_chill_proxy"] = (
        (w["wind_speed_10m"].clip(0, 50) / 50)
        * (1 - w["temperature_2m"].clip(-20, 30).add(20).div(50))
    ).astype(np.float32)

    return w


def attach_weather(
    panel: pd.DataFrame,
    weather: pd.DataFrame | None = None,
    hour_col: str = "hour",
) -> pd.DataFrame:
    """Merge derived weather features onto the (zone, hour) panel."""
    if weather is None:
        start = pd.to_datetime(panel[hour_col]).min().strftime("%Y-%m-%d")
        end = pd.to_datetime(panel[hour_col]).max().strftime("%Y-%m-%d")
        weather = fetch_weather(start, end)

    feats = derive_weather_features(weather)
    merged = panel.merge(feats, on=hour_col, how="left")
    # Fill any missing weather rows (edge of range) with zeros so models don't see NaN.
    for col in HOURLY_VARS + [
        "is_raining",
        "is_heavy_rain",
        "is_snowing",
        "is_freezing",
        "rolling_precip_3h",
        "wind_chill_proxy",
    ]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)
    return merged


if __name__ == "__main__":
    w = fetch_weather("2024-07-01", "2024-12-31")
    print(w.head())
    print(f"shape: {w.shape}")
