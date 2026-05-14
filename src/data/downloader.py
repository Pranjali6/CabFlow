"""Download NYC TLC trip records and zone metadata.

NYC TLC publishes monthly Parquet files at:
    https://d37ci6vzurychx.cloudfront.net/trip-data/<taxi_type>_tripdata_YYYY-MM.parquet

Yellow taxi data is the longest-running and most consistent series.
Each row is one trip; the columns we care about are:
    tpep_pickup_datetime, tpep_dropoff_datetime, PULocationID, DOLocationID
"""

from __future__ import annotations

import ssl
import urllib.request
from pathlib import Path
from typing import Iterable

import certifi
import yaml

_SSL_CTX = ssl.create_default_context(cafile=certifi.where())


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _download(url: str, dest: Path) -> None:
    """Stream a URL to disk with a small progress indicator."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  -> {url}")
    with urllib.request.urlopen(url, context=_SSL_CTX) as resp, open(dest, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk = 1024 * 256
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            f.write(buf)
            downloaded += len(buf)
            if total:
                pct = downloaded / total * 100
                print(
                    f"     {downloaded / 1024**2:6.1f} MB / {total / 1024**2:6.1f} MB  ({pct:5.1f}%)",
                    end="\r",
                )
        print()
    print(f"     saved to {dest} ({dest.stat().st_size / 1024**2:.1f} MB)")


def download_trip_files(
    config: dict | None = None,
    months: Iterable[str] | None = None,
    force: bool = False,
) -> list[Path]:
    """Download one Parquet file per month of trip data."""
    if config is None:
        config = load_config()

    months = list(months) if months is not None else list(config["data"]["months"])
    taxi_type = config["data"]["taxi_type"]
    base = config["data"]["base_url"].rstrip("/")
    raw_dir = Path(config["data"]["raw_dir"]) / "trips"

    print(f"Downloading {taxi_type} taxi data for {len(months)} month(s)...")
    paths: list[Path] = []
    for ym in months:
        fname = f"{taxi_type}_tripdata_{ym}.parquet"
        dest = raw_dir / fname
        if dest.exists() and not force:
            print(f"  {fname} already present ({dest.stat().st_size / 1024**2:.1f} MB)")
        else:
            _download(f"{base}/{fname}", dest)
        paths.append(dest)
    return paths


def download_zone_lookup(config: dict | None = None, force: bool = False) -> Path:
    """Download the taxi-zone lookup CSV (zone_id -> borough / zone name)."""
    if config is None:
        config = load_config()

    url = config["data"]["zone_lookup_url"]
    dest = Path(config["data"]["raw_dir"]) / "taxi_zone_lookup.csv"
    if dest.exists() and not force:
        print(f"Zone lookup already present at {dest}.")
        return dest

    print("Downloading taxi-zone lookup...")
    _download(url, dest)
    return dest


def download_zone_geojson(config: dict | None = None, force: bool = False) -> Path | None:
    """Download the GeoJSON outline of all 264 taxi zones.

    Tries each URL in ``zone_geojson_urls`` until one succeeds. Returns
    ``None`` if every source is unreachable - the map view degrades
    gracefully when the geojson is missing.
    """
    if config is None:
        config = load_config()

    urls = config["data"].get("zone_geojson_urls")
    if not urls:
        single = config["data"].get("zone_geojson_url")
        urls = [single] if single else []

    dest = Path(config["data"]["raw_dir"]) / "taxi_zones.geojson"
    if dest.exists() and not force:
        print(f"Zone GeoJSON already present at {dest}.")
        return dest

    print("Downloading taxi-zone GeoJSON...")
    for url in urls:
        try:
            _download(url, dest)
            return dest
        except Exception as exc:
            print(f"     {url} failed: {exc}")
            if dest.exists():
                dest.unlink()
    print("  WARNING: every GeoJSON source failed. Map tab will be disabled.")
    return None


def download_all(config: dict | None = None, force: bool = False) -> dict:
    """Convenience: download trips + lookup + geojson in one call."""
    if config is None:
        config = load_config()
    return {
        "trips": download_trip_files(config, force=force),
        "zone_lookup": download_zone_lookup(config, force=force),
        "zone_geojson": download_zone_geojson(config, force=force),
    }


if __name__ == "__main__":
    download_all()
