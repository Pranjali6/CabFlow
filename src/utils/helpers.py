import yaml
import numpy as np
import pandas as pd
from pathlib import Path


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Downcast numeric columns to reduce memory usage."""
    start_mem = df.memory_usage(deep=True).sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype
        if col_type in (np.float64, np.float32):
            df[col] = pd.to_numeric(df[col], downcast="float")
        elif col_type in (np.int64, np.int32, np.int16):
            df[col] = pd.to_numeric(df[col], downcast="integer")

    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        print(
            f"Memory: {start_mem:.1f} MB -> {end_mem:.1f} MB "
            f"({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)"
        )

    return df


def get_feature_columns(df: pd.DataFrame, exclude: list[str] | None = None) -> list[str]:
    """Numeric feature columns, excluding ID / target / date / metadata."""
    default_exclude = {
        # target & time index
        "pickup_count",
        "hour",
        # zone identifiers (categorical / string)
        "PULocationID",
        "DOLocationID",
        "Borough",
        "zone_name",
        "service_zone",
    }
    if exclude:
        default_exclude.update(exclude)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in default_exclude]
