import pytest
import pandas as pd
import numpy as np
from src.data.feature_engine import (
    add_lag_features,
    add_rolling_features,
    add_calendar_features,
    add_fourier_features,
    add_price_features,
)


@pytest.fixture
def sample_df():
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    return pd.DataFrame({
        "id": "item_1_store_1",
        "date": dates,
        "sales": np.random.poisson(5, 100).astype(float),
        "sell_price": np.random.uniform(1, 10, 100),
        "item_id": "item_1",
    })


def test_lag_features(sample_df):
    result = add_lag_features(sample_df.copy(), "sales", [7, 14])
    assert "lag_7" in result.columns
    assert "lag_14" in result.columns
    assert result["lag_7"].isna().sum() == 7
    assert result["lag_14"].isna().sum() == 14


def test_rolling_features(sample_df):
    result = add_rolling_features(sample_df.copy(), "sales", [7], ["mean", "std"])
    assert "rolling_mean_7" in result.columns
    assert "rolling_std_7" in result.columns


def test_calendar_features(sample_df):
    result = add_calendar_features(sample_df.copy())
    expected = ["day_of_week", "day_of_month", "week_of_year", "month", "year", "is_weekend"]
    for col in expected:
        assert col in result.columns
    assert result["is_weekend"].isin([0, 1]).all()


def test_fourier_features(sample_df):
    result = add_fourier_features(sample_df.copy(), order=3)
    assert "fourier_sin_1" in result.columns
    assert "fourier_cos_3" in result.columns
    assert result["fourier_sin_1"].between(-1, 1).all()


def test_price_features(sample_df):
    result = add_price_features(sample_df.copy())
    assert "price_change" in result.columns
    assert "price_relative" in result.columns
    assert "price_norm" in result.columns
