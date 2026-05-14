"""Tests for the cab-schema feature engineering."""

import numpy as np
import pandas as pd
import pytest

from src.data.feature_engine import (
    add_calendar_features,
    add_encoding_features,
    add_fourier_features,
    add_holiday_features,
    add_lag_features,
    add_rolling_features,
    add_target_encoding,
)


@pytest.fixture
def sample_panel():
    """Two-zone hourly panel with 14 days of data."""
    rng = np.random.default_rng(42)
    hours = pd.date_range("2024-07-01", periods=24 * 14, freq="h")
    rows = []
    for zone_id, borough in [(161, "Manhattan"), (132, "Queens")]:
        for h in hours:
            rows.append(
                {
                    "PULocationID": zone_id,
                    "hour": h,
                    "pickup_count": int(rng.poisson(20)),
                    "Borough": borough,
                    "zone_name": f"Zone_{zone_id}",
                    "service_zone": "Yellow Zone",
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Lag features
# ---------------------------------------------------------------------------

def test_lag_features_creates_expected_columns(sample_panel):
    result = add_lag_features(sample_panel.copy(), "pickup_count", [1, 24, 168])
    assert {"lag_1h", "lag_24h", "lag_168h"}.issubset(result.columns)


def test_lag_features_respect_zone_boundaries(sample_panel):
    """lag_1h on the first row of each zone must be NaN — never leak across zones."""
    result = add_lag_features(sample_panel.copy(), "pickup_count", [1])
    firsts = result.groupby("PULocationID").head(1)
    assert firsts["lag_1h"].isna().all()


def test_lag_168h_nans_match_expected(sample_panel):
    result = add_lag_features(sample_panel.copy(), "pickup_count", [168])
    # Two zones * 168 hours each = 336 NaNs
    assert result["lag_168h"].isna().sum() == 168 * 2


# ---------------------------------------------------------------------------
# Rolling features
# ---------------------------------------------------------------------------

def test_rolling_features_creates_expected_columns(sample_panel):
    result = add_rolling_features(
        sample_panel.copy(), "pickup_count", [3, 24], ["mean", "std"]
    )
    assert {"rolling_mean_3h", "rolling_std_3h", "rolling_mean_24h", "rolling_std_24h"}.issubset(
        result.columns
    )


def test_rolling_features_no_target_leak(sample_panel):
    """Rolling stats are shifted by 1, so the first row per zone must be NaN."""
    result = add_rolling_features(
        sample_panel.copy(), "pickup_count", [3], ["mean"]
    )
    firsts = result.groupby("PULocationID").head(1)
    assert firsts["rolling_mean_3h"].isna().all()


# ---------------------------------------------------------------------------
# Calendar features
# ---------------------------------------------------------------------------

def test_calendar_features_columns(sample_panel):
    result = add_calendar_features(sample_panel.copy())
    for col in [
        "hour_of_day",
        "day_of_week",
        "day_of_month",
        "month",
        "year",
        "quarter",
        "is_weekend",
        "is_rush_hour",
        "is_late_night",
    ]:
        assert col in result.columns


def test_calendar_flags_are_binary(sample_panel):
    result = add_calendar_features(sample_panel.copy())
    for flag in ("is_weekend", "is_rush_hour", "is_late_night"):
        assert set(result[flag].unique()).issubset({0, 1})


def test_rush_hour_excludes_weekends(sample_panel):
    """Rush-hour flag is only true on weekdays during commute hours."""
    result = add_calendar_features(sample_panel.copy())
    weekend_rush = result[(result["is_weekend"] == 1) & (result["is_rush_hour"] == 1)]
    assert len(weekend_rush) == 0


# ---------------------------------------------------------------------------
# Fourier features
# ---------------------------------------------------------------------------

def test_fourier_creates_daily_and_weekly_terms(sample_panel):
    result = add_fourier_features(
        sample_panel.copy(), daily_period=24, weekly_period=168, order=2
    )
    expected = {
        "fourier_day_sin_1",
        "fourier_day_cos_1",
        "fourier_day_sin_2",
        "fourier_day_cos_2",
        "fourier_week_sin_1",
        "fourier_week_cos_1",
        "fourier_week_sin_2",
        "fourier_week_cos_2",
    }
    assert expected.issubset(result.columns)


def test_fourier_values_bounded(sample_panel):
    result = add_fourier_features(sample_panel.copy(), order=2)
    for col in result.columns:
        if col.startswith("fourier_"):
            assert result[col].between(-1, 1).all()


# ---------------------------------------------------------------------------
# Holiday features
# ---------------------------------------------------------------------------

def test_holiday_features_columns(sample_panel):
    result = add_holiday_features(sample_panel.copy())
    for col in [
        "is_federal_holiday",
        "is_christmas_week",
        "is_nye",
        "is_nyd",
        "is_thanksgiving_week",
        "is_halloween_eve",
        "days_to_christmas",
    ]:
        assert col in result.columns


def test_july_4_marked_as_federal_holiday():
    """A row dated 2024-07-04 must have is_federal_holiday == 1."""
    df = pd.DataFrame(
        {
            "PULocationID": [161],
            "hour": [pd.Timestamp("2024-07-04 12:00")],
            "pickup_count": [10],
        }
    )
    result = add_holiday_features(df)
    assert result.loc[0, "is_federal_holiday"] == 1


def test_nye_flag_only_on_dec_31():
    """is_nye is exactly true on Dec 31 and false elsewhere."""
    df = pd.DataFrame(
        {
            "PULocationID": [1, 1, 1],
            "hour": pd.to_datetime(["2024-12-30", "2024-12-31", "2025-01-01"]),
            "pickup_count": [10, 10, 10],
        }
    )
    result = add_holiday_features(df)
    assert list(result["is_nye"]) == [0, 1, 0]
    assert list(result["is_nyd"]) == [0, 0, 1]


# ---------------------------------------------------------------------------
# Encoding features
# ---------------------------------------------------------------------------

def test_encoding_features_columns(sample_panel):
    result = add_encoding_features(sample_panel.copy())
    assert {"zone_enc", "borough_enc", "service_zone_enc"}.issubset(result.columns)


def test_target_encoding_creates_smooth_means(sample_panel):
    result = add_target_encoding(
        sample_panel.copy(), target="pickup_count", group_cols=["PULocationID", "Borough"]
    )
    assert {"PULocationID_target_enc", "Borough_target_enc"}.issubset(result.columns)
    # Every row must get a finite value
    assert result["PULocationID_target_enc"].notna().all()
    assert np.isfinite(result["PULocationID_target_enc"]).all()
