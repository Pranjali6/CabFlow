"""Tests for the cab-schema models."""

import numpy as np
import pandas as pd
import pytest

from src.models.base import BaseForecaster
from src.models.ml_models import (
    CatBoostForecaster,
    LightGBMForecaster,
    XGBoostForecaster,
    time_series_cv_split,
)


@pytest.fixture
def train_data():
    """500-row synthetic zone-hour panel with 3 features."""
    rng = np.random.default_rng(42)
    n = 500
    return pd.DataFrame(
        {
            "PULocationID": 161,
            "hour": pd.date_range("2024-07-01", periods=n, freq="h"),
            "feature_1": rng.normal(size=n),
            "feature_2": rng.normal(size=n),
            "feature_3": rng.uniform(0, 10, size=n),
            "pickup_count": rng.poisson(20, size=n).astype(float),
        }
    )


FEATURE_COLS = ["feature_1", "feature_2", "feature_3"]
TARGET = "pickup_count"


def test_base_forecaster_is_abstract():
    """BaseForecaster cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseForecaster()


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

def test_xgboost_fit_predict(train_data):
    model = XGBoostForecaster(n_estimators=10, max_depth=3, val_size=24)
    model.fit(train_data.iloc[:400], TARGET, FEATURE_COLS)
    preds = model.predict(train_data.iloc[400:])
    assert len(preds) == 100
    assert not np.isnan(preds).any()


def test_xgboost_feature_importance(train_data):
    model = XGBoostForecaster(n_estimators=10, max_depth=3, val_size=24)
    model.fit(train_data.iloc[:400], TARGET, FEATURE_COLS)
    imp = model.get_feature_importance(top_n=3)
    assert isinstance(imp, pd.DataFrame)
    assert len(imp) == 3
    assert {"feature", "importance"}.issubset(imp.columns)


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------

def test_lightgbm_fit_predict(train_data):
    model = LightGBMForecaster(n_estimators=10, max_depth=3, val_size=24)
    model.fit(train_data.iloc[:400], TARGET, FEATURE_COLS)
    preds = model.predict(train_data.iloc[400:])
    assert len(preds) == 100
    assert not np.isnan(preds).any()


# ---------------------------------------------------------------------------
# CatBoost
# ---------------------------------------------------------------------------

def test_catboost_fit_predict(train_data):
    model = CatBoostForecaster(iterations=20, depth=4, val_size=24)
    model.fit(train_data.iloc[:400], TARGET, FEATURE_COLS)
    preds = model.predict(train_data.iloc[400:])
    assert len(preds) == 100
    assert not np.isnan(preds).any()


def test_catboost_feature_importance(train_data):
    model = CatBoostForecaster(iterations=20, depth=4, val_size=24)
    model.fit(train_data.iloc[:400], TARGET, FEATURE_COLS)
    imp = model.get_feature_importance(top_n=3)
    assert len(imp) == 3
    assert imp["importance"].sum() > 0


# ---------------------------------------------------------------------------
# Time-series CV
# ---------------------------------------------------------------------------

def test_time_series_cv_split_no_overlap(train_data):
    """Validation indices must always be after training indices."""
    splits = list(time_series_cv_split(train_data, n_splits=3, test_size=24, group_col="PULocationID"))
    assert len(splits) == 3
    for train_idx, val_idx in splits:
        assert len(train_idx) > 0
        assert len(val_idx) > 0
        assert max(train_idx) < min(val_idx)


def test_time_series_cv_split_respects_horizon(train_data):
    """Each validation fold has exactly ``test_size`` rows."""
    splits = list(time_series_cv_split(train_data, n_splits=3, test_size=24, group_col="PULocationID"))
    for _, val_idx in splits:
        assert len(val_idx) == 24


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------

def test_xgboost_save_load_roundtrip(train_data, tmp_path):
    model = XGBoostForecaster(n_estimators=10, max_depth=3, val_size=24)
    model.fit(train_data.iloc[:400], TARGET, FEATURE_COLS)
    p = tmp_path / "xgb.pkl"
    model.save(str(p))
    loaded = BaseForecaster.load(str(p))
    assert loaded.name == model.name
    np.testing.assert_array_almost_equal(
        model.predict(train_data.iloc[400:]),
        loaded.predict(train_data.iloc[400:]),
    )


def test_catboost_save_load_roundtrip(train_data, tmp_path):
    model = CatBoostForecaster(iterations=20, depth=4, val_size=24)
    model.fit(train_data.iloc[:400], TARGET, FEATURE_COLS)
    p = tmp_path / "cat.pkl"
    model.save(str(p))
    loaded = BaseForecaster.load(str(p))
    assert loaded.name == model.name
    np.testing.assert_array_almost_equal(
        model.predict(train_data.iloc[400:]),
        loaded.predict(train_data.iloc[400:]),
    )
