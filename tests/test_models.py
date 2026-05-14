import pytest
import pandas as pd
import numpy as np
from src.models.base import BaseForecaster
from src.models.ml_models import XGBoostForecaster, LightGBMForecaster, time_series_cv_split


@pytest.fixture
def train_data():
    np.random.seed(42)
    n = 500
    return pd.DataFrame({
        "id": "item_1",
        "feature_1": np.random.randn(n),
        "feature_2": np.random.randn(n),
        "feature_3": np.random.uniform(0, 10, n),
        "sales": np.random.poisson(5, n).astype(float),
    })


def test_base_forecaster_is_abstract():
    with pytest.raises(TypeError):
        BaseForecaster()


def test_xgboost_fit_predict(train_data):
    model = XGBoostForecaster(n_estimators=10, max_depth=3)
    feature_cols = ["feature_1", "feature_2", "feature_3"]

    model.fit(train_data.iloc[:400], "sales", feature_cols)
    preds = model.predict(train_data.iloc[400:])

    assert len(preds) == 100
    assert not np.isnan(preds).any()
    assert preds.dtype == np.float32 or preds.dtype == np.float64


def test_lightgbm_fit_predict(train_data):
    model = LightGBMForecaster(n_estimators=10, max_depth=3)
    feature_cols = ["feature_1", "feature_2", "feature_3"]

    model.fit(train_data.iloc[:400], "sales", feature_cols)
    preds = model.predict(train_data.iloc[400:])

    assert len(preds) == 100
    assert not np.isnan(preds).any()


def test_feature_importance(train_data):
    model = XGBoostForecaster(n_estimators=10, max_depth=3)
    feature_cols = ["feature_1", "feature_2", "feature_3"]
    model.fit(train_data.iloc[:400], "sales", feature_cols)

    importance = model.get_feature_importance(top_n=3)
    assert isinstance(importance, pd.DataFrame)
    assert len(importance) == 3


def test_time_series_cv_split(train_data):
    splits = list(time_series_cv_split(train_data, n_splits=3, test_size=28))
    assert len(splits) == 3

    for train_idx, val_idx in splits:
        assert len(val_idx) > 0
        assert len(train_idx) > 0
        assert max(train_idx) < min(val_idx)


def test_model_save_load(train_data, tmp_path):
    model = XGBoostForecaster(n_estimators=10, max_depth=3)
    feature_cols = ["feature_1", "feature_2", "feature_3"]
    model.fit(train_data.iloc[:400], "sales", feature_cols)

    save_path = tmp_path / "model.pkl"
    model.save(str(save_path))

    loaded = BaseForecaster.load(str(save_path))
    assert loaded.name == model.name

    preds_original = model.predict(train_data.iloc[400:])
    preds_loaded = loaded.predict(train_data.iloc[400:])
    np.testing.assert_array_almost_equal(preds_original, preds_loaded)
