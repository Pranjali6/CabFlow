import pytest
import pandas as pd
import numpy as np
from src.evaluation.metrics import rmse, mae, smape, mase, compute_all_metrics


def test_rmse():
    y_true = np.array([1, 2, 3, 4, 5], dtype=float)
    y_pred = np.array([1.1, 2.2, 2.8, 4.1, 4.9], dtype=float)
    result = rmse(y_true, y_pred)
    assert 0 < result < 1


def test_mae():
    y_true = np.array([1, 2, 3], dtype=float)
    y_pred = np.array([1, 2, 3], dtype=float)
    assert mae(y_true, y_pred) == 0.0


def test_smape_perfect():
    y_true = np.array([1, 2, 3], dtype=float)
    y_pred = np.array([1, 2, 3], dtype=float)
    assert smape(y_true, y_pred) == 0.0


def test_smape_handles_zeros():
    y_true = np.array([0, 0, 0], dtype=float)
    y_pred = np.array([0, 0, 0], dtype=float)
    result = smape(y_true, y_pred)
    assert np.isfinite(result)


def test_mase():
    y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    y_true = np.array([11, 12, 13], dtype=float)
    y_pred = np.array([11.5, 12.5, 13.5], dtype=float)
    result = mase(y_true, y_pred, y_train, seasonality=1)
    assert np.isfinite(result)
    assert result > 0


def test_compute_all_metrics():
    y_true = np.array([1, 2, 3, 4, 5], dtype=float)
    y_pred = np.array([1.1, 2.2, 2.8, 4.1, 4.9], dtype=float)
    result = compute_all_metrics(y_true, y_pred)
    assert "rmse" in result
    assert "mae" in result
    assert "smape" in result
    assert all(np.isfinite(v) for v in result.values())
