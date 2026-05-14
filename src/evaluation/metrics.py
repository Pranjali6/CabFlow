"""Standalone forecast evaluation metrics.

All functions accept numpy arrays and return float values.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root Mean Squared Error."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean Absolute Error."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


def mase(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_train: ArrayLike,
    seasonality: int = 7,
) -> float:
    """Mean Absolute Scaled Error.

    Uses a naive seasonal forecast on the training set as the baseline
    denominator.  The naive seasonal forecast at time *t* is simply
    ``y_train[t - seasonality]``.

    Parameters
    ----------
    y_true : array-like
        Actual values for the evaluation period.
    y_pred : array-like
        Predicted values for the evaluation period.
    y_train : array-like
        Historical (training) series used to compute the naive baseline scale.
    seasonality : int, default 7
        Seasonal period for the naive forecast (e.g. 7 for daily data with
        weekly seasonality).

    Returns
    -------
    float
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)

    # Naive seasonal forecast errors on the training set
    naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    scale = np.mean(naive_errors)

    if scale == 0.0:
        return float("inf")

    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def smape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Symmetric Mean Absolute Percentage Error.

    Defined as::

        SMAPE = 100 * mean( 2 * |y - yhat| / (|y| + |yhat|) )

    When both ``y`` and ``yhat`` are zero for a given observation the
    contribution is treated as 0 (avoids 0/0).

    Returns a value in [0, 200].
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    numerator = 2.0 * np.abs(y_true - y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)

    # Handle division by zero: where both are 0 the ratio is 0
    ratio = np.where(denominator == 0.0, 0.0, numerator / denominator)

    return float(100.0 * np.mean(ratio))


def wrmsse(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    weights: ArrayLike,
    scale: ArrayLike,
) -> float:
    """Weighted Root Mean Squared Scaled Error (official M5 metric).

    Parameters
    ----------
    y_true : array-like, shape (n_series, horizon)
        Actual values for each series over the evaluation horizon.
    y_pred : array-like, shape (n_series, horizon)
        Predicted values for each series over the evaluation horizon.
    weights : array-like, shape (n_series,)
        Pre-computed revenue-based weights for each series (should sum to 1).
    scale : array-like, shape (n_series,)
        Pre-computed scale (mean squared error of the naive forecast on the
        training set) for each series.

    Returns
    -------
    float
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)

    # RMSSE per series: sqrt( mean( (y - yhat)^2 ) / scale )
    mse_per_series = np.mean((y_true - y_pred) ** 2, axis=-1)

    # Guard against zero scale
    safe_scale = np.where(scale == 0.0, 1.0, scale)
    rmsse_per_series = np.sqrt(mse_per_series / safe_scale)

    return float(np.sum(weights * rmsse_per_series))


def compute_all_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_train: ArrayLike | None = None,
    weights: ArrayLike | None = None,
    scale: ArrayLike | None = None,
) -> dict[str, float]:
    """Compute all available metrics and return them as a dictionary.

    ``mase`` is only computed when *y_train* is provided.
    ``wrmsse`` is only computed when both *weights* and *scale* are provided.

    Parameters
    ----------
    y_true : array-like
        Actual values.
    y_pred : array-like
        Predicted values.
    y_train : array-like or None
        Training series for MASE computation.
    weights : array-like or None
        Series weights for WRMSSE computation.
    scale : array-like or None
        Series scale values for WRMSSE computation.

    Returns
    -------
    dict[str, float]
    """
    results: dict[str, float] = {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }

    if y_train is not None:
        results["mase"] = mase(y_true, y_pred, y_train)

    if weights is not None and scale is not None:
        results["wrmsse"] = wrmsse(y_true, y_pred, weights, scale)

    return results
