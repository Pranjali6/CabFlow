"""Time-series cross-validation backtester.

Supports expanding-window and sliding-window strategies for both panel
(grouped) data and single series.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import compute_all_metrics


# ---------------------------------------------------------------------------
# Model protocol -- any object with fit / predict is accepted
# ---------------------------------------------------------------------------
@runtime_checkable
class _ForecastModel(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Any: ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class BacktestResult:
    """Container returned by :meth:`TimeSeriesBacktester.run`.

    Attributes
    ----------
    fold_metrics : list[dict[str, float]]
        Per-fold metric dictionaries.
    mean_metrics : dict[str, float]
        Metrics averaged across folds.
    predictions : dict[int, pd.DataFrame]
        Mapping of fold index to a DataFrame with columns
        ``['actual', 'predicted']`` (plus the group column when applicable).
    fold_dates : list[tuple[pd.Timestamp, pd.Timestamp]]
        ``(start, end)`` date range of the test set for each fold.
    """

    fold_metrics: list[dict[str, float]] = field(default_factory=list)
    mean_metrics: dict[str, float] = field(default_factory=dict)
    predictions: dict[int, pd.DataFrame] = field(default_factory=dict)
    fold_dates: list[tuple[pd.Timestamp, pd.Timestamp]] = field(
        default_factory=list
    )


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------
class TimeSeriesBacktester:
    """Walk-forward cross-validator for time-series forecasting models.

    Parameters
    ----------
    model : object
        Any model exposing ``fit(X, y)`` and ``predict(X)`` methods.
    n_splits : int, default 5
        Number of CV folds.
    test_size : int, default 28
        Length of the evaluation window (in rows / time steps) per fold.
    strategy : {'expanding_window', 'sliding_window'}
        How the training set grows or moves across folds.
    """

    VALID_STRATEGIES = {"expanding_window", "sliding_window"}

    def __init__(
        self,
        model: Any,
        n_splits: int = 5,
        test_size: int = 28,
        strategy: str = "expanding_window",
    ) -> None:
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"strategy must be one of {self.VALID_STRATEGIES}, "
                f"got '{strategy}'"
            )
        self.model = model
        self.n_splits = n_splits
        self.test_size = test_size
        self.strategy = strategy

    # ---- internal helpers ------------------------------------------------

    @staticmethod
    def _sorted_unique_dates(df: pd.DataFrame) -> np.ndarray:
        """Return sorted unique dates present in the DataFrame index."""
        if isinstance(df.index, pd.DatetimeIndex):
            return np.sort(df.index.unique())
        # Fall back: look for a column named 'date' or 'ds'
        for col in ("date", "ds", "Date", "DS"):
            if col in df.columns:
                return np.sort(df[col].unique())
        # Last resort: use the integer index
        return np.sort(df.index.unique())

    def _generate_splits(
        self, dates: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return ``(train_dates, test_dates)`` tuples for each fold."""
        n_dates = len(dates)
        required = self.n_splits * self.test_size
        if n_dates < required + self.test_size:
            raise ValueError(
                f"Not enough dates ({n_dates}) for {self.n_splits} splits "
                f"with test_size={self.test_size}. Need at least "
                f"{required + self.test_size}."
            )

        splits: list[tuple[np.ndarray, np.ndarray]] = []

        for i in range(self.n_splits):
            test_end_idx = n_dates - i * self.test_size
            test_start_idx = test_end_idx - self.test_size

            test_dates = dates[test_start_idx:test_end_idx]

            if self.strategy == "expanding_window":
                train_dates = dates[:test_start_idx]
            else:
                # sliding_window: keep the training length constant
                # (same as the first fold's training length)
                first_train_len = n_dates - self.n_splits * self.test_size
                train_start = test_start_idx - first_train_len
                train_dates = dates[max(train_start, 0) : test_start_idx]

            splits.append((train_dates, test_dates))

        # Reverse so folds go chronologically (earliest first)
        splits.reverse()
        return splits

    def _select_rows(
        self,
        df: pd.DataFrame,
        dates: np.ndarray,
        date_col: str | None,
    ) -> pd.DataFrame:
        """Select rows matching *dates* from *df*."""
        if isinstance(df.index, pd.DatetimeIndex):
            return df.loc[df.index.isin(dates)]
        if date_col is not None:
            return df.loc[df[date_col].isin(dates)]
        return df.loc[df.index.isin(dates)]

    @staticmethod
    def _detect_date_col(df: pd.DataFrame) -> str | None:
        if isinstance(df.index, pd.DatetimeIndex):
            return None
        for col in ("date", "ds", "Date", "DS"):
            if col in df.columns:
                return col
        return None

    # ---- public API ------------------------------------------------------

    def run(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: list[str],
        group_col: str = "id",
    ) -> BacktestResult:
        """Execute the backtest.

        Parameters
        ----------
        df : pd.DataFrame
            Input data sorted by time (and optionally by *group_col*).
        target_col : str
            Name of the column containing the target variable.
        feature_cols : list[str]
            Names of the feature columns to use for prediction.
        group_col : str, default ``'id'``
            Column identifying individual series in panel data. Ignored
            when the column is not present in *df* (single-series mode).

        Returns
        -------
        BacktestResult
        """
        is_panel = group_col in df.columns
        date_col = self._detect_date_col(df)
        all_dates = self._sorted_unique_dates(df)
        splits = self._generate_splits(all_dates)

        result = BacktestResult()

        for fold_idx, (train_dates, test_dates) in enumerate(splits):
            train_df = self._select_rows(df, train_dates, date_col)
            test_df = self._select_rows(df, test_dates, date_col)

            X_train = train_df[feature_cols]
            y_train = train_df[target_col]
            X_test = test_df[feature_cols]
            y_test = test_df[target_col]

            # Fit and predict
            self.model.fit(X_train, y_train)
            preds = self.model.predict(X_test)

            # Metrics
            fold_metrics = compute_all_metrics(
                y_true=y_test.values,
                y_pred=np.asarray(preds),
                y_train=y_train.values,
            )
            result.fold_metrics.append(fold_metrics)

            # Store predictions
            pred_df = pd.DataFrame(
                {"actual": y_test.values, "predicted": np.asarray(preds)},
                index=test_df.index,
            )
            if is_panel:
                pred_df[group_col] = test_df[group_col].values
            result.predictions[fold_idx] = pred_df

            # Date range
            result.fold_dates.append(
                (pd.Timestamp(test_dates.min()), pd.Timestamp(test_dates.max()))
            )

        # Mean metrics across folds
        all_keys = {k for m in result.fold_metrics for k in m}
        result.mean_metrics = {
            k: float(
                np.mean(
                    [m[k] for m in result.fold_metrics if k in m]
                )
            )
            for k in sorted(all_keys)
        }

        return result

    # ---- visualisation ---------------------------------------------------

    def plot_cv_results(self, result: BacktestResult) -> plt.Figure:
        """Plot actual vs predicted for each fold.

        Parameters
        ----------
        result : BacktestResult
            Output of :meth:`run`.

        Returns
        -------
        matplotlib.figure.Figure
        """
        n_folds = len(result.predictions)
        fig, axes = plt.subplots(
            n_folds, 1, figsize=(14, 4 * n_folds), squeeze=False
        )

        for fold_idx in range(n_folds):
            ax = axes[fold_idx, 0]
            pred_df = result.predictions[fold_idx]
            start, end = result.fold_dates[fold_idx]

            ax.plot(
                pred_df["actual"].values,
                label="Actual",
                linewidth=1.5,
            )
            ax.plot(
                pred_df["predicted"].values,
                label="Predicted",
                linewidth=1.5,
                linestyle="--",
            )

            metrics_str = "  ".join(
                f"{k}: {v:.4f}"
                for k, v in result.fold_metrics[fold_idx].items()
            )
            ax.set_title(
                f"Fold {fold_idx + 1}  [{start.date()} - {end.date()}]  "
                f"{metrics_str}",
                fontsize=10,
            )
            ax.legend(loc="upper right", fontsize=9)
            ax.grid(True, alpha=0.3)

        fig.suptitle("Backtest: Actual vs Predicted", fontsize=13, y=1.01)
        fig.tight_layout()
        return fig
