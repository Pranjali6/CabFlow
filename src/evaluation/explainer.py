"""Forecast explainability via SHAP values and STL decomposition."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from statsmodels.tsa.seasonal import STL


class ForecastExplainer:
    """Explain predictions from tree-based forecasting models.

    Supports XGBoost and LightGBM models (any model accepted by
    ``shap.TreeExplainer``).

    Parameters
    ----------
    model : object
        A fitted tree-based model (e.g. ``xgboost.XGBRegressor``,
        ``lightgbm.LGBMRegressor``).
    feature_names : list[str]
        Ordered list of feature names matching the columns used at
        training time.
    """

    def __init__(self, model: Any, feature_names: list[str]) -> None:
        self.model = model
        self.feature_names = list(feature_names)
        self._explainer = shap.TreeExplainer(self.model)

    # ------------------------------------------------------------------
    # SHAP helpers
    # ------------------------------------------------------------------

    def compute_shap_values(
        self, X: pd.DataFrame | np.ndarray
    ) -> shap.Explanation:
        """Compute SHAP values for the given feature matrix.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix with the same columns / ordering used during
            model training.

        Returns
        -------
        shap.Explanation
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)

        shap_values = self._explainer(X)
        return shap_values

    def plot_shap_summary(
        self,
        X: pd.DataFrame | np.ndarray,
        top_n: int = 20,
    ) -> plt.Figure:
        """SHAP beeswarm (summary) plot.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix.
        top_n : int, default 20
            Number of top features to display.

        Returns
        -------
        matplotlib.figure.Figure
        """
        shap_values = self.compute_shap_values(X)

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.plots.beeswarm(shap_values, max_display=top_n, show=False)
        fig = plt.gcf()
        fig.tight_layout()
        return fig

    def plot_shap_waterfall(
        self,
        X: pd.DataFrame | np.ndarray,
        idx: int = 0,
    ) -> plt.Figure:
        """SHAP waterfall plot for a single prediction.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix.
        idx : int, default 0
            Row index of the observation to explain.

        Returns
        -------
        matplotlib.figure.Figure
        """
        shap_values = self.compute_shap_values(X)

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.plots.waterfall(shap_values[idx], show=False)
        fig = plt.gcf()
        fig.tight_layout()
        return fig

    def plot_feature_importance(
        self,
        X: pd.DataFrame | np.ndarray | None = None,
        top_n: int = 20,
    ) -> plt.Figure:
        """Bar chart of mean absolute SHAP values (feature importance).

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray, optional
            Feature matrix. If provided, SHAP values are computed on the
            fly; otherwise the method raises.
        top_n : int, default 20
            Number of top features to display.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if X is None:
            raise ValueError(
                "Provide a feature matrix X so SHAP values can be computed."
            )

        shap_values = self.compute_shap_values(X)
        mean_abs = np.abs(shap_values.values).mean(axis=0)

        # Sort descending and take top_n
        indices = np.argsort(mean_abs)[::-1][:top_n]
        top_names = [self.feature_names[i] for i in indices]
        top_values = mean_abs[indices]

        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
        ax.barh(range(len(top_names)), top_values[::-1], align="center")
        ax.set_yticks(range(len(top_names)))
        ax.set_yticklabels(top_names[::-1])
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"Top {top_n} Feature Importances (mean |SHAP|)")
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # STL decomposition
    # ------------------------------------------------------------------

    def decompose_forecast(
        self,
        y: pd.Series | np.ndarray,
        period: int = 7,
    ) -> dict[str, np.ndarray]:
        """STL decomposition of a time series.

        Parameters
        ----------
        y : pd.Series or np.ndarray
            Univariate time series.
        period : int, default 7
            Seasonal period (e.g. 7 for daily data with weekly pattern).

        Returns
        -------
        dict[str, np.ndarray]
            Keys: ``'trend'``, ``'seasonal'``, ``'residual'``.
        """
        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        stl = STL(y, period=period)
        result = stl.fit()

        return {
            "trend": np.asarray(result.trend),
            "seasonal": np.asarray(result.seasonal),
            "residual": np.asarray(result.resid),
        }

    def plot_decomposition(
        self,
        y: pd.Series | np.ndarray,
        period: int = 7,
    ) -> plt.Figure:
        """Plot STL decomposition (observed, trend, seasonal, residual).

        Parameters
        ----------
        y : pd.Series or np.ndarray
            Univariate time series.
        period : int, default 7
            Seasonal period.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        components = self.decompose_forecast(y, period=period)

        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

        axes[0].plot(y.values, linewidth=1.2)
        axes[0].set_ylabel("Observed")
        axes[0].set_title("STL Decomposition")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(components["trend"], linewidth=1.2, color="tab:orange")
        axes[1].set_ylabel("Trend")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(components["seasonal"], linewidth=1.2, color="tab:green")
        axes[2].set_ylabel("Seasonal")
        axes[2].grid(True, alpha=0.3)

        axes[3].plot(components["residual"], linewidth=1.2, color="tab:red")
        axes[3].set_ylabel("Residual")
        axes[3].grid(True, alpha=0.3)

        fig.tight_layout()
        return fig
