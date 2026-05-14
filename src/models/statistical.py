"""Statistical forecasting models: SARIMAX, Prophet, and ETS."""

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.models.base import BaseForecaster

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SARIMAX
# ---------------------------------------------------------------------------

class SARIMAXForecaster(BaseForecaster):
    """Seasonal ARIMA with eXogenous regressors.

    Wraps :class:`statsmodels.tsa.statespace.sarimax.SARIMAX` and exposes the
    standard ``fit`` / ``predict`` / ``get_params`` interface.

    Parameters
    ----------
    order : tuple of int
        ``(p, d, q)`` order of the non-seasonal component.  Default ``(1, 1, 1)``.
    seasonal_order : tuple of int
        ``(P, D, Q, s)`` order of the seasonal component.  Default ``(1, 1, 1, 7)``.
    enforce_stationarity : bool
        Passed to the SARIMAX constructor.  Default ``True``.
    enforce_invertibility : bool
        Passed to the SARIMAX constructor.  Default ``True``.
    maxiter : int
        Maximum iterations for the optimizer.  Default ``200``.
    """

    name: str = "SARIMAX"

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 7),
        enforce_stationarity: bool = True,
        enforce_invertibility: bool = True,
        maxiter: int = 200,
    ) -> None:
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.maxiter = maxiter

        self._fitted_model = None
        self._target_col: Optional[str] = None
        self._feature_cols: List[str] = []

    # ----- interface -------------------------------------------------------

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str,
        feature_cols: Optional[List[str]] = None,
        **kwargs,
    ) -> "SARIMAXForecaster":
        """Fit a SARIMAX model on *train_df*.

        Parameters
        ----------
        train_df : DataFrame
            Training data.  Must contain *target_col* and any columns listed
            in *feature_cols*.
        target_col : str
            Name of the column to forecast.
        feature_cols : list[str], optional
            Exogenous regressors.  Pass ``None`` or ``[]`` for a pure SARIMA.
        **kwargs
            Forwarded to :meth:`SARIMAXResults.fit`.
        """
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        self._target_col = target_col
        self._feature_cols = list(feature_cols) if feature_cols else []

        endog = train_df[target_col].values
        exog = train_df[self._feature_cols].values if self._feature_cols else None

        model = SARIMAX(
            endog,
            exog=exog,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility,
        )

        fit_kwargs = {"maxiter": self.maxiter, "disp": False}
        fit_kwargs.update(kwargs)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*convergence.*", category=UserWarning
            )
            warnings.filterwarnings(
                "ignore",
                message=".*Maximum Likelihood optimization failed.*",
                category=UserWarning,
            )
            try:
                self._fitted_model = model.fit(**fit_kwargs)
            except Exception as exc:
                logger.warning(
                    "SARIMAX fit failed with order=%s seasonal_order=%s: %s",
                    self.order,
                    self.seasonal_order,
                    exc,
                )
                raise

        logger.info(
            "SARIMAX fitted  AIC=%.2f  BIC=%.2f",
            self._fitted_model.aic,
            self._fitted_model.bic,
        )
        return self

    def predict(self, df: Optional[pd.DataFrame] = None, **kwargs) -> np.ndarray:
        """Generate out-of-sample forecasts.

        Parameters
        ----------
        df : DataFrame, optional
            If the model was fitted with exogenous regressors, *df* must
            contain those columns for the forecast horizon.  When no exogenous
            variables were used, *df* can be ``None``.
        **kwargs
            ``horizon`` (int) -- number of steps ahead.  Default ``28``.

        Returns
        -------
        np.ndarray
            Forecast values.
        """
        if self._fitted_model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        horizon: int = kwargs.pop("horizon", 28)
        exog = None
        if self._feature_cols and df is not None:
            exog = df[self._feature_cols].values[:horizon]

        forecast = self._fitted_model.forecast(steps=horizon, exog=exog)
        return np.asarray(forecast, dtype=np.float64)

    def get_params(self) -> Dict[str, object]:
        return {
            "order": self.order,
            "seasonal_order": self.seasonal_order,
            "enforce_stationarity": self.enforce_stationarity,
            "enforce_invertibility": self.enforce_invertibility,
            "maxiter": self.maxiter,
        }


# ---------------------------------------------------------------------------
# Prophet
# ---------------------------------------------------------------------------

class ProphetForecaster(BaseForecaster):
    """Facebook / Meta Prophet forecaster.

    Wraps :class:`prophet.Prophet` and exposes the standard interface.

    Parameters
    ----------
    changepoint_prior_scale : float
        Flexibility of the trend changepoints.  Default ``0.05``.
    seasonality_prior_scale : float
        Strength of the seasonality model.  Default ``10``.
    yearly_seasonality : bool or str or int
        Whether to include yearly seasonality.  Default ``True``.
    weekly_seasonality : bool or str or int
        Whether to include weekly seasonality.  Default ``True``.
    regressors : list[str], optional
        Names of additional regressor columns to add to the Prophet model.
    """

    name: str = "Prophet"

    def __init__(
        self,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10,
        yearly_seasonality=True,
        weekly_seasonality=True,
        regressors: Optional[List[str]] = None,
    ) -> None:
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.regressors = list(regressors) if regressors else []

        self._model = None
        self._target_col: Optional[str] = None
        self._date_col: Optional[str] = None
        self._feature_cols: List[str] = []

    # ----- helpers ---------------------------------------------------------

    @staticmethod
    def _prepare_prophet_df(
        df: pd.DataFrame,
        date_col: str,
        target_col: str,
        regressor_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Return a DataFrame with ``ds``, ``y`` and optional regressor columns."""
        out = pd.DataFrame({"ds": pd.to_datetime(df[date_col]), "y": df[target_col]})
        for col in regressor_cols or []:
            out[col] = df[col].values
        return out

    # ----- interface -------------------------------------------------------

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str,
        feature_cols: Optional[List[str]] = None,
        **kwargs,
    ) -> "ProphetForecaster":
        """Fit Prophet on *train_df*.

        Parameters
        ----------
        train_df : DataFrame
            Must contain a date-like column (passed via ``date_col`` kwarg,
            default ``"date"``) and the *target_col*.
        target_col : str
            Column to forecast.
        feature_cols : list[str], optional
            Not used directly for regressors (use the ``regressors`` init
            parameter instead), but stored for compatibility.
        **kwargs
            ``date_col`` (str) -- name of the date column.  Default ``"date"``.
        """
        from prophet import Prophet

        date_col: str = kwargs.pop("date_col", "date")
        self._target_col = target_col
        self._date_col = date_col
        self._feature_cols = list(feature_cols) if feature_cols else []

        # Suppress the verbose Stan/cmdstanpy output.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model = Prophet(
                changepoint_prior_scale=self.changepoint_prior_scale,
                seasonality_prior_scale=self.seasonality_prior_scale,
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
            )

            # Register any custom regressors before fitting.
            for reg in self.regressors:
                self._model.add_regressor(reg)

            prophet_df = self._prepare_prophet_df(
                train_df,
                date_col=date_col,
                target_col=target_col,
                regressor_cols=self.regressors,
            )

            try:
                self._model.fit(prophet_df)
            except Exception as exc:
                logger.warning("Prophet fit failed: %s", exc)
                raise

        logger.info("Prophet model fitted successfully.")
        return self

    def predict(self, df: Optional[pd.DataFrame] = None, **kwargs) -> np.ndarray:
        """Generate forecasts.

        Parameters
        ----------
        df : DataFrame, optional
            If the model uses custom regressors, *df* must contain those
            columns as well as a date column for the forecast period.  When
            no regressors are present, *df* can be ``None`` and a future
            dataframe is created automatically.
        **kwargs
            ``horizon`` (int) -- forecast horizon in periods.  Default ``28``.
            ``freq`` (str) -- pandas frequency string.  Default ``"D"``.

        Returns
        -------
        np.ndarray
            The ``yhat`` column from Prophet's forecast.
        """
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        horizon: int = kwargs.pop("horizon", 28)
        freq: str = kwargs.pop("freq", "D")

        if self.regressors and df is not None:
            # User supplies their own future frame with regressor values.
            future = self._prepare_prophet_df(
                df,
                date_col=self._date_col or "date",
                target_col=self._target_col or "y",
                regressor_cols=self.regressors,
            )
            # Prophet does not need the 'y' column for prediction.
            future["y"] = np.nan
        else:
            future = self._model.make_future_dataframe(periods=horizon, freq=freq)

        forecast = self._model.predict(future)
        # Return only the last ``horizon`` rows (the out-of-sample portion).
        return forecast["yhat"].values[-horizon:]

    def get_params(self) -> Dict[str, object]:
        return {
            "changepoint_prior_scale": self.changepoint_prior_scale,
            "seasonality_prior_scale": self.seasonality_prior_scale,
            "yearly_seasonality": self.yearly_seasonality,
            "weekly_seasonality": self.weekly_seasonality,
            "regressors": self.regressors,
        }


# ---------------------------------------------------------------------------
# ETS (Exponential Smoothing)
# ---------------------------------------------------------------------------

class ETSForecaster(BaseForecaster):
    """Exponential Smoothing (Holt-Winters) forecaster.

    Wraps :class:`statsmodels.tsa.holtwinters.ExponentialSmoothing`.

    Parameters
    ----------
    seasonal_periods : int
        Number of periods in a seasonal cycle.  Default ``7``.
    trend : str or None
        Type of trend component: ``"add"``, ``"mul"``, or ``None``.
        Default ``"add"``.
    seasonal : str or None
        Type of seasonal component: ``"add"``, ``"mul"``, or ``None``.
        Default ``"add"``.
    damped_trend : bool
        Whether to damp the trend component.  Default ``False``.
    """

    name: str = "ETS"

    def __init__(
        self,
        seasonal_periods: int = 7,
        trend: Optional[str] = "add",
        seasonal: Optional[str] = "add",
        damped_trend: bool = False,
    ) -> None:
        self.seasonal_periods = seasonal_periods
        self.trend = trend
        self.seasonal = seasonal
        self.damped_trend = damped_trend

        self._fitted_model = None
        self._target_col: Optional[str] = None

    # ----- interface -------------------------------------------------------

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str,
        feature_cols: Optional[List[str]] = None,
        **kwargs,
    ) -> "ETSForecaster":
        """Fit an Exponential Smoothing model.

        Parameters
        ----------
        train_df : DataFrame
            Training data containing *target_col*.
        target_col : str
            Name of the column to forecast.
        feature_cols : list[str], optional
            Ignored -- ETS is a univariate method.
        **kwargs
            Forwarded to :meth:`ExponentialSmoothing.fit`.
        """
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        self._target_col = target_col
        endog = train_df[target_col].values.astype(np.float64)

        if len(endog) < 2 * self.seasonal_periods:
            logger.warning(
                "Series length (%d) is less than 2 * seasonal_periods (%d). "
                "Falling back to non-seasonal ETS.",
                len(endog),
                self.seasonal_periods,
            )
            seasonal = None
            seasonal_periods = None
        else:
            seasonal = self.seasonal
            seasonal_periods = self.seasonal_periods

        model = ExponentialSmoothing(
            endog,
            trend=self.trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
            damped_trend=self.damped_trend,
        )

        fit_kwargs = {"optimized": True}
        fit_kwargs.update(kwargs)

        try:
            self._fitted_model = model.fit(**fit_kwargs)
        except Exception as exc:
            logger.warning("ETS fit failed: %s", exc)
            raise

        logger.info(
            "ETS fitted  AIC=%.2f  BIC=%.2f",
            self._fitted_model.aic,
            self._fitted_model.bic,
        )
        return self

    def predict(self, df: Optional[pd.DataFrame] = None, **kwargs) -> np.ndarray:
        """Produce out-of-sample forecasts.

        Parameters
        ----------
        df : DataFrame, optional
            Ignored -- ETS is a univariate method.
        **kwargs
            ``horizon`` (int) -- number of steps ahead.  Default ``28``.

        Returns
        -------
        np.ndarray
            Forecast values.
        """
        if self._fitted_model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        horizon: int = kwargs.pop("horizon", 28)
        forecast = self._fitted_model.forecast(steps=horizon)
        return np.asarray(forecast, dtype=np.float64)

    def get_params(self) -> Dict[str, object]:
        return {
            "seasonal_periods": self.seasonal_periods,
            "trend": self.trend,
            "seasonal": self.seasonal,
            "damped_trend": self.damped_trend,
        }
