"""Ensemble forecasting models: weighted, stacking, and per-series selection."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.models.base import BaseForecaster
from src.models.ml_models import time_series_cv_split

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Weighted Ensemble
# ---------------------------------------------------------------------------

class WeightedEnsemble(BaseForecaster):
    """Weighted average of multiple forecasters.

    Fits all base models on the training data, then optimises the
    combination weights by minimising validation RMSE using
    :func:`scipy.optimize.minimize`.  Weights are constrained to be
    non-negative and sum to one.

    Parameters
    ----------
    forecasters : list[tuple[str, BaseForecaster]]
        Named base forecasters, e.g.
        ``[("xgb", XGBoostForecaster()), ("lgb", LightGBMForecaster())]``.
    val_size : int
        Number of rows held out from the tail of the training data for
        weight optimisation.  Default ``28``.
    """

    name: str = "WeightedEnsemble"

    def __init__(
        self,
        forecasters: List[Tuple[str, BaseForecaster]],
        val_size: int = 28,
    ) -> None:
        self.forecasters = forecasters
        self.val_size = val_size

        self._weights: Optional[np.ndarray] = None
        self._target_col: Optional[str] = None
        self._feature_cols: List[str] = []

    # ----- interface -------------------------------------------------------

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str,
        feature_cols: List[str],
        **kwargs: Any,
    ) -> "WeightedEnsemble":
        """Fit every base model and optimise combination weights.

        The last ``val_size`` rows of *train_df* are used as a hold-out
        set.  Each base model is fitted on the preceding rows, then its
        predictions on the hold-out set are collected.
        :func:`scipy.optimize.minimize` finds the weight vector that
        minimises the RMSE of the weighted average predictions on the
        hold-out set.

        Parameters
        ----------
        train_df : DataFrame
            Training data (must be sorted chronologically).
        target_col : str
            Name of the target column.
        feature_cols : list[str]
            Feature column names.
        **kwargs
            Forwarded to each base model's ``fit()``.
        """
        from scipy.optimize import minimize

        self._target_col = target_col
        self._feature_cols = list(feature_cols)

        # --- time-series split for weight optimisation -----------------------
        split_idx = len(train_df) - self.val_size
        df_train = train_df.iloc[:split_idx]
        df_val = train_df.iloc[split_idx:]
        y_val = df_val[target_col].values

        # Fit each base model and collect validation predictions.
        val_preds: List[np.ndarray] = []
        for name, forecaster in self.forecasters:
            logger.info("WeightedEnsemble: fitting base model %r", name)
            forecaster.fit(df_train, target_col, feature_cols, **kwargs)
            preds = forecaster.predict(df_val)
            val_preds.append(preds)

        val_preds_matrix = np.column_stack(val_preds)  # (n_val, n_models)
        n_models = len(self.forecasters)

        # --- optimise weights ------------------------------------------------
        def _rmse_objective(w: np.ndarray) -> float:
            combined = val_preds_matrix @ w
            return float(np.sqrt(np.mean((y_val - combined) ** 2)))

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        bounds = [(0.0, 1.0)] * n_models
        x0 = np.ones(n_models) / n_models

        result = minimize(
            _rmse_objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        self._weights = result.x

        weight_summary = {
            name: round(float(w), 4)
            for (name, _), w in zip(self.forecasters, self._weights)
        }
        logger.info(
            "WeightedEnsemble: optimal weights = %s  val_rmse = %.4f",
            weight_summary,
            result.fun,
        )

        # Re-fit every base model on the *full* training set so that
        # predict() benefits from the maximum amount of data.
        for name, forecaster in self.forecasters:
            logger.info(
                "WeightedEnsemble: re-fitting %r on full training data", name
            )
            forecaster.fit(train_df, target_col, feature_cols, **kwargs)

        return self

    def predict(self, df: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        """Weighted average of base model predictions.

        Parameters
        ----------
        df : DataFrame
            Must contain the feature columns used during :meth:`fit`.
        **kwargs
            Forwarded to each base model's ``predict()``.

        Returns
        -------
        np.ndarray
        """
        if self._weights is None:
            raise RuntimeError(
                "Ensemble has not been fitted yet. Call fit() first."
            )

        preds = np.column_stack(
            [f.predict(df, **kwargs) for _, f in self.forecasters]
        )
        return preds @ self._weights

    def get_params(self) -> Dict[str, object]:
        return {
            "ensemble_type": "weighted",
            "val_size": self.val_size,
            "n_models": len(self.forecasters),
            "model_names": [name for name, _ in self.forecasters],
            "weights": (
                {
                    name: round(float(w), 6)
                    for (name, _), w in zip(self.forecasters, self._weights)
                }
                if self._weights is not None
                else None
            ),
        }

    # ----- additional helpers -----------------------------------------------

    def get_model_weights(self) -> pd.DataFrame:
        """Return a DataFrame of model names and their optimised weights.

        Returns
        -------
        pd.DataFrame
            Columns: ``model``, ``weight``.

        Raises
        ------
        RuntimeError
            If the ensemble has not been fitted yet.
        """
        if self._weights is None:
            raise RuntimeError(
                "Weights not available. Call fit() first."
            )
        return pd.DataFrame(
            {
                "model": [name for name, _ in self.forecasters],
                "weight": self._weights,
            }
        ).sort_values("weight", ascending=False).reset_index(drop=True)

    def __repr__(self) -> str:
        model_names = [name for name, _ in self.forecasters]
        return f"WeightedEnsemble(models={model_names})"


# ---------------------------------------------------------------------------
# Stacking Ensemble
# ---------------------------------------------------------------------------

class StackingEnsemble(BaseForecaster):
    """Two-level stacking ensemble with out-of-fold meta-training.

    Level 0 (base models) generate out-of-fold predictions using
    expanding-window time-series cross-validation.  Level 1 (meta-learner)
    is trained on those stacked predictions to produce the final forecast.

    Parameters
    ----------
    forecasters : list[tuple[str, BaseForecaster]]
        Named base forecasters.
    meta_learner : str or object
        Either ``'ridge'`` (default) or ``'xgboost'``, or any
        scikit-learn-compatible estimator with ``fit`` / ``predict``.
    n_splits : int
        Number of expanding-window CV folds for generating
        out-of-fold predictions.  Default ``5``.
    test_size : int
        Validation window size per fold.  Default ``28``.
    """

    name: str = "StackingEnsemble"

    def __init__(
        self,
        forecasters: List[Tuple[str, BaseForecaster]],
        meta_learner: Any = "ridge",
        n_splits: int = 5,
        test_size: int = 28,
    ) -> None:
        self.forecasters = forecasters
        self._meta_learner_spec = meta_learner
        self.n_splits = n_splits
        self.test_size = test_size

        self._meta_learner: Optional[Any] = None
        self._target_col: Optional[str] = None
        self._feature_cols: List[str] = []

    # ----- helpers ---------------------------------------------------------

    @staticmethod
    def _build_meta_learner(spec: Any) -> Any:
        """Instantiate a meta-learner from a string or return as-is.

        Parameters
        ----------
        spec : str or estimator
            ``'ridge'`` or ``'xgboost'``, or a pre-built estimator.

        Returns
        -------
        A scikit-learn-compatible estimator.
        """
        if isinstance(spec, str):
            key = spec.lower()
            if key == "ridge":
                from sklearn.linear_model import Ridge

                return Ridge(alpha=1.0)
            elif key == "xgboost":
                from xgboost import XGBRegressor

                return XGBRegressor(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0,
                )
            else:
                raise ValueError(
                    f"Unknown meta_learner string: {spec!r}. "
                    "Use 'ridge' or 'xgboost'."
                )
        # Assume it is an already-instantiated estimator.
        return spec

    # ----- interface -------------------------------------------------------

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str,
        feature_cols: List[str],
        **kwargs: Any,
    ) -> "StackingEnsemble":
        """Fit the stacking ensemble using out-of-fold predictions.

        1. Generate expanding-window CV splits.
        2. For each fold, fit every base model on the training portion
           and record its predictions on the validation portion.
        3. Assemble the out-of-fold prediction matrix and train the
           meta-learner on it.
        4. Re-fit every base model on the full training data so that
           :meth:`predict` has access to fully trained base models.

        Parameters
        ----------
        train_df : DataFrame
            Training data (sorted chronologically).
        target_col : str
            Name of the target column.
        feature_cols : list[str]
            Feature column names.
        **kwargs
            Forwarded to each base model's ``fit()``.
        """
        self._target_col = target_col
        self._feature_cols = list(feature_cols)
        n_models = len(self.forecasters)

        # --- out-of-fold predictions -----------------------------------------
        cv_splits = list(
            time_series_cv_split(
                train_df,
                n_splits=self.n_splits,
                test_size=self.test_size,
            )
        )

        oof_indices: List[np.ndarray] = []
        oof_preds: List[np.ndarray] = []  # each element: (fold_val_size, n_models)

        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            logger.info(
                "StackingEnsemble: fold %d/%d  train=%d  val=%d",
                fold_idx + 1,
                len(cv_splits),
                len(train_idx),
                len(val_idx),
            )

            df_train_fold = train_df.iloc[train_idx]
            df_val_fold = train_df.iloc[val_idx]

            fold_preds = np.empty((len(val_idx), n_models))
            for m_idx, (name, forecaster) in enumerate(self.forecasters):
                forecaster.fit(
                    df_train_fold, target_col, feature_cols, **kwargs
                )
                fold_preds[:, m_idx] = forecaster.predict(df_val_fold)

            oof_indices.append(val_idx)
            oof_preds.append(fold_preds)

        # Combine across folds.
        all_val_idx = np.concatenate(oof_indices)
        meta_X = np.vstack(oof_preds)
        meta_y = train_df[target_col].values[all_val_idx]

        # --- train meta-learner on the stacked OOF predictions ---------------
        self._meta_learner = self._build_meta_learner(self._meta_learner_spec)
        self._meta_learner.fit(meta_X, meta_y)

        meta_pred = self._meta_learner.predict(meta_X)
        meta_rmse = float(np.sqrt(np.mean((meta_y - meta_pred) ** 2)))
        logger.info(
            "StackingEnsemble: meta-learner trained  oof_rmse=%.4f", meta_rmse
        )

        # --- re-fit base models on full training data ------------------------
        for name, forecaster in self.forecasters:
            logger.info(
                "StackingEnsemble: re-fitting %r on full training data", name
            )
            forecaster.fit(train_df, target_col, feature_cols, **kwargs)

        return self

    def predict(self, df: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        """Generate predictions via the stacking meta-learner.

        Base model predictions are assembled into a feature matrix and
        passed to the trained meta-learner.

        Parameters
        ----------
        df : DataFrame
            Must contain the feature columns used during :meth:`fit`.
        **kwargs
            Forwarded to each base model's ``predict()``.

        Returns
        -------
        np.ndarray
        """
        if self._meta_learner is None:
            raise RuntimeError(
                "Ensemble has not been fitted yet. Call fit() first."
            )

        base_preds = np.column_stack(
            [f.predict(df, **kwargs) for _, f in self.forecasters]
        )
        return self._meta_learner.predict(base_preds)

    def get_params(self) -> Dict[str, object]:
        meta_name: str
        if isinstance(self._meta_learner_spec, str):
            meta_name = self._meta_learner_spec
        else:
            meta_name = type(self._meta_learner_spec).__name__

        return {
            "ensemble_type": "stacking",
            "n_models": len(self.forecasters),
            "model_names": [name for name, _ in self.forecasters],
            "meta_learner": meta_name,
            "n_splits": self.n_splits,
            "test_size": self.test_size,
        }

    # ----- additional helpers -----------------------------------------------

    def get_model_weights(self) -> pd.DataFrame:
        """Return meta-learner coefficients (available for linear meta-learners).

        For Ridge regression the coefficients directly indicate each
        base model's contribution.  For non-linear meta-learners a
        placeholder DataFrame is returned.

        Returns
        -------
        pd.DataFrame
            Columns: ``model``, ``coefficient``.

        Raises
        ------
        RuntimeError
            If the ensemble has not been fitted yet.
        """
        if self._meta_learner is None:
            raise RuntimeError(
                "Meta-learner not available. Call fit() first."
            )

        model_names = [name for name, _ in self.forecasters]

        if hasattr(self._meta_learner, "coef_"):
            coefs = np.asarray(self._meta_learner.coef_).ravel()
            return pd.DataFrame(
                {"model": model_names, "coefficient": coefs}
            ).sort_values("coefficient", ascending=False).reset_index(
                drop=True
            )

        if hasattr(self._meta_learner, "feature_importances_"):
            importances = self._meta_learner.feature_importances_
            return pd.DataFrame(
                {"model": model_names, "importance": importances}
            ).sort_values("importance", ascending=False).reset_index(
                drop=True
            )

        return pd.DataFrame(
            {"model": model_names, "coefficient": [np.nan] * len(model_names)}
        )

    def __repr__(self) -> str:
        model_names = [name for name, _ in self.forecasters]
        meta = (
            self._meta_learner_spec
            if isinstance(self._meta_learner_spec, str)
            else type(self._meta_learner_spec).__name__
        )
        return f"StackingEnsemble(models={model_names}, meta={meta})"


# ---------------------------------------------------------------------------
# Per-Series Model Selector
# ---------------------------------------------------------------------------

class PerSeriesModelSelector(BaseForecaster):
    """Select the best forecasting model per individual time series.

    For panel (multi-series) datasets this evaluator fits every
    candidate model on each series independently, measures validation
    RMSE, and retains the best model per series.  At prediction time
    each series is routed to its winning model.

    Parameters
    ----------
    forecasters : list[tuple[str, BaseForecaster]]
        Named candidate forecasters.
    series_col : str
        Column that identifies individual time series.
        Default ``"id"``.
    val_size : int
        Number of time steps held out per series for evaluation.
        Default ``28``.
    metric : str
        Evaluation metric (must be supported by
        :meth:`BaseForecaster.evaluate`).  Default ``"rmse"``.
    """

    name: str = "PerSeriesModelSelector"

    def __init__(
        self,
        forecasters: List[Tuple[str, BaseForecaster]],
        series_col: str = "id",
        val_size: int = 28,
        metric: str = "rmse",
    ) -> None:
        self.forecasters = forecasters
        self.series_col = series_col
        self.val_size = val_size
        self.metric = metric

        # Mapping: series_id -> (model_name, fitted_forecaster)
        self._best_models: Dict[Any, Tuple[str, BaseForecaster]] = {}
        self._selection_df: Optional[pd.DataFrame] = None
        self._target_col: Optional[str] = None
        self._feature_cols: List[str] = []

    # ----- helpers ---------------------------------------------------------

    @staticmethod
    def _clone_forecaster(forecaster: BaseForecaster) -> BaseForecaster:
        """Create a fresh copy of a forecaster via pickle round-trip.

        This ensures each series gets its own independent model instance
        without shared mutable state.
        """
        import pickle

        return pickle.loads(pickle.dumps(forecaster))

    # ----- interface -------------------------------------------------------

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str,
        feature_cols: List[str],
        **kwargs: Any,
    ) -> "PerSeriesModelSelector":
        """Evaluate every model on every series and select the best.

        For each unique value of ``series_col``, the last ``val_size``
        rows are held out as validation.  Every candidate model is
        fitted on the training portion and scored on the validation
        portion.  The model with the lowest validation metric wins.

        Parameters
        ----------
        train_df : DataFrame
            Panel training data containing ``series_col``.
        target_col : str
            Name of the target column.
        feature_cols : list[str]
            Feature column names.
        **kwargs
            Forwarded to each base model's ``fit()``.
        """
        if self.series_col not in train_df.columns:
            raise ValueError(
                f"series_col {self.series_col!r} not found in train_df. "
                f"Available columns: {list(train_df.columns)}"
            )

        self._target_col = target_col
        self._feature_cols = list(feature_cols)

        series_ids = train_df[self.series_col].unique()
        selection_records: List[Dict[str, Any]] = []

        for series_id in series_ids:
            series_mask = train_df[self.series_col] == series_id
            df_series = train_df.loc[series_mask].copy()

            if len(df_series) <= self.val_size:
                logger.warning(
                    "PerSeriesModelSelector: series %r has only %d rows "
                    "(val_size=%d); skipping.",
                    series_id,
                    len(df_series),
                    self.val_size,
                )
                continue

            split_idx = len(df_series) - self.val_size
            df_train_s = df_series.iloc[:split_idx]
            df_val_s = df_series.iloc[split_idx:]
            y_val = df_val_s[target_col].values

            best_score = np.inf
            best_name: Optional[str] = None
            best_model: Optional[BaseForecaster] = None
            model_scores: Dict[str, float] = {}

            for model_name, forecaster in self.forecasters:
                cloned = self._clone_forecaster(forecaster)
                try:
                    cloned.fit(df_train_s, target_col, feature_cols, **kwargs)
                    preds = cloned.predict(df_val_s)
                    score = cloned.evaluate(y_val, preds, metrics=[self.metric])[
                        self.metric
                    ]
                except Exception as exc:
                    logger.warning(
                        "PerSeriesModelSelector: model %r failed on "
                        "series %r: %s",
                        model_name,
                        series_id,
                        exc,
                    )
                    score = np.inf

                model_scores[model_name] = score

                if score < best_score:
                    best_score = score
                    best_name = model_name
                    best_model = cloned

            if best_name is None:
                logger.warning(
                    "PerSeriesModelSelector: no model succeeded for "
                    "series %r; skipping.",
                    series_id,
                )
                continue

            # Re-fit the winner on the *full* series data.
            best_model_full = self._clone_forecaster(
                dict(self.forecasters)[best_name]
            )
            best_model_full.fit(df_series, target_col, feature_cols, **kwargs)
            self._best_models[series_id] = (best_name, best_model_full)

            record: Dict[str, Any] = {
                "series": series_id,
                "best_model": best_name,
                f"best_{self.metric}": best_score,
            }
            for mn, sc in model_scores.items():
                record[f"{mn}_{self.metric}"] = sc
            selection_records.append(record)

        self._selection_df = pd.DataFrame(selection_records)

        if not self._selection_df.empty:
            winner_counts = self._selection_df["best_model"].value_counts()
            logger.info(
                "PerSeriesModelSelector: selection summary\n%s",
                winner_counts.to_string(),
            )

        return self

    def predict(self, df: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        """Route each series to its best model and concatenate predictions.

        The returned array follows the row order of *df*.

        Parameters
        ----------
        df : DataFrame
            Must contain ``series_col`` and the feature columns used
            during :meth:`fit`.
        **kwargs
            Forwarded to each model's ``predict()``.

        Returns
        -------
        np.ndarray
        """
        if not self._best_models:
            raise RuntimeError(
                "Selector has not been fitted yet. Call fit() first."
            )

        predictions = np.empty(len(df), dtype=np.float64)
        predictions[:] = np.nan

        for series_id, (model_name, model) in self._best_models.items():
            mask = df[self.series_col] == series_id
            if not mask.any():
                continue
            df_series = df.loc[mask]
            preds = model.predict(df_series, **kwargs)
            predictions[mask.values] = preds

        # Warn if any series were unseen during fit.
        n_nan = int(np.isnan(predictions).sum())
        if n_nan > 0:
            unseen = set(df[self.series_col].unique()) - set(
                self._best_models.keys()
            )
            logger.warning(
                "PerSeriesModelSelector: %d rows have NaN predictions "
                "(unseen series: %s). Consider adding a fallback model.",
                n_nan,
                unseen,
            )

        return predictions

    def get_params(self) -> Dict[str, object]:
        return {
            "ensemble_type": "per_series_selector",
            "series_col": self.series_col,
            "val_size": self.val_size,
            "metric": self.metric,
            "n_candidates": len(self.forecasters),
            "model_names": [name for name, _ in self.forecasters],
            "n_series_fitted": len(self._best_models),
        }

    # ----- additional helpers -----------------------------------------------

    def get_selection_summary(self) -> pd.DataFrame:
        """Return a DataFrame showing which model won for each series.

        Columns include ``series``, ``best_model``,
        ``best_<metric>``, and per-model metric scores.

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        RuntimeError
            If the selector has not been fitted yet.
        """
        if self._selection_df is None:
            raise RuntimeError(
                "Selection summary not available. Call fit() first."
            )
        return self._selection_df.copy()

    def __repr__(self) -> str:
        model_names = [name for name, _ in self.forecasters]
        return (
            f"PerSeriesModelSelector(models={model_names}, "
            f"series_col={self.series_col!r})"
        )
