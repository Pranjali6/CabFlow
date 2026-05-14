"""Machine-learning forecasting models: XGBoost and LightGBM."""

import logging
import warnings
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.models.base import BaseForecaster

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility: time-series cross-validation splits
# ---------------------------------------------------------------------------

def time_series_cv_split(
    df: pd.DataFrame,
    n_splits: int = 5,
    test_size: int = 28,
    group_col: str = "id",
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Expanding window CV splits for time series.

    If *group_col* exists in *df*, the split respects group boundaries so
    that all rows belonging to the same group stay together.  Otherwise
    the split operates on raw row indices, assuming the dataframe is
    already sorted chronologically.

    Parameters
    ----------
    df : DataFrame
        Input data, sorted by time (and optionally by *group_col*).
    n_splits : int
        Number of train/validation folds to produce.
    test_size : int
        Number of time steps (rows per group, or total rows when no
        *group_col*) in each validation window.
    group_col : str
        Column that identifies individual time-series.  If not present
        in *df*, a single-series layout is assumed.

    Yields
    ------
    (train_idx, val_idx) : tuple[np.ndarray, np.ndarray]
        Integer position indices into *df* for each fold.
    """
    n = len(df)

    if group_col in df.columns:
        # ---- grouped (panel) data ----
        groups = df[group_col].unique()
        n_groups = len(groups)

        # Build a mapping: group -> sorted positional indices
        group_to_idx: Dict[Any, np.ndarray] = {}
        for g in groups:
            group_to_idx[g] = df.index[df[group_col] == g].to_numpy()

        # The number of time steps per group (assume equal-length series).
        series_len = len(group_to_idx[groups[0]])

        for fold in range(n_splits):
            # Validation window slides backwards from the end.
            val_end = series_len - fold * test_size
            val_start = val_end - test_size
            if val_start <= 0:
                break

            train_idx_parts: List[np.ndarray] = []
            val_idx_parts: List[np.ndarray] = []
            for g in groups:
                idx = group_to_idx[g]
                train_idx_parts.append(idx[:val_start])
                val_idx_parts.append(idx[val_start:val_end])

            yield (
                np.concatenate(train_idx_parts),
                np.concatenate(val_idx_parts),
            )
    else:
        # ---- single series / flat layout ----
        for fold in range(n_splits):
            val_end = n - fold * test_size
            val_start = val_end - test_size
            if val_start <= 0:
                break
            train_idx = np.arange(0, val_start)
            val_idx = np.arange(val_start, val_end)
            yield train_idx, val_idx


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

class XGBoostForecaster(BaseForecaster):
    """XGBoost gradient-boosted tree forecaster.

    Wraps :class:`xgboost.XGBRegressor` and adds Optuna-based
    hyperparameter tuning with time-series-aware cross-validation.

    Default parameters are read from ``config/config.yaml`` under
    ``models.ml.xgboost``.

    Parameters
    ----------
    n_estimators : int
        Number of boosting rounds.  Default ``1000``.
    max_depth : int
        Maximum tree depth.  Default ``8``.
    learning_rate : float
        Step-size shrinkage.  Default ``0.05``.
    subsample : float
        Row subsampling ratio.  Default ``0.8``.
    colsample_bytree : float
        Column subsampling ratio per tree.  Default ``0.8``.
    early_stopping_rounds : int
        Rounds without validation improvement before stopping.
        Default ``50``.
    val_size : int
        Number of rows held out from the tail of *train_df* during
        :meth:`fit` for early-stopping validation.  Default ``28``.
    random_state : int
        Random seed.  Default ``42``.
    **kwargs
        Forwarded to :class:`xgboost.XGBRegressor`.
    """

    name: str = "XGBoost"

    def __init__(
        self,
        n_estimators: int = 1000,
        max_depth: int = 8,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        early_stopping_rounds: int = 50,
        val_size: int = 28,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.early_stopping_rounds = early_stopping_rounds
        self.val_size = val_size
        self.random_state = random_state
        self.extra_params = kwargs

        self._model: Optional[Any] = None
        self._target_col: Optional[str] = None
        self._feature_cols: List[str] = []
        self._feature_importance: Optional[pd.DataFrame] = None

    # ----- interface -------------------------------------------------------

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str,
        feature_cols: List[str],
        **kwargs: Any,
    ) -> "XGBoostForecaster":
        """Fit an XGBoost model with time-series-aware validation.

        The last ``val_size`` rows of *train_df* are used as a
        validation set for early stopping -- no random splitting is
        performed.

        Parameters
        ----------
        train_df : DataFrame
            Training data (must be sorted chronologically).
        target_col : str
            Name of the target column.
        feature_cols : list[str]
            Feature column names.
        **kwargs
            Forwarded to :meth:`XGBRegressor.fit`.
        """
        from xgboost import XGBRegressor

        self._target_col = target_col
        self._feature_cols = list(feature_cols)

        # --- time-series split (last val_size rows = validation) -----------
        split_idx = len(train_df) - self.val_size
        df_train = train_df.iloc[:split_idx]
        df_val = train_df.iloc[split_idx:]

        X_train = df_train[self._feature_cols].values
        y_train = df_train[target_col].values
        X_val = df_val[self._feature_cols].values
        y_val = df_val[target_col].values

        self._model = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0,
            **self.extra_params,
        )

        fit_kwargs: Dict[str, Any] = {
            "eval_set": [(X_val, y_val)],
            "verbose": False,
        }
        fit_kwargs.update(kwargs)

        self._model.set_params(
            early_stopping_rounds=self.early_stopping_rounds,
        )
        self._model.fit(X_train, y_train, **fit_kwargs)

        # --- store feature importance -------------------------------------
        importances = self._model.feature_importances_
        self._feature_importance = (
            pd.DataFrame(
                {"feature": self._feature_cols, "importance": importances}
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

        best_iter = getattr(self._model, "best_iteration", self.n_estimators)
        logger.info(
            "XGBoost fitted  best_iteration=%d  n_features=%d",
            best_iter,
            len(self._feature_cols),
        )
        return self

    def predict(self, df: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        """Generate predictions for *df*.

        Parameters
        ----------
        df : DataFrame
            Must contain the feature columns used during :meth:`fit`.
        **kwargs
            Forwarded to :meth:`XGBRegressor.predict`.

        Returns
        -------
        np.ndarray
        """
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        X = df[self._feature_cols].values
        return self._model.predict(X, **kwargs)

    def get_params(self) -> Dict[str, object]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "early_stopping_rounds": self.early_stopping_rounds,
            "val_size": self.val_size,
            "random_state": self.random_state,
            **self.extra_params,
        }

    # ----- feature importance ---------------------------------------------

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Return the top-*n* features sorted by importance.

        Parameters
        ----------
        top_n : int
            How many features to return.  Default ``20``.

        Returns
        -------
        pd.DataFrame
            Columns: ``feature``, ``importance``.
        """
        if self._feature_importance is None:
            raise RuntimeError(
                "Feature importance not available. Call fit() first."
            )
        return self._feature_importance.head(top_n).copy()

    # ----- Optuna tuning ---------------------------------------------------

    def tune(
        self,
        train_df: pd.DataFrame,
        target_col: str,
        feature_cols: List[str],
        n_trials: int = 50,
    ) -> Dict[str, Any]:
        """Run Optuna hyperparameter search with time-series CV.

        Uses an expanding-window cross-validation strategy via
        :func:`time_series_cv_split` so that validation folds are
        always in the future relative to their training set.

        Parameters
        ----------
        train_df : DataFrame
            Full training data (sorted chronologically).
        target_col : str
            Target column name.
        feature_cols : list[str]
            Feature column names.
        n_trials : int
            Number of Optuna trials.  Default ``50``.

        Returns
        -------
        dict
            Best hyperparameters found by the study.
        """
        import optuna
        from xgboost import XGBRegressor

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        self._target_col = target_col
        self._feature_cols = list(feature_cols)

        X = train_df[self._feature_cols].values
        y = train_df[target_col].values

        cv_splits = list(
            time_series_cv_split(
                train_df, n_splits=5, test_size=self.val_size
            )
        )

        def objective(trial: optuna.Trial) -> float:
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.5, 1.0
                ),
                "min_child_weight": trial.suggest_int(
                    "min_child_weight", 1, 10
                ),
                "reg_alpha": trial.suggest_float(
                    "reg_alpha", 1e-8, 10.0, log=True
                ),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda", 1e-8, 10.0, log=True
                ),
            }

            fold_scores: List[float] = []
            for train_idx, val_idx in cv_splits:
                X_tr, y_tr = X[train_idx], y[train_idx]
                X_va, y_va = X[val_idx], y[val_idx]

                model = XGBRegressor(
                    **params,
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbosity=0,
                    early_stopping_rounds=self.early_stopping_rounds,
                )

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(
                        X_tr,
                        y_tr,
                        eval_set=[(X_va, y_va)],
                        verbose=False,
                    )

                preds = model.predict(X_va)
                rmse = float(np.sqrt(np.mean((y_va - preds) ** 2)))
                fold_scores.append(rmse)

            return float(np.mean(fold_scores))

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        logger.info(
            "XGBoost Optuna tuning complete  best_rmse=%.4f  best_params=%s",
            study.best_value,
            best_params,
        )

        # Re-fit with the best params on the full training set.
        self.n_estimators = best_params.pop("n_estimators")
        self.max_depth = best_params.pop("max_depth")
        self.learning_rate = best_params.pop("learning_rate")
        self.subsample = best_params.pop("subsample")
        self.colsample_bytree = best_params.pop("colsample_bytree")
        self.extra_params.update(best_params)

        self.fit(train_df, target_col, feature_cols)

        return study.best_params

    def __repr__(self) -> str:
        return (
            f"XGBoostForecaster(n_estimators={self.n_estimators}, "
            f"max_depth={self.max_depth}, lr={self.learning_rate})"
        )


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------

class LightGBMForecaster(BaseForecaster):
    """LightGBM gradient-boosted tree forecaster.

    Wraps :class:`lightgbm.LGBMRegressor` and adds Optuna-based
    hyperparameter tuning with time-series-aware cross-validation.

    Default parameters mirror ``config/config.yaml`` under
    ``models.ml.lightgbm``.

    Parameters
    ----------
    n_estimators : int
        Number of boosting rounds.  Default ``1000``.
    max_depth : int
        Maximum tree depth.  Default ``8``.
    learning_rate : float
        Step-size shrinkage.  Default ``0.05``.
    subsample : float
        Row subsampling ratio (``bagging_fraction``).  Default ``0.8``.
    colsample_bytree : float
        Column subsampling ratio per tree.  Default ``0.8``.
    early_stopping_rounds : int
        Rounds without validation improvement before stopping.
        Default ``50``.
    val_size : int
        Number of rows held out from the tail of *train_df* during
        :meth:`fit`.  Default ``28``.
    random_state : int
        Random seed.  Default ``42``.
    **kwargs
        Forwarded to :class:`lightgbm.LGBMRegressor`.
    """

    name: str = "LightGBM"

    def __init__(
        self,
        n_estimators: int = 1000,
        max_depth: int = 8,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        early_stopping_rounds: int = 50,
        val_size: int = 28,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.early_stopping_rounds = early_stopping_rounds
        self.val_size = val_size
        self.random_state = random_state
        self.extra_params = kwargs

        self._model: Optional[Any] = None
        self._target_col: Optional[str] = None
        self._feature_cols: List[str] = []
        self._feature_importance: Optional[pd.DataFrame] = None

    # ----- interface -------------------------------------------------------

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str,
        feature_cols: List[str],
        **kwargs: Any,
    ) -> "LightGBMForecaster":
        """Fit a LightGBM model with time-series-aware validation.

        The last ``val_size`` rows of *train_df* serve as the
        early-stopping validation set.

        Parameters
        ----------
        train_df : DataFrame
            Training data (must be sorted chronologically).
        target_col : str
            Name of the target column.
        feature_cols : list[str]
            Feature column names.
        **kwargs
            Forwarded to :meth:`LGBMRegressor.fit`.
        """
        import lightgbm as lgb

        self._target_col = target_col
        self._feature_cols = list(feature_cols)

        # --- time-series split ------------------------------------------------
        split_idx = len(train_df) - self.val_size
        df_train = train_df.iloc[:split_idx]
        df_val = train_df.iloc[split_idx:]

        X_train = df_train[self._feature_cols].values
        y_train = df_train[target_col].values
        X_val = df_val[self._feature_cols].values
        y_val = df_val[target_col].values

        callbacks = [
            lgb.early_stopping(self.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=-1),
        ]

        self._model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            n_jobs=1,
            verbosity=-1,
            **self.extra_params,
        )

        fit_kwargs: Dict[str, Any] = {
            "eval_set": [(X_val, y_val)],
            "callbacks": callbacks,
        }
        fit_kwargs.update(kwargs)

        self._model.fit(X_train, y_train, **fit_kwargs)

        # --- store feature importance -----------------------------------------
        importances = self._model.feature_importances_
        self._feature_importance = (
            pd.DataFrame(
                {"feature": self._feature_cols, "importance": importances}
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

        best_iter = getattr(self._model, "best_iteration_", self.n_estimators)
        logger.info(
            "LightGBM fitted  best_iteration=%d  n_features=%d",
            best_iter,
            len(self._feature_cols),
        )
        return self

    def predict(self, df: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        """Generate predictions for *df*.

        Parameters
        ----------
        df : DataFrame
            Must contain the feature columns used during :meth:`fit`.
        **kwargs
            Forwarded to :meth:`LGBMRegressor.predict`.

        Returns
        -------
        np.ndarray
        """
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        X = df[self._feature_cols].values
        return self._model.predict(X, **kwargs)

    def get_params(self) -> Dict[str, object]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "early_stopping_rounds": self.early_stopping_rounds,
            "val_size": self.val_size,
            "random_state": self.random_state,
            **self.extra_params,
        }

    # ----- feature importance ---------------------------------------------

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Return the top-*n* features sorted by importance.

        Parameters
        ----------
        top_n : int
            How many features to return.  Default ``20``.

        Returns
        -------
        pd.DataFrame
            Columns: ``feature``, ``importance``.
        """
        if self._feature_importance is None:
            raise RuntimeError(
                "Feature importance not available. Call fit() first."
            )
        return self._feature_importance.head(top_n).copy()

    # ----- Optuna tuning ---------------------------------------------------

    def tune(
        self,
        train_df: pd.DataFrame,
        target_col: str,
        feature_cols: List[str],
        n_trials: int = 50,
    ) -> Dict[str, Any]:
        """Run Optuna hyperparameter search with time-series CV.

        Uses an expanding-window cross-validation strategy via
        :func:`time_series_cv_split`.

        Parameters
        ----------
        train_df : DataFrame
            Full training data (sorted chronologically).
        target_col : str
            Target column name.
        feature_cols : list[str]
            Feature column names.
        n_trials : int
            Number of Optuna trials.  Default ``50``.

        Returns
        -------
        dict
            Best hyperparameters found by the study.
        """
        import lightgbm as lgb
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        self._target_col = target_col
        self._feature_cols = list(feature_cols)

        X = train_df[self._feature_cols].values
        y = train_df[target_col].values

        cv_splits = list(
            time_series_cv_split(
                train_df, n_splits=5, test_size=self.val_size
            )
        )

        def objective(trial: optuna.Trial) -> float:
            params = {
                "num_leaves": trial.suggest_int("num_leaves", 20, 300),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.5, 1.0
                ),
                "min_child_samples": trial.suggest_int(
                    "min_child_samples", 5, 100
                ),
                "reg_alpha": trial.suggest_float(
                    "reg_alpha", 1e-8, 10.0, log=True
                ),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda", 1e-8, 10.0, log=True
                ),
            }

            fold_scores: List[float] = []
            for train_idx, val_idx in cv_splits:
                X_tr, y_tr = X[train_idx], y[train_idx]
                X_va, y_va = X[val_idx], y[val_idx]

                callbacks = [
                    lgb.early_stopping(
                        self.early_stopping_rounds, verbose=False
                    ),
                    lgb.log_evaluation(period=-1),
                ]

                model = lgb.LGBMRegressor(
                    **params,
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbosity=-1,
                )
                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_va, y_va)],
                    callbacks=callbacks,
                )

                preds = model.predict(X_va)
                rmse = float(np.sqrt(np.mean((y_va - preds) ** 2)))
                fold_scores.append(rmse)

            return float(np.mean(fold_scores))

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        logger.info(
            "LightGBM Optuna tuning complete  best_rmse=%.4f  best_params=%s",
            study.best_value,
            best_params,
        )

        # Re-fit with the best params on the full training set.
        self.n_estimators = best_params.pop("n_estimators")
        self.learning_rate = best_params.pop("learning_rate")
        self.subsample = best_params.pop("subsample")
        self.colsample_bytree = best_params.pop("colsample_bytree")
        self.extra_params.update(best_params)

        self.fit(train_df, target_col, feature_cols)

        return study.best_params

    def __repr__(self) -> str:
        return (
            f"LightGBMForecaster(n_estimators={self.n_estimators}, "
            f"max_depth={self.max_depth}, lr={self.learning_rate})"
        )


# ---------------------------------------------------------------------------
# CatBoost
# ---------------------------------------------------------------------------

class CatBoostForecaster(BaseForecaster):
    """CatBoost gradient-boosted tree forecaster.

    Wraps :class:`catboost.CatBoostRegressor`. CatBoost often beats
    LightGBM on tabular data with mixed categorical + numerical features
    and tends to need less hyperparameter tuning. Adds useful model
    diversity to the XGBoost / LightGBM ensemble.
    """

    name: str = "CatBoost"

    def __init__(
        self,
        iterations: int = 1000,
        depth: int = 8,
        learning_rate: float = 0.05,
        l2_leaf_reg: float = 3.0,
        early_stopping_rounds: int = 50,
        val_size: int = 168,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.early_stopping_rounds = early_stopping_rounds
        self.val_size = val_size
        self.random_state = random_state
        self.extra_params = kwargs

        self._model: Optional[Any] = None
        self._target_col: Optional[str] = None
        self._feature_cols: List[str] = []
        self._feature_importance: Optional[pd.DataFrame] = None

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str,
        feature_cols: List[str],
        **kwargs: Any,
    ) -> "CatBoostForecaster":
        from catboost import CatBoostRegressor, Pool

        self._target_col = target_col
        self._feature_cols = list(feature_cols)

        split_idx = len(train_df) - self.val_size
        df_train = train_df.iloc[:split_idx]
        df_val = train_df.iloc[split_idx:]

        X_train = df_train[self._feature_cols].values
        y_train = df_train[target_col].values
        X_val = df_val[self._feature_cols].values
        y_val = df_val[target_col].values

        self._model = CatBoostRegressor(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            random_seed=self.random_state,
            verbose=False,
            allow_writing_files=False,
            **self.extra_params,
        )
        self._model.fit(
            Pool(X_train, y_train),
            eval_set=Pool(X_val, y_val),
            early_stopping_rounds=self.early_stopping_rounds,
            use_best_model=True,
        )

        importances = self._model.get_feature_importance()
        self._feature_importance = (
            pd.DataFrame({"feature": self._feature_cols, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

        logger.info(
            "CatBoost fitted  best_iteration=%s  n_features=%d",
            self._model.get_best_iteration(),
            len(self._feature_cols),
        )
        return self

    def predict(self, df: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        X = df[self._feature_cols].values
        return self._model.predict(X, **kwargs)

    def get_params(self) -> Dict[str, object]:
        return {
            "iterations": self.iterations,
            "depth": self.depth,
            "learning_rate": self.learning_rate,
            "l2_leaf_reg": self.l2_leaf_reg,
            "early_stopping_rounds": self.early_stopping_rounds,
            "val_size": self.val_size,
            "random_state": self.random_state,
            **self.extra_params,
        }

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        if self._feature_importance is None:
            raise RuntimeError("Feature importance not available. Call fit() first.")
        return self._feature_importance.head(top_n).copy()

    def __repr__(self) -> str:
        return (
            f"CatBoostForecaster(iterations={self.iterations}, "
            f"depth={self.depth}, lr={self.learning_rate})"
        )
