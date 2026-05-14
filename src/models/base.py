"""Abstract base class for all forecasting models."""

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


class BaseForecaster(ABC):
    """Abstract base class that every forecasting model must extend.

    Subclasses are required to set the ``name`` attribute and implement
    :meth:`fit`, :meth:`predict`, and :meth:`get_params`.
    """

    name: str = "BaseForecaster"

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(
        self,
        train_df,
        target_col: str,
        feature_cols: List[str],
        **kwargs,
    ) -> "BaseForecaster":
        """Train the model on *train_df*.

        Parameters
        ----------
        train_df : DataFrame
            Training data.
        target_col : str
            Name of the target column.
        feature_cols : list[str]
            Names of the feature columns.
        **kwargs
            Additional model-specific training options.

        Returns
        -------
        self
        """

    @abstractmethod
    def predict(self, df, **kwargs) -> np.ndarray:
        """Generate predictions for the rows in *df*.

        Parameters
        ----------
        df : DataFrame
            Input data (must contain the feature columns used during fit).
        **kwargs
            Additional model-specific prediction options.

        Returns
        -------
        np.ndarray
            Array of predicted values.
        """

    @abstractmethod
    def get_params(self) -> Dict[str, object]:
        """Return the current model parameters as a dictionary."""

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metrics: Optional[Sequence[str]] = None,
    ) -> Dict[str, float]:
        """Compute evaluation metrics.

        Parameters
        ----------
        y_true : array-like
            Ground-truth values.
        y_pred : array-like
            Predicted values.
        metrics : sequence of str, optional
            Which metrics to compute.  Defaults to
            ``["rmse", "mae", "smape"]``.

        Returns
        -------
        dict[str, float]
            Mapping of metric name to computed value.
        """
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)

        if metrics is None:
            metrics = ["rmse", "mae", "smape"]

        results: Dict[str, float] = {}

        for metric in metrics:
            key = metric.lower()
            if key == "rmse":
                results["rmse"] = float(
                    np.sqrt(np.mean((y_true - y_pred) ** 2))
                )
            elif key == "mae":
                results["mae"] = float(np.mean(np.abs(y_true - y_pred)))
            elif key == "smape":
                denominator = np.abs(y_true) + np.abs(y_pred)
                # Avoid division by zero: where both are zero the error is 0.
                smape_values = np.where(
                    denominator == 0,
                    0.0,
                    2.0 * np.abs(y_true - y_pred) / denominator,
                )
                results["smape"] = float(np.mean(smape_values) * 100)
            else:
                raise ValueError(f"Unknown metric: {metric!r}")

        return results

    def save(self, path: str) -> None:
        """Persist the model to disk using pickle.

        Parameters
        ----------
        path : str or Path
            Destination file path.
        """
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "BaseForecaster":
        """Load a previously saved model from disk.

        Parameters
        ----------
        path : str or Path
            Path to the pickled model file.

        Returns
        -------
        BaseForecaster
            The deserialized model instance.
        """
        with open(Path(path), "rb") as f:
            model = pickle.load(f)
        if not isinstance(model, cls):
            raise TypeError(
                f"Loaded object is {type(model).__name__}, "
                f"expected an instance of {cls.__name__}"
            )
        return model

    def __repr__(self) -> str:
        return self.name
