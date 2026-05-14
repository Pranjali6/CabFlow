"""Deep-learning forecasting models: LSTM, N-BEATS, and TFT."""

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.models.base import BaseForecaster

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------

class LSTMForecaster(BaseForecaster):
    """Pure PyTorch LSTM forecaster.

    Uses a sliding-window approach to create input/target sequence pairs
    from a univariate (or multivariate) time series and trains a stacked
    LSTM network with dropout regularisation.

    Parameters
    ----------
    hidden_size : int
        Number of features in the LSTM hidden state.  Default ``128``.
    num_layers : int
        Number of stacked LSTM layers.  Default ``2``.
    dropout : float
        Dropout probability applied between LSTM layers.  Default ``0.2``.
    seq_length : int
        Length of the input sliding window.  Default ``28``.
    batch_size : int
        Mini-batch size for training.  Default ``64``.
    epochs : int
        Maximum number of training epochs.  Default ``50``.
    lr : float
        Initial learning rate for Adam.  Default ``0.001``.
    val_fraction : float
        Fraction of the training data held out (from the tail) for
        validation / early stopping.  Default ``0.15``.
    patience : int
        Number of epochs without validation improvement before early
        stopping triggers.  Default ``10``.
    random_state : int
        Random seed.  Default ``42``.
    """

    name: str = "LSTM"

    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        seq_length: int = 28,
        batch_size: int = 64,
        epochs: int = 50,
        lr: float = 0.001,
        val_fraction: float = 0.15,
        patience: int = 10,
        random_state: int = 42,
    ) -> None:
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.val_fraction = val_fraction
        self.patience = patience
        self.random_state = random_state

        self._model = None
        self._device = None
        self._target_col: Optional[str] = None
        self._feature_cols: List[str] = []
        self._target_mean: float = 0.0
        self._target_std: float = 1.0

    # ----- inner network ------------------------------------------------------

    @staticmethod
    def _build_network(input_size, hidden_size, num_layers, dropout):
        """Return an ``_LSTMNetwork`` instance.

        Defined as a static factory so that the import of ``torch`` is
        deferred until the method is actually called.
        """
        import torch
        import torch.nn as nn

        class _LSTMNetwork(nn.Module):
            """Stacked LSTM followed by a fully-connected output layer."""

            def __init__(self, input_size, hidden_size, num_layers, dropout):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0.0,
                    batch_first=True,
                )
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x):
                # x: (batch, seq_length, input_size)
                lstm_out, _ = self.lstm(x)
                # Use the output of the last time step.
                last_hidden = lstm_out[:, -1, :]
                return self.fc(last_hidden).squeeze(-1)

        return _LSTMNetwork(input_size, hidden_size, num_layers, dropout)

    # ----- helpers ------------------------------------------------------------

    @staticmethod
    def _create_sequences(
        data: np.ndarray, seq_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding-window input/target pairs.

        Parameters
        ----------
        data : np.ndarray
            2-D array of shape ``(n_timesteps, n_features)``.  The first
            column is assumed to be the target.
        seq_length : int
            Number of past time steps in each input window.

        Returns
        -------
        X : np.ndarray
            Shape ``(n_samples, seq_length, n_features)``.
        y : np.ndarray
            Shape ``(n_samples,)`` -- the target value immediately after
            each window.
        """
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            xs.append(data[i : i + seq_length])
            ys.append(data[i + seq_length, 0])  # target is column 0
        return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

    # ----- interface ----------------------------------------------------------

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str,
        feature_cols: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> "LSTMForecaster":
        """Train the LSTM network on *train_df*.

        The target column is placed first, followed by any additional
        feature columns.  Data is normalised to zero mean / unit variance
        using training-set statistics.

        Parameters
        ----------
        train_df : DataFrame
            Training data, sorted chronologically.
        target_col : str
            Column to forecast.
        feature_cols : list[str], optional
            Additional feature columns.  Pass ``None`` or ``[]`` for a
            univariate forecast.
        **kwargs
            Not used; accepted for interface compatibility.
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self._target_col = target_col
        self._feature_cols = list(feature_cols) if feature_cols else []
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # -- assemble feature matrix (target as first column) ------------------
        cols = [target_col] + self._feature_cols
        raw = train_df[cols].values.astype(np.float32)

        # Normalise
        self._target_mean = float(raw[:, 0].mean())
        self._target_std = float(raw[:, 0].std()) or 1.0
        normalised = raw.copy()
        normalised[:, 0] = (normalised[:, 0] - self._target_mean) / self._target_std

        # -- create sequences --------------------------------------------------
        X_all, y_all = self._create_sequences(normalised, self.seq_length)

        if len(X_all) == 0:
            raise ValueError(
                f"Not enough data to create sequences.  Need at least "
                f"{self.seq_length + 1} rows, got {len(raw)}."
            )

        # -- train / val split (last portion) ----------------------------------
        val_size = max(1, int(len(X_all) * self.val_fraction))
        X_train = torch.tensor(X_all[:-val_size], device=self._device)
        y_train = torch.tensor(y_all[:-val_size], device=self._device)
        X_val = torch.tensor(X_all[-val_size:], device=self._device)
        y_val = torch.tensor(y_all[-val_size:], device=self._device)

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=self.batch_size,
            shuffle=True,
        )

        # -- build model -------------------------------------------------------
        input_size = X_all.shape[2]
        self._model = self._build_network(
            input_size, self.hidden_size, self.num_layers, self.dropout
        ).to(self._device)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=False
        )
        criterion = nn.MSELoss()

        # -- training loop with early stopping ---------------------------------
        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_state = None

        for epoch in range(1, self.epochs + 1):
            self._model.train()
            epoch_loss = 0.0
            n_batches = 0

            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                preds = self._model(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)

            # Validation
            self._model.eval()
            with torch.no_grad():
                val_preds = self._model(X_val)
                val_loss = criterion(val_preds, y_val).item()

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_state = {
                    k: v.clone() for k, v in self._model.state_dict().items()
                }
            else:
                epochs_no_improve += 1

            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    "LSTM epoch %d/%d  train_loss=%.6f  val_loss=%.6f  lr=%.2e",
                    epoch,
                    self.epochs,
                    avg_train_loss,
                    val_loss,
                    optimizer.param_groups[0]["lr"],
                )

            if epochs_no_improve >= self.patience:
                logger.info(
                    "Early stopping at epoch %d (patience=%d).",
                    epoch,
                    self.patience,
                )
                break

        # Restore best weights
        if best_state is not None:
            self._model.load_state_dict(best_state)

        logger.info(
            "LSTM fitted  best_val_loss=%.6f  device=%s",
            best_val_loss,
            self._device,
        )
        return self

    def predict(self, df: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        """Generate predictions for each row in *df*.

        The method uses a sliding window of length ``seq_length`` drawn
        from the end of *df* to produce one prediction per remaining time
        step.  If *df* has exactly ``seq_length`` rows the result is a
        single-element array.

        Parameters
        ----------
        df : DataFrame
            Must contain the target and feature columns used during
            :meth:`fit`.  The last ``seq_length`` rows are used as the
            initial context window.
        **kwargs
            ``horizon`` (int) -- number of steps to predict via
            autoregressive roll-out.  Default is
            ``len(df) - seq_length`` (i.e., one prediction per remaining
            row).

        Returns
        -------
        np.ndarray
            Predicted values (de-normalised to the original scale).
        """
        import torch

        if self._model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        self._model.eval()

        cols = [self._target_col] + self._feature_cols
        raw = df[cols].values.astype(np.float32)

        # Normalise target column
        normalised = raw.copy()
        normalised[:, 0] = (normalised[:, 0] - self._target_mean) / self._target_std

        horizon: int = kwargs.pop(
            "horizon", max(1, len(normalised) - self.seq_length)
        )

        predictions: List[float] = []
        window = normalised[: self.seq_length].copy()

        with torch.no_grad():
            for step in range(horizon):
                x = torch.tensor(
                    window[-self.seq_length :][np.newaxis, ...],
                    device=self._device,
                )
                pred_norm = self._model(x).item()
                pred_original = pred_norm * self._target_std + self._target_mean
                predictions.append(pred_original)

                # Roll the window forward
                next_idx = self.seq_length + step
                if next_idx < len(normalised):
                    new_row = normalised[next_idx].copy()
                else:
                    new_row = np.zeros(normalised.shape[1], dtype=np.float32)
                    new_row[0] = pred_norm
                window = np.vstack([window, new_row[np.newaxis, :]])

        return np.array(predictions, dtype=np.float64)

    def get_params(self) -> Dict[str, object]:
        return {
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "seq_length": self.seq_length,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "lr": self.lr,
            "val_fraction": self.val_fraction,
            "patience": self.patience,
            "random_state": self.random_state,
        }

    def __repr__(self) -> str:
        return (
            f"LSTMForecaster(hidden={self.hidden_size}, "
            f"layers={self.num_layers}, seq_len={self.seq_length})"
        )


# ---------------------------------------------------------------------------
# N-BEATS
# ---------------------------------------------------------------------------

class NBEATSForecaster(BaseForecaster):
    """N-BEATS forecaster via the ``neuralforecast`` library.

    Wraps :class:`neuralforecast.models.NBEATS` inside a
    :class:`neuralforecast.NeuralForecast` pipeline and exposes the
    standard ``fit`` / ``predict`` / ``get_params`` interface.

    Parameters
    ----------
    input_size : int
        Number of past observations fed to the model.  Default ``28``.
    h : int
        Forecast horizon.  Default ``28``.
    max_steps : int
        Maximum training steps.  Default ``1000``.
    learning_rate : float
        Initial learning rate.  Default ``0.001``.
    num_stacks : int
        Number of N-BEATS stacks.  Default ``10``.
    num_blocks : list[int]
        Blocks per stack.  Default ``[1, 1]``.
    random_state : int
        Random seed.  Default ``42``.
    """

    name: str = "NBEATS"

    def __init__(
        self,
        input_size: int = 28,
        h: int = 28,
        max_steps: int = 1000,
        learning_rate: float = 0.001,
        num_stacks: int = 10,
        num_blocks: Optional[List[int]] = None,
        random_state: int = 42,
    ) -> None:
        self.input_size = input_size
        self.h = h
        self.max_steps = max_steps
        self.learning_rate = learning_rate
        self.num_stacks = num_stacks
        self.num_blocks = num_blocks if num_blocks is not None else [1, 1]
        self.random_state = random_state

        self._nf = None
        self._target_col: Optional[str] = None
        self._feature_cols: List[str] = []
        self._device: Optional[str] = None

    # ----- helpers ------------------------------------------------------------

    @staticmethod
    def _prepare_nixtla_df(
        df: pd.DataFrame,
        target_col: str,
        date_col: str = "date",
        unique_id: str = "series_1",
    ) -> pd.DataFrame:
        """Convert a standard dataframe to nixtla long format.

        The resulting dataframe has columns ``unique_id``, ``ds``, and
        ``y`` as expected by :class:`neuralforecast.NeuralForecast`.
        """
        out = pd.DataFrame(
            {
                "unique_id": unique_id,
                "ds": pd.to_datetime(df[date_col]),
                "y": df[target_col].values.astype(np.float64),
            }
        )
        return out

    # ----- interface ----------------------------------------------------------

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str,
        feature_cols: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> "NBEATSForecaster":
        """Fit an N-BEATS model via ``neuralforecast``.

        Parameters
        ----------
        train_df : DataFrame
            Training data containing at least a date column and
            *target_col*.
        target_col : str
            Column to forecast.
        feature_cols : list[str], optional
            Not used directly by N-BEATS (univariate), but stored for
            compatibility.
        **kwargs
            ``date_col`` (str) -- name of the date column.
                Default ``"date"``.
            ``unique_id`` (str) -- series identifier.
                Default ``"series_1"``.
        """
        import torch
        from neuralforecast import NeuralForecast
        from neuralforecast.models import NBEATS

        torch.manual_seed(self.random_state)

        self._target_col = target_col
        self._feature_cols = list(feature_cols) if feature_cols else []
        self._device = "gpu" if torch.cuda.is_available() else "cpu"

        date_col: str = kwargs.pop("date_col", "date")
        unique_id: str = kwargs.pop("unique_id", "series_1")

        nixtla_df = self._prepare_nixtla_df(
            train_df, target_col, date_col=date_col, unique_id=unique_id
        )

        model = NBEATS(
            input_size=self.input_size,
            h=self.h,
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            num_stacks=self.num_stacks,
            num_blocks=self.num_blocks,
            random_seed=self.random_state,
        )

        self._nf = NeuralForecast(models=[model], freq="D")

        try:
            self._nf.fit(df=nixtla_df)
        except Exception as exc:
            logger.error("NBEATS fit failed: %s", exc)
            raise

        logger.info(
            "NBEATS fitted  input_size=%d  h=%d  max_steps=%d  device=%s",
            self.input_size,
            self.h,
            self.max_steps,
            self._device,
        )
        return self

    def predict(self, df: Optional[pd.DataFrame] = None, **kwargs: Any) -> np.ndarray:
        """Generate forecasts for the next ``h`` time steps.

        Parameters
        ----------
        df : DataFrame, optional
            Ignored for N-BEATS (the library forecasts from the end of
            the fitted data automatically).
        **kwargs
            Not used; accepted for interface compatibility.

        Returns
        -------
        np.ndarray
            Forecast values of length ``h``.
        """
        if self._nf is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        try:
            forecast_df = self._nf.predict()
        except Exception as exc:
            logger.error("NBEATS predict failed: %s", exc)
            raise

        # The result dataframe has columns like 'NBEATS'; grab the first
        # model column (excluding 'unique_id' and 'ds').
        value_cols = [
            c for c in forecast_df.columns if c not in ("unique_id", "ds")
        ]
        if not value_cols:
            raise RuntimeError(
                "neuralforecast predict returned no value columns."
            )

        return forecast_df[value_cols[0]].values.astype(np.float64)

    def get_params(self) -> Dict[str, object]:
        return {
            "input_size": self.input_size,
            "h": self.h,
            "max_steps": self.max_steps,
            "learning_rate": self.learning_rate,
            "num_stacks": self.num_stacks,
            "num_blocks": self.num_blocks,
            "random_state": self.random_state,
        }

    def __repr__(self) -> str:
        return (
            f"NBEATSForecaster(input_size={self.input_size}, "
            f"h={self.h}, max_steps={self.max_steps})"
        )


# ---------------------------------------------------------------------------
# Temporal Fusion Transformer
# ---------------------------------------------------------------------------

class TFTForecaster(BaseForecaster):
    """Temporal Fusion Transformer forecaster.

    Wraps :class:`pytorch_forecasting.TemporalFusionTransformer` and uses
    :class:`pytorch_forecasting.TimeSeriesDataSet` for data preparation.

    Parameters
    ----------
    hidden_size : int
        Hidden state size of the TFT.  Default ``64``.
    attention_head_size : int
        Number of attention heads.  Default ``4``.
    dropout : float
        Dropout rate.  Default ``0.1``.
    max_prediction_length : int
        Forecast horizon.  Default ``28``.
    max_encoder_length : int
        Maximum number of past observations provided to the encoder.
        Default ``60``.
    max_epochs : int
        Maximum number of training epochs.  Default ``50``.
    batch_size : int
        Mini-batch size.  Default ``128``.
    lr : float
        Learning rate.  Default ``0.001``.
    time_varying_known_reals : list[str], optional
        Known real-valued covariates (e.g., day-of-week encoded as
        float).  Default ``[]``.
    time_varying_unknown_reals : list[str], optional
        Unknown real-valued covariates (typically the target plus lags).
        Defaults to ``[target_col]`` during :meth:`fit`.
    static_categoricals : list[str], optional
        Static categorical features (e.g., store ID).  Default ``[]``.
    random_state : int
        Random seed.  Default ``42``.
    """

    name: str = "TFT"

    def __init__(
        self,
        hidden_size: int = 64,
        attention_head_size: int = 4,
        dropout: float = 0.1,
        max_prediction_length: int = 28,
        max_encoder_length: int = 60,
        max_epochs: int = 50,
        batch_size: int = 128,
        lr: float = 0.001,
        time_varying_known_reals: Optional[List[str]] = None,
        time_varying_unknown_reals: Optional[List[str]] = None,
        static_categoricals: Optional[List[str]] = None,
        random_state: int = 42,
    ) -> None:
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.dropout = dropout
        self.max_prediction_length = max_prediction_length
        self.max_encoder_length = max_encoder_length
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.time_varying_known_reals = (
            list(time_varying_known_reals) if time_varying_known_reals else []
        )
        self.time_varying_unknown_reals = (
            list(time_varying_unknown_reals) if time_varying_unknown_reals else []
        )
        self.static_categoricals = (
            list(static_categoricals) if static_categoricals else []
        )
        self.random_state = random_state

        self._model = None
        self._trainer = None
        self._training_dataset = None
        self._target_col: Optional[str] = None
        self._feature_cols: List[str] = []
        self._device: Optional[str] = None

    # ----- interface ----------------------------------------------------------

    def fit(
        self,
        train_df: pd.DataFrame,
        target_col: str,
        feature_cols: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> "TFTForecaster":
        """Fit a Temporal Fusion Transformer on *train_df*.

        The dataframe must contain:
        - A ``time_idx`` column (integer time index).
        - A ``group_id`` column (string/categorical series identifier).
        - The *target_col*.

        If these columns are missing they are synthesised automatically
        (single-series, sequential integer index).

        Parameters
        ----------
        train_df : DataFrame
            Training data sorted chronologically.
        target_col : str
            Column to forecast.
        feature_cols : list[str], optional
            Additional feature columns.
        **kwargs
            ``time_idx_col`` (str) -- name of the integer time-index
                column.  Default ``"time_idx"``.
            ``group_col`` (str) -- name of the group column.
                Default ``"group_id"``.
            ``val_fraction`` (float) -- fraction of data used for
                validation.  Default ``0.15``.
        """
        import torch
        import pytorch_lightning as pl
        from pytorch_forecasting import (
            TemporalFusionTransformer,
            TimeSeriesDataSet,
        )
        from pytorch_forecasting.metrics import QuantileLoss

        pl.seed_everything(self.random_state, workers=True)

        self._target_col = target_col
        self._feature_cols = list(feature_cols) if feature_cols else []
        self._device = "gpu" if torch.cuda.is_available() else "cpu"

        time_idx_col: str = kwargs.pop("time_idx_col", "time_idx")
        group_col: str = kwargs.pop("group_col", "group_id")
        val_fraction: float = kwargs.pop("val_fraction", 0.15)

        # -- ensure required columns exist -------------------------------------
        df = train_df.copy()
        if time_idx_col not in df.columns:
            df[time_idx_col] = np.arange(len(df))
        if group_col not in df.columns:
            df[group_col] = "series_0"

        # Make sure group column is a string type for pytorch_forecasting.
        df[group_col] = df[group_col].astype(str)

        # -- unknown reals: default to target if not specified -----------------
        unknown_reals = list(self.time_varying_unknown_reals)
        if not unknown_reals:
            unknown_reals = [target_col]
        elif target_col not in unknown_reals:
            unknown_reals = [target_col] + unknown_reals

        # -- train / val split -------------------------------------------------
        max_time_idx = int(df[time_idx_col].max())
        val_cutoff = max_time_idx - int(
            (max_time_idx - df[time_idx_col].min()) * val_fraction
        )

        training_cutoff = val_cutoff

        # -- TimeSeriesDataSet -------------------------------------------------
        try:
            self._training_dataset = TimeSeriesDataSet(
                df[df[time_idx_col] <= training_cutoff],
                time_idx=time_idx_col,
                target=target_col,
                group_ids=[group_col],
                max_encoder_length=self.max_encoder_length,
                max_prediction_length=self.max_prediction_length,
                time_varying_known_reals=self.time_varying_known_reals,
                time_varying_unknown_reals=unknown_reals,
                static_categoricals=self.static_categoricals,
                add_relative_time_idx=True,
                add_target_scales=True,
                add_encoder_length=True,
            )
        except Exception as exc:
            logger.error("Failed to create TimeSeriesDataSet: %s", exc)
            raise

        validation_dataset = TimeSeriesDataSet.from_dataset(
            self._training_dataset,
            df,
            predict=True,
            stop_randomization=True,
        )

        train_dataloader = self._training_dataset.to_dataloader(
            train=True, batch_size=self.batch_size, num_workers=0
        )
        val_dataloader = validation_dataset.to_dataloader(
            train=False, batch_size=self.batch_size, num_workers=0
        )

        # -- build TFT model ---------------------------------------------------
        self._model = TemporalFusionTransformer.from_dataset(
            self._training_dataset,
            hidden_size=self.hidden_size,
            attention_head_size=self.attention_head_size,
            dropout=self.dropout,
            learning_rate=self.lr,
            loss=QuantileLoss(),
            log_interval=10,
            reduce_on_plateau_patience=5,
        )

        logger.info(
            "TFT network has %d parameters.", self._model.size() if hasattr(self._model, 'size') else sum(
                p.numel() for p in self._model.parameters()
            )
        )

        # -- trainer -----------------------------------------------------------
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        self._trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator=accelerator,
            devices=1,
            gradient_clip_val=0.1,
            enable_model_summary=False,
            enable_progress_bar=True,
            callbacks=[
                pl.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=10,
                    mode="min",
                ),
            ],
        )

        try:
            self._trainer.fit(
                self._model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
            )
        except Exception as exc:
            logger.error("TFT training failed: %s", exc)
            raise

        # Load best checkpoint if available.
        best_path = self._trainer.checkpoint_callback
        if best_path is not None and hasattr(best_path, "best_model_path"):
            best_model_path = best_path.best_model_path
            if best_model_path:
                try:
                    self._model = TemporalFusionTransformer.load_from_checkpoint(
                        best_model_path
                    )
                except Exception:
                    logger.warning(
                        "Could not load best checkpoint; using last epoch weights."
                    )

        logger.info(
            "TFT fitted  epochs=%d  device=%s", self.max_epochs, self._device
        )
        return self

    def predict(self, df: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        """Generate predictions using the trained TFT.

        Parameters
        ----------
        df : DataFrame
            Must contain the same columns used during :meth:`fit`
            (including ``time_idx`` and ``group_id`` or their configured
            equivalents).
        **kwargs
            ``time_idx_col`` (str) -- Default ``"time_idx"``.
            ``group_col`` (str) -- Default ``"group_id"``.

        Returns
        -------
        np.ndarray
            Point predictions (median quantile).
        """
        from pytorch_forecasting import TimeSeriesDataSet

        if self._model is None or self._training_dataset is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        time_idx_col: str = kwargs.pop("time_idx_col", "time_idx")
        group_col: str = kwargs.pop("group_col", "group_id")

        df_pred = df.copy()
        if time_idx_col not in df_pred.columns:
            df_pred[time_idx_col] = np.arange(len(df_pred))
        if group_col not in df_pred.columns:
            df_pred[group_col] = "series_0"
        df_pred[group_col] = df_pred[group_col].astype(str)

        try:
            predict_dataset = TimeSeriesDataSet.from_dataset(
                self._training_dataset,
                df_pred,
                predict=True,
                stop_randomization=True,
            )
            predict_dataloader = predict_dataset.to_dataloader(
                train=False, batch_size=self.batch_size, num_workers=0
            )

            predictions = self._model.predict(
                predict_dataloader, return_x=False
            )
            return predictions.cpu().numpy().flatten().astype(np.float64)

        except Exception as exc:
            logger.error("TFT predict failed: %s", exc)
            raise

    def get_params(self) -> Dict[str, object]:
        return {
            "hidden_size": self.hidden_size,
            "attention_head_size": self.attention_head_size,
            "dropout": self.dropout,
            "max_prediction_length": self.max_prediction_length,
            "max_encoder_length": self.max_encoder_length,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "time_varying_known_reals": self.time_varying_known_reals,
            "time_varying_unknown_reals": self.time_varying_unknown_reals,
            "static_categoricals": self.static_categoricals,
            "random_state": self.random_state,
        }

    def __repr__(self) -> str:
        return (
            f"TFTForecaster(hidden={self.hidden_size}, "
            f"heads={self.attention_head_size}, "
            f"horizon={self.max_prediction_length})"
        )
