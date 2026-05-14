"""Generate forecasts for upcoming hours per zone."""

from pathlib import Path

import pandas as pd

from src.models.base import BaseForecaster
from src.utils.helpers import get_feature_columns, load_config
from src.utils.logger import get_logger

logger = get_logger("predict_pipeline")


def load_model(model_path: str) -> BaseForecaster:
    """Load a saved model from disk."""
    return BaseForecaster.load(model_path)


def run_prediction_pipeline(
    model_path: str = "models/xgboost.pkl",
    data_path: str | None = None,
    config_path: str = "config/config.yaml",
    output_path: str | None = None,
) -> pd.DataFrame:
    """Generate forecasts using a trained model."""
    config = load_config(config_path)

    logger.info(f"Loading model from {model_path}")
    model = load_model(model_path)
    logger.info(f"Model: {model.name}")

    if data_path is None:
        data_path = str(Path(config["data"]["processed_dir"]) / "trips_featured.parquet")

    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)

    feature_cols = get_feature_columns(df)
    target_col = config["data"]["target_col"]
    date_col = config["data"]["date_col"]
    horizon = config["data"]["forecast_horizon"]

    df = df.sort_values(["PULocationID", date_col]).reset_index(drop=True)
    max_hour = df[date_col].max()
    cutoff = max_hour - pd.Timedelta(hours=horizon)
    forecast_df = df[df[date_col] > cutoff].copy()

    logger.info(
        f"Generating predictions for {len(forecast_df):,} rows (horizon={horizon}h)"
    )
    predictions = model.predict(forecast_df)

    forecast_df = forecast_df.copy()
    forecast_df["prediction"] = predictions
    forecast_df["residual"] = forecast_df[target_col] - forecast_df["prediction"]

    result_cols = [
        "PULocationID",
        "Borough",
        "zone_name",
        date_col,
        target_col,
        "prediction",
        "residual",
    ]
    available = [c for c in result_cols if c in forecast_df.columns]
    result = forecast_df[available].copy()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")

    logger.info(
        f"Prediction stats: mean={predictions.mean():.2f}, std={predictions.std():.2f}"
    )
    return result


if __name__ == "__main__":
    result = run_prediction_pipeline(
        output_path="data/processed/predictions.parquet"
    )
    print(result.head(20))
