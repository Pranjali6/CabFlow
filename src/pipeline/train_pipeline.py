"""End-to-end training pipeline for CabFlow.

Steps:
    1. Load (or build) the hourly per-zone featured panel.
    2. Optionally sample N zones for faster experimentation.
    3. Walk-forward split: last ``test_size`` hours per zone are held out.
    4. Train XGBoost, LightGBM, WeightedEnsemble, StackingEnsemble.
    5. Log everything to MLflow and save the best models to disk.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from src.data.feature_engine import build_features
from src.data.preprocessor import preprocess_pipeline
from src.evaluation.metrics import compute_all_metrics
from src.models.ensemble import StackingEnsemble, WeightedEnsemble
from src.models.ml_models import CatBoostForecaster, LightGBMForecaster, XGBoostForecaster
from src.utils.helpers import get_feature_columns, load_config, set_seed
from src.utils.logger import get_logger

logger = get_logger("train_pipeline")


def run_training_pipeline(
    config_path: str = "config/config.yaml",
    n_zones: int | None = None,
):
    """Train all models on the hourly TLC panel.

    Parameters
    ----------
    config_path
        Path to YAML config.
    n_zones
        If set, randomly sample this many zones for faster iteration.
        ``None`` uses all zones present in the data.
    """
    config = load_config(config_path)
    set_seed(config["project"]["seed"])

    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"]
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    processed_dir = Path(config["data"]["processed_dir"])
    featured_path = processed_dir / "trips_featured.parquet"
    hourly_path = processed_dir / "trips_hourly.parquet"

    if featured_path.exists():
        logger.info(f"Loading featured panel from {featured_path}")
        df = pd.read_parquet(featured_path)
    elif hourly_path.exists():
        logger.info(f"Loading hourly panel from {hourly_path}, building features...")
        df = pd.read_parquet(hourly_path)
        df = build_features(df, config)
        df.to_parquet(featured_path, index=False)
    else:
        logger.info("No processed data found, running full preprocessing + features...")
        df = preprocess_pipeline(config, save=True)
        df = build_features(df, config)
        df.to_parquet(featured_path, index=False)

    if n_zones is not None:
        all_zones = df["PULocationID"].unique()
        sampled = np.random.choice(
            all_zones, size=min(n_zones, len(all_zones)), replace=False
        )
        df = df[df["PULocationID"].isin(sampled)].reset_index(drop=True)
        logger.info(f"Sampled {len(sampled)} zones for training")

    logger.info(f"Data shape: {df.shape}")

    feature_cols = get_feature_columns(df)
    target_col = config["data"]["target_col"]
    date_col = config["data"]["date_col"]
    test_size = config["evaluation"]["test_size"]

    logger.info(
        f"Features: {len(feature_cols)}  target: {target_col}  test_size: {test_size}h"
    )

    df = df.sort_values(["PULocationID", date_col]).reset_index(drop=True)
    max_hour = df[date_col].max()
    cutoff = max_hour - pd.Timedelta(hours=test_size)
    train_df = df[df[date_col] <= cutoff].copy()
    test_df = df[df[date_col] > cutoff].copy()

    logger.info(f"Train: {len(train_df):,} rows  Test: {len(test_df):,} rows")

    results = {}

    with mlflow.start_run(run_name="xgboost"):
        logger.info("Training XGBoost...")
        xgb = XGBoostForecaster()
        xgb.fit(train_df, target_col, feature_cols)
        xgb_preds = xgb.predict(test_df)
        xgb_metrics = compute_all_metrics(test_df[target_col].values, xgb_preds)
        results["XGBoost"] = xgb_metrics
        mlflow.log_params(xgb.get_params())
        mlflow.log_metrics(xgb_metrics)
        logger.info(f"XGBoost metrics: {xgb_metrics}")

    with mlflow.start_run(run_name="lightgbm"):
        logger.info("Training LightGBM...")
        lgb = LightGBMForecaster()
        lgb.fit(train_df, target_col, feature_cols)
        lgb_preds = lgb.predict(test_df)
        lgb_metrics = compute_all_metrics(test_df[target_col].values, lgb_preds)
        results["LightGBM"] = lgb_metrics
        mlflow.log_params(lgb.get_params())
        mlflow.log_metrics(lgb_metrics)
        logger.info(f"LightGBM metrics: {lgb_metrics}")

    with mlflow.start_run(run_name="catboost"):
        logger.info("Training CatBoost...")
        cat = CatBoostForecaster()
        cat.fit(train_df, target_col, feature_cols)
        cat_preds = cat.predict(test_df)
        cat_metrics = compute_all_metrics(test_df[target_col].values, cat_preds)
        results["CatBoost"] = cat_metrics
        mlflow.log_params(cat.get_params())
        mlflow.log_metrics(cat_metrics)
        logger.info(f"CatBoost metrics: {cat_metrics}")

    with mlflow.start_run(run_name="weighted_ensemble"):
        logger.info("Training Weighted Ensemble...")
        ens = WeightedEnsemble(
            [
                ("xgboost", XGBoostForecaster()),
                ("lightgbm", LightGBMForecaster()),
                ("catboost", CatBoostForecaster()),
            ]
        )
        ens.fit(train_df, target_col, feature_cols)
        ens_preds = ens.predict(test_df)
        ens_metrics = compute_all_metrics(test_df[target_col].values, ens_preds)
        results["WeightedEnsemble"] = ens_metrics
        mlflow.log_metrics(ens_metrics)
        weights_df = ens.get_model_weights()
        logger.info(f"Ensemble weights:\n{weights_df}")
        logger.info(f"Ensemble metrics: {ens_metrics}")

    with mlflow.start_run(run_name="stacking_ensemble"):
        logger.info("Training Stacking Ensemble...")
        stack = StackingEnsemble(
            [
                ("xgboost", XGBoostForecaster()),
                ("lightgbm", LightGBMForecaster()),
                ("catboost", CatBoostForecaster()),
            ],
            meta_learner="ridge",
        )
        stack.fit(train_df.reset_index(drop=True), target_col, feature_cols)
        stack_preds = stack.predict(test_df)
        stack_metrics = compute_all_metrics(test_df[target_col].values, stack_preds)
        results["StackingEnsemble"] = stack_metrics
        mlflow.log_metrics(stack_metrics)
        logger.info(f"Stacking metrics: {stack_metrics}")

    logger.info("\n=== Final Results ===")
    results_df = pd.DataFrame(results).T.round(4)
    logger.info(f"\n{results_df.to_string()}")

    best_model = results_df["rmse"].idxmin()
    logger.info(
        f"\nBest model by RMSE: {best_model} ({results_df.loc[best_model, 'rmse']:.4f})"
    )

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    xgb.save(models_dir / "xgboost.pkl")
    lgb.save(models_dir / "lightgbm.pkl")
    cat.save(models_dir / "catboost.pkl")
    logger.info(f"Models saved to {models_dir}")

    return results_df


if __name__ == "__main__":
    run_training_pipeline()
