"""CabFlow FastAPI service.

Endpoints:
    GET  /health         - liveness + whether a model is loaded
    GET  /model-info     - current model name + hyperparameters
    GET  /zones          - list of pickup zones (id, name, borough)
    POST /predict        - per-zone hourly forecast

The model file path and config are read from ``config/config.yaml`` at
startup. Trained models live in ``models/`` and are produced by
``python -m src.pipeline.train_pipeline``.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.models.base import BaseForecaster
from src.utils.helpers import get_feature_columns, load_config

app = FastAPI(
    title="CabFlow API",
    description="NYC yellow-taxi hourly pickup demand forecasts.",
    version="1.0.0",
)

config = load_config()
MODEL_PATH = Path("models/xgboost.pkl")
model: BaseForecaster | None = None


def get_model() -> BaseForecaster:
    global model
    if model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="No trained model found. Run `python -m src.pipeline.train_pipeline` first.",
            )
        model = BaseForecaster.load(MODEL_PATH)
    return model


def _load_featured() -> pd.DataFrame:
    p = Path(config["data"]["processed_dir"]) / "trips_featured.parquet"
    if not p.exists():
        raise HTTPException(
            status_code=503,
            detail="Featured panel missing. Run preprocessing + feature engine first.",
        )
    return pd.read_parquet(p)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class PredictionRequest(BaseModel):
    zone_id: int
    horizon: int = 24


class PredictionResponse(BaseModel):
    zone_id: int
    zone_name: str | None
    borough: str | None
    hours: list[str]
    predictions: list[float]
    model_name: str
    inference_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str | None


class ModelInfoResponse(BaseModel):
    model_name: str
    parameters: dict
    feature_count: int


class ZoneInfo(BaseModel):
    zone_id: int
    zone_name: str
    borough: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health_check():
    m = model
    return HealthResponse(
        status="healthy",
        model_loaded=m is not None,
        model_name=m.name if m else None,
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    m = get_model()
    params = m.get_params()
    return ModelInfoResponse(
        model_name=m.name,
        parameters={k: str(v) for k, v in params.items()},
        feature_count=len(getattr(m, "_feature_cols", []) or []),
    )


@app.get("/zones", response_model=list[ZoneInfo])
def list_zones():
    df = _load_featured()
    zones = (
        df[["PULocationID", "zone_name", "Borough"]]
        .drop_duplicates()
        .sort_values(["Borough", "zone_name"])
    )
    return [
        ZoneInfo(zone_id=int(r.PULocationID), zone_name=r.zone_name, borough=r.Borough)
        for r in zones.itertuples()
    ]


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    m = get_model()
    start = time.time()

    df = _load_featured()
    zone_df = df[df["PULocationID"] == request.zone_id]
    if zone_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Zone {request.zone_id} not found. Use GET /zones to list valid ids.",
        )

    forecast_df = zone_df.sort_values("hour").tail(request.horizon).copy()
    feature_cols = get_feature_columns(forecast_df)

    preds = m.predict(forecast_df[feature_cols] if hasattr(m, "_feature_cols") else forecast_df)
    elapsed = (time.time() - start) * 1000

    zone_name = str(forecast_df["zone_name"].iloc[0]) if "zone_name" in forecast_df else None
    borough = str(forecast_df["Borough"].iloc[0]) if "Borough" in forecast_df else None
    hours = [pd.Timestamp(t).isoformat() for t in forecast_df["hour"]]

    return PredictionResponse(
        zone_id=request.zone_id,
        zone_name=zone_name,
        borough=borough,
        hours=hours,
        predictions=[round(float(max(p, 0)), 2) for p in preds[: request.horizon]],
        model_name=m.name,
        inference_time_ms=round(elapsed, 2),
    )


@app.on_event("startup")
async def startup():
    if MODEL_PATH.exists():
        get_model()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config["api"]["host"], port=config["api"]["port"])
