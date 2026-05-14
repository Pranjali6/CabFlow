---
title: CabFlow
emoji: 🚕
colorFrom: yellow
colorTo: red
sdk: streamlit
sdk_version: 1.29.0
app_file: dashboard/app.py
pinned: false
license: mit
---

# CabFlow 🚕

**Real-time NYC Yellow Taxi Demand Forecasting**

Hourly pickup forecasts for 263 NYC taxi zones, built on the public NYC TLC trip-record dataset. Statistical, gradient-boosted, and deep-learning models combined via ensembling, served as a Streamlit dashboard + FastAPI endpoint, deployable end-to-end via Docker Compose / Kubernetes.

Forked from the [ForecastFlow](../ForecastFlow) M5 demand-forecasting project — same architecture, retargeted at urban ride-hailing demand.

---

## Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│  NYC TLC        │───▶│  Hourly      │───▶│  Model      │───▶│  Evaluation  │
│  Parquet files  │    │  zone panel  │    │  Training   │    │  & SHAP      │
└─────────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
                                                  │                   │
                                                  ▼                   ▼
                                           ┌─────────────┐    ┌──────────────┐
                                           │  MLflow     │    │  AI Agents   │
                                           │  Tracking   │    │  (Claude)    │
                                           └─────────────┘    └──────────────┘
                                                  │                   │
                                        ┌─────────┴───────────────────┘
                                        ▼
                                 ┌─────────────┐    ┌──────────────┐
                                 │  FastAPI    │    │  Streamlit   │
                                 │  /predict   │    │  Dashboard   │
                                 └─────────────┘    └──────────────┘
```

## What it forecasts

For every NYC taxi zone (1..263, ~258 active), predict the number of yellow-taxi pickups in the next 1..168 hours. Each zone is treated as one time series; the panel has 258 zones × hours-of-data rows. Target column: `pickup_count`.

## Models

| Family | Models | Approach |
|--------|--------|----------|
| **Statistical** | SARIMAX, Prophet, ETS | Hourly seasonality (24, 168) |
| **Machine Learning** | XGBoost, LightGBM | Gradient boosting with Optuna tuning |
| **Deep Learning** | LSTM, N-BEATS, TFT | Sequence modelling |
| **Ensemble** | Weighted, Stacking, Per-zone | Meta-learning combination |

## Features engineered

- **Lags**: 1h, 2h, 3h (very recent), 24h (yesterday), 48h, 168h (last week)
- **Rolling stats**: mean/std/min/max at 3, 6, 24, 168 hour windows
- **Calendar**: hour-of-day, day-of-week, is_weekend, is_rush_hour, is_late_night
- **Fourier**: daily (24) + weekly (168) seasonality, order 3
- **Zone metadata**: borough / service_zone label encodings + smoothed target encoding

## Quick Start

```bash
# 0. Activate / create a venv (or reuse ForecastFlow's)
source ../ForecastFlow/venv/bin/activate

# 1. Download TLC trip data + zone metadata (~50 MB per month)
python -m src.data.downloader

# 2. Aggregate to (zone, hour) panel
python -m src.data.preprocessor

# 3. Engineer features
python -m src.data.feature_engine

# 4. Train XGBoost / LightGBM / ensembles
python -m src.pipeline.train_pipeline

# 5. Launch dashboard
streamlit run dashboard/app.py

# 6. Launch API
uvicorn api.app:app --reload
```

## Live replay

The dashboard can simulate a real-time feed by advancing a clock through the historical panel. Use `src/data/replay.py` to construct a `ReplayClock` and feed only "visible" rows to the dashboard at a configurable speed (`speed_x=60` = one real minute advances one simulated hour).

## API

| Endpoint | Description |
|---|---|
| `GET /health` | Liveness + whether a model is loaded |
| `GET /model-info` | Model name + hyperparameters |
| `GET /zones` | List all zones with names and boroughs |
| `POST /predict` | `{zone_id, horizon}` → hourly pickup forecast |

Example:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"zone_id": 161, "horizon": 24}'
```

## Dashboard tabs

1. **City Overview** — total pickups, peak-hour heatmap, borough breakdown, top zones
2. **Zone Forecast** — pick a zone, see actual + predicted + 95% confidence band
3. **Zone Map** — choropleth map of NYC zones (requires geojson)
4. **Model Comparison** — XGBoost vs LightGBM (+ ensembles) on the same zone
5. **Data Explorer** — distribution, hour-of-day, weekend vs weekday
6. **Statistical Analysis** — ADF, ACF/PACF (lags 24/168 visible), STL decomposition
7. **Trends & Patterns** — DoW × hour heatmap, borough hourly profiles, MA crossovers
8. **Agent Insights** — Claude-powered data-quality audit, forecast insights, exec summary

## MLOps

- **Experiment tracking**: MLflow logs all hyperparameters, metrics, and per-model runs
- **Backtesting**: walk-forward (expanding-window) CV with `TimeSeriesBacktester`
- **Explainability**: SHAP TreeExplainer + STL decomposition for any zone
- **Metrics**: RMSE, MAE, MASE, SMAPE

## Deployment

### Docker Compose

```bash
make build     # build forecastflow/api and forecastflow/dashboard images
make up        # dashboard:8501, api:8000, mlflow:5000
make logs      # tail all three containers
make down
```

### Kubernetes

`k8s/` ships: namespace, ConfigMap, PVCs, deployments + services for API / Dashboard / MLflow, HPAs, Ingress, secret stub.

```bash
make lint       # client-side dry-run
make deploy     # kubectl apply -f k8s/
make undeploy
```

### CI

`.github/workflows/ci.yml` runs pytest, lints all manifests, and builds both Docker images on every push / PR to `main`.

## Tech Stack

**Core**: Python 3.11+, Pandas, NumPy, Scikit-learn
**ML**: XGBoost, LightGBM, Optuna
**Deep Learning**: PyTorch, PyTorch Forecasting, NeuralForecast
**Explainability**: SHAP
**MLOps**: MLflow
**Serving**: FastAPI, Streamlit, Plotly
**AI**: Claude API (Anthropic)
**Infra**: Docker, Kubernetes, GitHub Actions

## Data source

NYC TLC publishes monthly Parquet files of every yellow / green / FHV trip. Free, no API key, no rate limits.

- Trip records: <https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page>
- Direct CloudFront: `https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_YYYY-MM.parquet`
- Zone lookup: `https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv`

---

Built on top of [ForecastFlow](../ForecastFlow). Same code patterns, different domain.
