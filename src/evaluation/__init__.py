from .metrics import compute_all_metrics, mae, mase, rmse, smape, wrmsse
from .backtester import BacktestResult, TimeSeriesBacktester

# ForecastExplainer is intentionally NOT eagerly imported here: it pulls in
# `shap`, which is a research-only dependency (in requirements-dev.txt, not
# requirements-deploy.txt). Import it directly when you need it:
#     from src.evaluation.explainer import ForecastExplainer

__all__ = [
    "rmse",
    "mae",
    "mase",
    "smape",
    "wrmsse",
    "compute_all_metrics",
    "TimeSeriesBacktester",
    "BacktestResult",
]
