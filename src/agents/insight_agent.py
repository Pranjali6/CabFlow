"""Insight Agent -- turns forecast metrics + SHAP into operational insights."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class InsightAgent:
    """Analyses forecast results and SHAP values to produce operational insights."""

    def __init__(self, model: str = "claude-sonnet-4-6", max_tokens: int = 4096) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.client = self._init_client()

    @staticmethod
    def _init_client():
        try:
            import anthropic

            return anthropic.Anthropic()
        except Exception as exc:
            logger.warning("Anthropic client unavailable (%s).", exc)
            return None

    def _call_claude(self, prompt: str) -> str:
        if self.client is None:
            return "[Claude API unavailable -- set ANTHROPIC_API_KEY to enable analysis]"
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def analyze_forecasts(
        self,
        results_df: pd.DataFrame,
        shap_importance: pd.DataFrame | None = None,
    ) -> str:
        return self._call_claude(self._build_forecast_prompt(results_df, shap_importance))

    def explain_anomalies(self, anomaly_df: pd.DataFrame) -> str:
        return self._call_claude(self._build_anomaly_prompt(anomaly_df))

    def _build_forecast_prompt(
        self,
        results_df: pd.DataFrame,
        shap_importance: pd.DataFrame | None,
    ) -> str:
        model_table = (
            results_df.to_markdown(index=True)
            if hasattr(results_df, "to_markdown")
            else results_df.to_string()
        )
        shap_section = ""
        if shap_importance is not None and not shap_importance.empty:
            shap_table = (
                shap_importance.head(15).to_markdown(index=False)
                if hasattr(shap_importance, "to_markdown")
                else shap_importance.head(15).to_string()
            )
            shap_section = f"\n\n## Top SHAP Feature Importances\n\n{shap_table}\n"

        return (
            "You are a senior demand-forecasting analyst at a ride-hailing / taxi "
            "operations team. Below are hourly NYC yellow-taxi pickup forecast "
            "results.\n\n"
            "## Model Comparison\n\n"
            f"{model_table}\n"
            f"{shap_section}\n\n"
            "Please provide:\n"
            "1. **What drives demand** -- translate features (e.g. lag_24h, "
            "rolling_mean_168h, fourier_day_*, is_rush_hour, borough_target_enc) "
            "into operational language (recent past, weekly seasonality, daily "
            "rhythm, commute peaks, zone characteristics).\n"
            "2. **Which model is most reliable for fleet decisions** -- and why "
            "(accuracy vs. variance vs. tail risk).\n"
            "3. **Actionable recommendations** for fleet rebalancing, driver "
            "incentives, and surge pricing windows based on these forecasts.\n\n"
            "Keep it concise, business-oriented, and ops-grade."
        )

    @staticmethod
    def _build_anomaly_prompt(anomaly_df: pd.DataFrame) -> str:
        summary: dict[str, Any] = {
            "n_anomalies": len(anomaly_df),
            "columns": anomaly_df.columns.tolist(),
        }
        if "residual" in anomaly_df.columns:
            summary["mean_abs_residual"] = round(
                float(anomaly_df["residual"].abs().mean()), 4
            )
            summary["max_abs_residual"] = round(
                float(anomaly_df["residual"].abs().max()), 4
            )

        sample = anomaly_df.head(20)
        sample_table = (
            sample.to_markdown(index=False)
            if hasattr(sample, "to_markdown")
            else sample.to_string()
        )

        return (
            "You are a senior ride-hailing demand analyst. The rows below are hours "
            "where the forecast residual on NYC taxi pickup volume was unusually "
            "large.\n\n"
            f"**Summary:** {json.dumps(summary, default=str)}\n\n"
            "**Sample anomaly rows:**\n\n"
            f"{sample_table}\n\n"
            "Provide a root-cause analysis:\n"
            "1. Likely causes per cluster of anomalies (weather, events, transit "
            "outages, road closures, sports/concerts at MSG/Yankee Stadium, "
            "airport delays, NYE/holidays).\n"
            "2. Whether each is a genuine demand spike or a data artefact.\n"
            "3. How to improve the model: add features, hold out events, blend in "
            "weather, increase regularisation in high-variance zones.\n\n"
            "Be concise and operational."
        )
