"""Report Agent -- writes the executive summary for taxi-demand stakeholders."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ReportAgent:
    """Generates executive summary reports from forecast results."""

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

    def generate_report(
        self,
        metrics: dict[str, Any],
        model_comparison: pd.DataFrame,
        top_features: list[str],
        forecast_summary: dict[str, Any],
    ) -> str:
        return self._call_claude(
            self._build_report_prompt(
                metrics, model_comparison, top_features, forecast_summary
            )
        )

    def _build_report_prompt(
        self,
        metrics: dict[str, Any],
        model_comparison: pd.DataFrame,
        top_features: list[str],
        forecast_summary: dict[str, Any],
    ) -> str:
        comparison_table = (
            model_comparison.to_markdown(index=True)
            if hasattr(model_comparison, "to_markdown")
            else model_comparison.to_string()
        )

        return (
            "You are a senior analytics consultant briefing the operations leadership "
            "team at a ride-hailing / taxi company. Using the data below, produce a "
            "polished markdown report on NYC yellow-taxi hourly demand forecasting.\n\n"
            "---\n\n"
            "## Input Data\n\n"
            "### Best Model Metrics (hourly pickup forecast)\n\n"
            f"```json\n{json.dumps(metrics, indent=2, default=str)}\n```\n\n"
            "### Model Comparison\n\n"
            f"{comparison_table}\n\n"
            "### Top Demand Drivers (features)\n\n"
            + "\n".join(f"- {f}" for f in top_features)
            + "\n\n"
            "### Forecast Summary\n\n"
            f"```json\n{json.dumps(forecast_summary, indent=2, default=str)}\n```\n\n"
            "---\n\n"
            "Generate the report with these **exact** sections:\n\n"
            "1. **Executive Summary** -- 3-4 sentence high-level overview for the "
            "VP of Operations.\n"
            "2. **Model Performance** -- compare candidates, declare a winner, "
            "explain accuracy vs. tail-risk trade-offs.\n"
            "3. **Key Demand Drivers** -- translate feature names into ops language "
            "(time-of-day rhythm, weekly seasonality, zone effects, recent demand).\n"
            "4. **Recommendations** -- 3-5 concrete prioritised actions: fleet "
            "rebalancing windows, driver incentives, surge pricing tuning, airport "
            "hex strategies.\n"
            "5. **Risk Factors** -- weather sensitivity, single-month training "
            "bias, drift risk, model-zone mismatch, holidays.\n\n"
            "Professional but accessible tone. Bullets and tables where helpful."
        )
