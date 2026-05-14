"""Agent Orchestrator -- coordinates the multi-agent workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.agents.data_quality_agent import DataQualityAgent
from src.agents.insight_agent import InsightAgent
from src.agents.report_agent import ReportAgent
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AgentOrchestrator:
    """Coordinates the multi-agent workflow.

    The orchestrator reads agent configuration (model name, max_tokens) from
    the project config file and wires up the three specialised agents:

    * :class:`DataQualityAgent` -- data quality analysis and drift detection
    * :class:`InsightAgent` -- forecast insight generation and anomaly explanation
    * :class:`ReportAgent` -- executive summary report generation
    """

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        self.config = self._load_config(config_path)

        agent_cfg = self.config.get("agents", {})
        model = agent_cfg.get("model", "claude-sonnet-4-6")
        max_tokens = agent_cfg.get("max_tokens", 4096)

        self.data_quality_agent = DataQualityAgent(model=model, max_tokens=max_tokens)
        self.insight_agent = InsightAgent(model=model, max_tokens=max_tokens)
        self.report_agent = ReportAgent(model=model, max_tokens=max_tokens)

        logger.info("AgentOrchestrator initialised (model=%s, max_tokens=%d).", model, max_tokens)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_full_analysis(
        self,
        df: pd.DataFrame,
        results_df: pd.DataFrame,
        shap_importance: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Run all agents in sequence: data quality -> insights -> report.

        Parameters
        ----------
        df : pd.DataFrame
            The raw / processed dataset used for training.
        results_df : pd.DataFrame
            Model comparison / evaluation results table.
        shap_importance : pd.DataFrame | None
            Optional SHAP feature-importance table (columns: feature, importance).

        Returns
        -------
        dict
            Combined outputs from every agent::

                {
                    "data_quality": { ... },  # DataQualityAgent.analyze() output
                    "insights": "...",         # InsightAgent.analyze_forecasts() output
                    "report": "...",           # ReportAgent.generate_report() output
                }
        """
        logger.info("Starting full multi-agent analysis pipeline.")

        # Step 1 -- Data Quality
        logger.info("[1/3] Running DataQualityAgent...")
        dq_output = self.run_data_quality(df)

        # Step 2 -- Insights
        logger.info("[2/3] Running InsightAgent...")
        insights_output = self.run_insights(results_df, shap_importance)

        # Step 3 -- Report
        logger.info("[3/3] Running ReportAgent...")
        report_output = self.run_report(
            results_df=results_df,
            shap_importance=shap_importance,
        )

        logger.info("Full analysis pipeline complete.")
        return {
            "data_quality": dq_output,
            "insights": insights_output,
            "report": report_output,
        }

    def run_data_quality(self, df: pd.DataFrame) -> dict[str, Any]:
        """Run only the data-quality agent."""
        return self.data_quality_agent.analyze(df)

    def run_insights(
        self,
        results_df: pd.DataFrame,
        shap_importance: pd.DataFrame | None = None,
    ) -> str:
        """Run only the insight agent."""
        return self.insight_agent.analyze_forecasts(results_df, shap_importance)

    def run_report(
        self,
        results_df: pd.DataFrame,
        shap_importance: pd.DataFrame | None = None,
        metrics: dict[str, Any] | None = None,
        forecast_summary: dict[str, Any] | None = None,
    ) -> str:
        """Run only the report agent.

        If ``metrics`` or ``forecast_summary`` are not supplied, they are
        derived automatically from ``results_df``.
        """
        if metrics is None:
            metrics = self._derive_metrics(results_df)
        if forecast_summary is None:
            forecast_summary = self._derive_forecast_summary(results_df)

        top_features: list[str] = []
        if shap_importance is not None and not shap_importance.empty:
            top_features = shap_importance.head(10)["feature"].tolist()

        return self.report_agent.generate_report(
            metrics=metrics,
            model_comparison=results_df,
            top_features=top_features,
            forecast_summary=forecast_summary,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_config(config_path: str) -> dict[str, Any]:
        """Load YAML configuration file."""
        path = Path(config_path)
        if not path.exists():
            logger.warning("Config file %s not found. Using defaults.", config_path)
            return {}
        with open(path) as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def _derive_metrics(results_df: pd.DataFrame) -> dict[str, Any]:
        """Best-effort extraction of metrics from a model-comparison dataframe."""
        metrics: dict[str, Any] = {}
        metric_cols = [c for c in results_df.columns if c.lower() not in ("model", "model_name", "name")]
        if metric_cols and len(results_df) > 0:
            best_idx = 0
            for col in metric_cols:
                if results_df[col].dtype.kind in ("f", "i"):
                    best_idx = int(results_df[col].idxmin())
                    break
            for col in metric_cols:
                val = results_df.iloc[best_idx][col]
                metrics[col] = round(float(val), 4) if results_df[col].dtype.kind in ("f", "i") else val
        return metrics

    @staticmethod
    def _derive_forecast_summary(results_df: pd.DataFrame) -> dict[str, Any]:
        """Build a minimal forecast summary from available data."""
        return {
            "n_models_evaluated": len(results_df),
            "metrics_tracked": [
                c for c in results_df.columns if c.lower() not in ("model", "model_name", "name")
            ],
        }
