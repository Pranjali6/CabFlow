"""Data Quality Agent -- audits the TLC hourly demand panel via Claude."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataQualityAgent:
    """Inspects the hourly zone-level demand panel for problems."""

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
            logger.warning("Anthropic client unavailable (%s). Returning placeholder results.", exc)
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

    def analyze(self, df: pd.DataFrame) -> dict[str, Any]:
        quality_stats = self._compute_quality_stats(df)
        prompt = self._build_analysis_prompt(quality_stats, df.shape)
        raw_analysis = self._call_claude(prompt)
        recommendations = self._extract_recommendations(raw_analysis)
        return {
            "stats": quality_stats,
            "analysis": raw_analysis,
            "recommendations": recommendations,
        }

    def check_drift(self, train_df: pd.DataFrame, new_df: pd.DataFrame) -> dict[str, Any]:
        drift_stats = self._compute_drift_stats(train_df, new_df)
        prompt = self._build_drift_prompt(drift_stats)
        return {
            "drift_stats": drift_stats,
            "interpretation": self._call_claude(prompt),
        }

    def _compute_quality_stats(self, df: pd.DataFrame) -> dict[str, Any]:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        datetime_cols = df.select_dtypes(include="datetime").columns.tolist()

        missing = {
            c: {
                "count": int(df[c].isna().sum()),
                "pct": round(df[c].isna().sum() / max(len(df), 1) * 100, 2),
            }
            for c in df.columns
        }
        zero_rates = {
            c: round((df[c] == 0).mean() * 100, 2) for c in numeric_cols if len(df) > 0
        }

        outlier_counts = {}
        for c in numeric_cols:
            s = df[c].dropna()
            if s.empty:
                outlier_counts[c] = 0
                continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            outlier_counts[c] = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())

        coverage = {}
        if "PULocationID" in df.columns:
            coverage["unique_zones"] = int(df["PULocationID"].nunique())
        if "hour" in df.columns:
            ts = pd.to_datetime(df["hour"]).dropna().sort_values()
            if len(ts) >= 2:
                step = ts.diff().mode().iloc[0]
                gaps = ts.diff().dropna()
                coverage["expected_step_hours"] = float(step.total_seconds() / 3600)
                coverage["gap_count"] = int((gaps > step).sum())
                coverage["time_range"] = f"{ts.min()} -> {ts.max()}"

        return {
            "row_count": len(df),
            "column_count": len(df.columns),
            "datetime_columns": datetime_cols,
            "numeric_columns": numeric_cols[:15],
            "missing_values": {k: v for k, v in missing.items() if v["count"] > 0},
            "zero_rates_top": dict(
                sorted(zero_rates.items(), key=lambda kv: -kv[1])[:5]
            ),
            "outlier_counts_top": dict(
                sorted(outlier_counts.items(), key=lambda kv: -kv[1])[:5]
            ),
            "coverage": coverage,
            "duplicate_rows": int(df.duplicated().sum()),
        }

    def _compute_drift_stats(
        self, train_df: pd.DataFrame, new_df: pd.DataFrame
    ) -> dict[str, Any]:
        numeric_cols = train_df.select_dtypes(include="number").columns
        ks_results = {}
        for c in numeric_cols:
            if c not in new_df.columns:
                continue
            a, b = train_df[c].dropna(), new_df[c].dropna()
            if a.empty or b.empty:
                continue
            stat, pv = stats.ks_2samp(a, b)
            ks_results[c] = {"ks_statistic": round(stat, 4), "p_value": round(pv, 6)}
        return {"ks_test_results": ks_results}

    def _build_analysis_prompt(self, qstats: dict, shape: tuple) -> str:
        return (
            "You are a senior data analyst auditing a panel of hourly NYC yellow-taxi "
            f"pickup counts. The dataset has shape {shape} (rows x columns) with one "
            "row per (taxi_zone, hour). Quality statistics:\n\n"
            f"```json\n{json.dumps(qstats, indent=2, default=str)}\n```\n\n"
            "Provide a concise audit:\n"
            "1. **Coverage issues** -- missing zones, time gaps, duplicate rows.\n"
            "2. **Distributional red flags** -- unrealistic zero rates, outliers that "
            "look like data errors vs. legitimate demand spikes (e.g. airport hexes).\n"
            "3. **Severity** (high/medium/low) for each issue.\n"
            "4. **Recommended fixes** -- what to do before training.\n\n"
            "Be specific and operational."
        )

    def _build_drift_prompt(self, drift_stats: dict) -> str:
        return (
            "You are a senior ML engineer checking distribution drift between a "
            "training window and recent NYC taxi data. Drift statistics:\n\n"
            f"```json\n{json.dumps(drift_stats, indent=2, default=str)}\n```\n\n"
            "For each feature with significant drift:\n"
            "1. Likely cause (seasonality, regulation change, weather, special event).\n"
            "2. Impact on forecast accuracy.\n"
            "3. Recommendation: monitor, recalibrate, or retrain.\n\n"
            "Be concise and actionable."
        )

    @staticmethod
    def _extract_recommendations(text: str) -> list[str]:
        recs: list[str] = []
        for line in text.splitlines():
            s = line.strip()
            if s.startswith(("- ", "* ", ">> ")) or (
                len(s) > 3 and s[0].isdigit() and s[1] in (".", ")")
            ):
                recs.append(s.lstrip("-*> 0123456789.)").strip())
        return recs if recs else [text.strip()]
