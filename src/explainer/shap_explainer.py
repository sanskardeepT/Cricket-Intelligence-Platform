"""SHAP and fallback feature-attribution explanations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Reason:
    """A single additive prediction reason."""

    feature: str
    contribution: float
    text: str


def fallback_reasons(features: dict[str, float], probability: float) -> list[Reason]:
    """Generate deterministic scientific reasons when SHAP artifacts are unavailable."""

    reasons: list[Reason] = []
    if features.get("elo_prior", 0.5) >= 0.55:
        reasons.append(Reason("elo_prior", 0.08, "Team strength prior is above opponent baseline."))
    if features.get("current_run_rate", 0.0) >= features.get("required_run_rate", 99.0):
        reasons.append(Reason("run_rate_edge", 0.10, "Current run rate is meeting or beating the chase rate."))
    if features.get("pressure", 50.0) >= 70:
        reasons.append(Reason("pressure", -0.09, "Pressure index is high because wickets or required rate are rising."))
    if features.get("balls_left", 0.0) <= 30 and probability >= 0.65:
        reasons.append(Reason("death_over_convergence", 0.07, "Late-innings simulations converge toward this result."))
    if not reasons:
        reasons.append(Reason("balanced_state", 0.0, "Match state is balanced; no single factor dominates."))
    return reasons


def explain_with_shap(model, features: pd.DataFrame) -> list[Reason]:
    """Return SHAP feature contributions for a fitted tree model."""

    if features.empty:
        raise ValueError("features cannot be empty")
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("shap is not installed") from exc
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(features)
    row_values = values[1][0] if isinstance(values, list) else values[0]
    reasons: list[Reason] = []
    for feature, contribution in zip(features.columns, row_values, strict=False):
        reasons.append(
            Reason(
                feature=str(feature),
                contribution=round(float(contribution), 4),
                text=f"{feature} contributed {float(contribution):+.3f} log-odds.",
            )
        )
    return sorted(reasons, key=lambda reason: abs(reason.contribution), reverse=True)


def net_explanation(probability: float, reasons: list[Reason]) -> dict[str, object]:
    """Format final explanation payload."""

    baseline = 0.5
    net = probability - baseline
    return {
        "baseline": baseline,
        "probability": probability,
        "net_change": round(net, 4),
        "reasons": [reason.__dict__ for reason in reasons],
        "summary": f"Baseline 50% moved {net:+.1%} to {probability:.1%}.",
    }

