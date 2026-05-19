"""Per-ball outcome prediction and explanation."""

from __future__ import annotations

from src.models.monte_carlo import OUTCOMES, ball_probabilities


OUTCOME_LABELS = {
    0: "Dot ball",
    1: "Single",
    2: "Two runs",
    4: "Four",
    6: "Six",
    -1: "Wicket",
}


def predict_ball_outcome(pressure: float, batter_settle_score: float, bowler_fatigue: float) -> dict[str, object]:
    """Predict next-ball outcome probabilities with reason text."""

    probs = ball_probabilities(pressure, batter_settle_score, bowler_fatigue)
    distribution = {
        OUTCOME_LABELS[int(outcome)]: round(float(prob), 4)
        for outcome, prob in zip(OUTCOMES, probs, strict=False)
    }
    likely = max(distribution, key=distribution.get)
    reasons: list[str] = []
    if pressure >= 70:
        reasons.append("High pressure raises dot-ball and wicket probability.")
    if batter_settle_score >= 65:
        reasons.append("Set batter increases boundary probability.")
    if bowler_fatigue >= 65:
        reasons.append("Bowler fatigue lifts four/six probability.")
    if not reasons:
        reasons.append("Base T20 outcome rates dominate this ball.")
    return {"most_likely": likely, "distribution": distribution, "reasons": reasons}

