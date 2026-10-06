"""Fixed R3 play derivations from one executed score matrix."""

from __future__ import annotations

import math
from typing import Any

from erguoyuan_football.models.score_matrix import ScoreMatrix
from erguoyuan_football.prediction_heads.heads import (
    score_top2,
    total_goals,
    total_top2,
)

DERIVATION_VERSION = "R3_SCORE_MATRIX_DERIVATION_V1"


def _top2(values: dict[str, float]) -> list[dict[str, Any]]:
    return [{"selection": name, "probability": probability} for name, probability in
            sorted(values.items(), key=lambda item: (-item[1], item[0]))[:2]]


def standardize(raw: dict[str, Any], home_handicap: int) -> dict[str, Any]:
    """Derive plays from model output; unavailable heads remain explicit."""
    if raw.get("execution_status") != "SUCCESS":
        raise ValueError("R3_MODEL_NOT_SUCCESSFUL")
    if isinstance(home_handicap, bool) or not isinstance(home_handicap, int):
        raise TypeError("R3_INTEGER_HANDICAP_REQUIRED")
    metadata = raw.get("metadata")
    if not isinstance(metadata, dict) or "score_matrix" not in metadata:
        raise ValueError("R3_SCORE_MATRIX_METADATA_MISSING")
    matrix = ScoreMatrix.model_validate(metadata["score_matrix"])
    outcome = matrix.outcome()
    expected = (raw.get("p_home"), raw.get("p_draw"), raw.get("p_away"))
    actual = (outcome.p_home, outcome.p_draw, outcome.p_away)
    if any(not isinstance(value, (float, int)) or not math.isclose(
            float(value), result, abs_tol=1e-8) for value, result in zip(expected, actual)):
        raise ValueError("R3_OUTCOME_MATRIX_MISMATCH")
    one_x_two = {"HOME": outcome.p_home, "DRAW": outcome.p_draw,
                 "AWAY": outcome.p_away}
    handicap = {"HOME": 0.0, "DRAW": 0.0, "AWAY": 0.0}
    for home, row in enumerate(matrix.values):
        for away, probability in enumerate(row):
            delta = home + home_handicap - away
            handicap["HOME" if delta > 0 else "DRAW" if delta == 0 else "AWAY"] += probability
    totals = total_goals(matrix)
    total_best = total_top2(matrix)
    score_best = score_top2(matrix)
    return {
        "derivation_version": DERIVATION_VERSION,
        "status": "AVAILABLE_PARTIAL",
        "one_x_two": {"probabilities": one_x_two, "top2": _top2(one_x_two)},
        "handicap_one_x_two": {"official_handicap": home_handicap,
            "handicap_source": "USER_SCREENSHOT", "market_verified": False,
            "probabilities": handicap, "top2": _top2(handicap)},
        "total_goals": {"probabilities": totals,
            "top2": [{"selection": name, "probability": probability} for name, probability
                     in zip(total_best.selections, total_best.probabilities)]},
        "half_full_time": {"status": "UNAVAILABLE",
            "reason": "INDEPENDENT_HTFT_MODEL_NOT_FROZEN"},
        "exact_score": {"top2": [{"selection": name, "probability": probability}
            for name, probability in zip(score_best.selections, score_best.probabilities)]},
        "score_matrix": {"max_goals": matrix.max_goals,
            "retained_mass": matrix.retained_mass, "tail_mass": matrix.tail_mass,
            "tail_policy": matrix.tail_policy},
        "daily_combinations": {"status": "UNAVAILABLE",
            "reason": "R3_COMBINATION_SELECTION_RULE_NOT_FROZEN"},
        "reference_plans_400_100_20": {"status": "UNAVAILABLE",
            "reason": "R3_PLAN_RULE_AND_VERIFIED_MARKET_UNAVAILABLE"},
        "market_verified": False,
    }
