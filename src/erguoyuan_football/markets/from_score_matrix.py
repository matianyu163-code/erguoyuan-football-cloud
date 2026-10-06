"""All projections share the exact support and tail policy of ScoreMatrix."""

from __future__ import annotations

from typing import Any

from erguoyuan_football.models.score_matrix import ScoreMatrix


def derive_markets(matrix: ScoreMatrix) -> dict[str, Any]:
    """Return 1X2, double chance, BTTS, half-goal totals, totals and exact scores."""
    outcome = matrix.outcome()
    totals = {goals: 0.0 for goals in range(2 * matrix.max_goals + 1)}
    exact: dict[str, float] = {}
    btts = 0.0
    for home, row in enumerate(matrix.values):
        for away, probability in enumerate(row):
            totals[home + away] += probability
            exact[f"{home}-{away}"] = probability
            if home > 0 and away > 0:
                btts += probability
    grouped_totals = {str(goals): totals.get(goals, 0.0) for goals in range(7)}
    grouped_totals["7+"] = sum(probability for goals, probability in totals.items() if goals >= 7)
    over_under = {}
    for line in (0.5, 1.5, 2.5, 3.5):
        under = sum(p for goals, p in totals.items() if goals < line)
        over_under[str(line)] = {"UNDER": under, "OVER": 1 - under}
    return {"1X2": outcome.model_dump(), "DOUBLE_CHANCE": {
        "1X": outcome.p_home + outcome.p_draw, "X2": outcome.p_draw + outcome.p_away,
        "12": outcome.p_home + outcome.p_away}, "BTTS": {"YES": btts, "NO": 1 - btts},
        "OVER_UNDER": over_under, "TOTAL_GOALS": totals, "EXACT_SCORE": exact,
        "TOTAL_GOALS_0_TO_7_PLUS": grouped_totals,
        "tail_policy": matrix.tail_policy, "tail_mass": matrix.tail_mass}
