"""Mathematical play heads from one reconciled score matrix."""

from __future__ import annotations

import math
from dataclasses import dataclass

from erguoyuan_football.models.score_matrix import ScoreMatrix


@dataclass(frozen=True)
class PlayTop2:
    selections: tuple[str, str]
    probabilities: tuple[float, float]
    coverage: float


def score_top2(matrix: ScoreMatrix) -> PlayTop2:
    """Select the two largest score cells with stable tie breaks."""
    cells = [(f"{home}-{away}", float(p)) for home, row in enumerate(matrix.values)
             for away, p in enumerate(row)]
    top = sorted(cells, key=lambda item: (-item[1], item[0]))[:2]
    return PlayTop2((top[0][0], top[1][0]), (top[0][1], top[1][1]), top[0][1] + top[1][1])


def total_goals(matrix: ScoreMatrix) -> dict[str, float]:
    """Aggregate 0..6 and 7+ from the same final score matrix."""
    totals = {str(index): 0.0 for index in range(7)}
    totals["7+"] = 0.0
    for home, row in enumerate(matrix.values):
        for away, p in enumerate(row):
            key = str(home + away) if home + away < 7 else "7+"
            totals[key] += float(p)
    if not math.isclose(math.fsum(totals.values()), 1, abs_tol=1e-8):
        raise ValueError("TOTAL_GOALS_MASS_INVALID")
    return totals


def total_top2(matrix: ScoreMatrix) -> PlayTop2:
    totals = total_goals(matrix)
    top = sorted(totals.items(), key=lambda item: (-item[1], item[0]))[:2]
    return PlayTop2((top[0][0], top[1][0]), (top[0][1], top[1][1]), top[0][1] + top[1][1])


def official_handicap_1x2(matrix: ScoreMatrix, home_handicap: int | None, *,
                          verified_source: bool) -> dict[str, float]:
    """Apply an actual integer official handicap from the home-team perspective."""
    if home_handicap is None or not verified_source or isinstance(home_handicap, bool):
        raise ValueError("OFFICIAL_HANDICAP_UNAVAILABLE")
    if not isinstance(home_handicap, int):
        raise TypeError("OFFICIAL_HANDICAP_REQUIRES_INTEGER_LINE")
    outcome = {"HOME": 0.0, "DRAW": 0.0, "AWAY": 0.0}
    for home, row in enumerate(matrix.values):
        for away, p in enumerate(row):
            delta = home + home_handicap - away
            outcome["HOME" if delta > 0 else "DRAW" if delta == 0 else "AWAY"] += float(p)
    return outcome


def htft_top2(distribution: dict[str, float] | None, *, independent_oos_model: bool) -> PlayTop2:
    """Require nine outcomes from an independently trained half/full-time model."""
    keys = {a + b for a in "HDA" for b in "HDA"}
    if not independent_oos_model or distribution is None:
        raise ValueError("HTFT_UNAVAILABLE:INDEPENDENT_MODEL_REQUIRED")
    if set(distribution) != keys or any(not math.isfinite(p) or p < 0 for p in distribution.values()):
        raise ValueError("HTFT_DISTRIBUTION_INVALID")
    if not math.isclose(math.fsum(distribution.values()), 1, abs_tol=1e-8):
        raise ValueError("HTFT_DISTRIBUTION_INVALID")
    top = sorted(distribution.items(), key=lambda item: (-item[1], item[0]))[:2]
    return PlayTop2((top[0][0], top[1][0]), (top[0][1], top[1][1]), top[0][1] + top[1][1])
