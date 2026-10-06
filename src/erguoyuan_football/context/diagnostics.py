"""Compact diagnostics for data availability and context state quality."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from erguoyuan_football.context.schemas import ContextAssessment


def summarize_assessments(assessments: Iterable[ContextAssessment]) -> dict[str, object]:
    rows = tuple(assessments)
    return {
        "target_count": len(rows),
        "context_status": dict(Counter(item.context_status.value for item in rows)),
        "tournament_rules_verified": sum(item.tournament_state.rule_version is not None for item in rows),
        "lineup_rows": sum(len(item.lineup_evidence) for item in rows),
        "injury_rows": sum(len(item.injury_evidence) for item in rows),
        "adjustment_rows": sum(item.adjusted_probability.tournament_adjustment_applied
                                or item.adjusted_probability.lineup_adjustment_applied for item in rows),
        "no_adjustment_rows": sum(not item.ledger.adjustment_applied for item in rows),
        "uncertainty_distribution": {
            "min": min((item.feature_vector.context_uncertainty for item in rows), default=None),
            "mean": (sum(item.feature_vector.context_uncertainty for item in rows) / len(rows)
                     if rows else None),
            "max": max((item.feature_vector.context_uncertainty for item in rows), default=None),
        },
    }
