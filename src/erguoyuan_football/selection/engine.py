"""Deterministic highest-hit selections from separate matches and play heads."""

from __future__ import annotations

import json
from hashlib import sha256
from itertools import combinations

from erguoyuan_football.selection.schemas import (
    CandidateSlot,
    MatchHeads,
    RankedCandidate,
)

HIGHEST_HIT_ORDER = ("MATCH_1X2_PAIR", "OFFICIAL_HANDICAP_PAIR", "TOTAL_GOALS_TOP2",
                     "HTFT_TOP2", "EXACT_SCORE_TOP2_PAIR")
ACCOUNT_400_ORDER = ("MATCH_1X2_PAIR", "HTFT_TOP2", "TOTAL_GOALS_TOP2",
                     "OFFICIAL_HANDICAP_PAIR")


def _candidate(play_type: str, matches: tuple[MatchHeads, ...], selections: tuple[str, ...],
               component: tuple[float, ...], *, coverage: float | None = None,
               rank: int = 1, handicap_lines: tuple[int, ...] = ()) -> RankedCandidate:
    probability = 1.0
    for value in component:
        probability *= value
    actual_coverage = probability if coverage is None else coverage
    identity = sha256(json.dumps([play_type, [m.match_id for m in matches], selections, handicap_lines],
                                 sort_keys=True).encode()).hexdigest()[:24]
    return RankedCandidate(candidate_id=identity,
        prediction_snapshot_ids=tuple(m.prediction_snapshot_id for m in matches),
        match_ids=tuple(m.match_id for m in matches), objective="HIGHEST_HIT",
        play_type=play_type, selections=selections,
        component_probabilities=component, joint_probability=actual_coverage,
        coverage_probability=actual_coverage,
        probability_method="INDEPENDENCE_ASSUMPTION" if len(matches) > 1 else "EXACT_MARGINAL_SUM",
        data_quality="REAL_OOS_DEVELOPMENT", context_quality="EVALUATED_OR_UNAVAILABLE",
        correlation_status="INDEPENDENCE_ASSUMPTION" if len(matches) > 1 else "NOT_APPLICABLE",
        rank=rank, validation_status=matches[0].validation_status,
        production_status=matches[0].production_status,
        reason_codes=("NO_VALIDATED_CROSS_MATCH_CORRELATION",) if len(matches) > 1 else (),
        official_handicap_lines=handicap_lines)


class SelectionEngine:
    """Rank every supported fixed slot by hit probability, never by EV."""

    def highest_hit(self, matches: tuple[MatchHeads, ...]) -> tuple[CandidateSlot, ...]:
        slots: list[CandidateSlot] = []
        for play_type in HIGHEST_HIT_ORDER:
            options = self._options(play_type, matches)
            if not options:
                reason = ("HTFT_INDEPENDENT_MODEL_UNAVAILABLE" if play_type == "HTFT_TOP2" else
                          "OFFICIAL_HANDICAP_UNAVAILABLE" if play_type == "OFFICIAL_HANDICAP_PAIR" else
                          "SCORE_MATRIX_UNAVAILABLE" if play_type in {"TOTAL_GOALS_TOP2", "EXACT_SCORE_TOP2_PAIR"}
                          else "INSUFFICIENT_DISTINCT_MATCHES")
                slots.append(CandidateSlot(play_type=play_type, status="UNAVAILABLE",
                                           candidate=None, reason=reason))
                continue
            best = min(options, key=lambda c: (-c.joint_probability, c.candidate_id))
            slots.append(CandidateSlot(play_type=play_type, status="AVAILABLE", candidate=best))
        return tuple(slots)

    def _options(self, play_type: str, matches: tuple[MatchHeads, ...]) -> list[RankedCandidate]:
        result: list[RankedCandidate] = []
        if play_type in {"MATCH_1X2_PAIR", "OFFICIAL_HANDICAP_PAIR"}:
            for a, b in combinations(matches, 2):
                if a.match_id == b.match_id:
                    continue
                probabilities = (a.probability, b.probability) if play_type == "MATCH_1X2_PAIR" else (
                    a.handicap_probabilities, b.handicap_probabilities)
                if any(item is None for item in probabilities):
                    continue
                def best(item):
                    if isinstance(item, dict):
                        return max(item.items(), key=lambda pair: (pair[1], pair[0]))
                    return max((("HOME", item.p_home), ("DRAW", item.p_draw), ("AWAY", item.p_away)),
                               key=lambda pair: (pair[1], pair[0]))
                selected = (best(probabilities[0]), best(probabilities[1]))
                lines: tuple[int, ...] = ()
                if play_type == "OFFICIAL_HANDICAP_PAIR":
                    if a.official_home_handicap is None or b.official_home_handicap is None:
                        continue
                    lines = (a.official_home_handicap, b.official_home_handicap)
                result.append(_candidate(play_type, (a, b),
                    (selected[0][0], selected[1][0]), (selected[0][1], selected[1][1]),
                    handicap_lines=lines))
        elif play_type in {"TOTAL_GOALS_TOP2", "HTFT_TOP2"}:
            for match in matches:
                head = match.totals_top2 if play_type == "TOTAL_GOALS_TOP2" else match.htft_top2
                if head is not None:
                    result.append(_candidate(play_type, (match,), head.selections,
                                             head.probabilities, coverage=head.coverage))
        elif play_type == "EXACT_SCORE_TOP2_PAIR":
            for a, b in combinations(matches, 2):
                if a.match_id == b.match_id or a.score_top2 is None or b.score_top2 is None:
                    continue
                combos = tuple(f"{x}&{y}" for x in a.score_top2.selections for y in b.score_top2.selections)
                component = tuple(x * y for x in a.score_top2.probabilities
                                  for y in b.score_top2.probabilities)
                result.append(_candidate(play_type, (a, b), combos, component,
                    coverage=a.score_top2.coverage * b.score_top2.coverage))
        return result
