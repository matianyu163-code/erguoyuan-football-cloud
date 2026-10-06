"""Aggregate state for explicitly identified, verified two-leg ties."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date

from erguoyuan_football.context.schemas import (
    CompetitionRule,
    HistoricalMatchEvent,
    TieState,
    TournamentType,
)


def aggregate_before_second_leg(*, target_match_id: str, target_date: date,
                                tie_id: str | None, leg_number: int | None,
                                home_team_id: str, away_team_id: str,
                                events: Iterable[HistoricalMatchEvent],
                                rule: CompetitionRule | None
                                ) -> tuple[TieState, int | None, tuple[str, ...]]:
    """Rebuild the first-leg goal margin, rejecting unversioned or ambiguous ties."""
    if (rule is None or not rule.verified or rule.tournament_type != TournamentType.TWO_LEG_KNOCKOUT
            or rule.legs != 2 or tie_id is None or leg_number != 2):
        return TieState.UNKNOWN, None, ("VERIFIED_TWO_LEG_RULE_OR_TIE_ID_UNAVAILABLE",)
    prior_legs = [event for event in events if event.tie_id == tie_id and event.match_id != target_match_id
                  and event.leg_number == 1 and event.match_date < target_date]
    if len(prior_legs) != 1:
        return TieState.UNKNOWN, None, ("FIRST_LEG_RESULT_UNAVAILABLE_OR_AMBIGUOUS",)
    first = prior_legs[0]
    if {first.home_team_id, first.away_team_id} != {home_team_id, away_team_id}:
        return TieState.UNKNOWN, None, ("FIRST_LEG_TEAM_ID_MISMATCH",)
    margin = ((first.home_goals - first.away_goals) if first.home_team_id == home_team_id
              else first.away_goals - first.home_goals)
    state = TieState.LEADING if margin > 0 else TieState.TRAILING if margin < 0 else TieState.LEVEL
    return state, margin, ()
