"""Build a sourced tournament state without parsing match-name strings."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime

from erguoyuan_football.context.competition_rules import CompetitionRuleRegistry
from erguoyuan_football.context.schemas import (
    HistoricalMatchEvent,
    TieState,
    TournamentState,
    TournamentType,
)
from erguoyuan_football.context.standings import StandingsReconstructor
from erguoyuan_football.context.tie_state import aggregate_before_second_leg


class TournamentStateBuilder:
    """Join verified competition-season rules to strictly prior event records."""

    def __init__(self, rules: CompetitionRuleRegistry) -> None:
        self.rules = rules
        self.standings = StandingsReconstructor()

    def build(self, *, match_id: str, competition_id: str, season_id: str,
              target_date: date, prediction_time: datetime, home_team_id: str,
              away_team_id: str, events: Iterable[HistoricalMatchEvent],
              round_name: str | None = None, tie_id: str | None = None,
              leg_number: int | None = None) -> TournamentState:
        event_rows = tuple(events)
        rule = self.rules.resolve(competition_id, season_id, target_date,
                                  prediction_time=prediction_time)
        standings = self.standings.reconstruct(competition_id=competition_id,
            season_id=season_id, target_date=target_date, events=event_rows, rule=rule,
            prediction_boundary=prediction_time)
        if rule and rule.tournament_type == TournamentType.TWO_LEG_KNOCKOUT and leg_number == 2:
            tie_state, aggregate_margin, tie_reasons = aggregate_before_second_leg(
                target_match_id=match_id, target_date=target_date, tie_id=tie_id,
                leg_number=leg_number, home_team_id=home_team_id, away_team_id=away_team_id,
                events=event_rows, rule=rule)
        else:
            tie_state, aggregate_margin, tie_reasons = TieState.UNKNOWN, None, ()
        reasons = list(tie_reasons)
        if rule is None:
            reasons.append("COMPETITION_RULE_UNAVAILABLE")
        if not standings.source_match_ids:
            reasons.append("NO_PRIOR_STANDINGS_EVENTS")
        return TournamentState(
            match_id=match_id, competition_id=competition_id, season_id=season_id,
            stage=None, round_name=round_name,
            tournament_type=rule.tournament_type if rule else TournamentType.UNKNOWN,
            leg_number=leg_number, tie_id=tie_id,
            aggregate_state_before_match=tie_state,
            aggregate_goal_margin_before_match=aggregate_margin,
            standings_state=standings,
            source_ids=tuple(dict.fromkeys((*standings.source_match_ids,
                *((rule.source,) if rule else ())))),
            rule_version=rule.rule_version if rule else None,
            rule_source_id=rule.source if rule else None,
            as_of_time=prediction_time, quality_status=("PARTIAL" if reasons else "AVAILABLE"),
            unavailable_reasons=tuple(dict.fromkeys(reasons)),
        )
