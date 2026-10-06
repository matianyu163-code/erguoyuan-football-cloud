"""Point-in-time standings rebuilt from immutable prior results only."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, datetime

from erguoyuan_football.context.schemas import (
    Availability,
    CompetitionRule,
    HistoricalMatchEvent,
    PositionQuality,
    StandingsSnapshot,
    TeamStanding,
)


@dataclass
class _MutableStanding:
    """Typed accumulator for one team's prior results."""

    played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_for: int = 0
    goals_against: int = 0
    source_match_ids: list[str] = field(default_factory=list)


class StandingsReconstructor:
    """Build W/D/L and goal totals; compute points only with verified scoring rules."""

    def reconstruct(self, *, competition_id: str, season_id: str, target_date: date,
                    events: Iterable[HistoricalMatchEvent],
                    rule: CompetitionRule | None,
                    prediction_boundary: datetime | None = None) -> StandingsSnapshot:
        eligible = sorted(
            (event for event in events if event.competition_id == competition_id
             and event.season_id == season_id and event.match_date < target_date),
            key=lambda event: (event.match_date, event.match_id),
        )
        rows: dict[str, _MutableStanding] = {}
        for event in eligible:
            home = rows.setdefault(event.home_team_id, _MutableStanding())
            away = rows.setdefault(event.away_team_id, _MutableStanding())
            self._apply(home, event.home_goals, event.away_goals, event.match_id)
            self._apply(away, event.away_goals, event.home_goals, event.match_id)
        verified_rule = (rule is not None and rule.verified
                         and (prediction_boundary is None or rule.retrieved_at <= prediction_boundary))
        standing_rows: list[TeamStanding] = []
        for team_id, values in rows.items():
            points = None
            if verified_rule and rule is not None and rule.points_win is not None and rule.points_draw is not None:
                points = values.wins * rule.points_win + values.draws * rule.points_draw
            gd = values.goals_for - values.goals_against
            standing_rows.append(TeamStanding(
                team_id=team_id, played=values.played, wins=values.wins,
                draws=values.draws, losses=values.losses,
                goals_for=values.goals_for, goals_against=values.goals_against,
                goal_difference=gd, points=points, position=None,
                position_quality=(PositionQuality.APPROXIMATE_POINTS_ONLY if points is not None
                                  else PositionQuality.UNAVAILABLE),
                source_match_ids=tuple(values.source_match_ids),
            ))
        position_quality = PositionQuality.UNAVAILABLE
        if verified_rule and rule is not None and rule.points_win is not None:
            standing_rows.sort(key=lambda row: (
                -(row.points if row.points is not None else -1), row.team_id))
            ranked: list[TeamStanding] = []
            prior_points: int | None = None
            position = 0
            for index, row in enumerate(standing_rows, start=1):
                if row.points != prior_points:
                    position = index
                    prior_points = row.points
                ranked.append(row.model_copy(update={
                    "position": position,
                    "position_quality": PositionQuality.APPROXIMATE_POINTS_ONLY,
                }))
            standing_rows = ranked
            position_quality = PositionQuality.APPROXIMATE_POINTS_ONLY
        reason = None if eligible else "NO_PRIOR_MATCH_RESULTS"
        if not verified_rule:
            reason = "VERIFIED_POINTS_RULE_UNAVAILABLE"
        return StandingsSnapshot(
            competition_id=competition_id, season_id=season_id, as_of_date=target_date,
            boundary_mode="DATE_SAFE_PRIOR_DATE_ONLY",
            rule_version=rule.rule_version if verified_rule and rule else None,
            position_quality=position_quality, rows=tuple(standing_rows),
            source_match_ids=tuple(event.match_id for event in eligible),
            availability=Availability.AVAILABLE if eligible else Availability.UNAVAILABLE,
            reason=reason,
        )

    @staticmethod
    def _apply(team: _MutableStanding, goals_for: int, goals_against: int,
               match_id: str) -> None:
        team.played += 1
        team.goals_for += goals_for
        team.goals_against += goals_against
        if goals_for > goals_against:
            team.wins += 1
        elif goals_for == goals_against:
            team.draws += 1
        else:
            team.losses += 1
        team.source_match_ids.append(match_id)
