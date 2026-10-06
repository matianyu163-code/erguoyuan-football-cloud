"""Mathematical incentive-state derivations; never infer subjective motivation."""

from __future__ import annotations

from erguoyuan_football.context.schemas import (
    Availability,
    CompetitionRule,
    StandingsSnapshot,
    TournamentIncentiveState,
)


def title_mathematically_impossible(*, team_points: int | None, leader_points: int | None,
                                   remaining_matches: int | None,
                                   rule: CompetitionRule | None) -> bool | None:
    """Compare the maximum reachable points only when rules/schedule are complete."""
    if (team_points is None or leader_points is None or remaining_matches is None
            or rule is None or not rule.verified or rule.points_win is None):
        return None
    maximum = team_points + max(remaining_matches, 0) * rule.points_win
    return maximum < leader_points


class TournamentIncentiveEngine:
    """Build explicit unknown states when mathematical inputs are incomplete."""

    def build(self, *, match_id: str, home_team_id: str, away_team_id: str,
              standings: StandingsSnapshot | None,
              rule: CompetitionRule | None,
              remaining_matches: dict[str, int] | None = None) -> TournamentIncentiveState:
        rows = {row.team_id: row for row in standings.rows} if standings else {}
        leader_points = max((row.points for row in rows.values() if row.points is not None),
                            default=None)
        def impossible(team_id: str) -> bool | None:
            row = rows.get(team_id)
            left = remaining_matches.get(team_id) if remaining_matches is not None else None
            return title_mathematically_impossible(team_points=row.points if row else None,
                leader_points=leader_points, remaining_matches=left, rule=rule)
        reasons = ("NO_VERIFIED_REMAINING_FIXTURE_SNAPSHOT",) if remaining_matches is None else ()
        return TournamentIncentiveState(
            match_id=match_id,
            title_mathematically_impossible_home=impossible(home_team_id),
            title_mathematically_impossible_away=impossible(away_team_id),
            mai=None, mai_status=Availability.UNAVAILABLE, reason_codes=reasons,
            source_ids=standings.source_match_ids if standings else (),
        )
