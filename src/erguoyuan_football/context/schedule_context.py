"""Point-in-time rest and congestion features from prior finished matches."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date

from erguoyuan_football.context.schemas import (
    Availability,
    HistoricalMatchEvent,
    ScheduleContext,
)


class ScheduleContextBuilder:
    """Build prior-date schedule counts; unknown future fixtures stay null."""

    def build(self, *, match_id: str, competition_id: str, season_id: str,
              target_date: date, home_team_id: str, away_team_id: str,
              events: Iterable[HistoricalMatchEvent],
              neutral_venue: bool | None = None,
              schedule_coverage_verified: bool = False) -> ScheduleContext:
        del competition_id, season_id
        prior = tuple(event for event in events if event.match_date < target_date)
        home_events = tuple(event for event in prior
                            if home_team_id in (event.home_team_id, event.away_team_id))
        away_events = tuple(event for event in prior
                            if away_team_id in (event.home_team_id, event.away_team_id))
        coverage = schedule_coverage_verified
        return ScheduleContext(
            match_id=match_id, prediction_date=target_date,
            rest_days_home=self._rest(home_events, target_date) if coverage else None,
            rest_days_away=self._rest(away_events, target_date) if coverage else None,
            matches_last_7d_home=self._window(home_events, target_date, 7) if coverage else None,
            matches_last_7d_away=self._window(away_events, target_date, 7) if coverage else None,
            matches_last_14d_home=self._window(home_events, target_date, 14) if coverage else None,
            matches_last_14d_away=self._window(away_events, target_date, 14) if coverage else None,
            matches_last_21d_home=self._window(home_events, target_date, 21) if coverage else None,
            matches_last_21d_away=self._window(away_events, target_date, 21) if coverage else None,
            # The canonical source has no historically snapshotted future fixture list or venue data.
            future_schedule_count_home=None, future_schedule_count_away=None,
            travel_distance_home_km=None, travel_distance_away_km=None,
            neutral_venue=neutral_venue,
            source_match_ids_home=tuple(event.match_id for event in home_events),
            source_match_ids_away=tuple(event.match_id for event in away_events),
            availability=(Availability.AVAILABLE if coverage and home_events and away_events
                          else Availability.UNAVAILABLE),
            reason_codes=tuple(reason for condition, reason in (
                (not coverage, "CROSS_COMPETITION_SCHEDULE_COVERAGE_UNVERIFIED"),
                (not home_events, "HOME_PRIOR_RESULTS_UNAVAILABLE"),
                (not away_events, "AWAY_PRIOR_RESULTS_UNAVAILABLE"),
                (True, "FUTURE_SCHEDULE_SNAPSHOT_UNAVAILABLE"),
                (neutral_venue is None, "NEUTRAL_VENUE_UNVERIFIED"),
                (True, "VENUE_COORDINATES_UNAVAILABLE"),
            ) if condition),
        )

    @staticmethod
    def _rest(events: tuple[HistoricalMatchEvent, ...], target_date: date) -> int | None:
        if not events:
            return None
        previous = max(event.match_date for event in events)
        return (target_date - previous).days

    @staticmethod
    def _window(events: tuple[HistoricalMatchEvent, ...], target_date: date, days: int) -> int | None:
        if not events:
            return None
        lower = target_date.fromordinal(target_date.toordinal() - days)
        return sum(lower < event.match_date < target_date for event in events)
