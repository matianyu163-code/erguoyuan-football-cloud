"""Build versioned context features with explicit missing masks and source lineage."""

from __future__ import annotations

import json
from datetime import date
from hashlib import sha256

from erguoyuan_football.context.schemas import (
    ContextFeatureVector,
    ScheduleContext,
    TournamentIncentiveState,
    TournamentState,
)
from erguoyuan_football.context.uncertainty import ContextUncertaintyEngine

FEATURE_FIELDS = (
    "tournament_type", "stage", "leg_number", "aggregate_margin",
    "standings_features", "mai", "rest_days_home", "rest_days_away",
    "matches_last_7d_home", "matches_last_7d_away",
    "matches_last_14d_home", "matches_last_14d_away",
    "matches_last_21d_home", "matches_last_21d_away",
    "rotation_risk", "lineup_strength_delta", "injury_strength_delta",
)
FEATURE_SCHEMA_HASH = sha256(json.dumps({"version": "CONTEXT_FEATURE_V1",
    "fields": FEATURE_FIELDS}, separators=(",", ":")).encode()).hexdigest()


class ContextFeatureBuilder:
    """Convert independently gated states into a null-aware feature record."""

    def __init__(self) -> None:
        self.uncertainty = ContextUncertaintyEngine()

    def build(self, *, match_id: str, prediction_snapshot_id: str,
              home_team_id: str, away_team_id: str,
              prediction_date: date,
              tournament: TournamentState, incentive: TournamentIncentiveState,
              schedule: ScheduleContext, lineup_available: bool,
              injury_available: bool, player_strength_available: bool,
              source_event_dates: dict[str, date],
              lineup_source_ids: tuple[str, ...] = (),
              injury_source_ids: tuple[str, ...] = (),
              player_strength_source_ids: tuple[str, ...] = (),
              venue_source_ids: tuple[str, ...] = (),
              rotation_risk: float | None = None,
              lineup_strength_delta: float | None = None,
              injury_strength_delta: float | None = None) -> ContextFeatureVector:
        standings = tournament.standings_state
        availability = {
            "tournament_rule": tournament.rule_version is not None,
            "standings": standings is not None and bool(standings.source_match_ids),
            "schedule_history": schedule.availability.value == "AVAILABLE",
            "lineup": lineup_available,
            "injury": injury_available,
            "player_strength": player_strength_available,
            "venue": schedule.neutral_venue is not None and bool(venue_source_ids),
        }
        uncertainty = self.uncertainty.evaluate(availability)
        lineage = {
            "tournament_rule": (),
            "standings": standings.source_match_ids if standings else (),
            "schedule_history": tuple(dict.fromkeys((
                *schedule.source_match_ids_home, *schedule.source_match_ids_away))),
            "standings_features": standings.source_match_ids if standings else (),
            "rest_days_home": schedule.source_match_ids_home,
            "rest_days_away": schedule.source_match_ids_away,
            "matches_last_7d_home": schedule.source_match_ids_home,
            "matches_last_7d_away": schedule.source_match_ids_away,
            "matches_last_14d_home": schedule.source_match_ids_home,
            "matches_last_14d_away": schedule.source_match_ids_away,
            "matches_last_21d_home": schedule.source_match_ids_home,
            "matches_last_21d_away": schedule.source_match_ids_away,
            "lineup": lineup_source_ids,
            "injury": injury_source_ids,
            "player_strength": player_strength_source_ids,
            "venue": venue_source_ids,
            "lineup_strength_delta": (*lineup_source_ids, *player_strength_source_ids),
            "injury_strength_delta": injury_source_ids,
        }
        if availability["tournament_rule"] and tournament.rule_source_id:
            lineage["tournament_rule"] = (tournament.rule_source_id,)
        home_standing = next((row for row in standings.rows if row.team_id == home_team_id), None) if standings else None
        away_standing = next((row for row in standings.rows if row.team_id == away_team_id), None) if standings else None
        standings_features: dict[str, float | None] = {}
        for side, row in (("home", home_standing), ("away", away_standing)):
            standings_features[f"{side}_played"] = float(row.played) if row else None
            standings_features[f"{side}_points"] = float(row.points) if row and row.points is not None else None
        return ContextFeatureVector(
            match_id=match_id, prediction_snapshot_id=prediction_snapshot_id,
            prediction_date=prediction_date,
            tournament_type=tournament.tournament_type, stage=tournament.stage,
            leg_number=tournament.leg_number,
            aggregate_margin=tournament.aggregate_goal_margin_before_match,
            standings_features=standings_features, mai=incentive.mai,
            rest_days_home=schedule.rest_days_home, rest_days_away=schedule.rest_days_away,
            matches_last_7d_home=schedule.matches_last_7d_home,
            matches_last_7d_away=schedule.matches_last_7d_away,
            matches_last_14d_home=schedule.matches_last_14d_home,
            matches_last_14d_away=schedule.matches_last_14d_away,
            matches_last_21d_home=schedule.matches_last_21d_home,
            matches_last_21d_away=schedule.matches_last_21d_away,
            rotation_risk=rotation_risk, lineup_strength_delta=lineup_strength_delta,
            injury_strength_delta=injury_strength_delta,
            context_uncertainty=uncertainty.score,
            availability_mask=availability, lineage=lineage,
            source_event_dates=source_event_dates,
            rule_versions=(tournament.rule_version,) if tournament.rule_version else (),
            feature_schema_hash=FEATURE_SCHEMA_HASH,
        )
