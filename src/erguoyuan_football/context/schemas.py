"""Typed, auditable schemas for pre-match context and probability lineage."""

from __future__ import annotations

import math
from datetime import date
from enum import StrEnum
from typing import Any

from pydantic import Field, model_validator

from erguoyuan_football.contracts.common import (
    Contract,
    Identifier,
    Probability,
    UTCTime,
)
from erguoyuan_football.contracts.predictions import ProbabilityVector


class ContextDataClass(StrEnum):
    RECONSTRUCTIBLE_EVENT_CONTEXT = "RECONSTRUCTIBLE_EVENT_CONTEXT"
    STATIC_COMPETITION_RULE = "STATIC_COMPETITION_RULE"
    HISTORICAL_SNAPSHOT_CONTEXT = "HISTORICAL_SNAPSHOT_CONTEXT"
    LIVE_PREMATCH_CONTEXT = "LIVE_PREMATCH_CONTEXT"
    POST_MATCH_CONTEXT = "POST_MATCH_CONTEXT"
    UNKNOWN_CONTEXT = "UNKNOWN_CONTEXT"


class ContextStatus(StrEnum):
    NOT_EVALUATED = "NOT_EVALUATED"
    EVALUATED_NO_ADJUSTMENT = "EVALUATED_NO_ADJUSTMENT"
    ADJUSTED = "ADJUSTED"
    UNAVAILABLE = "UNAVAILABLE"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"


class TournamentType(StrEnum):
    LEAGUE = "LEAGUE"
    GROUP_STAGE = "GROUP_STAGE"
    SINGLE_ELIMINATION = "SINGLE_ELIMINATION"
    TWO_LEG_KNOCKOUT = "TWO_LEG_KNOCKOUT"
    ROUND_ROBIN = "ROUND_ROBIN"
    UNKNOWN = "UNKNOWN"


class TieState(StrEnum):
    LEADING = "LEADING"
    TRAILING = "TRAILING"
    LEVEL = "LEVEL"
    UNKNOWN = "UNKNOWN"


class LineupStatus(StrEnum):
    CONFIRMED_STARTER = "CONFIRMED_STARTER"
    CONFIRMED_BENCH = "CONFIRMED_BENCH"
    EXPECTED_STARTER = "EXPECTED_STARTER"
    EXPECTED_BENCH = "EXPECTED_BENCH"
    OUT = "OUT"
    DOUBTFUL = "DOUBTFUL"
    SUSPENDED = "SUSPENDED"
    UNKNOWN = "UNKNOWN"


class InjuryStatus(StrEnum):
    OUT = "OUT"
    DOUBTFUL = "DOUBTFUL"
    RETURNING = "RETURNING"
    UNKNOWN = "UNKNOWN"


class Availability(StrEnum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"


class PositionQuality(StrEnum):
    OFFICIAL_TIEBREAK = "OFFICIAL_TIEBREAK"
    APPROXIMATE_POINTS_ONLY = "APPROXIMATE_POINTS_ONLY"
    UNAVAILABLE = "UNAVAILABLE"


class CompetitionRule(Contract):
    """Versioned, source-backed competition settings, never inferred from labels."""

    competition_id: Identifier
    season_id: Identifier
    valid_from: date
    valid_to: date
    rule_version: Identifier
    points_win: int | None = Field(default=None, ge=0)
    points_draw: int | None = Field(default=None, ge=0)
    tie_break_policy: str | None = None
    tournament_type: TournamentType = TournamentType.UNKNOWN
    knockout_format: TournamentType = TournamentType.UNKNOWN
    legs: int | None = Field(default=None, ge=1)
    away_goals_rule: bool | None = None
    extra_time_rule: str | None = None
    source: Identifier
    retrieved_at: UTCTime
    verified: bool = False

    @model_validator(mode="after")
    def valid_interval(self) -> CompetitionRule:
        if self.valid_to < self.valid_from:
            raise ValueError("COMPETITION_RULE_INVALID_INTERVAL")
        if self.legs == 2 and self.knockout_format != TournamentType.TWO_LEG_KNOCKOUT:
            raise ValueError("COMPETITION_RULE_LEG_FORMAT_CONFLICT")
        if self.verified and not self.source:
            raise ValueError("VERIFIED_RULE_REQUIRES_SOURCE")
        return self


class HistoricalMatchEvent(Contract):
    """Immutable result fact with explicit event and retrieval dates."""

    match_id: Identifier
    competition_id: Identifier
    season_id: Identifier
    home_team_id: Identifier
    away_team_id: Identifier
    match_date: date
    home_goals: int = Field(ge=0)
    away_goals: int = Field(ge=0)
    source: Identifier
    retrieved_at: UTCTime
    as_of_time: UTCTime
    round_name: str | None = None
    tie_id: str | None = None
    leg_number: int | None = Field(default=None, ge=1)
    event_class: ContextDataClass = ContextDataClass.RECONSTRUCTIBLE_EVENT_CONTEXT


class TeamStanding(Contract):
    """Point-in-time record; unavailable rules leave points and rank null."""

    team_id: Identifier
    played: int = Field(ge=0)
    wins: int = Field(ge=0)
    draws: int = Field(ge=0)
    losses: int = Field(ge=0)
    goals_for: int = Field(ge=0)
    goals_against: int = Field(ge=0)
    goal_difference: int
    points: int | None = None
    position: int | None = None
    position_quality: PositionQuality
    source_match_ids: tuple[Identifier, ...] = ()


class StandingsSnapshot(Contract):
    competition_id: Identifier
    season_id: Identifier
    as_of_date: date
    boundary_mode: str
    rule_version: str | None = None
    position_quality: PositionQuality
    rows: tuple[TeamStanding, ...]
    source_match_ids: tuple[Identifier, ...]
    availability: Availability
    reason: str | None = None


class TournamentState(Contract):
    match_id: Identifier
    competition_id: Identifier
    season_id: Identifier
    stage: str | None = None
    round_name: str | None = None
    tournament_type: TournamentType = TournamentType.UNKNOWN
    leg_number: int | None = Field(default=None, ge=1)
    tie_id: str | None = None
    aggregate_state_before_match: TieState = TieState.UNKNOWN
    aggregate_goal_margin_before_match: int | None = None
    standings_state: StandingsSnapshot | None = None
    qualification_state: str | None = None
    elimination_state: str | None = None
    relegation_state: str | None = None
    title_state: str | None = None
    source_ids: tuple[Identifier, ...] = ()
    rule_version: str | None = None
    rule_source_id: str | None = None
    as_of_time: UTCTime
    quality_status: str
    unavailable_reasons: tuple[str, ...] = ()


class TournamentIncentiveState(Contract):
    """Only mathematically derivable states; no subjective motivation score."""

    match_id: Identifier
    mathematically_eliminated_home: bool | None = None
    mathematically_eliminated_away: bool | None = None
    qualification_secured_home: bool | None = None
    qualification_secured_away: bool | None = None
    relegation_confirmed_home: bool | None = None
    relegation_confirmed_away: bool | None = None
    title_mathematically_impossible_home: bool | None = None
    title_mathematically_impossible_away: bool | None = None
    mai: float | None = None
    mai_status: Availability = Availability.UNAVAILABLE
    reason_codes: tuple[str, ...] = ()
    source_ids: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def mai_is_feature_only(self) -> TournamentIncentiveState:
        if self.mai is not None and not math.isfinite(self.mai):
            raise ValueError("MAI_MUST_BE_FINITE")
        if self.mai_status == Availability.UNAVAILABLE and self.mai is not None:
            raise ValueError("UNAVAILABLE_MAI_MUST_BE_NULL")
        return self


class ScheduleContext(Contract):
    match_id: Identifier
    prediction_date: date
    rest_days_home: int | None = Field(default=None, ge=0)
    rest_days_away: int | None = Field(default=None, ge=0)
    matches_last_7d_home: int | None = Field(default=None, ge=0)
    matches_last_7d_away: int | None = Field(default=None, ge=0)
    matches_last_14d_home: int | None = Field(default=None, ge=0)
    matches_last_14d_away: int | None = Field(default=None, ge=0)
    matches_last_21d_home: int | None = Field(default=None, ge=0)
    matches_last_21d_away: int | None = Field(default=None, ge=0)
    future_schedule_count_home: int | None = Field(default=None, ge=0)
    future_schedule_count_away: int | None = Field(default=None, ge=0)
    travel_distance_home_km: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    travel_distance_away_km: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    neutral_venue: bool | None = None
    source_match_ids_home: tuple[Identifier, ...] = ()
    source_match_ids_away: tuple[Identifier, ...] = ()
    availability: Availability
    reason_codes: tuple[str, ...] = ()


class LineupEvidenceSnapshot(Contract):
    match_id: Identifier
    team_id: Identifier
    player_id: Identifier
    evidence_type: str
    lineup_status: LineupStatus
    source_id: Identifier
    source: Identifier
    source_time: UTCTime
    retrieved_at: UTCTime
    confidence: Probability
    is_official: bool
    quality_status: str
    data_class: ContextDataClass = ContextDataClass.HISTORICAL_SNAPSHOT_CONTEXT


class InjuryEvidence(Contract):
    match_id: Identifier
    team_id: Identifier
    player_id: Identifier
    status: InjuryStatus
    source_id: Identifier
    source: Identifier
    published_at: UTCTime
    retrieved_at: UTCTime
    confidence: Probability
    statement: str | None = None


class PlayerStrengthEvidence(Contract):
    player_id: Identifier
    value: float = Field(allow_inf_nan=False)
    model_id: Identifier
    model_version: Identifier
    trained_until: UTCTime
    source_ids: tuple[Identifier, ...]
    available: bool


class ContextFeatureVector(Contract):
    match_id: Identifier
    prediction_snapshot_id: Identifier
    prediction_date: date
    tournament_type: TournamentType
    stage: str | None = None
    leg_number: int | None = None
    aggregate_margin: int | None = None
    standings_features: dict[str, float | None] = Field(default_factory=dict)
    mai: float | None = None
    rest_days_home: int | None = None
    rest_days_away: int | None = None
    matches_last_7d_home: int | None = None
    matches_last_7d_away: int | None = None
    matches_last_14d_home: int | None = None
    matches_last_14d_away: int | None = None
    matches_last_21d_home: int | None = None
    matches_last_21d_away: int | None = None
    rotation_risk: float | None = None
    lineup_strength_delta: float | None = None
    injury_strength_delta: float | None = None
    context_uncertainty: float = Field(ge=0, le=1)
    availability_mask: dict[str, bool]
    lineage: dict[str, tuple[str, ...]]
    source_event_dates: dict[str, date]
    rule_versions: tuple[str, ...] = ()
    feature_schema_hash: Identifier

    @model_validator(mode="after")
    def available_features_have_lineage(self) -> ContextFeatureVector:
        for name, is_available in self.availability_mask.items():
            if is_available and not self.lineage.get(name):
                raise ValueError(f"CONTEXT_FEATURE_LINEAGE_MISSING:{name}")
        if any(event_date >= self.prediction_date for event_date in self.source_event_dates.values()):
            raise ValueError("CONTEXT_FEATURE_SOURCE_NOT_BEFORE_TARGET_DATE")
        return self


class ContextDataAvailabilityReport(Contract):
    """Per-match component status with explicit evidence IDs for available inputs."""

    match_id: Identifier
    prediction_snapshot_id: Identifier
    prediction_time: UTCTime
    component_status: dict[str, Availability]
    source_ids: dict[str, tuple[Identifier, ...]]
    reason_codes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def available_components_have_sources(self) -> ContextDataAvailabilityReport:
        for component, status in self.component_status.items():
            if status == Availability.AVAILABLE and not self.source_ids.get(component):
                raise ValueError(f"CONTEXT_AVAILABILITY_LINEAGE_MISSING:{component}")
        return self


class ContextAdjustedCoreProbability(Contract):
    prediction_id: Identifier
    prediction_snapshot_id: Identifier
    match_id: Identifier
    base_p_home: Probability
    base_p_draw: Probability
    base_p_away: Probability
    tournament_p_home: Probability
    tournament_p_draw: Probability
    tournament_p_away: Probability
    lineup_p_home: Probability
    lineup_p_draw: Probability
    lineup_p_away: Probability
    final_p_home: Probability
    final_p_draw: Probability
    final_p_away: Probability
    tournament_adjustment_applied: bool
    lineup_adjustment_applied: bool
    context_model_id: str
    context_model_version: str
    lineup_model_id: str
    lineup_model_version: str
    mai: float | None = None
    rotation_risk: float | None = None
    lineup_strength_delta: float | None = None
    context_uncertainty: float = Field(ge=0, le=1)
    evidence_quality: str
    validation_status: str = "DEVELOPMENT_ONLY"
    production_status: str = "NOT_PROMOTED"
    created_at: UTCTime

    @model_validator(mode="after")
    def valid_probability_layers(self) -> ContextAdjustedCoreProbability:
        for stage in ("base", "tournament", "lineup", "final"):
            ProbabilityVector(
                p_home=getattr(self, f"{stage}_p_home"),
                p_draw=getattr(self, f"{stage}_p_draw"),
                p_away=getattr(self, f"{stage}_p_away"),
            )
        if not self.tournament_adjustment_applied and (
            self.tournament_p_home, self.tournament_p_draw, self.tournament_p_away
        ) != (self.base_p_home, self.base_p_draw, self.base_p_away):
            raise ValueError("TOURNAMENT_BASE_MUST_BE_PRESERVED_WITHOUT_ADJUSTMENT")
        if not self.lineup_adjustment_applied and (
            self.lineup_p_home, self.lineup_p_draw, self.lineup_p_away
        ) != (self.tournament_p_home, self.tournament_p_draw, self.tournament_p_away):
            raise ValueError("LINEUP_BASE_MUST_BE_PRESERVED_WITHOUT_ADJUSTMENT")
        if self.production_status != "NOT_PROMOTED":
            raise ValueError("PHASE10_CANNOT_PROMOTE_CONTEXT")
        return self


class AdjustmentLedger(Contract):
    ledger_id: Identifier
    match_id: Identifier
    prediction_snapshot_id: Identifier
    base_probability: dict[str, float]
    tournament_probability: dict[str, float]
    lineup_probability: dict[str, float]
    final_probability: dict[str, float]
    base_logits: tuple[float, float, float]
    context_delta_logits: tuple[float, float, float]
    lineup_delta_logits: tuple[float, float, float]
    final_logits: tuple[float, float, float]
    context_features: dict[str, Any]
    evidence_ids: tuple[str, ...]
    rule_versions: tuple[str, ...]
    reason_codes: tuple[str, ...]
    models: dict[str, str]
    adjustment_applied: bool
    created_at: UTCTime


class ContextAssessment(Contract):
    match_id: Identifier
    prediction_snapshot_id: Identifier
    context_status: ContextStatus
    tournament_state: TournamentState
    incentive: TournamentIncentiveState
    schedule: ScheduleContext
    data_availability: ContextDataAvailabilityReport
    lineup_status: Availability
    lineup_evidence: tuple[LineupEvidenceSnapshot, ...] = ()
    injury_evidence: tuple[InjuryEvidence, ...] = ()
    feature_vector: ContextFeatureVector
    adjusted_probability: ContextAdjustedCoreProbability
    ledger: AdjustmentLedger
    context_model_status: str
    lineup_model_status: str
    reason_codes: tuple[str, ...] = ()
