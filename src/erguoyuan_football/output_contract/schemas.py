"""Phase 8 CORE V2 records. A base-model probability is never a final CORE probability."""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, model_validator

from erguoyuan_football.contracts.common import (
    Contract,
    Identifier,
    Probability,
    UTCTime,
)
from erguoyuan_football.contracts.predictions import ProbabilityVector


class OutputVersion(StrEnum):
    LEGACY_V51 = "LEGACY_V51"
    CORE_OUTPUT_V2 = "CORE_OUTPUT_V2"


class ProbabilityStage(StrEnum):
    BASE_MODEL = "BASE_MODEL"
    ML_MODEL = "ML_MODEL"
    PRE_META = "PRE_META"
    FINAL_CORE_CALIBRATED = "FINAL_CORE_CALIBRATED"
    CONTEXT_ADJUSTED_CORE = "CONTEXT_ADJUSTED_CORE"


class AvailabilityStatus(StrEnum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"


class CanonicalPredictionResult(Contract):
    """Audited probability stage; calibrated development output is not live proof."""

    output_version: OutputVersion = OutputVersion.CORE_OUTPUT_V2
    prediction_id: Identifier
    prediction_snapshot_id: Identifier
    match_id: Identifier
    competition_id: Identifier
    kickoff_time: UTCTime | None
    match_date: date | None = None
    prediction_temporal_mode: Literal["EXACT_UTC", "DATE_SAFE_BATCH"] = "EXACT_UTC"
    prediction_time: UTCTime
    prediction_horizon: str
    data_origin: Literal["REAL", "SYNTHETIC", "TEST_FIXTURE", "UNKNOWN"]
    status: AvailabilityStatus
    data_quality_status: str
    pit_status: str
    probability_stage: ProbabilityStage | None = None
    validation_status: Literal["UNVALIDATED", "DEVELOPMENT_ONLY", "FINAL_HOLDOUT_VALIDATED"] = "UNVALIDATED"
    production_status: Literal["NOT_PROMOTED", "PROMOTED"] = "NOT_PROMOTED"
    p_home: Probability | None = None
    p_draw: Probability | None = None
    p_away: Probability | None = None
    score_matrix_ref: str | None = None
    score_matrix_status: AvailabilityStatus = AvailabilityStatus.UNAVAILABLE
    total_goals_distribution: dict[str, Probability] | None = None
    total_goals_status: AvailabilityStatus = AvailabilityStatus.NOT_IMPLEMENTED
    htft_distribution: dict[str, Probability] | None = None
    htft_status: AvailabilityStatus = AvailabilityStatus.NOT_IMPLEMENTED
    official_handicap: float | None = None
    handicap_probabilities: dict[str, Probability] | None = None
    handicap_status: AvailabilityStatus = AvailabilityStatus.UNAVAILABLE
    market_snapshot_id: str | None = None
    market_probabilities: dict[str, Probability] | None = None
    market_status: AvailabilityStatus = AvailabilityStatus.UNAVAILABLE
    models_used: tuple[str, ...] = ()
    models_unavailable: tuple[str, ...] = ()
    models_failed: tuple[str, ...] = ()
    calibration_status: AvailabilityStatus = AvailabilityStatus.NOT_IMPLEMENTED
    context_status: Literal["NOT_IMPLEMENTED", "NOT_EVALUATED", "EVALUATED_NO_ADJUSTMENT",
                            "ADJUSTED", "UNAVAILABLE", "BLOCKED", "FAILED"] = "NOT_IMPLEMENTED"
    context_adjustment_status: str = "NOT_APPLIED"
    context: dict[str, Any] | None = None
    data_lineage: dict[str, Any] = Field(default_factory=dict)
    model_lineage: dict[str, Any] = Field(default_factory=dict)
    created_at: UTCTime

    @model_validator(mode="after")
    def check_integrity(self) -> CanonicalPredictionResult:
        if self.prediction_temporal_mode == "EXACT_UTC":
            if self.kickoff_time is None or not self.prediction_time < self.kickoff_time:
                raise ValueError("prediction_time must precede kickoff with a verified exact UTC time")
        elif (self.kickoff_time is not None or self.match_date is None or
              self.prediction_time.date() != self.match_date):
            raise ValueError("DATE_SAFE_BATCH requires match_date boundary and null exact kickoff")
        values = (self.p_home, self.p_draw, self.p_away)
        if self.status == AvailabilityStatus.AVAILABLE:
            if self.data_origin != "REAL":
                raise ValueError("production canonical probability requires REAL data origin")
            if self.probability_stage is None:
                raise ValueError("available probability requires its intermediate stage")
            if any(value is None for value in values):
                raise ValueError("available probability requires all three values")
            assert self.p_home is not None and self.p_draw is not None and self.p_away is not None
            ProbabilityVector(p_home=self.p_home, p_draw=self.p_draw, p_away=self.p_away)
            if not self.models_used or self.pit_status != "PASS":
                raise ValueError("available probability requires sourced model and PIT pass")
            if self.probability_stage == ProbabilityStage.FINAL_CORE_CALIBRATED and (
                    self.calibration_status != AvailabilityStatus.AVAILABLE or
                    self.validation_status == "UNVALIDATED"):
                raise ValueError("calibrated CORE requires calibration and validation status")
            if self.production_status == "PROMOTED" and (
                    self.probability_stage != ProbabilityStage.FINAL_CORE_CALIBRATED or
                    self.validation_status != "FINAL_HOLDOUT_VALIDATED" or
                    self.prediction_temporal_mode != "EXACT_UTC"):
                raise ValueError("production promotion requires final validation and exact UTC")
        elif any(value is not None for value in values) or self.probability_stage is not None:
            raise ValueError("unavailable output cannot contain probabilities")
        if self.score_matrix_status == AvailabilityStatus.AVAILABLE and not self.score_matrix_ref:
            raise ValueError("available score matrix needs a reference")
        if self.market_status != AvailabilityStatus.AVAILABLE and self.market_probabilities is not None:
            raise ValueError("unavailable market probabilities must be null")
        if self.context_status == "ADJUSTED" and (
                self.probability_stage != ProbabilityStage.CONTEXT_ADJUSTED_CORE or
                self.context is None):
            raise ValueError("context-adjusted output requires its stage and context block")
        if self.context_status == "EVALUATED_NO_ADJUSTMENT" and (
                self.probability_stage == ProbabilityStage.CONTEXT_ADJUSTED_CORE):
            raise ValueError("no-adjustment output must retain its input probability stage")
        if self.handicap_status != AvailabilityStatus.AVAILABLE and self.handicap_probabilities is not None:
            raise ValueError("unavailable handicap probabilities must be null")
        if self.htft_status != AvailabilityStatus.AVAILABLE and self.htft_distribution is not None:
            raise ValueError("unavailable HTFT probabilities must be null")
        return self


class RankedCandidate(Contract):
    """Selection output shape only; Phase 8 does not rank or select bets."""

    candidate_id: Identifier
    match_id: Identifier
    prediction_snapshot_id: Identifier
    selection: Identifier
    rank: int = Field(ge=1)
    probability: Probability | None = None
    status: AvailabilityStatus


class BetAdvice(Contract):
    """Recommendation shape only; NO_BET is distinct from data unavailable."""

    candidate_id: Identifier | None = None
    decision: Literal["BET", "NO_BET", "UNAVAILABLE"]
    stake: float = Field(ge=0, allow_inf_nan=False)
    reason: str

    @model_validator(mode="after")
    def no_bet_has_zero_stake(self) -> BetAdvice:
        if self.decision == "NO_BET" and self.stake != 0:
            raise ValueError("NO_BET requires zero stake")
        if self.decision == "BET" and (self.stake <= 0 or self.candidate_id is None):
            raise ValueError("BET requires a candidate and positive stake")
        if self.decision == "UNAVAILABLE" and self.stake != 0:
            raise ValueError("UNAVAILABLE requires zero stake")
        return self
