"""Explicit research-stage META, calibrated CORE and diagnostic contracts."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import Field, model_validator

from erguoyuan_football.contracts.common import (
    Contract,
    Identifier,
    Probability,
    UTCTime,
)
from erguoyuan_football.contracts.predictions import ProbabilityVector


class MetaRawPrediction(Contract):
    """Fitted META output before independent calibration."""

    match_id: Identifier
    prediction_snapshot_id: Identifier
    meta_model_id: Literal["META_NO_MARKET_V1"]
    meta_model_version: Identifier
    p_home: Probability
    p_draw: Probability
    p_away: Probability
    models_used: tuple[Identifier, ...]
    models_missing: tuple[Identifier, ...]
    availability_pattern: str
    dependency_summary: dict[str, Any]
    temporal_mode: Literal["DATE_SAFE_BATCH"]
    validation_status: Literal["DEVELOPMENT_VALIDATED_DATE_SAFE"]
    probability_stage: Literal["META_RAW"] = "META_RAW"

    @model_validator(mode="after")
    def valid_probability(self) -> MetaRawPrediction:
        ProbabilityVector(p_home=self.p_home, p_draw=self.p_draw, p_away=self.p_away)
        if not self.models_used or self.dependency_summary.get("market_dependency_count") != 0:
            raise ValueError("META_RAW_SOURCE_OR_MARKET_INVALID")
        return self


class CalibratedCoreProbability(Contract):
    """Highest current pipeline probability, explicitly not promoted to production."""

    prediction_id: Identifier
    prediction_snapshot_id: Identifier
    match_id: Identifier
    meta_model_id: Literal["META_NO_MARKET_V1"]
    meta_model_version: Identifier
    meta_raw_p_home: Probability
    meta_raw_p_draw: Probability
    meta_raw_p_away: Probability
    calibrated_p_home: Probability
    calibrated_p_draw: Probability
    calibrated_p_away: Probability
    calibrator_id: Identifier
    calibrator_version: Identifier
    models_used: tuple[Identifier, ...]
    models_missing: tuple[Identifier, ...]
    dependency_summary: dict[str, Any]
    temporal_mode: Literal["DATE_SAFE_BATCH"]
    validation_status: Literal["DEVELOPMENT_VALIDATED_DATE_SAFE"]
    production_status: Literal["NOT_PROMOTED"] = "NOT_PROMOTED"
    created_at: UTCTime

    @model_validator(mode="after")
    def valid_probabilities(self) -> CalibratedCoreProbability:
        ProbabilityVector(p_home=self.meta_raw_p_home, p_draw=self.meta_raw_p_draw,
                          p_away=self.meta_raw_p_away)
        ProbabilityVector(p_home=self.calibrated_p_home, p_draw=self.calibrated_p_draw,
                          p_away=self.calibrated_p_away)
        if not self.models_used or self.dependency_summary.get("market_dependency_count") != 0:
            raise ValueError("CALIBRATED_CORE_SOURCE_OR_MARKET_INVALID")
        return self


class ModelDisagreementReport(Contract):
    """Descriptive probability spread; it is never a bet or selection signal."""

    match_id: Identifier
    available_model_count: int = Field(ge=1)
    probability_std: dict[str, float]
    probability_range: dict[str, float]
    mean_pairwise_distance: float = Field(ge=0, allow_inf_nan=False)
    entropy: float = Field(ge=0, allow_inf_nan=False)
    purpose: Literal["DIAGNOSTIC_ONLY"] = "DIAGNOSTIC_ONLY"

    @model_validator(mode="after")
    def finite_spread(self) -> ModelDisagreementReport:
        if any(not math.isfinite(value) or value < 0 for value in (
                *self.probability_std.values(), *self.probability_range.values())):
            raise ValueError("NONFINITE_DISAGREEMENT")
        return self


class MetaPromotionDecision(Contract):
    """Fail-closed model-promotion decision, separate from pipeline stage."""

    engineering_ready: bool
    artifact_ready: bool
    development_validation_ready: bool
    final_holdout_ready: bool
    superiority_evidence: Literal["ESTABLISHED", "NOT_ESTABLISHED", "INSUFFICIENT_EVIDENCE"]
    live_temporal_ready: bool
    market_ready: bool
    production_promoted: bool
    reason: str

    @model_validator(mode="after")
    def no_unearned_promotion(self) -> MetaPromotionDecision:
        prerequisites = (self.engineering_ready, self.artifact_ready,
                         self.development_validation_ready, self.final_holdout_ready,
                         self.live_temporal_ready, self.market_ready)
        if self.production_promoted and (not all(prerequisites) or
                                         self.superiority_evidence != "ESTABLISHED"):
            raise ValueError("PROMOTION_PREREQUISITES_NOT_MET")
        return self
