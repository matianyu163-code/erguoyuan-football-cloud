"""Future prediction records; validation never computes probabilities."""

import math
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field, model_validator

from erguoyuan_football.contracts.common import (
    Availability,
    Contract,
    ExecutionStatus,
    Identifier,
    ImplementationType,
    NonNegative,
    Probability,
    UTCTime,
)

TOLERANCE = 1e-8


def validate_matrix(matrix):
    if matrix is None:
        return
    if not matrix or not matrix[0] or any(len(row) != len(matrix[0]) for row in matrix):
        raise ValueError("score matrix must be non-empty and rectangular")
    if not math.isclose(sum(map(sum, matrix)), 1, rel_tol=0, abs_tol=TOLERANCE):
        raise ValueError("score matrix mass must sum to 1; tail handling must be explicit upstream")


class ProbabilityVector(Contract):
    p_home: Probability
    p_draw: Probability
    p_away: Probability

    @model_validator(mode="after")
    def total(self):
        if not math.isclose(self.p_home + self.p_draw + self.p_away, 1, rel_tol=0, abs_tol=TOLERANCE):
            raise ValueError("probabilities must sum to 1")
        return self


class ModelPrediction(Contract):
    prediction_id: Identifier = Field(default_factory=lambda: str(uuid4()))
    match_id: Identifier
    prediction_snapshot_id: Identifier
    model_id: Identifier
    model_version: Identifier
    implementation_type: ImplementationType
    training_end_time: UTCTime | None
    trained_until: UTCTime | None
    prediction_time: UTCTime
    input_data_version: Identifier
    p_home: Probability | None = None
    p_draw: Probability | None = None
    p_away: Probability | None = None
    lambda_home: NonNegative | None = None
    lambda_away: NonNegative | None = None
    score_matrix: tuple[tuple[Probability, ...], ...] | None = None
    data_source: tuple[Identifier, ...]
    data_status: Availability
    execution_status: ExecutionStatus
    reason: str | None = None
    failure_code: str | None = None
    is_oos: bool = False
    expected_home_goals: NonNegative | None = None
    expected_away_goals: NonNegative | None = None
    correlation_parameter: NonNegative | None = None
    execution_time_ms: NonNegative = 0
    warnings: tuple[str, ...] = ()
    dependency_tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def audit(self):
        if self.model_id.endswith("_LIKE") and self.implementation_type != ImplementationType.LIKE_IMPLEMENTATION:
            raise ValueError("LIKE model must declare LIKE_IMPLEMENTATION")
        if self.training_end_time != self.trained_until:
            raise ValueError("trained_until and training_end_time must denote the same cutoff")
        future_cutoff = self.training_end_time is not None and self.training_end_time > self.prediction_time
        if future_cutoff and self.execution_status == ExecutionStatus.SUCCESS:
            raise ValueError("POINT_IN_TIME_GUARD_V1 violation: SUCCESS prediction cannot have training cutoff after prediction_time.")
        if future_cutoff and self.execution_status != ExecutionStatus.FAILED:
            raise ValueError("POINT_IN_TIME_GUARD_V1 violation: only FAILED may retain a future cutoff as audit evidence.")
        if future_cutoff and self.failure_code != "TRAINING_CUTOFF_AFTER_PREDICTION":
            raise ValueError("future cutoff audit evidence requires TRAINING_CUTOFF_AFTER_PREDICTION")
        values = (self.p_home, self.p_draw, self.p_away)
        if self.execution_status == ExecutionStatus.SUCCESS:
            if self.data_status != Availability.AVAILABLE or not self.data_source:
                raise ValueError("successful prediction requires available, sourced inputs")
            ProbabilityVector(p_home=self.p_home, p_draw=self.p_draw, p_away=self.p_away)
            if self.training_end_time is None and not self.reason:
                raise ValueError("unknown training cutoff requires explanation")
        else:
            if any(v is not None for v in (*values, self.lambda_home, self.lambda_away, self.score_matrix,
                                          self.expected_home_goals, self.expected_away_goals, self.correlation_parameter)):
                raise ValueError("non-success output must have null prediction values")
            if not self.reason:
                raise ValueError("non-success requires reason")
        if (self.lambda_home is None) != (self.lambda_away is None):
            raise ValueError("expected goals must be present as a pair")
        validate_matrix(self.score_matrix)
        if self.score_matrix is not None:
            totals = [0.0, 0.0, 0.0]
            for home, row in enumerate(self.score_matrix):
                for away, probability in enumerate(row):
                    totals[0 if home > away else 1 if home == away else 2] += probability
            if any(not math.isclose(total, p, rel_tol=0, abs_tol=TOLERANCE) for total, p in zip(totals, values)):
                raise ValueError("score matrix and 1X2 probabilities disagree")
        return self


class ExpectedGoals(Contract):
    home: NonNegative
    away: NonNegative


class CorePrediction(Contract):
    match_id: Identifier
    prediction_snapshot_id: Identifier
    prediction_time: UTCTime
    validation_status: Literal["UNVALIDATED", "DEVELOPMENT_VALIDATED_DATE_SAFE",
                               "FINAL_HOLDOUT_VALIDATED"] = "UNVALIDATED"
    source_prediction_ids: tuple[Identifier, ...] = ()
    source_snapshot_ids: tuple[Identifier, ...] = ()
    meta_artifact_id: Identifier | None = None
    base_model_predictions: tuple[ModelPrediction, ...] = ()
    external_predictions: tuple[ModelPrediction, ...] = ()
    market_probabilities: dict[str, ProbabilityVector] = Field(default_factory=dict)
    meta_raw_probability: ProbabilityVector | None = None
    calibrated_probability: ProbabilityVector | None = None
    # Signed adjustments have no calculation semantics in phase 2.
    context_adjustment: dict[str, float] | None = None
    lineup_adjustment: dict[str, float] | None = None
    final_core_probability: ProbabilityVector | None = None
    final_score_matrix: tuple[tuple[Probability, ...], ...] | None = None
    expected_goals: ExpectedGoals | None = None
    market_edge: float | None = None
    confidence: str | None = None
    URS: float | None = None
    MDI: float | None = None
    rating: str | None = None
    execution_status: ExecutionStatus = ExecutionStatus.SKIPPED
    reason: str | None = "NOT_IMPLEMENTED"

    @model_validator(mode="after")
    def lineage(self):
        for prediction in self.base_model_predictions + self.external_predictions:
            if (prediction.match_id, prediction.prediction_snapshot_id, prediction.prediction_time) != (
                self.match_id, self.prediction_snapshot_id, self.prediction_time
            ):
                raise ValueError("core has mixed snapshot lineage")
        if any(p.model_id == "OPTA_SUPERCOMPUTER_LIKE" for p in self.base_model_predictions):
            raise ValueError("simulation engine is not a base-model signal")
        if any(p.implementation_type == ImplementationType.EXTERNAL for p in self.base_model_predictions):
            raise ValueError("external output belongs in external_predictions")
        if any(p.implementation_type != ImplementationType.EXTERNAL for p in self.external_predictions):
            raise ValueError("external predictions must be EXTERNAL")
        if self.execution_status == ExecutionStatus.SUCCESS:
            if self.final_core_probability is None or self.calibrated_probability is None or self.meta_raw_probability is None:
                raise ValueError("successful CORE requires full probability stages")
        else:
            computed = (self.meta_raw_probability, self.calibrated_probability, self.context_adjustment,
                        self.lineup_adjustment, self.final_core_probability, self.final_score_matrix,
                        self.expected_goals, self.market_edge, self.confidence, self.URS, self.MDI, self.rating)
            if any(item is not None for item in computed) or not self.reason:
                raise ValueError("unexecuted CORE cannot contain fabricated outputs")
        validate_matrix(self.final_score_matrix)
        if self.final_score_matrix is not None:
            totals = [0.0, 0.0, 0.0]
            for home, row in enumerate(self.final_score_matrix):
                for away, probability in enumerate(row):
                    totals[0 if home > away else 1 if home == away else 2] += probability
            vector = self.final_core_probability
            if vector is None or any(abs(a - b) > TOLERANCE for a, b in zip(totals, (vector.p_home, vector.p_draw, vector.p_away))):
                raise ValueError("CORE matrix and probabilities disagree")
        return self
