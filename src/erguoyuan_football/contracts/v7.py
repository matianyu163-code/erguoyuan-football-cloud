"""Stable V7 internal contract for audited match-level model outputs."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from erguoyuan_football.contracts.common import (
    Contract,
    Identifier,
    Probability,
    UTCTime,
)


class V7ModelStatus(StrEnum):
    """Per-model execution state; data status is not substituted for execution."""

    EXECUTED = "EXECUTED"
    BLOCKED = "BLOCKED"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"


class V7Probability(Contract):
    """Validated 1X2 probability names used by V7."""

    home: Probability
    draw: Probability
    away: Probability

    @model_validator(mode="after")
    def check_sum(self) -> V7Probability:
        if not math.isclose(self.home + self.draw + self.away, 1,
                            rel_tol=0, abs_tol=1e-8):
            raise ValueError("V7_PROBABILITIES_MUST_SUM_TO_ONE")
        return self


class V7TeamPair(Contract):
    """Canonical participant identities and display names."""

    home_id: Identifier
    home_name: Identifier
    away_id: Identifier
    away_name: Identifier


class V7Evidence(Contract):
    """Source reference required to audit any emitted probability."""

    provider: Identifier
    source: Identifier
    observed_at: UTCTime
    evidence_id: Identifier


class V7PredictionContract(Contract):
    """Fixed required field order for a V7 match-level prediction record."""

    match: Identifier
    competition: Identifier
    teams: V7TeamPair
    probabilities: V7Probability | None
    model_status: dict[Identifier, V7ModelStatus] = Field(min_length=1)
    confidence: Literal["LOW", "MEDIUM", "HIGH", "VERY_HIGH", "UNAVAILABLE"]
    evidence: tuple[V7Evidence, ...] = ()
    timestamp: UTCTime

    @model_validator(mode="after")
    def check_evidence_and_null_outputs(self) -> V7PredictionContract:
        has_success = any(status in {V7ModelStatus.EXECUTED, V7ModelStatus.DEGRADED}
                          for status in self.model_status.values())
        if has_success and self.probabilities is None:
            raise ValueError("V7_EXECUTED_MODEL_REQUIRES_PROBABILITIES")
        if not has_success and self.probabilities is not None:
            raise ValueError("V7_BLOCKED_OR_FAILED_MODELS_CANNOT_HAVE_PROBABILITIES")
        if self.probabilities is not None and not self.evidence:
            raise ValueError("V7_PROBABILITY_REQUIRES_EVIDENCE")
        if any(item.observed_at > self.timestamp for item in self.evidence):
            raise ValueError("V7_EVIDENCE_AFTER_OUTPUT_TIMESTAMP")
        if self.probabilities is None and self.confidence != "UNAVAILABLE":
            raise ValueError("V7_UNAVAILABLE_OUTPUT_REQUIRES_UNAVAILABLE_CONFIDENCE")
        return self
