"""Choose the final legal probability without promoting development evidence."""

from __future__ import annotations

from dataclasses import dataclass

from erguoyuan_football.contracts.predictions import ProbabilityVector
from erguoyuan_football.output_contract.schemas import (
    AvailabilityStatus,
    CanonicalPredictionResult,
    ProbabilityStage,
)


@dataclass(frozen=True)
class EffectiveCoreProbability:
    probability: ProbabilityVector
    source_stage: str
    source_id: str
    validation_status: str
    production_status: str
    temporal_mode: str


class EffectiveCoreProbabilityResolver:
    """Reject base-model or unvalidated output; accept actual context adjustment only."""

    def resolve(self, row: CanonicalPredictionResult) -> EffectiveCoreProbability:
        if row.status != AvailabilityStatus.AVAILABLE or row.pit_status != "PASS":
            raise ValueError("CORE_PROBABILITY_UNAVAILABLE")
        if row.probability_stage == ProbabilityStage.CONTEXT_ADJUSTED_CORE:
            if row.context_status != "ADJUSTED" or row.context is None:
                raise ValueError("CONTEXT_ADJUSTMENT_NOT_VERIFIED")
        elif row.probability_stage != ProbabilityStage.FINAL_CORE_CALIBRATED:
            raise ValueError("FINAL_CALIBRATED_CORE_REQUIRED")
        if row.validation_status == "UNVALIDATED":
            raise ValueError("UNVALIDATED_CORE_PROBABILITY")
        assert row.p_home is not None and row.p_draw is not None and row.p_away is not None
        return EffectiveCoreProbability(
            ProbabilityVector(p_home=row.p_home, p_draw=row.p_draw, p_away=row.p_away),
            row.probability_stage.value, row.prediction_id,
            row.validation_status, row.production_status, row.prediction_temporal_mode,
        )
