"""Conservative context adjustment in centered logit space, with a no-fit default."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Protocol

from erguoyuan_football.context.schemas import (
    AdjustmentLedger,
    ContextAdjustedCoreProbability,
    ContextFeatureVector,
)
from erguoyuan_football.contracts.common import now
from erguoyuan_football.contracts.predictions import ProbabilityVector
from erguoyuan_football.ml.schemas import stable_hash


class TrainedContextResidualModel(Protocol):
    """A fitted, chronologically validated artifact may provide learned logit deltas."""

    model_id: str
    model_version: str
    trained_until: datetime
    artifact_verified: bool

    def predict_delta_logits(self, features: ContextFeatureVector) -> tuple[float, float, float]: ...


def probability_logits(probabilities: tuple[float, float, float], *, epsilon: float
                       ) -> tuple[float, float, float]:
    """Compute stable centered log-ratio coordinates for H/D/A probabilities."""
    if epsilon <= 0 or epsilon >= 1 / 3:
        raise ValueError("CONTEXT_LOGIT_EPSILON_INVALID")
    ProbabilityVector(p_home=probabilities[0], p_draw=probabilities[1], p_away=probabilities[2])
    values = [math.log(max(value, epsilon)) for value in probabilities]
    center = sum(values) / 3
    return tuple(value - center for value in values)  # type: ignore[return-value]


def softmax_logits(logits: tuple[float, float, float]) -> tuple[float, float, float]:
    if any(not math.isfinite(value) for value in logits):
        raise ValueError("CONTEXT_LOGITS_NONFINITE")
    maximum = max(logits)
    exponentials = [math.exp(value - maximum) for value in logits]
    total = sum(exponentials)
    return tuple(value / total for value in exponentials)  # type: ignore[return-value]


class ContextAdjustmentEngine:
    """Preserve base probabilities unless a verified trained residual model exists."""

    def __init__(self, *, logit_epsilon: float = 1e-12) -> None:
        self.logit_epsilon = logit_epsilon

    def evaluate(self, *, prediction_id: str, prediction_snapshot_id: str, match_id: str,
                 base: tuple[float, float, float], features: ContextFeatureVector,
                 tournament_adjustment_enabled: bool = False,
                 lineup_adjustment_enabled: bool = False,
                 context_model: TrainedContextResidualModel | None = None,
                 lineup_model_id: str = "UNAVAILABLE",
                 lineup_model_version: str = "NOT_TRAINED"
                 ) -> tuple[ContextAdjustedCoreProbability, AdjustmentLedger, tuple[str, ...]]:
        base_vector = ProbabilityVector(p_home=base[0], p_draw=base[1], p_away=base[2])
        base_logits = probability_logits(base, epsilon=self.logit_epsilon)
        context_delta = (0.0, 0.0, 0.0)
        lineup_delta = (0.0, 0.0, 0.0)
        tournament = base
        lineup = base
        final = base
        applied_context = False
        reasons: list[str] = []
        model_id = "CONTEXT_RESIDUAL_V1"
        model_version = "NOT_TRAINED_INSUFFICIENT_PIT_CONTEXT"
        if context_model is None or not context_model.artifact_verified:
            reasons.append("NO_ADJUSTMENT_DUE_TO_INSUFFICIENT_EVIDENCE")
        elif tournament_adjustment_enabled:
            context_delta = context_model.predict_delta_logits(features)
            tournament_logits = tuple(a + b for a, b in zip(base_logits, context_delta, strict=True))
            tournament = softmax_logits((tournament_logits[0], tournament_logits[1],
                                          tournament_logits[2]))
            applied_context = True
            model_id, model_version = context_model.model_id, context_model.model_version
        if lineup_adjustment_enabled:
            # A separate learned and verified lineup model is not configured in this release.
            reasons.append("LINEUP_ADJUSTMENT_MODEL_UNAVAILABLE")
        else:
            lineup_delta = (0.0, 0.0, 0.0)
        lineup = tournament
        final = lineup
        tournament_vector = ProbabilityVector(p_home=tournament[0], p_draw=tournament[1], p_away=tournament[2])
        lineup_vector = ProbabilityVector(p_home=lineup[0], p_draw=lineup[1], p_away=lineup[2])
        final_vector = ProbabilityVector(p_home=final[0], p_draw=final[1], p_away=final[2])
        result = ContextAdjustedCoreProbability(
            prediction_id=prediction_id, prediction_snapshot_id=prediction_snapshot_id,
            match_id=match_id, base_p_home=base_vector.p_home, base_p_draw=base_vector.p_draw,
            base_p_away=base_vector.p_away, tournament_p_home=tournament_vector.p_home,
            tournament_p_draw=tournament_vector.p_draw, tournament_p_away=tournament_vector.p_away,
            lineup_p_home=lineup_vector.p_home, lineup_p_draw=lineup_vector.p_draw,
            lineup_p_away=lineup_vector.p_away, final_p_home=final_vector.p_home,
            final_p_draw=final_vector.p_draw, final_p_away=final_vector.p_away,
            tournament_adjustment_applied=applied_context,
            lineup_adjustment_applied=False, context_model_id=model_id,
            context_model_version=model_version, lineup_model_id=lineup_model_id,
            lineup_model_version=lineup_model_version, mai=features.mai,
            rotation_risk=features.rotation_risk,
            lineup_strength_delta=features.lineup_strength_delta,
            context_uncertainty=features.context_uncertainty,
            evidence_quality="DEVELOPMENT_ONLY",
            created_at=datetime.now(UTC),
        )
        final_logits = (base_logits if not applied_context else
                        probability_logits(final, epsilon=self.logit_epsilon))
        ledger = AdjustmentLedger(
            ledger_id=stable_hash((match_id, prediction_snapshot_id, "PHASE10_CONTEXT"))[:32],
            match_id=match_id, prediction_snapshot_id=prediction_snapshot_id,
            base_probability={"home": base[0], "draw": base[1], "away": base[2]},
            tournament_probability={"home": tournament[0], "draw": tournament[1], "away": tournament[2]},
            lineup_probability={"home": lineup[0], "draw": lineup[1], "away": lineup[2]},
            final_probability={"home": final[0], "draw": final[1], "away": final[2]},
            base_logits=base_logits, context_delta_logits=context_delta,
            lineup_delta_logits=lineup_delta, final_logits=final_logits,
            context_features=features.model_dump(mode="json"),
            evidence_ids=tuple(sorted({source for ids in features.lineage.values() for source in ids})),
            rule_versions=features.rule_versions, reason_codes=tuple(dict.fromkeys(reasons)),
            models={"context": model_id, "lineup": lineup_model_id},
            adjustment_applied=applied_context, created_at=now(),
        )
        return result, ledger, tuple(dict.fromkeys(reasons))
