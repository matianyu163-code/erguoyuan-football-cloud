"""Preserve conditional score patterns while matching calibrated outcome masses."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from hashlib import sha256

from erguoyuan_football.contracts.predictions import ProbabilityVector
from erguoyuan_football.models.score_matrix import ScoreMatrix


@dataclass(frozen=True)
class ReconciledMatrix:
    reference_matrix: ScoreMatrix
    reference_1x2: ProbabilityVector
    effective_core_1x2: ProbabilityVector
    scaling_factors: dict[str, float]
    final_matrix: ScoreMatrix
    reference_matrix_hash: str
    final_matrix_hash: str
    reconciliation_status: str = "SUCCESS"


def matrix_hash(matrix: ScoreMatrix) -> str:
    """Content hash over the exact serialized matrix contract."""
    return sha256(json.dumps(matrix.model_dump(mode="json"), sort_keys=True).encode()).hexdigest()


class OutcomeMassReconciler:
    """Scale home/draw/away score regions, rejecting impossible zero-mass regions."""

    def reconcile(self, reference: ScoreMatrix, target: ProbabilityVector) -> ReconciledMatrix:
        source = reference.outcome()
        original = (source.p_home, source.p_draw, source.p_away)
        desired = (target.p_home, target.p_draw, target.p_away)
        if any(old == 0 and new > 0 for old, new in zip(original, desired, strict=True)):
            raise ValueError("RECONCILIATION_FAILED:ZERO_REFERENCE_REGION")
        factors = tuple(new / old if old else 0.0 for old, new in zip(original, desired, strict=True))
        values = tuple(tuple(float(value * factors[0 if home > away else 1 if home == away else 2])
                             for away, value in enumerate(row))
                       for home, row in enumerate(reference.values))
        final = ScoreMatrix(values=values, max_goals=reference.max_goals,
                            retained_mass=reference.retained_mass, tail_mass=reference.tail_mass)
        observed = final.outcome()
        if any(not math.isclose(a, b, abs_tol=1e-8, rel_tol=0) for a, b in zip(
                (observed.p_home, observed.p_draw, observed.p_away), desired, strict=True)):
            raise ValueError("RECONCILIATION_FAILED:MASS_MISMATCH")
        return ReconciledMatrix(reference, source, target,
            dict(zip(("HOME", "DRAW", "AWAY"), factors, strict=True)),
            final, matrix_hash(reference), matrix_hash(final))
