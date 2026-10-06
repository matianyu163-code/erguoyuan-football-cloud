"""Rotation-risk interface; no untrained heuristic is exposed as a probability."""

from __future__ import annotations

from dataclasses import dataclass

from erguoyuan_football.context.schemas import Availability, LineupEvidenceSnapshot


@dataclass(frozen=True)
class RotationRiskResult:
    probability: float | None
    availability: Availability
    model_id: str
    model_version: str
    reason: str


class RotationRiskEngine:
    """Return unavailable until a real chronological rotation model is trained."""

    def evaluate(self, *, historical_lineups: tuple[LineupEvidenceSnapshot, ...],
                 current_lineup: tuple[LineupEvidenceSnapshot, ...],
                 schedule_features_available: bool) -> RotationRiskResult:
        del historical_lineups, current_lineup, schedule_features_available
        return RotationRiskResult(None, Availability.UNAVAILABLE,
            "ROTATION_RISK_V1", "NOT_TRAINED", "ROTATION_MODEL_NOT_TRAINED")
