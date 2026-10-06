"""Lineup evidence boundary with strict pre-kickoff and retrieval-time checks."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Protocol

from erguoyuan_football.context.schemas import Availability, LineupEvidenceSnapshot
from erguoyuan_football.context.temporal import ContextTemporalPolicy


class LineupProvider(Protocol):
    """Authorized provider protocol; providers must preserve source timestamps."""

    def get_evidence(self, match_id: str, as_of_time: datetime) -> tuple[LineupEvidenceSnapshot, ...]: ...


class UnavailableLineupProvider:
    """Explicit unavailable provider until a licensed historical source is configured."""

    def get_evidence(self, match_id: str, as_of_time: datetime) -> tuple[LineupEvidenceSnapshot, ...]:
        return ()


class LineupEvidenceLayer:
    def evaluate(self, *, match_id: str, prediction_time: datetime,
                 kickoff_time: datetime | None,
                 evidence: Iterable[LineupEvidenceSnapshot]
                 ) -> tuple[tuple[LineupEvidenceSnapshot, ...], tuple[str, ...]]:
        accepted: list[LineupEvidenceSnapshot] = []
        rejected: list[str] = []
        for item in evidence:
            try:
                ContextTemporalPolicy.require_lineup(item, target_match_id=match_id,
                    prediction_time=prediction_time, kickoff_time=kickoff_time)
            except ValueError as error:
                rejected.append(str(error))
            else:
                accepted.append(item)
        return tuple(accepted), tuple(rejected)

    @staticmethod
    def availability(evidence: Iterable[LineupEvidenceSnapshot]) -> Availability:
        return Availability.AVAILABLE if tuple(evidence) else Availability.UNAVAILABLE
