"""Source-backed, time-bound injury evidence; missing feeds remain unavailable."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

from erguoyuan_football.context.schemas import Availability, InjuryEvidence
from erguoyuan_football.context.temporal import ContextTemporalPolicy


class InjuryEvidenceLayer:
    def evaluate(self, *, match_id: str, prediction_time: datetime,
                 evidence: Iterable[InjuryEvidence]
                 ) -> tuple[tuple[InjuryEvidence, ...], tuple[str, ...]]:
        accepted: list[InjuryEvidence] = []
        rejected: list[str] = []
        for item in evidence:
            try:
                ContextTemporalPolicy.require_injury(item, target_match_id=match_id,
                                                       prediction_time=prediction_time)
            except ValueError as error:
                rejected.append(str(error))
            else:
                accepted.append(item)
        return tuple(accepted), tuple(rejected)

    @staticmethod
    def availability(evidence: Iterable[InjuryEvidence]) -> Availability:
        return Availability.AVAILABLE if tuple(evidence) else Availability.UNAVAILABLE
