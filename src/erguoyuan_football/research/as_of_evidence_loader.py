"""Read persisted evidence strictly as it was knowable at a historical cutoff."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from erguoyuan_football.web_research.evidence.evidence_schema import EvidenceRecord
from erguoyuan_football.web_research.evidence.evidence_store import EvidenceStore


class ReplayEvidenceUnavailable(ValueError):
    """Raised when no persisted, cutoff-valid fixture evidence exists."""


@dataclass(frozen=True)
class AsOfEvidenceBundle:
    """The evidence records that were persisted and observable by the requested time."""

    match_id: str
    prediction_cutoff: datetime
    fixture: tuple[EvidenceRecord, ...]
    historical_results: tuple[EvidenceRecord, ...]
    other_evidence: tuple[EvidenceRecord, ...]


class AsOfEvidenceLoader:
    """Never refetch or relabel present-day evidence into a historical replay."""

    def __init__(self, store: EvidenceStore) -> None:
        self.store = store

    def load_as_of(self, match_id: str, prediction_cutoff: datetime) -> AsOfEvidenceBundle:
        """Return only evidence already stored by cutoff or fail explicitly."""
        if not match_id:
            raise ValueError("MATCH_ID_REQUIRED")
        if prediction_cutoff.tzinfo is None or prediction_cutoff.utcoffset() is None:
            raise ValueError("UTC_CUTOFF_REQUIRED")
        key = f"MATCH:{match_id}"
        records = tuple(record for record in self.store.available_at(prediction_cutoff)
                        if record.match_key == key)
        fixtures = tuple(record for record in records if record.data_type == "FIXTURE")
        if not fixtures:
            raise ReplayEvidenceUnavailable("REPLAY_EVIDENCE_UNAVAILABLE")
        history = tuple(record for record in records
                        if record.data_type == "HISTORICAL_RESULT")
        other = tuple(record for record in records
                      if record.data_type not in {"FIXTURE", "HISTORICAL_RESULT"})
        return AsOfEvidenceBundle(match_id, prediction_cutoff, fixtures, history, other)
