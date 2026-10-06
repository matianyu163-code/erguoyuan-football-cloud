"""Immutable evidence with both publication and actual retrieval time."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from erguoyuan_football.web_research.time_utils import parse_utc


@dataclass(frozen=True)
class EvidenceRecord:
    """Unverified research evidence, never a model feature by itself."""

    evidence_id: str
    data_type: str
    value: Any
    source_id: str
    published_time: str | None
    fetched_time: str
    confidence: str
    source_url: str
    as_of_time: str
    provider_id: str = ""
    source_tier: int | None = None
    observed_time: str | None = None
    match_key: str | None = None
    claim_type: str | None = None

    def __post_init__(self) -> None:
        if not self.evidence_id or not self.data_type or not self.source_id or not self.source_url:
            raise ValueError("EVIDENCE_PROVENANCE_REQUIRED")
        if self.confidence not in {"HIGH", "MEDIUM", "LOW"}:
            raise ValueError("INVALID_CONFIDENCE")
        fetched = parse_utc(self.fetched_time)
        timestamps = [parse_utc(self.as_of_time)]
        if self.published_time is not None:
            timestamps.append(parse_utc(self.published_time))
        if self.observed_time is not None:
            timestamps.append(parse_utc(self.observed_time))
        if max(timestamps) > fetched:
            raise ValueError("EVIDENCE_TIME_AFTER_RETRIEVAL")
        if self.claim_type is not None and self.claim_type not in {"FACT", "REPORT"}:
            raise ValueError("INVALID_NEWS_CLAIM_TYPE")
