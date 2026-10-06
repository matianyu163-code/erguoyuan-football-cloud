"""Auditable provider response, with source and actual retrieval time."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from erguoyuan_football.web_research.time_utils import parse_utc


@dataclass(frozen=True)
class ProviderResult:
    """Provider output; a success is evidence, never a production model input."""

    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    source_name: str = ""
    source_url: str = ""
    fetched_time: str = ""
    confidence: str = "LOW"
    as_of_time: str | None = None
    error_code: str | None = None
    provider_id: str = ""
    source_tier: int | None = None
    task_type: str = ""
    published_time: str | None = None
    observed_time: str | None = None
    is_cache_hit: bool = False
    cache_age_seconds: int | None = None
    cache_status: str = "MISS"
    error_message: str | None = None
    http_status: int | None = None
    retry_count: int = 0
    endpoint_id: str | None = None

    def __post_init__(self) -> None:
        if self.confidence not in {"HIGH", "MEDIUM", "LOW"}:
            raise ValueError("INVALID_CONFIDENCE")
        if self.cache_status not in {"MISS", "LIVE", "CACHE_FRESH", "CACHE_STALE"}:
            raise ValueError("INVALID_CACHE_STATUS")
        if self.success:
            if not self.source_name or not self.source_url or not self.fetched_time:
                raise ValueError("PROVIDER_PROVENANCE_REQUIRED")
            fetched = parse_utc(self.fetched_time)
            if self.as_of_time is None:
                raise ValueError("PROVIDER_AS_OF_TIME_REQUIRED")
            if parse_utc(self.as_of_time) > fetched:
                raise ValueError("SOURCE_TIME_AFTER_RETRIEVAL")
            if self.published_time is not None and parse_utc(self.published_time) > fetched:
                raise ValueError("PUBLICATION_AFTER_RETRIEVAL")
            if self.observed_time is not None and parse_utc(self.observed_time) > fetched:
                raise ValueError("OBSERVATION_AFTER_RETRIEVAL")
        elif self.data:
            raise ValueError("FAILED_PROVIDER_MUST_HAVE_NO_DATA")
        elif self.fetched_time:
            parse_utc(self.fetched_time)

    @property
    def fetched_at(self) -> datetime | None:
        """UTC retrieval time, preserving the Phase 13.2 string field."""
        return parse_utc(self.fetched_time) if self.fetched_time else None

    @property
    def published_at(self) -> datetime | None:
        """Source publication time, if the provider actually supplies it."""
        return parse_utc(self.published_time) if self.published_time else None
