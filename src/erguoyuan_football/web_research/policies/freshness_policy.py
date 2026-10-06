"""Configurable per-data-type freshness and point-in-time checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from erguoyuan_football.web_research.evidence.evidence_schema import EvidenceRecord
from erguoyuan_football.web_research.time_utils import parse_utc

DEFAULT_TTL_SECONDS = {
    "FIXTURE": 86400, "TEAM_STATS": 86400, "STANDINGS": 21600,
    "ELO": 86400, "XG": 86400, "NEWS": 21600,
    "INJURY": 21600, "ODDS": 900, "LINEUP": 900, "RESULT": 86400,
}


@dataclass(frozen=True)
class FreshnessPolicy:
    """A source-age limit, separate from cache age and model eligibility."""

    ttl_seconds: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_TTL_SECONDS))

    def ttl(self, data_type: str) -> timedelta:
        """Return a configured positive TTL for a supported data type."""
        seconds = self.ttl_seconds.get(data_type)
        if seconds is None or seconds <= 0:
            raise ValueError("FRESHNESS_TTL_NOT_CONFIGURED")
        return timedelta(seconds=seconds)

    def is_fresh(self, data_type: str, observed_at: datetime, now: datetime) -> bool:
        """Reject future observations and values beyond the type-specific TTL."""
        if observed_at.tzinfo is None or now.tzinfo is None:
            raise ValueError("UTC_TIMESTAMP_REQUIRED")
        age = now - observed_at
        return timedelta(0) <= age <= self.ttl(data_type)


def is_valid_for_cutoff(evidence: EvidenceRecord, prediction_cutoff: datetime) -> bool:
    """Require every known source, retrieval and observation time by cutoff."""
    if prediction_cutoff.tzinfo is None or prediction_cutoff.utcoffset() is None:
        raise ValueError("UTC_TIMESTAMP_REQUIRED")
    times = [parse_utc(evidence.fetched_time), parse_utc(evidence.as_of_time)]
    if evidence.published_time is not None:
        times.append(parse_utc(evidence.published_time))
    if evidence.observed_time is not None:
        times.append(parse_utc(evidence.observed_time))
    return max(times) <= prediction_cutoff
