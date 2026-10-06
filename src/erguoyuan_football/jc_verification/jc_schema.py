"""Stable data contracts for JC fixture evidence and verification."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum


class JCStatus(StrEnum):
    """Evidence-backed classification of a requested match."""

    CONFIRMED = "CONFIRMED"
    LIKELY = "LIKELY"
    USER_CONFIRMED = "USER_CONFIRMED"
    UNKNOWN = "UNKNOWN"
    NOT_JC = "NOT_JC"


@dataclass(frozen=True)
class JCMatch:
    """One fixture returned by an explicitly registered JC provider."""

    match_id: str
    home_team: str
    away_team: str
    competition: str
    kickoff_time: datetime
    market_types: tuple[str, ...]
    source: str
    evidence_id: str
    provider_tier: int = 1
    competition_id: str | None = None

    def __post_init__(self) -> None:
        if not all((self.match_id, self.home_team, self.away_team, self.competition,
                    self.source, self.evidence_id)):
            raise ValueError("JC_MATCH_IDENTITY_AND_EVIDENCE_REQUIRED")
        if self.kickoff_time.tzinfo is None or self.kickoff_time.utcoffset() is None:
            raise ValueError("JC_KICKOFF_TIMEZONE_REQUIRED")
        if not 1 <= self.provider_tier <= 3:
            raise ValueError("JC_PROVIDER_TIER_INVALID")


@dataclass(frozen=True)
class JCMatchQuery:
    """Canonicalized request identity; names are retained for audit only."""

    home_team_id: str
    home_team_name: str
    away_team_id: str
    away_team_name: str
    competition_id: str | None
    competition_name: str | None
    kickoff_time: datetime | None

    def __post_init__(self) -> None:
        if not self.home_team_id or not self.away_team_id:
            raise ValueError("JC_QUERY_TEAM_IDS_REQUIRED")
        if self.home_team_id == self.away_team_id:
            raise ValueError("JC_QUERY_TEAMS_MUST_DIFFER")
        if self.kickoff_time is not None and self.kickoff_time.utcoffset() is None:
            raise ValueError("JC_QUERY_KICKOFF_TIMEZONE_REQUIRED")

    @property
    def match_date(self) -> date | None:
        """Use UTC calendar date consistently for provider/cache lookup."""
        return self.kickoff_time.astimezone(UTC).date() if self.kickoff_time else None


@dataclass(frozen=True)
class JCVerificationResult:
    """Auditable JC status, with evidence required for positive confirmation."""

    status: JCStatus
    confidence_score: float
    competition_id: str | None
    competition_name: str | None
    kickoff_time: datetime | None
    source: str | None
    evidence_id: str | None
    checked_at: datetime
    reason: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError("JC_CONFIDENCE_OUT_OF_RANGE")
        if self.checked_at.tzinfo is None or self.checked_at.utcoffset() is None:
            raise ValueError("JC_CHECKED_AT_TIMEZONE_REQUIRED")
        if self.kickoff_time is not None and self.kickoff_time.utcoffset() is None:
            raise ValueError("JC_KICKOFF_TIMEZONE_REQUIRED")
        if self.status in {JCStatus.CONFIRMED, JCStatus.LIKELY,
                           JCStatus.USER_CONFIRMED} and not (
            self.source and self.evidence_id
        ):
            raise ValueError("POSITIVE_JC_STATUS_REQUIRES_SOURCE_EVIDENCE")
        if self.status == JCStatus.UNKNOWN and self.confidence_score != 0:
            raise ValueError("UNKNOWN_STATUS_CONFIDENCE_MUST_BE_ZERO")

    @classmethod
    def unknown(cls, checked_at: datetime, reason: str) -> JCVerificationResult:
        """Create an explicit unknown result without inventing data."""
        return cls(JCStatus.UNKNOWN, 0.0, None, None, None, None, None,
                   checked_at, reason)
