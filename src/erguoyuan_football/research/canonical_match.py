"""Stable canonical identity for an exact, provider-verified fixture."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class CanonicalMatchIdentity:
    """Canonical teams and kickoff, retaining every provider identity used."""

    match_id: str
    home_team_id: str
    away_team_id: str
    competition_id: str | None
    kickoff_at: datetime
    provider_match_ids: tuple[tuple[str, str], ...]
    venue_type: str
    verified: bool
    verification_evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.kickoff_at.tzinfo is None or self.kickoff_at.utcoffset() is None:
            raise ValueError("CANONICAL_MATCH_KICKOFF_MUST_BE_AWARE")
        if not self.match_id or not self.home_team_id or not self.away_team_id:
            raise ValueError("CANONICAL_MATCH_IDENTITIES_REQUIRED")
        if self.home_team_id == self.away_team_id:
            raise ValueError("CANONICAL_MATCH_TEAMS_MUST_DIFFER")
        if self.verified and (not self.provider_match_ids
                              or not self.verification_evidence_ids):
            raise ValueError("VERIFIED_MATCH_REQUIRES_PROVIDER_EVIDENCE")

