"""Typed discovery candidates with no automatic fuzzy identity selection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from erguoyuan_football.knowledge.entities.competition_hint_parser import (
    ParsedCompetitionHint,
)
from erguoyuan_football.knowledge.entities.team_entity_parser import (
    ParsedTeamEntityHint,
)


@dataclass(frozen=True)
class TeamDiscoveryCandidate:
    """Provider-observed team and its explicit identity dimensions."""

    provider_id: str
    provider_team_id: str
    official_name: str
    country: str | None
    federation: str | None
    entity_type: str
    gender: str
    age_group: str
    squad_level: str
    aliases: tuple[str, ...]
    source_url: str
    source_tier: int
    fetched_at: datetime
    confidence: str
    evidence_id: str


class TeamDiscoveryProvider(Protocol):
    """Only adapters declaring TEAM_DISCOVERY may satisfy this protocol."""

    provider_id: str
    supports_team_discovery: bool

    def can_cover(self, hint: ParsedTeamEntityHint,
                  competition: ParsedCompetitionHint | None = None) -> bool:
        """Declare the actual source scope before network access."""

    def discover(self, hint: ParsedTeamEntityHint,
                 competition: ParsedCompetitionHint | None = None
                 ) -> tuple[TeamDiscoveryCandidate, ...]:
        """Return actual provider candidates, never model-generated entities."""
