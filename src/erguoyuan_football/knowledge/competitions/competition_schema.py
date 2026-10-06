"""Canonical competition identity, independent of individual fixtures."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompetitionIdentity:
    """Offline identity record supplied by the Phase 13.1 specification."""

    competition_id: str
    name: str
    federation: str
    country: str
    level: str

    def __post_init__(self) -> None:
        if not all((self.competition_id, self.name, self.federation,
                    self.country, self.level)):
            raise ValueError("COMPETITION_IDENTITY_FIELDS_REQUIRED")
