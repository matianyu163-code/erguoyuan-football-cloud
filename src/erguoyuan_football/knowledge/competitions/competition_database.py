"""Versioned, process-local competition seed; no database migration."""

from __future__ import annotations

from collections.abc import Iterable

from erguoyuan_football.knowledge.competitions.competition_schema import (
    CompetitionIdentity,
)

COMPETITION_KNOWLEDGE_VERSION = "PHASE13_1_USER_SPEC_V1"
COMPETITION_KNOWLEDGE_SOURCE = "USER_SPECIFICATION"

_SEED = (
    CompetitionIdentity("FIFA_WORLD_CUP", "FIFA World Cup", "FIFA", "WORLD",
                        "INTERNATIONAL"),
    CompetitionIdentity("UEFA_CL", "UEFA Champions League", "UEFA", "EUROPE",
                        "CLUB"),
    CompetitionIdentity("OFC_NATIONS", "OFC Nations Cup", "OFC", "OCEANIA",
                        "INTERNATIONAL"),
)

_ALIASES = {
    "FIFA_WORLD_CUP": ("FIFA World Cup", "World Cup", "世界杯"),
    "UEFA_CL": ("UEFA Champions League", "Champions League", "欧冠"),
    "OFC_NATIONS": ("OFC Nations Cup", "大洋洲国家杯"),
}


class CompetitionDatabase:
    """Catalog of known competition IDs and explicitly registered aliases."""

    source = COMPETITION_KNOWLEDGE_SOURCE
    version = COMPETITION_KNOWLEDGE_VERSION

    def __init__(
        self,
        competitions: Iterable[CompetitionIdentity] | None = None,
        aliases: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        records = tuple(competitions) if competitions is not None else _SEED
        ids = [competition.competition_id for competition in records]
        if len(ids) != len(set(ids)):
            raise ValueError("DUPLICATE_COMPETITION_ID")
        self._records = {competition.competition_id: competition for competition in records}
        configured = aliases if aliases is not None else (
            _ALIASES if competitions is None else {}
        )
        if any(competition_id not in self._records for competition_id in configured):
            raise ValueError("UNKNOWN_COMPETITION_ALIAS_TARGET")
        self._aliases = {key: tuple(values) for key, values in configured.items()}

    def all(self) -> tuple[CompetitionIdentity, ...]:
        """List only the catalog's registered competition identities."""
        return tuple(self._records.values())

    def get(self, competition_id: str) -> CompetitionIdentity | None:
        """Look up a canonical competition ID."""
        return self._records.get(competition_id)

    def aliases_for(self, competition_id: str) -> tuple[str, ...]:
        """Return explicitly registered names for one competition."""
        return self._aliases.get(competition_id, ())
