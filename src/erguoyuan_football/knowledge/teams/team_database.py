"""Small, versioned offline seed from the Phase 13.1 user specification."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

from erguoyuan_football.knowledge.teams.team_schema import TeamIdentity

TEAM_KNOWLEDGE_VERSION = "PHASE13_1_USER_SPEC_V1"
TEAM_KNOWLEDGE_SOURCE = "USER_SPECIFICATION"

_SEED = (
    TeamIdentity("ENG_ARS", "Arsenal FC", "England", "UEFA",
                 ["Arsenal", "Arsenal FC", "阿森纳"], entity_type="CLUB"),
    TeamIdentity("ENG_LIV", "Liverpool FC", "England", "UEFA",
                 ["Liverpool", "Liverpool FC", "利物浦"], entity_type="CLUB"),
    TeamIdentity("OFC_COK", "Cook Islands", "Cook Islands", "OFC",
                 ["Cook Islands", "库克群岛", "COK"], entity_type="NATIONAL",
                 gender="MEN"),
    TeamIdentity("OFC_TAH", "Tahiti", "Tahiti", "OFC",
                 ["Tahiti", "塔西提", "TAH"], entity_type="NATIONAL",
                 gender="MEN"),
)


class TeamDatabase:
    """Read-only process-local catalog; it does not migrate or modify DuckDB."""

    source = TEAM_KNOWLEDGE_SOURCE
    version = TEAM_KNOWLEDGE_VERSION

    def __init__(self, teams: Iterable[TeamIdentity] | None = None):
        records = tuple(teams) if teams is not None else _SEED
        ids = [team.team_id for team in records]
        if len(ids) != len(set(ids)):
            raise ValueError("DUPLICATE_TEAM_ID")
        self._records = {team.team_id: replace(team, aliases=list(team.aliases))
                         for team in records}

    def all(self) -> tuple[TeamIdentity, ...]:
        """Return defensive copies so callers cannot mutate catalog aliases."""
        return tuple(replace(team, aliases=list(team.aliases))
                     for team in self._records.values())

    def get(self, team_id: str) -> TeamIdentity | None:
        """Look up a canonical ID without fuzzy matching."""
        team = self._records.get(team_id)
        return replace(team, aliases=list(team.aliases)) if team is not None else None
