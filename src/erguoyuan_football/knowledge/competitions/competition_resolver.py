"""Conservative competition resolution over the offline seed."""

from __future__ import annotations

import re

from erguoyuan_football.knowledge.competitions.competition_database import (
    CompetitionDatabase,
)
from erguoyuan_football.knowledge.competitions.competition_schema import (
    CompetitionIdentity,
)
from erguoyuan_football.knowledge.teams.alias_matcher import normalize_alias

_TRAILING_YEAR = re.compile(r"\s+(?:19|20)\d{2}\s*$")


class CompetitionResolver:
    """Resolve an exact registered name, optionally followed by a calendar year."""

    def __init__(self, database: CompetitionDatabase | None = None) -> None:
        self.database = database or CompetitionDatabase()
        index: dict[str, set[str]] = {}
        for competition in self.database.all():
            for alias in (competition.name, *self.database.aliases_for(
                competition.competition_id
            )):
                key = normalize_alias(alias)
                if key:
                    index.setdefault(key, set()).add(competition.competition_id)
        self._index = index

    def resolve(self, query: str) -> CompetitionIdentity | None:
        """Return one known identity; unknown or colliding names return None."""
        canonical_query = _TRAILING_YEAR.sub("", query.strip())
        candidate_ids = self._index.get(normalize_alias(canonical_query), set())
        if len(candidate_ids) != 1:
            return None
        return self.database.get(next(iter(candidate_ids)))
