"""Exact canonical alias matching with collision rejection."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable

from erguoyuan_football.knowledge.teams.team_schema import TeamIdentity


def normalize_alias(value: str) -> str:
    """NFKC/case-fold then remove only whitespace and dash variants."""
    folded = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"[\s\-\u2010-\u2015]+", "", folded)


def normalize_club_alias(value: str) -> str:
    """Normalize typography and edge club designators, retaining squad markers."""
    folded = unicodedata.normalize("NFKC", value).casefold().strip()
    folded = re.sub(r"^(?:f\.?c\.?|c\.?f\.?|a\.?f\.?c\.?|s\.?c\.?)\s+", "", folded)
    folded = re.sub(r"\s+(?:f\.?c\.?|c\.?f\.?|a\.?f\.?c\.?|s\.?c\.?)$", "", folded)
    return re.sub(r"[\s.()\-\u2010-\u2015]+", "", folded)


class AliasMatcher:
    """An alias resolves only when it maps to exactly one canonical team."""

    def __init__(self, teams: Iterable[TeamIdentity]):
        self._teams = {team.team_id: team for team in teams}
        index: dict[str, set[str]] = defaultdict(set)
        for team in self._teams.values():
            for alias in (team.official_name, *team.aliases):
                keys = {normalize_alias(alias)}
                if team.entity_type == "CLUB":
                    keys.add(normalize_club_alias(alias))
                for key in keys:
                    if key:
                        index[key].add(team.team_id)
        self._index = dict(index)

    def candidate_ids(self, name: str) -> tuple[str, ...]:
        """Expose all exact normalized matches for ambiguity auditing."""
        ids = set(self._index.get(normalize_alias(name), set()))
        ids.update(self._index.get(normalize_club_alias(name), set()))
        return tuple(sorted(ids))

    def match(self, name: str) -> TeamIdentity | None:
        """Return a unique identity, or None for unknown/colliding aliases."""
        ids = self.candidate_ids(name)
        return self._teams[ids[0]] if len(ids) == 1 else None
