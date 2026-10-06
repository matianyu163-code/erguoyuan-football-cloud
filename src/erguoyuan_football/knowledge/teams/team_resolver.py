"""Public team identity resolver over the offline Phase 13.1 catalog."""

from __future__ import annotations

from erguoyuan_football.knowledge.teams.alias_matcher import AliasMatcher
from erguoyuan_football.knowledge.teams.team_database import TeamDatabase
from erguoyuan_football.knowledge.teams.team_schema import TeamIdentity


class TeamResolver:
    """Resolve exact aliases; never select a merely similar team."""

    def __init__(self, database: TeamDatabase | None = None):
        self.database = database or TeamDatabase()
        self.matcher = AliasMatcher(self.database.all())

    def resolve(self, name: str) -> TeamIdentity | None:
        """Return a canonical team or None when unknown/ambiguous."""
        match = self.matcher.match(name)
        return self.database.get(match.team_id) if match is not None else None
