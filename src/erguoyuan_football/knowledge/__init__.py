"""Offline, deterministic football identity knowledge for Phase 13.1."""

from erguoyuan_football.knowledge.competitions.competition_resolver import (
           CompetitionResolver,
)
from erguoyuan_football.knowledge.match_identity import (
           GlobalKnowledgeResolver,
           KnowledgeMatchIdentity,
)
from erguoyuan_football.knowledge.teams.team_resolver import TeamResolver

__all__ = ("CompetitionResolver", "GlobalKnowledgeResolver", "KnowledgeMatchIdentity",
           "TeamResolver")
