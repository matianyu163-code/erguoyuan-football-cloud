"""Identity resolution result; never implies a verified fixture or prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from erguoyuan_football.app.input.match_input import MatchInputParserV2, MatchRequest
from erguoyuan_football.knowledge.competitions.competition_resolver import (
    CompetitionResolver,
)
from erguoyuan_football.knowledge.competitions.competition_schema import (
    CompetitionIdentity,
)
from erguoyuan_football.knowledge.entities.historical_club_directory import (
    load_historical_clubs,
)
from erguoyuan_football.knowledge.entities.universal_team_resolver import (
    EntityResolution,
    UniversalTeamResolver,
)
from erguoyuan_football.knowledge.entities.verified_entity_store import (
    VerifiedEntityStore,
)
from erguoyuan_football.knowledge.teams.alias_matcher import normalize_club_alias
from erguoyuan_football.knowledge.teams.team_database import TeamDatabase
from erguoyuan_football.knowledge.teams.team_resolver import TeamResolver
from erguoyuan_football.knowledge.teams.team_schema import TeamIdentity


@dataclass(frozen=True)
class KnowledgeMatchIdentity:
    """Resolved participants without a fixture ID, kickoff, or model inputs."""

    home_team: TeamIdentity | None
    away_team: TeamIdentity | None
    competition: CompetitionIdentity | None
    status: str
    home_resolution: EntityResolution | None = None
    away_resolution: EntityResolution | None = None
    warnings: tuple[str, ...] = ()


class GlobalKnowledgeResolver:
    """Compose the universal team resolver and competition aliases offline."""

    def __init__(
        self,
        team_resolver: TeamResolver | None = None,
        competition_resolver: CompetitionResolver | None = None,
        entity_resolver: UniversalTeamResolver | None = None,
        entity_store: VerifiedEntityStore | None = None,
    ) -> None:
        if team_resolver is None:
            root = Path(__file__).resolve().parents[3]
            seed = TeamDatabase().all()
            historical = load_historical_clubs(root / "data" / "football.duckdb")
            seed_keys = {normalize_club_alias(alias) for team in seed
                         for alias in (team.official_name, *team.aliases)}
            historical = tuple(team for team in historical
                               if normalize_club_alias(team.official_name) not in seed_keys)
            known = {team.team_id: team for team in (*seed, *historical)}
            team_resolver = TeamResolver(TeamDatabase(known.values()))
        self.teams = team_resolver
        self.entity_resolver = entity_resolver or UniversalTeamResolver(
            static=self.teams, store=entity_store)
        self.competitions = competition_resolver or CompetitionResolver()

    def resolve_names(
        self,
        home_name: str,
        away_name: str,
        competition_query: str | None = None,
        *,
        allow_discovery: bool = False,
    ) -> KnowledgeMatchIdentity:
        """Resolve identities only; require a separate fixture verification step."""
        home_resolution = self.entity_resolver.resolve(
            home_name, allow_discovery=allow_discovery)
        away_resolution = self.entity_resolver.resolve(
            away_name, allow_discovery=allow_discovery)
        home, away = home_resolution.identity, away_resolution.identity
        competition = (self.competitions.resolve(competition_query)
                       if competition_query else None)
        if home is None or away is None:
            status = "TEAM_NOT_FOUND"
        elif home.team_id == away.team_id:
            status = "INVALID_MATCH"
        elif competition_query and competition is None:
            status = "COMPETITION_NOT_FOUND"
        else:
            status = "IDENTITIES_RESOLVED_NO_FIXTURE"
        warnings = (("DIFFERENT_AGE_GROUPS_CONTEXT_REVIEW",)
                    if home is not None and away is not None
                    and home.age_group != away.age_group else ())
        return KnowledgeMatchIdentity(home, away, competition, status,
                                      home_resolution, away_resolution, warnings)

    def resolve_request(self, request: MatchRequest) -> KnowledgeMatchIdentity:
        """Use the unchanged V2 request fields without verifying a fixture."""
        if (request.validation_status != "VALID" or request.home_team is None
                or request.away_team is None):
            return KnowledgeMatchIdentity(None, None, None,
                                          request.error_code or "MATCH_SYNTAX_INVALID")
        return self.resolve_names(request.home_team, request.away_team, request.competition)

    def resolve_text(self, text: str) -> KnowledgeMatchIdentity:
        """Parse one match text with INPUT_PARSER_V2, then resolve identities."""
        return self.resolve_request(MatchInputParserV2().parse(text))
