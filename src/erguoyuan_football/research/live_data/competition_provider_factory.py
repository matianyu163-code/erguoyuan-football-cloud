"""Create a match-data provider only from a verified reviewed team directory."""

from __future__ import annotations

from dataclasses import dataclass

from erguoyuan_football.knowledge.entities.entity_fingerprint import (
    canonical_entity_id,
    entity_fingerprint,
)
from erguoyuan_football.knowledge.teams.team_database import TeamDatabase
from erguoyuan_football.knowledge.teams.team_resolver import TeamResolver
from erguoyuan_football.knowledge.teams.team_schema import TeamIdentity
from erguoyuan_football.research.live_data.openligadb import (
    OpenLigaDBConfig,
    OpenLigaDBProvider,
    TeamBinding,
)
from erguoyuan_football.research.live_data.openligadb_directory import (
    DirectoryFetch,
    OpenLigaDBDirectoryProvider,
)
from erguoyuan_football.research.provider_coverage_profile import CoverageStatus


@dataclass(frozen=True)
class BoundCompetitionProvider:
    """A match provider and exact team resolver derived from verified directory rows."""

    provider: OpenLigaDBProvider
    team_resolver: TeamResolver
    team_evidence_ids: tuple[str, ...]


def bind_verified_competition(directory: OpenLigaDBDirectoryProvider,
                              scope_id: str, team_result: DirectoryFetch
                              ) -> BoundCompetitionProvider:
    """Bind all teams exactly; incomplete or unverified directories fail closed."""
    scope = directory.scopes.get(scope_id)
    if scope is None:
        raise ValueError("PROVIDER_COVERAGE_MISSING")
    if (team_result.status != CoverageStatus.VERIFIED or not team_result.rows
            or any(row.competition.scope_id != scope_id for row in team_result.rows)):
        raise ValueError("COMPETITION_TEAM_DIRECTORY_NOT_VERIFIED")
    bindings: dict[str, TeamBinding] = {}
    identities: list[TeamIdentity] = []
    evidence_ids: list[str] = []
    for row in team_result.rows:
        if row.provider_team_id in {item.provider_team_id for item in bindings.values()}:
            raise ValueError("DUPLICATE_PROVIDER_TEAM_ID")
        fingerprint = entity_fingerprint(
            country=scope.country, entity_type=scope.entity_type,
            gender=scope.gender, age_group=scope.age_group,
            squad_level=scope.squad_level, name=row.provider_name,
            provider_id=directory.provider_id,
            provider_team_id=str(row.provider_team_id))
        team_id = canonical_entity_id(
            country=scope.country, federation=scope.federation,
            entity_type=scope.entity_type, gender=scope.gender,
            age_group=scope.age_group, fingerprint=fingerprint,
            provider_id=directory.provider_id,
            provider_team_id=str(row.provider_team_id))
        if team_id in bindings:
            raise ValueError("CANONICAL_TEAM_ID_COLLISION")
        bindings[team_id] = TeamBinding(row.provider_team_id, row.provider_name)
        identities.append(TeamIdentity(
            team_id, row.provider_name, scope.country, scope.federation,
            [row.provider_name], entity_type=scope.entity_type,
            gender=scope.gender, age_group=scope.age_group,
            squad_level=scope.squad_level,
            provider_ids={directory.provider_id: str(row.provider_team_id)},
            identity_status="PROVIDER_VERIFIED",
            verification_evidence_ids=[row.evidence_id]))
        evidence_ids.append(row.evidence_id)
    aliases = (("Bundesliga", "German Bundesliga", "德甲")
               if scope.shortcut == "bl1" else
               ("2. Bundesliga", "Bundesliga 2", "德乙")
               if scope.shortcut == "bl2" else (scope.competition_name,))
    config = OpenLigaDBConfig(
        provider_id=directory.provider_id, base_url=directory.base_url,
        league_shortcut=scope.shortcut, league_season=scope.season,
        competition_id=scope.competition_id,
        competition_name=scope.competition_name, team_bindings=bindings,
        enabled=True, license="ODbL-1.0",
        license_url="https://openligadb.de/lizenz",
        team_country=scope.country, team_federation=scope.federation,
        competition_gender=scope.gender, competition_age_group=scope.age_group,
        competition_entity_type=scope.entity_type,
        history_seasons=tuple(season for season in (scope.season - 1, scope.season - 2)
                              if season >= 2000),
        competition_type=scope.competition_type,
        competition_aliases=aliases,
    )
    return BoundCompetitionProvider(OpenLigaDBProvider(config),
                                    TeamResolver(TeamDatabase(identities)),
                                    tuple(sorted(evidence_ids)))
