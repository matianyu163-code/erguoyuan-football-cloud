"""League-scoped OpenLigaDB team discovery through the existing network gate."""

from __future__ import annotations

import hashlib

from erguoyuan_football.knowledge.entities.competition_hint_parser import (
    ParsedCompetitionHint,
)
from erguoyuan_football.knowledge.entities.discovery import TeamDiscoveryCandidate
from erguoyuan_football.knowledge.entities.team_entity_parser import (
    ParsedTeamEntityHint,
    UniversalTeamNameParser,
)
from erguoyuan_football.knowledge.teams.alias_matcher import normalize_club_alias
from erguoyuan_football.research.live_data.openligadb import OpenLigaDBProvider
from erguoyuan_football.web_research.evidence.evidence_schema import EvidenceRecord
from erguoyuan_football.web_research.evidence.evidence_store import EvidenceStore
from erguoyuan_football.web_research.providers.capabilities import ProviderCapability
from erguoyuan_football.web_research.time_utils import utc_iso


class OpenLigaDBTeamDiscovery:
    """Discover only names in the reviewed configured league and season."""

    supports_team_discovery = True
    capabilities = frozenset({ProviderCapability.TEAM_DISCOVERY})

    def __init__(self, provider: OpenLigaDBProvider,
                 store: EvidenceStore) -> None:
        self.provider = provider
        self.store = store
        self.provider_id = provider.config.provider_id
        self.parser = UniversalTeamNameParser()

    def can_cover(self, hint: ParsedTeamEntityHint,
                  competition: ParsedCompetitionHint | None = None) -> bool:
        """Reject known non-PL dimensions before any network call."""
        config = self.provider.config
        if hint.country_hint and hint.country_hint != config.team_country:
            return False
        if hint.entity_type_hint != config.competition_entity_type or (
            hint.age_group_hint != config.competition_age_group or
            hint.squad_level_hint != "FIRST_TEAM" or
            hint.gender_hint not in {"UNKNOWN", config.competition_gender}):
            return False
        return competition is None or not any((
            competition.country not in {None, config.team_country},
            competition.gender not in {None, config.competition_gender},
            competition.age_group not in {None, config.competition_age_group},
            competition.entity_type not in {None, config.competition_entity_type},
        ))

    def discover(self, hint: ParsedTeamEntityHint,
                 competition: ParsedCompetitionHint | None = None
                 ) -> tuple[TeamDiscoveryCandidate, ...]:
        """Persist exact real provider team rows as identity evidence."""
        if not self.can_cover(hint, competition):
            return ()
        response = self.provider.client.fetch_json(
            self.provider_id, "available_teams", params={}, bypass_cache=True)
        if not isinstance(response.body, list):
            raise TypeError("INVALID_TEAM_DISCOVERY_SCHEMA")
        url = response.final_url or self.provider.teams_endpoint
        self.provider.sources.validate_result_url(self.provider_id, url)
        found: list[TeamDiscoveryCandidate] = []
        for raw in response.body:
            if not isinstance(raw, dict):
                raise TypeError("INVALID_TEAM_DISCOVERY_SCHEMA")
            team_id, name = raw.get("teamId"), raw.get("teamName")
            if not isinstance(team_id, int) or team_id <= 0 or not isinstance(name, str):
                raise ValueError("INVALID_TEAM_DISCOVERY_SCHEMA")
            parsed = self.parser.parse(name)
            if normalize_club_alias(parsed.base_name) != normalize_club_alias(hint.base_name):
                continue
            # Provider row has identity and league membership, not individual gender/age fields.
            if parsed.gender_hint not in {"UNKNOWN", self.provider.config.competition_gender}:
                continue
            if parsed.age_group_hint != self.provider.config.competition_age_group or (
                parsed.squad_level_hint != "FIRST_TEAM"):
                continue
            stamp = utc_iso(response.retrieved_at)
            evidence_id = "TEAM_DISCOVERY_" + hashlib.sha256(
                f"{self.provider_id}:{team_id}:{name}:{stamp}".encode()).hexdigest()
            evidence = EvidenceRecord(
                evidence_id, "TEAM_DISCOVERY",
                {"provider_team_id": team_id, "official_name": name,
                 "league_shortcut": self.provider.config.league_shortcut,
                 "league_season": self.provider.config.league_season,
                 "reviewed_context": {
                     "country": self.provider.config.team_country,
                     "gender": self.provider.config.competition_gender,
                     "age_group": self.provider.config.competition_age_group,
                 }},
                self.provider_id, None, stamp, "HIGH", url, stamp,
                provider_id=self.provider_id, source_tier=2, observed_time=stamp,
            )
            self.store.save(evidence)
            found.append(TeamDiscoveryCandidate(
                self.provider_id, str(team_id), name,
                self.provider.config.team_country,
                self.provider.config.team_federation,
                self.provider.config.competition_entity_type,
                self.provider.config.competition_gender,
                self.provider.config.competition_age_group,
                "FIRST_TEAM", (), url, 2, response.retrieved_at,
                "HIGH", evidence_id,
            ))
        return tuple(found)
