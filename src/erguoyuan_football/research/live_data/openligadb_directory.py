"""Allowlisted OpenLigaDB competition and team directories for multiple scopes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from erguoyuan_football.network.client import CoreNetworkClient
from erguoyuan_football.network.config import RateLimitPolicy
from erguoyuan_football.network.schemas import SourceDefinition
from erguoyuan_football.network.source_registry import ExternalSourceRegistry
from erguoyuan_football.research.provider_coverage_profile import (
    CoverageStatus,
    ProviderCoverageProfile,
)
from erguoyuan_football.web_research.evidence.evidence_schema import EvidenceRecord
from erguoyuan_football.web_research.evidence.evidence_store import EvidenceStore
from erguoyuan_football.web_research.sources.source_registry import SourceRegistry
from erguoyuan_football.web_research.sources.source_schema import SourceRecord
from erguoyuan_football.web_research.time_utils import utc_iso


@dataclass(frozen=True)
class DirectoryCompetition:
    """Configured competition scope and provider's exact seasonal identity."""

    scope_id: str
    shortcut: str
    season: int
    competition_id: str
    competition_name: str
    country: str
    federation: str
    gender: str
    age_group: str
    entity_type: str
    squad_level: str
    competition_type: str


@dataclass(frozen=True)
class DirectoryTeam:
    """Team directory row with reviewed competition dimensions and evidence."""

    provider_team_id: int
    provider_name: str
    competition: DirectoryCompetition
    evidence_id: str
    retrieved_at: datetime
    source_url: str


@dataclass(frozen=True)
class DirectoryFetch:
    """A directory response and the time it was actually retrieved."""

    status: CoverageStatus
    rows: tuple[DirectoryTeam, ...]
    retrieved_at: datetime
    evidence_ids: tuple[str, ...]
    reason: str | None = None
    capability: str = "UNKNOWN"
    competition_rows: tuple[dict[str, Any], ...] = ()


class OpenLigaDBDirectoryProvider:
    """Fetch only reviewed league scopes via documented endpoints and CoreNetworkClient."""

    provider_id = "OPENLIGADB_GLOBAL_DIRECTORY"
    base_url = "https://api.openligadb.de"

    def __init__(self, config_path: Path | str, evidence_store: EvidenceStore | None = None,
                 *, client: CoreNetworkClient | None = None) -> None:
        raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("source_tier") != 2:
            raise ValueError("INVALID_OPENLIGADB_DIRECTORY_CONFIG")
        parsed = urlparse(str(raw.get("base_url", "")))
        if (parsed.scheme != "https" or parsed.netloc != "api.openligadb.de"
                or parsed.path not in {"", "/"} or parsed.query or parsed.fragment):
            raise ValueError("OPENLIGADB_HOST_NOT_ALLOWLISTED")
        scopes_raw = raw.get("competitions")
        if not isinstance(scopes_raw, dict) or not scopes_raw:
            raise ValueError("DIRECTORY_COMPETITIONS_REQUIRED")
        scopes: dict[str, DirectoryCompetition] = {}
        endpoints: dict[str, str] = {}
        for scope_id, value in scopes_raw.items():
            if not isinstance(value, dict):
                raise TypeError("INVALID_DIRECTORY_COMPETITION")
            scope = DirectoryCompetition(str(scope_id), str(value["shortcut"]),
                int(value["season"]), str(value["competition_id"]),
                str(value["competition_name"]), str(value["country"]),
                str(value["federation"]), str(value["gender"]),
                str(value["age_group"]), str(value["entity_type"]),
                str(value["squad_level"]), str(value["competition_type"]))
            if not scope.shortcut.isalnum() or not 2000 <= scope.season <= 2100:
                raise ValueError("INVALID_DIRECTORY_SCOPE_ID")
            scopes[scope.scope_id] = scope
            endpoints[f"competitions_{scope.season}"] = (
                f"{self.base_url}/getavailableleagues/{scope.season}")
            endpoints[f"teams_{scope.scope_id}"] = (
                f"{self.base_url}/getavailableteams/{scope.shortcut}/{scope.season}")
            endpoints[f"matches_{scope.scope_id}"] = (
                f"{self.base_url}/getmatchdata/{scope.shortcut}/{scope.season}")
        definition = SourceDefinition(
            source_id=self.provider_id, display_name="OpenLigaDB Global Directory",
            category="WEB_RESEARCH", base_url=self.base_url, endpoints=endpoints,
            schema_version="OPENLIGADB_DIRECTORY_V1", supports_live=True,
            supports_history=True,
            rate_limit_policy=RateLimitPolicy(requests_per_second=1,
                requests_per_minute=60, burst=1),
        )
        network_registry = ExternalSourceRegistry((definition,))
        self.network_registry = network_registry
        self.sources = SourceRegistry(network_registry)
        self.sources.add(SourceRecord(self.provider_id, "OpenLigaDB Global Directory",
            "STRUCTURED", 2, self.base_url, raw.get("enabled") is True,
            allowed_domains=(parsed.hostname or "",)))
        self.scopes = scopes
        self._owns_evidence_store = evidence_store is None
        self.evidence_store = evidence_store or EvidenceStore(":memory:", self.sources)
        self._owns_client = client is None
        self.client = client or CoreNetworkClient(network_registry)

    def close(self) -> None:
        """Close the pooled HTTP client when this provider created it."""
        if self._owns_client:
            self.client.close()
        if self._owns_evidence_store:
            self.evidence_store.close()

    def fetch_competitions(self, season: int) -> DirectoryFetch:
        """Fetch the documented seasonal competition directory and persist its evidence."""
        endpoint_id = f"competitions_{season}"
        if endpoint_id not in self.network_registry.get(self.provider_id).endpoints:
            return DirectoryFetch(CoverageStatus.UNSUPPORTED, (), datetime.now(UTC), (),
                                  "SEASON_NOT_ALLOWLISTED")
        response = self.client.fetch_json(self.provider_id, endpoint_id,
            params={}, bypass_cache=True)
        body = response.body
        if not isinstance(body, list) or any(not isinstance(row, dict) for row in body):
            return DirectoryFetch(CoverageStatus.DEGRADED, (), response.retrieved_at, (),
                                  "INVALID_COMPETITION_DIRECTORY")
        stamp = utc_iso(response.retrieved_at)
        evidence_id = "OPENLIGADB_COMPETITIONS_" + hashlib.sha256(
            f"{season}:{response.content_hash}:{stamp}".encode()).hexdigest()
        self.evidence_store.save(EvidenceRecord(evidence_id, "COMPETITION_DIRECTORY",
            {"season": season, "rows": body}, self.provider_id, None, stamp, "HIGH",
            response.final_url or f"{self.base_url}/getavailableleagues",
            stamp, provider_id=self.provider_id, source_tier=2, observed_time=stamp))
        return DirectoryFetch(CoverageStatus.VERIFIED if body else CoverageStatus.PARTIAL,
                              (), response.retrieved_at, (evidence_id,),
                              None if body else "EMPTY_COMPETITION_DIRECTORY",
                              "COMPETITION_DIRECTORY", tuple(body))

    def fetch_teams(self, scope_id: str) -> DirectoryFetch:
        """Verify a configured category from live typed rows; never infer dimensions."""
        if scope_id not in self.scopes:
            return DirectoryFetch(CoverageStatus.UNSUPPORTED, (), datetime.now(UTC), (),
                                  "PROVIDER_COVERAGE_MISSING")
        scope = self.scopes[scope_id]
        response = self.client.fetch_json(self.provider_id, f"teams_{scope_id}",
            params={}, bypass_cache=True)
        if not isinstance(response.body, list):
            return DirectoryFetch(CoverageStatus.DEGRADED, (), response.retrieved_at, (),
                                  "INVALID_TEAM_DIRECTORY")
        if not response.body:
            return DirectoryFetch(CoverageStatus.PARTIAL, (), response.retrieved_at, (),
                                  "EMPTY_TEAM_DIRECTORY")
        records: list[DirectoryTeam] = []
        for value in response.body:
            if not isinstance(value, dict):
                return DirectoryFetch(CoverageStatus.DEGRADED, (), response.retrieved_at, (),
                                      "INVALID_TEAM_DIRECTORY_ROW")
            team_id, name = value.get("teamId"), value.get("teamName")
            if not isinstance(team_id, int) or team_id <= 0 or not isinstance(name, str) or not name.strip():
                return DirectoryFetch(CoverageStatus.DEGRADED, (), response.retrieved_at, (),
                                      "INVALID_TEAM_DIRECTORY_ROW")
            stamp = utc_iso(response.retrieved_at)
            evidence_id = "OPENLIGADB_TEAM_" + hashlib.sha256(
                f"{scope.scope_id}:{team_id}:{response.content_hash}:{stamp}".encode()).hexdigest()
            url = response.final_url or f"{self.base_url}/getavailableteams/{scope.shortcut}/{scope.season}"
            self.evidence_store.save(EvidenceRecord(evidence_id, "TEAM_DIRECTORY",
                {"team_id": team_id, "team_name": name, "scope_id": scope.scope_id,
                 "competition_id": scope.competition_id,
                 "gender": scope.gender, "age_group": scope.age_group,
                 "entity_type": scope.entity_type, "squad_level": scope.squad_level},
                self.provider_id, None, stamp, "HIGH", url, stamp,
                provider_id=self.provider_id, source_tier=2, observed_time=stamp))
            records.append(DirectoryTeam(team_id, name, scope, evidence_id,
                                         response.retrieved_at, url))
        return DirectoryFetch(CoverageStatus.VERIFIED, tuple(records),
                              response.retrieved_at,
                              tuple(row.evidence_id for row in records),
                              capability="TEAM_DIRECTORY")

    @staticmethod
    def profile_after_live_tests(scope: DirectoryCompetition,
                                 competition_result: DirectoryFetch,
                                 team_result: DirectoryFetch) -> ProviderCoverageProfile:
        """Promote only the exact capabilities proven by nonempty live responses."""
        statuses: list[tuple[str, CoverageStatus]] = []
        if competition_result.capability == "COMPETITION_DIRECTORY":
            statuses.append(("COMPETITION_DIRECTORY", competition_result.status))
        if (team_result.capability == "TEAM_DIRECTORY" and team_result.rows
                and all(row.competition.scope_id == scope.scope_id for row in team_result.rows)):
            statuses.append(("TEAM_DIRECTORY", CoverageStatus.VERIFIED))
        verified_times = [result.retrieved_at for result in (competition_result, team_result)
                          if result.status == CoverageStatus.VERIFIED]
        return ProviderCoverageProfile(
            OpenLigaDBDirectoryProvider.provider_id, 2, frozenset({"EUROPE"}),
            frozenset({scope.country}), frozenset({scope.federation}),
            frozenset({scope.gender}), frozenset({scope.age_group}),
            frozenset({scope.entity_type}), frozenset({scope.squad_level}),
            frozenset({scope.competition_type}),
            frozenset({"COMPETITION_DIRECTORY", "TEAM_DIRECTORY", "FIXTURE",
                       "RESULTS", "HISTORICAL_RESULTS", "STANDINGS"}),
            None, True, False, True,
            "Open data under ODbL-1.0; attribution/share-alike obligations apply",
            max(verified_times) if verified_times else None, tuple(statuses),
        )

    @staticmethod
    def profile_declaration(scope: DirectoryCompetition) -> ProviderCoverageProfile:
        """Declare allowlisted directory endpoints without claiming live verification."""
        return ProviderCoverageProfile(
            OpenLigaDBDirectoryProvider.provider_id, 2, frozenset({"EUROPE"}),
            frozenset({scope.country}), frozenset({scope.federation}),
            frozenset({scope.gender}), frozenset({scope.age_group}),
            frozenset({scope.entity_type}), frozenset({scope.squad_level}),
            frozenset({scope.competition_type}),
            frozenset({"COMPETITION_DIRECTORY", "TEAM_DIRECTORY", "FIXTURE",
                       "RESULTS", "HISTORICAL_RESULTS", "STANDINGS"}),
            None, True, False, True,
            "ODbL-1.0; community-maintained; respect documented request limit",
        )
