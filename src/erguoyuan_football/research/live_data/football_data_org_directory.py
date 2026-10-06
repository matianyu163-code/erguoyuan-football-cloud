"""Unauthenticated public competition directory from football-data.org v4."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any

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
class CompetitionDirectoryResult:
    """Validated current competition directory and its source evidence."""

    status: CoverageStatus
    competitions: tuple[dict[str, Any], ...]
    provider_id: str
    retrieved_at: datetime
    source_url: str
    evidence_id: str | None
    reason: str | None = None


class FootballDataOrgDirectoryProvider:
    """Read only the API's documented unauthenticated competition-list endpoint."""

    provider_id = "FOOTBALL_DATA_ORG_DIRECTORY"
    base_url = "https://api.football-data.org"

    def __init__(self, evidence_store: EvidenceStore | None = None,
                 *, client: CoreNetworkClient | None = None) -> None:
        definition = SourceDefinition(
            source_id=self.provider_id, display_name="football-data.org Competition Directory",
            category="WEB_RESEARCH", base_url=self.base_url,
            endpoints={"competition_directory": f"{self.base_url}/v4/competitions"},
            schema_version="FOOTBALL_DATA_ORG_V4_COMPETITIONS",
            supports_live=True,
            rate_limit_policy=RateLimitPolicy(requests_per_second=1,
                requests_per_minute=1, burst=1),
        )
        self.network_registry = ExternalSourceRegistry((definition,))
        self.sources = SourceRegistry(self.network_registry)
        self.sources.add(SourceRecord(self.provider_id, "football-data.org",
            "STRUCTURED", 2, self.base_url, True,
            allowed_domains=("api.football-data.org",)))
        self._owns_client = client is None
        self.client = client or CoreNetworkClient(self.network_registry)
        self._owns_evidence_store = evidence_store is None
        self.evidence_store = evidence_store or EvidenceStore(":memory:", self.sources)

    def close(self) -> None:
        """Close owned transport and evidence connection."""
        if self._owns_client:
            self.client.close()
        if self._owns_evidence_store:
            self.evidence_store.close()

    def fetch_competitions(self) -> CompetitionDirectoryResult:
        """Fetch competition-list JSON without a credential or guessed resource."""
        response = self.client.fetch_json(self.provider_id, "competition_directory",
            params={}, bypass_cache=True)
        body = response.body
        rows = body.get("competitions") if isinstance(body, dict) else None
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            return CompetitionDirectoryResult(CoverageStatus.DEGRADED, (), self.provider_id,
                response.retrieved_at, response.final_url or f"{self.base_url}/v4/competitions",
                None, "INVALID_COMPETITION_DIRECTORY_SCHEMA")
        stamp = utc_iso(response.retrieved_at)
        evidence_id = "FOOTBALL_DATA_COMPETITIONS_" + hashlib.sha256(
            f"{response.content_hash}:{stamp}".encode()).hexdigest()
        url = response.final_url or f"{self.base_url}/v4/competitions"
        if rows:
            self.evidence_store.save(EvidenceRecord(evidence_id, "COMPETITION_DIRECTORY",
                {"competitions": rows, "provider_api_version": "v4"}, self.provider_id,
                None, stamp, "HIGH", url, stamp, provider_id=self.provider_id,
                source_tier=2, observed_time=stamp))
        return CompetitionDirectoryResult(
            CoverageStatus.VERIFIED if rows else CoverageStatus.PARTIAL,
            tuple(rows), self.provider_id, response.retrieved_at, url,
            evidence_id if rows else None,
            None if rows else "EMPTY_COMPETITION_DIRECTORY")

    @staticmethod
    def profile_after_live_test(result: CompetitionDirectoryResult
                                ) -> ProviderCoverageProfile:
        """Verify only broad competition enumeration, not fixture or team coverage."""
        verified = (result.status == CoverageStatus.VERIFIED and bool(result.competitions)
                    and bool(result.evidence_id))
        return ProviderCoverageProfile(
            FootballDataOrgDirectoryProvider.provider_id, 2,
            frozenset({"*"}), frozenset({"*"}), frozenset({"*"}),
            frozenset({"*"}), frozenset({"*"}), frozenset({"*"}),
            frozenset({"*"}), frozenset({"*"}),
            frozenset({"COMPETITION_DIRECTORY"}), None, True, False, True,
            "football-data.org terms apply; public competition-list endpoint only; no blanket open-data license asserted",
            result.retrieved_at if verified else None,
            (("COMPETITION_DIRECTORY", CoverageStatus.VERIFIED if verified
              else result.status),),
        )

    @staticmethod
    def profile_declaration() -> ProviderCoverageProfile:
        """Declare the documented global list endpoint without claiming its response."""
        return ProviderCoverageProfile(
            FootballDataOrgDirectoryProvider.provider_id, 2,
            frozenset({"*"}), frozenset({"*"}), frozenset({"*"}),
            frozenset({"*"}), frozenset({"*"}), frozenset({"*"}),
            frozenset({"*"}), frozenset({"*"}),
            frozenset({"COMPETITION_DIRECTORY"}), None, True, False, True,
            "Provider terms apply; unauthenticated competition directory only",
        )
