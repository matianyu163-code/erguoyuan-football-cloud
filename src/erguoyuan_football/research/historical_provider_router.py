"""PIT-safe, coverage-aware routing for configured historical-result sources."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum, StrEnum

from erguoyuan_football.research.live_data.openligadb import SeasonBatch
from erguoyuan_football.research.live_data.schemas import VerifiedFixture
from erguoyuan_football.web_research.evidence.evidence_schema import EvidenceRecord


class HistoricalProviderTier(IntEnum):
    """Explicit provider preference; ordering is Tier A, Tier B, then backup."""

    TIER_A = 1
    TIER_B = 2
    BACKUP = 3


class HistoricalCapability(StrEnum):
    """Historical data product requested from a provider."""

    RESULTS = "HISTORICAL_RESULTS"
    FIXTURES = "FIXTURES"


@dataclass(frozen=True)
class HistoricalProviderOption:
    """One reviewed source with exact competition/season coverage and fetch method."""

    provider_id: str
    tier: HistoricalProviderTier
    competition_id: str
    country: str
    seasons: frozenset[int]
    capabilities: frozenset[str]
    coverage_status: str
    fetch: Callable[[int], SeasonBatch]
    historical_results: Callable[[SeasonBatch, VerifiedFixture, object],
                                 tuple[EvidenceRecord, ...]]
    canonical_team_map_complete: bool
    canonical_team_ids: dict[int, str]


class HistoricalProviderRouter:
    """Return only configured sources with exact, verified requested coverage."""

    def resolve(self, *, competition: str, country: str, season: int,
                capability: HistoricalCapability,
                providers: tuple[HistoricalProviderOption, ...]
                ) -> tuple[HistoricalProviderOption, ...]:
        """Order exact scope candidates by provider tier then stable provider ID."""
        if season < 1900 or season > 2100:
            raise ValueError("HISTORICAL_SEASON_OUT_OF_RANGE")
        matches = [row for row in providers
                   if row.competition_id == competition
                   and row.country == country
                   and season in row.seasons
                   and capability.value in row.capabilities
                   and row.coverage_status == "VERIFIED"
                   and row.canonical_team_map_complete
                   and bool(row.canonical_team_ids)]
        return tuple(sorted(matches, key=lambda row: (row.tier, row.provider_id)))
