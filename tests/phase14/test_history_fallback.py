"""Historical fallback requires verified exact coverage and PIT-safe retrieval."""

from datetime import UTC, datetime

import pytest

from erguoyuan_football.research.historical_data_acquirer import (
    HistoricalDataAcquirer,
    HistoricalProviderUnavailable,
)
from erguoyuan_football.research.historical_provider_router import (
    HistoricalCapability,
    HistoricalProviderOption,
    HistoricalProviderRouter,
    HistoricalProviderTier,
)
from erguoyuan_football.research.historical_result_cache import HistoricalResultCache
from erguoyuan_football.research.live_data.openligadb import SeasonBatch


def _option(provider_id: str, tier: HistoricalProviderTier, *, season: int = 2025,
            competition: str = "COMP", country: str = "DEU",
            retrieved: datetime | None = None) -> HistoricalProviderOption:
    return HistoricalProviderOption(
        provider_id=provider_id, tier=tier, competition_id=competition,
        country=country, seasons=frozenset({season}),
        capabilities=frozenset({"HISTORICAL_RESULTS"}), coverage_status="VERIFIED",
        fetch=lambda requested: SeasonBatch((), retrieved or datetime(2026, 1, 1, tzinfo=UTC),
                                             "https://provider.example/history", 200, 1.0,
                                             requested),
        historical_results=lambda batch, fixture, cutoff: (),
        canonical_team_map_complete=True, canonical_team_ids={1: "CANON_TEAM"},
    )


def test_router_orders_only_exact_verified_coverage() -> None:
    rows = (
        _option("BACKUP", HistoricalProviderTier.BACKUP),
        _option("TIER_B", HistoricalProviderTier.TIER_B),
        _option("OTHER_COMP", HistoricalProviderTier.TIER_A, competition="OTHER"),
        _option("OTHER_SEASON", HistoricalProviderTier.TIER_A, season=2024),
    )
    result = HistoricalProviderRouter().resolve(
        competition="COMP", country="DEU", season=2025,
        capability=HistoricalCapability.RESULTS, providers=rows,
    )
    assert [row.provider_id for row in result] == ["TIER_B", "BACKUP"]


def test_fallback_fetch_merges_sources_in_priority_order_and_rejects_future() -> None:
    class Config:
        competition_id = "COMP"
        team_country = "DEU"

    class Primary:
        config = Config()

    providers = (
        _option("TIER_B", HistoricalProviderTier.TIER_B),
        _option("BACKUP", HistoricalProviderTier.BACKUP),
        _option("LATE", HistoricalProviderTier.TIER_A,
                retrieved=datetime(2026, 3, 1, tzinfo=UTC)),
    )
    acquirer = HistoricalDataAcquirer(cache=HistoricalResultCache(":memory:"),
                                      fallback_providers=providers)
    try:
        attempts: list[tuple[str, str]] = []
        rows = acquirer._fetch_fallback(
            Primary(), 2025, datetime(2026, 2, 1, tzinfo=UTC), attempts,
            RuntimeError("primary unavailable"),
        )
        assert [provider for _, provider in rows] == ["TIER_B", "BACKUP"]
        assert ("LATE", "ValueError:FALLBACK_EVIDENCE_AFTER_CUTOFF") in attempts
    finally:
        acquirer.close()


def test_no_eligible_provider_returns_structured_unavailable() -> None:
    class Config:
        competition_id = "COMP"
        team_country = "DEU"

    class Primary:
        config = Config()

    acquirer = HistoricalDataAcquirer(cache=HistoricalResultCache(":memory:"))
    try:
        with pytest.raises(HistoricalProviderUnavailable,
                           match="HISTORY_PROVIDER_UNAVAILABLE"):
            acquirer._fetch_fallback(Primary(), 2025,
                datetime(2026, 2, 1, tzinfo=UTC), [], RuntimeError("primary unavailable"))
    finally:
        acquirer.close()
