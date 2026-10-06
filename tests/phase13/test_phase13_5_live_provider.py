"""Real OpenLigaDB smoke, opt-in only; never replaced by a local fixture file."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from erguoyuan_football.research.live_data.model_requirements import (
    load_model_requirements,
)
from erguoyuan_football.research.live_data.openligadb import (
    OpenLigaDBConfig,
    OpenLigaDBProvider,
)
from erguoyuan_football.research.live_data.provider_coverage import build_coverage
from erguoyuan_football.research.live_data.provider_health import ProviderHealthChecker
from erguoyuan_football.research.live_data.readiness import (
    LiveDataReadinessGate,
    ReadinessConfig,
)
from erguoyuan_football.research.live_data.service import LiveResearchService
from erguoyuan_football.web_research.evidence.evidence_store import EvidenceStore
from erguoyuan_football.web_research.time_utils import parse_utc

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.live_network
def test_real_fixture_stats_package_and_readiness(request: pytest.FixtureRequest) -> None:
    """One real request validates fixture, results, persisted evidence and audit."""
    if "live_network" not in request.config.option.markexpr:
        pytest.skip("explicit -m live_network required")
    config = OpenLigaDBConfig.from_yaml(ROOT / "config/phase13_5_provider.yaml")
    provider = OpenLigaDBProvider(config)
    store = EvidenceStore(":memory:", provider.sources)
    try:
        batch = provider.fetch_season()
        cutoff = datetime.now(UTC)
        assert ProviderHealthChecker.check(config.provider_id, config.enabled, batch).status == "READY"
        fixtures = [row for row in batch.matches
                    if row.get("team1", {}).get("teamId") == 2617
                    and row.get("team2", {}).get("teamId") == 370
                    and isinstance(row.get("matchDateTimeUTC"), str)
                    and parse_utc(row["matchDateTimeUTC"]) > cutoff]
        assert fixtures, "no future exact Arsenal-home Liverpool fixture in configured season"
        hint = parse_utc(fixtures[0]["matchDateTimeUTC"])
        gate = LiveDataReadinessGate(
            ReadinessConfig.from_yaml(ROOT / "config/phase13_5_readiness.yaml"),
            load_model_requirements(ROOT / "config/model_registry.yaml"),
        )
        result = LiveResearchService(provider, store, gate).build(
            batch, "ENG_ARS", "ENG_LIV", cutoff=cutoff, date_hint=hint)
        assert result.fixture_check.status == "VERIFIED"
        assert result.fixture_check.fixture is not None
        assert result.fixture_check.fixture.provider_match_id
        assert result.fixture_check.fixture.source_url.startswith("https://api.openligadb.de/")
        assert result.historical_count > 0
        assert result.recent_form_count == 2
        assert result.package is not None and result.readiness is not None
        assert store.get(result.package.available_data["fixture"][0].evidence_id)
        assert result.readiness.item("HISTORICAL_RESULTS").evidence_ids
        assert result.package.prediction_executed is False
        assert result.readiness.prediction_executed is False
        assert provider.client.audit.records()
        coverage = build_coverage(config.provider_id,
                                  declared=frozenset({"FIXTURE", "RESULTS"}),
                                  verified=frozenset({"FIXTURE", "RESULTS"}))
        assert coverage.statuses["FIXTURE"] == "VERIFIED"
        assert coverage.statuses["RESULTS"] == "VERIFIED"
        assert coverage.statuses["STATS"] == "UNSUPPORTED"
        assert coverage.statuses["XG"] == "UNSUPPORTED"
    finally:
        store.close()
        provider.close()
