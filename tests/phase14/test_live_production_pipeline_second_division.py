"""Opt-in directory check; team coverage is not promoted to match coverage."""

from pathlib import Path

import pytest

from erguoyuan_football.research.live_data.openligadb_directory import (
    OpenLigaDBDirectoryProvider,
)
from erguoyuan_football.research.provider_coverage_profile import CoverageStatus

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.live_network
def test_live_second_division_directory_remains_partial_pipeline_coverage(
        request: pytest.FixtureRequest) -> None:
    if "live_network" not in request.config.option.markexpr:
        pytest.skip("explicit -m live_network required")
    provider = OpenLigaDBDirectoryProvider(ROOT / "config/phase13_7_openligadb.yaml")
    try:
        teams = provider.fetch_teams("germany_men_second_division_2026")
        assert teams.status == CoverageStatus.VERIFIED
        assert teams.rows
        # No Phase 14.1 resolver yet binds this directory to verified fixtures/results.
        assert "matches_germany_men_second_division_2026" in (
            provider.network_registry.get(provider.provider_id).endpoints)
    finally:
        provider.close()

