"""Opt-in women's directory check; fixture/history coverage stays separate."""

from pathlib import Path

import pytest

from erguoyuan_football.research.live_data.openligadb_directory import (
    OpenLigaDBDirectoryProvider,
)
from erguoyuan_football.research.provider_coverage_profile import CoverageStatus

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.live_network
def test_live_womens_directory_is_not_a_fixture_or_history_claim(
        request: pytest.FixtureRequest) -> None:
    if "live_network" not in request.config.option.markexpr:
        pytest.skip("explicit -m live_network required")
    provider = OpenLigaDBDirectoryProvider(ROOT / "config/phase13_7_openligadb.yaml")
    try:
        teams = provider.fetch_teams("germany_women_top_division_2026")
        assert teams.status == CoverageStatus.VERIFIED
        assert teams.rows and all(row.competition.gender == "WOMEN" for row in teams.rows)
        profile = provider.profile_declaration(provider.scopes[
            "germany_women_top_division_2026"])
        assert profile.status_for("FIXTURE") == CoverageStatus.UNVERIFIED
        assert profile.status_for("HISTORICAL_RESULTS") == CoverageStatus.UNVERIFIED
    finally:
        provider.close()

