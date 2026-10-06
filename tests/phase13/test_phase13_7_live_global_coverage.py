"""Real opt-in OpenLigaDB coverage expansion beyond the Phase 13.6 EPL scope."""

from __future__ import annotations

from pathlib import Path

import pytest

from erguoyuan_football.research.global_provider_registry import (
    GlobalProviderRegistry,
    RegisteredProvider,
)
from erguoyuan_football.research.live_data.football_data_org_directory import (
    FootballDataOrgDirectoryProvider,
)
from erguoyuan_football.research.live_data.openligadb_directory import (
    OpenLigaDBDirectoryProvider,
)
from erguoyuan_football.research.provider_coverage_profile import CoverageStatus

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.live_network
def test_real_global_competition_and_womens_team_directory(
        request: pytest.FixtureRequest) -> None:
    """Verify an actual women's competition and directory response with evidence."""
    if "live_network" not in request.config.option.markexpr:
        pytest.skip("explicit -m live_network required")
    provider = OpenLigaDBDirectoryProvider(ROOT / "config/phase13_7_openligadb.yaml")
    try:
        competitions = provider.fetch_competitions(2026)
        assert competitions.status == CoverageStatus.VERIFIED
        frauen_comp = [row for row in competitions.competition_rows
                       if row.get("leagueShortcut") == "ffb1"
                       and str(row.get("leagueSeason")) == "2026"]
        assert frauen_comp, "configured German women's competition absent from real directory"
        women_scope = provider.scopes["germany_women_top_division_2026"]
        women = provider.fetch_teams(women_scope.scope_id)
        assert women.status == CoverageStatus.VERIFIED
        assert len(women.rows) >= 10
        assert all(row.competition.gender == "WOMEN" for row in women.rows)
        assert all(provider.evidence_store.get(row.evidence_id) is not None for row in women.rows)
        second_division_scope = provider.scopes["germany_men_second_division_2026"]
        second_division = provider.fetch_teams(second_division_scope.scope_id)
        assert second_division.status == CoverageStatus.VERIFIED
        assert len(second_division.rows) >= 16
        assert all(row.competition.gender == "MEN" for row in second_division.rows)
        profile = provider.profile_after_live_tests(women_scope, competitions, women)
        assert profile.status_for("TEAM_DIRECTORY") == CoverageStatus.VERIFIED
        assert profile.status_for("COMPETITION_DIRECTORY") == CoverageStatus.VERIFIED
        assert profile.status_for("XG") == CoverageStatus.UNSUPPORTED
        registered = RegisteredProvider(profile, frozenset({"api.openligadb.de"}), True,
                                        "HEALTHY", provider)
        assert GlobalProviderRegistry((registered,)).get(provider.provider_id) == registered
        assert provider.client.audit.records()
    finally:
        provider.close()


@pytest.mark.live_network
def test_real_second_provider_public_competition_directory(
        request: pytest.FixtureRequest) -> None:
    """Verify a second source's documented unauthenticated directory endpoint."""
    if "live_network" not in request.config.option.markexpr:
        pytest.skip("explicit -m live_network required")
    provider = FootballDataOrgDirectoryProvider()
    try:
        result = provider.fetch_competitions()
        assert result.status == CoverageStatus.VERIFIED
        codes = {row.get("code") for row in result.competitions}
        assert "PL" in codes
        assert len(result.competitions) >= 10
        assert result.evidence_id is not None
        assert provider.evidence_store.get(result.evidence_id) is not None
        assert provider.client.audit.records()
        profile = provider.profile_after_live_test(result)
        assert profile.status_for("COMPETITION_DIRECTORY") == CoverageStatus.VERIFIED
        assert profile.status_for("TEAM_DIRECTORY") == CoverageStatus.UNSUPPORTED
    finally:
        provider.close()
