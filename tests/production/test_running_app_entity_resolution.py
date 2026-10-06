"""Entity resolution stays consistent through source, runner, and desktop dispatch."""

from __future__ import annotations

from pathlib import Path

import pytest

from erguoyuan_football.app.installer.version import source_fingerprints
from erguoyuan_football.app.ui.window import run_desktop_text
from erguoyuan_football.knowledge.entities.universal_team_resolver import (
    UniversalTeamResolver,
)
from production.runner import ProductionRunner

_MATCHES = (
    ("爱尔兰U18VS荷兰U18", "NATIONAL_IRL_M_U18", "NATIONAL_NLD_M_U18"),
    ("德国U19VS捷克U19", "NATIONAL_DEU_M_U19", "NATIONAL_CZE_M_U19"),
    ("德国VS捷克", "NATIONAL_DEU_M_SENIOR", "NATIONAL_CZE_M_SENIOR"),
    ("Germany U19 VS Czechia U19", "NATIONAL_DEU_M_U19", "NATIONAL_CZE_M_U19"),
    ("哈萨克斯坦VS摩尔多瓦", "NATIONAL_KAZ_M_SENIOR", "NATIONAL_MDA_M_SENIOR"),
    ("俄罗斯U19VS沙特U20", "NATIONAL_RUS_M_U19", "NATIONAL_SAU_M_U20"),
    ("日本U23VS韩国U23", "NATIONAL_JPN_M_U23", "NATIONAL_KOR_M_U23"),
)


def test_build_fingerprint_covers_production_and_desktop_wiring() -> None:
    fingerprints = source_fingerprints(Path(__file__).resolve().parents[2])
    assert len(fingerprints["production_runner"]) == 64
    assert len(fingerprints["fixture_router"]) == 64
    assert len(fingerprints["global_fixture_research"]) == 64
    assert len(fingerprints["association_sources"]) == 64
    assert len(fingerprints["v7_diagnostic"]) == 64
    assert len(fingerprints["desktop_dispatch"]) == 64
    assert len(fingerprints["identity_resolver"]) == 64


@pytest.mark.parametrize(("match_text", "home_id", "away_id"), _MATCHES)
def test_running_app_entity_resolution_is_consistent(
    production_config, match_text: str, home_id: str, away_id: str,
) -> None:
    """Check the actual resolver, production runner, and desktop dispatch path."""
    home_name, away_name = match_text.split("VS")
    resolver = UniversalTeamResolver()
    home = resolver.resolve(home_name.strip())
    away = resolver.resolve(away_name.strip())
    assert home.identity is not None
    assert away.identity is not None
    assert home.identity.team_id == home_id
    assert away.identity.team_id == away_id

    runner = ProductionRunner(production_config)
    try:
        production_result = runner.run(match_text)
    finally:
        runner.close()
    entity_stage = next(
        stage for stage in production_result.stages
        if stage.stage == "ENTITY_RESOLVER"
    )
    assert entity_stage.status == "READY"
    assert entity_stage.detail == f"{home_id};{away_id}"

    desktop_result = run_desktop_text(match_text, production_config,
                                      mode="AUTO_RESEARCH")
    assert desktop_result.status == "PRODUCTION_TRIAL"
    assert "TEAM_NOT_FOUND" not in "\n".join(desktop_result.reasons)
    assert "OFFICIAL_SOURCE_COVERAGE_MISSING" in desktop_result.report_text
    assert "Research Fixture   OFFICIAL_SOURCE_COVERAGE_MISSING" in desktop_result.report_text
