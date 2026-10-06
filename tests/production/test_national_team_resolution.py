"""Production and desktop resolution for country-backed national teams."""

from pathlib import Path

import pytest

from erguoyuan_football.app.config import AppConfig
from erguoyuan_football.app.input.match_input import MatchInputParserV2
from erguoyuan_football.app.input.parser import MatchInputResolver
from erguoyuan_football.knowledge.entities.country_alias_registry import (
    CountryAliasRegistry,
)
from erguoyuan_football.knowledge.entities.universal_team_resolver import (
    UniversalTeamResolver,
)
from erguoyuan_football.knowledge.match_identity import GlobalKnowledgeResolver
from erguoyuan_football.knowledge.teams.team_resolver import TeamResolver
from production.runner import ProductionRunner

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(("name", "iso3", "country"), [
    ("哈萨克斯坦", "KAZ", "Kazakhstan"),
    ("Kazakhstan", "KAZ", "Kazakhstan"),
    ("摩尔多瓦", "MDA", "Moldova"),
    ("Moldova", "MDA", "Moldova"),
])
def test_four_reported_names_resolve_to_senior_mens_national_team(
    name: str, iso3: str, country: str,
) -> None:
    """Country aliases construct a team entity without a static team row."""
    local = TeamResolver().resolve(name)
    country_result = CountryAliasRegistry().resolve(name)
    result = UniversalTeamResolver().resolve(name)

    assert local is None
    assert country_result is not None
    assert country_result.iso3 == iso3 and country_result.name == country
    assert result.status == "VERIFIED"
    assert result.source == "COUNTRY_REGISTRY"
    assert result.identity is not None
    assert result.identity.country == country
    assert result.identity.entity_type == "NATIONAL"
    assert result.identity.gender == "MEN"
    assert result.identity.age_group == "SENIOR"


@pytest.mark.parametrize(("raw", "home_age", "away_age"), [
    ("哈萨克斯坦VS摩尔多瓦", "SENIOR", "SENIOR"),
    ("德国VS法国", "SENIOR", "SENIOR"),
    ("日本VS韩国", "SENIOR", "SENIOR"),
    ("英格兰VS苏格兰", "SENIOR", "SENIOR"),
    ("俄罗斯U19VS沙特U20", "U19", "U20"),
    ("日本U23VS韩国U23", "U23", "U23"),
    ("德国U16VS希腊U16", "U16", "U16"),
    ("Saudi Arabia VS Japan", "SENIOR", "SENIOR"),
    ("Kazakhstan VS Moldova", "SENIOR", "SENIOR"),
])
def test_national_match_names_parse_and_resolve(
    raw: str, home_age: str, away_age: str,
) -> None:
    request = MatchInputParserV2().parse(raw)
    identity = GlobalKnowledgeResolver().resolve_request(request)

    assert request.validation_status == "VALID"
    assert identity.status == "IDENTITIES_RESOLVED_NO_FIXTURE"
    assert identity.home_team is not None and identity.away_team is not None
    assert identity.home_team.entity_type == identity.away_team.entity_type == "NATIONAL"
    assert identity.home_team.gender == identity.away_team.gender == "MEN"
    assert identity.home_team.age_group == home_age
    assert identity.away_team.age_group == away_age
    assert identity.home_resolution is not None
    assert identity.away_resolution is not None
    assert identity.home_resolution.source == "COUNTRY_REGISTRY"
    assert identity.away_resolution.source == "COUNTRY_REGISTRY"


def test_production_runner_uses_universal_resolver_and_continues_after_identity(
    production_config,
) -> None:
    runner = ProductionRunner(production_config)
    assert isinstance(runner.identity_resolver.entity_resolver, UniversalTeamResolver)

    result = runner.run("哈萨克斯坦VS摩尔多瓦", research_test=True)

    entity_stage = next(stage for stage in result.stages if stage.stage == "ENTITY_RESOLVER")
    pipeline_stage = next(stage for stage in result.stages if stage.stage == "DATA_PIPELINE")
    assert entity_stage.status == "READY"
    assert entity_stage.detail == "NATIONAL_KAZ_M_SENIOR;NATIONAL_MDA_M_SENIOR"
    assert pipeline_stage.detail == "VERIFIED_FIXTURE_AND_PIT_SNAPSHOT_UNAVAILABLE"
    assert result.request.validation_status == "VALID"


def test_desktop_historical_input_reports_fixture_gap_after_entity_resolution() -> None:
    config = AppConfig.load(ROOT / "config/application.yaml")

    request = MatchInputResolver(config).parse_text("哈萨克斯坦VS摩尔多瓦")[0]

    assert request.validation_status == "NOT_FOUND"
    assert request.error_code == "FIXTURE_NOT_FOUND"
