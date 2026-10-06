"""Systematic input and identity regression without fabricated match data."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from erguoyuan_football.app.input.match_input import MatchInputParserV2
from erguoyuan_football.app.installer.version import runtime_build_info
from erguoyuan_football.app.ui.window import run_desktop_text
from erguoyuan_football.knowledge.entities.country_alias_registry import (
    CountryAliasRegistry,
)
from erguoyuan_football.knowledge.entities.team_entity_parser import (
    UniversalTeamNameParser,
)
from erguoyuan_football.knowledge.entities.universal_team_resolver import (
    UniversalTeamResolver,
)
from erguoyuan_football.knowledge.entities.verified_entity_store import (
    VerifiedEntityStore,
)
from erguoyuan_football.knowledge.match_identity import GlobalKnowledgeResolver
from erguoyuan_football.knowledge.teams.team_schema import TeamIdentity
from erguoyuan_football.match_source.match_source import MatchSourceType
from production.config import ProductionConfig
from production.runner import ProductionRunner


@pytest.mark.parametrize("raw,home,away,competition,code", [
    ("哈萨克斯坦VS摩尔多瓦", "哈萨克斯坦", "摩尔多瓦", None, None),
    ("欧国联哈萨克斯坦VS摩尔多瓦", "哈萨克斯坦", "摩尔多瓦", "欧国联", None),
    ("欧国联 哈萨克斯坦 VS 摩尔多瓦", "哈萨克斯坦", "摩尔多瓦", "欧国联", None),
    ("周五001 欧国联 哈萨克斯坦VS摩尔多瓦", "哈萨克斯坦", "摩尔多瓦", "欧国联", "周五001"),
    ("竞彩足球 周五001 欧国联 哈萨克斯坦 VS 摩尔多瓦", "哈萨克斯坦", "摩尔多瓦", "欧国联", "周五001"),
    ("Russia U19 vs Saudi Arabia U20", "Russia U19", "Saudi Arabia U20", None, None),
])
def test_structure_never_passes_metadata_as_team(raw: str, home: str, away: str,
                                                   competition: str | None,
                                                   code: str | None) -> None:
    parsed = MatchInputParserV2().parse_structure(raw)
    assert (parsed.home_raw, parsed.away_raw, parsed.competition_raw,
            parsed.jc_code) == (home, away, competition, code)


@pytest.mark.parametrize("raw,base,age,level", [
    ("Russia U-19", "Russia", "U19", "FIRST_TEAM"),
    ("Russia U 19", "Russia", "U19", "FIRST_TEAM"),
    ("Russia Under 19", "Russia", "U19", "FIRST_TEAM"),
    ("Manchester United U21", "Manchester United", "U21", "YOUTH"),
    ("Barcelona B", "Barcelona", "SENIOR", "B_TEAM"),
    ("Bayern Munich II", "Bayern Munich", "SENIOR", "SECOND_TEAM"),
    ("拜仁II", "拜仁", "SENIOR", "SECOND_TEAM"),
    ("Chelsea Reserves", "Chelsea", "SENIOR", "RESERVE"),
    ("Ajax Academy", "Ajax", "SENIOR", "ACADEMY"),
    ("Ajax Youth", "Ajax", "SENIOR", "YOUTH"),
])
def test_structured_team_rules(raw: str, base: str, age: str, level: str) -> None:
    parsed = UniversalTeamNameParser().parse(raw)
    assert (parsed.base_name, parsed.age_group_hint,
            parsed.squad_level_hint) == (base, age, level)


def test_all_country_names_construct_senior_and_youth() -> None:
    resolver = UniversalTeamResolver()
    countries = CountryAliasRegistry().all()
    assert len(countries) >= 250
    for country in countries:
        names = (country.name, country.canonical_name_zh)
        for name in names:
            if not name:
                continue
            senior = resolver.resolve(name)
            assert senior.identity is not None, country.iso3
            assert senior.resolution_status in {"RESOLVED_LOCAL", "RESOLVED_STRUCTURED"}
            for age in ("U23", "U21", "U20", "U19", "U17"):
                youth = resolver.resolve(f"{name} {age}")
                assert youth.identity is not None, (country.iso3, age)
                assert senior.identity.team_id != youth.identity.team_id


@pytest.mark.parametrize("raw", [
    "哈萨克斯坦VS摩尔多瓦", "Kazakhstan VS Moldova", "德国VS法国",
    "Germany VS France", "日本VS韩国", "俄罗斯U19VS沙特U20",
    "Germany U16 VS Greece U16", "Japan U23 VS Korea Republic U23",
    "England U21 VS France U21", "Barcelona VS Real Madrid",
])
def test_minimum_identity_matrix(raw: str) -> None:
    result = GlobalKnowledgeResolver().resolve_text(raw)
    assert result.status == "IDENTITIES_RESOLVED_NO_FIXTURE"
    assert result.home_team is not None and result.away_team is not None


def test_mixed_youth_age_is_context_warning_not_identity_block() -> None:
    result = GlobalKnowledgeResolver().resolve_text("Russia U19 VS Saudi Arabia U20")
    assert result.status == "IDENTITIES_RESOLVED_NO_FIXTURE"
    assert result.warnings == ("DIFFERENT_AGE_GROUPS_CONTEXT_REVIEW",)


@pytest.mark.parametrize("name,expected", [
    ("德国U19", "NATIONAL_DEU_M_U19"),
    ("捷克U19", "NATIONAL_CZE_M_U19"),
    ("Germany U19", "NATIONAL_DEU_M_U19"),
    ("Czechia U19", "NATIONAL_CZE_M_U19"),
    ("Czech Republic U19", "NATIONAL_CZE_M_U19"),
    ("捷克共和国U19", "NATIONAL_CZE_M_U19"),
    ("DEU", "NATIONAL_DEU_M_SENIOR"),
    ("CZE", "NATIONAL_CZE_M_SENIOR"),
])
def test_source_resolver_country_registry_aliases(name: str, expected: str) -> None:
    result = UniversalTeamResolver().resolve(name)
    assert result.identity is not None
    assert result.identity.team_id == expected
    assert result.resolution_status == "RESOLVED_STRUCTURED"


def test_unknown_second_team_does_not_fall_back_to_senior() -> None:
    resolver = GlobalKnowledgeResolver()
    senior = resolver.entity_resolver.resolve("Barcelona")
    reserve = resolver.entity_resolver.resolve("Barcelona B")
    assert senior.identity is not None
    assert reserve.identity is None
    assert reserve.parsed.squad_level_hint == "B_TEAM"
    assert reserve.resolution_status == "DISCOVERY_REQUIRED"


@pytest.mark.parametrize("raw", [
    "Barcelona B VS Real Madrid Castilla",
    "Bayern Munich II VS Borussia Dortmund II",
])
def test_uncovered_second_team_needs_discovery(raw: str) -> None:
    result = GlobalKnowledgeResolver().resolve_text(raw)
    assert result.home_resolution is not None and result.away_resolution is not None
    assert result.home_resolution.identity is None
    assert result.away_resolution.identity is None
    assert result.home_resolution.resolution_status == "DISCOVERY_REQUIRED"
    assert result.away_resolution.resolution_status == "DISCOVERY_REQUIRED"


@pytest.mark.parametrize("raw", [
    "哈萨克斯坦VS摩尔多瓦", "俄罗斯U19VS沙特U20", "德国VS法国",
    "日本U23VS韩国U23", "Barcelona VS Real Madrid",
])
def test_real_desktop_text_entry_reaches_production_resolver(
    tmp_path: Path, raw: str,
) -> None:
    config = ProductionConfig("TRIAL", False, False, True, False, False, False,
                              tmp_path / "trial.sqlite", tmp_path / "updates.jsonl")
    result = run_desktop_text(raw, config, mode="AUTO_RESEARCH")
    assert result.status == "PRODUCTION_TRIAL"
    assert result.requests[0].home_team and result.requests[0].away_team
    assert "TEAM_NOT_FOUND_OR_AMBIGUOUS" not in result.report_text
    assert (tmp_path / "trial.sqlite").is_file()


def test_production_reads_provider_verified_dynamic_cache(tmp_path: Path) -> None:
    store = VerifiedEntityStore(tmp_path / "dynamic_entity_store.sqlite")
    identity = TeamIdentity(
        "PROVIDER_CLUB_42", "Verified Research Club", "Spain", "UEFA",
        ["Verified Research Club"], entity_type="CLUB", gender="MEN",
        provider_ids={"TEST_PROVIDER": "42"}, identity_status="PROVIDER_VERIFIED",
        verification_evidence_ids=["EVIDENCE_42"],
    )
    store.save(identity, provider_id="TEST_PROVIDER", provider_team_id="42",
               verified_at=datetime.now(UTC))
    store.close()
    config = ProductionConfig("TRIAL", False, False, True, False, False, False,
                              tmp_path / "trial.sqlite", tmp_path / "updates.jsonl")
    runner = ProductionRunner(config)
    try:
        resolved = runner.identity_resolver.resolve_names(
            "Verified Research Club", "Barcelona")
        assert resolved.home_team is not None
        assert resolved.home_team.team_id == "PROVIDER_CLUB_42"
        assert resolved.home_resolution is not None
        assert resolved.home_resolution.resolution_status == "RESOLVED_DYNAMIC"
    finally:
        runner.close()
    result = run_desktop_text("Verified Research Club VS Barcelona", config,
                              mode="AUTO_RESEARCH")
    assert "TEAM_NOT_FOUND" not in result.report_text


def test_jc_code_and_competition_remain_separate_from_identity(tmp_path: Path) -> None:
    config = ProductionConfig("TRIAL", False, False, True, False, False, False,
                              tmp_path / "trial.sqlite", tmp_path / "updates.jsonl")
    runner = ProductionRunner(config)
    try:
        result = runner.run("竞彩足球 周五001 欧国联 哈萨克斯坦VS摩尔多瓦")
    finally:
        runner.close()
    assert result.source_type == MatchSourceType.USER_JC_CONFIRMED
    assert result.request.home_team == "哈萨克斯坦"
    assert result.request.away_team == "摩尔多瓦"
    assert result.request.competition == "欧国联"
    assert next(s for s in result.stages if s.stage == "ENTITY_RESOLVER").status == "READY"


def test_production_and_desktop_keep_germany_czechia_resolved(tmp_path: Path) -> None:
    config = ProductionConfig("TRIAL", False, False, True, False, False, False,
                              tmp_path / "trial.sqlite", tmp_path / "updates.jsonl")
    runner = ProductionRunner(config)
    try:
        production = runner.run("德国U19VS捷克U19")
    finally:
        runner.close()
    assert next(s for s in production.stages if s.stage == "ENTITY_RESOLVER").status == "READY"
    desktop = run_desktop_text("德国U19VS捷克U19", config,
                               mode="AUTO_RESEARCH")
    assert desktop.requests[0].home_team == "德国U19"
    assert desktop.requests[0].away_team == "捷克U19"
    assert "TEAM_NOT_FOUND" not in desktop.report_text


def test_runtime_build_info_exposes_resolver_and_country_versions() -> None:
    info = runtime_build_info()
    assert info["build_id"]
    assert info["build_time"]
    assert info["resolver_version"] == UniversalTeamResolver.version
    assert len(info["country_registry_hash"]) == 64
