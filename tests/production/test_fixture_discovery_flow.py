"""Competition-less requests resolve teams and enter fixture discovery safely."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from erguoyuan_football.knowledge.teams.team_schema import TeamIdentity
from erguoyuan_football.match_source.match_source import MatchSourceType
from erguoyuan_football.research.global_provider_registry import (
    GlobalProviderRegistry,
    RegisteredProvider,
)
from erguoyuan_football.research.provider_coverage_profile import (
    CoverageStatus,
    ProviderCoverageProfile,
)
from production.fixture_discovery import (
    FixtureCandidate,
    FixtureDiscoveryRequest,
    FixtureDiscoveryRouter,
)
from production.runner import ProductionRunner

_PAIRS = (
    "爱尔兰U18VS荷兰U18",
    "法国U17VS美国U17",
    "德国VS法国",
    "哈萨克斯坦VS摩尔多瓦",
    "俄罗斯U19VS沙特U20",
    "Barcelona VS Real Madrid",
)


@pytest.mark.parametrize("match_text", _PAIRS)
def test_missing_competition_reaches_fixture_discovery(
    production_config,
    match_text: str,
) -> None:
    result = ProductionRunner(production_config).run(match_text)
    stages = {stage.stage: stage for stage in result.stages}
    assert stages["ENTITY_RESOLVER"].status == "READY"
    assert stages["COMPETITION_RESOLVER"].detail == "COMPETITION_DISCOVERY_REQUIRED"
    assert stages["FIXTURE_DISCOVERY"].status == "WARNING"
    assert "OFFICIAL_SOURCE_COVERAGE_MISSING" in stages["FIXTURE_DISCOVERY"].detail
    assert "COMPETITION_NOT_FOUND_OR_MISSING" not in result.rendered_output
    assert "PRIMARY BLOCKER: OFFICIAL_SOURCE_COVERAGE_MISSING" in (
        result.rendered_output
    )
    assert (
        "Research Fixture   OFFICIAL_SOURCE_COVERAGE_MISSING" in result.rendered_output
    )
    assert "global_research=CALLED" in stages["FIXTURE_DISCOVERY"].detail
    assert "research_sources=" in stages["FIXTURE_DISCOVERY"].detail
    assert "NO_PROVIDER_CONFIGURED" in stages["FIXTURE_DISCOVERY"].detail
    if match_text == "爱尔兰U18VS荷兰U18":
        assert "FAI:NO_PROVIDER_CONFIGURED" in stages["FIXTURE_DISCOVERY"].detail
        assert "KNVB:NO_PROVIDER_CONFIGURED" in stages["FIXTURE_DISCOVERY"].detail
    assert "SECONDARY CONSEQUENCES: COMPETITION_UNRESOLVED;SNAPSHOT_NOT_CREATED" in (
        result.rendered_output
    )
    assert next(
        stage for stage in result.stages if stage.stage == "JC_VERIFICATION"
    ).detail == ("NOT_REQUIRED_FOR_AUTO_RESEARCH")


def test_fixture_discovery_accepts_optional_competition_and_date_hints(
    production_config,
) -> None:
    runner = ProductionRunner(production_config)
    result = runner.run("法国U17VS美国U17 2026-11-01", competition="国际友谊赛")
    stages = {stage.stage: stage for stage in result.stages}
    assert result.request.date == date(2026, 11, 1)
    assert stages["FIXTURE_DISCOVERY"].status == "WARNING"
    assert "OFFICIAL_SOURCE_COVERAGE_MISSING" in result.rendered_output
    assert "COMPETITION_NOT_FOUND_OR_MISSING" not in result.rendered_output


def test_jc_confirmation_survives_competitionless_fixture_search(
    production_config,
) -> None:
    result = ProductionRunner(production_config).run(
        "中国竞彩 周001 法国U17VS美国U17", jc_confirmed=True
    )
    assert result.source_type == MatchSourceType.USER_JC_CONFIRMED
    stages = {stage.stage: stage for stage in result.stages}
    assert stages["JC_VERIFICATION"].detail == (
        "USER_CONFIRMED; OFFICIAL_LOOKUP_BYPASSED"
    )
    assert stages["FIXTURE_DISCOVERY"].status == "WARNING"
    assert "OFFICIAL_SOURCE_COVERAGE_MISSING" in result.rendered_output


def _profile() -> ProviderCoverageProfile:
    return ProviderCoverageProfile(
        "TEST_FIXTURE_PROVIDER",
        2,
        frozenset({"*"}),
        frozenset({"*"}),
        frozenset({"*"}),
        frozenset({"MEN"}),
        frozenset({"U17"}),
        frozenset({"NATIONAL"}),
        frozenset({"FIRST_TEAM"}),
        frozenset({"FRIENDLY"}),
        frozenset({"FIXTURE"}),
        0,
        True,
        False,
        True,
        "test capability",
        datetime(2026, 10, 1, tzinfo=UTC),
        (("FIXTURE", CoverageStatus.VERIFIED),),
    )


class _FixtureProvider:
    def __init__(self, candidates: tuple[FixtureCandidate, ...]) -> None:
        self.candidates = candidates

    def find_fixtures(
        self, request: FixtureDiscoveryRequest
    ) -> tuple[FixtureCandidate, ...]:
        return self.candidates


def _request() -> FixtureDiscoveryRequest:
    home = TeamIdentity(
        "NATIONAL_FRA_M_U17",
        "France U17",
        "France",
        "UEFA",
        ["France U17"],
        "NATIONAL",
        "MEN",
        "U17",
    )
    away = TeamIdentity(
        "NATIONAL_USA_M_U17",
        "USA U17",
        "United States",
        "CONCACAF",
        ["USA U17"],
        "NATIONAL",
        "MEN",
        "U17",
    )
    return FixtureDiscoveryRequest(home, away, datetime(2026, 10, 3, tzinfo=UTC))


def _candidate(day: int, *, home: str = "NATIONAL_FRA_M_U17") -> FixtureCandidate:
    return FixtureCandidate(
        f"fixture-{day}",
        home,
        "NATIONAL_USA_M_U17",
        "FIFA_U17_FRIENDLY",
        "International U17 Friendly",
        datetime(2026, 10, day, 18, tzinfo=UTC),
        "TEST_FIXTURE_PROVIDER",
        f"evidence-{day}",
    )


def _registry(candidates: tuple[FixtureCandidate, ...]) -> GlobalProviderRegistry:
    return GlobalProviderRegistry(
        (
            RegisteredProvider(
                _profile(),
                frozenset({"fixtures.example.test"}),
                True,
                provider=_FixtureProvider(candidates),
            ),
        )
    )


def test_fixture_router_uses_exact_registered_provider_and_optional_hints() -> None:
    request = _request()
    result = FixtureDiscoveryRouter(_registry((_candidate(4),))).discover(request)
    assert result.status == "FOUND"
    assert result.providers_queried == ("TEST_FIXTURE_PROVIDER",)
    assert result.candidates[0].competition_name == "International U17 Friendly"
    assert result.candidates[0].kickoff_time == datetime(2026, 10, 4, 18, tzinfo=UTC)


def test_multiple_fixtures_are_ambiguous_and_future_kickoff_is_filtered() -> None:
    request = _request()
    ambiguous = FixtureDiscoveryRouter(
        _registry((_candidate(4), _candidate(5)))
    ).discover(request)
    assert ambiguous.status == "AMBIGUOUS_FIXTURE"
    assert len(ambiguous.candidates) == 2
    future = FixtureDiscoveryRouter(_registry((_candidate(2),))).discover(request)
    assert future.status == "NO_VERIFIED_FIXTURE_FOUND"


def test_fixture_router_rejects_wrong_team_orientation() -> None:
    result = FixtureDiscoveryRouter(
        _registry((_candidate(4, home="NATIONAL_USA_M_U17"),))
    ).discover(_request())
    assert result.status == "NO_VERIFIED_FIXTURE_FOUND"
    assert result.candidates == ()
