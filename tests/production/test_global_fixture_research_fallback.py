"""Fallback routing, official-source evidence, and point-in-time regression matrix."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from erguoyuan_football.knowledge.teams.team_schema import TeamIdentity
from erguoyuan_football.research.global_provider_registry import (
    GlobalProviderRegistry,
    RegisteredProvider,
)
from erguoyuan_football.research.provider_coverage_profile import (
    CoverageStatus,
    ProviderCoverageProfile,
)
from erguoyuan_football.web_research.evidence.evidence_store import EvidenceStore
from production.association_sources import AssociationSourceRegistry
from production.fixture_discovery import (
    FixtureCandidate,
    FixtureDiscoveryRequest,
    FixtureDiscoveryRouter,
)
from production.global_fixture_research import (
    FixtureResearchEvidence,
    GlobalFixtureResearch,
    build_research_source_registry,
    localized_fixture_queries,
)

_AS_OF = datetime(2026, 10, 2, 12, tzinfo=UTC)


def _request(*, prediction_time: datetime = _AS_OF) -> FixtureDiscoveryRequest:
    home = TeamIdentity(
        "NATIONAL_IRL_M_U18",
        "Republic of Ireland U18",
        "Ireland",
        "UEFA",
        ["Ireland U18", "Republic of Ireland U18"],
        "NATIONAL",
        "MEN",
        "U18",
        "YOUTH",
    )
    away = TeamIdentity(
        "NATIONAL_NLD_M_U18",
        "Netherlands U18",
        "Netherlands",
        "UEFA",
        ["Netherlands U18", "Holland U18"],
        "NATIONAL",
        "MEN",
        "U18",
        "YOUTH",
    )
    return FixtureDiscoveryRequest(home, away, prediction_time)


def _fixture(day: int, *, fixture_id: str | None = None) -> FixtureCandidate:
    return FixtureCandidate(
        fixture_id or f"fixture-{day}",
        "NATIONAL_IRL_M_U18",
        "NATIONAL_NLD_M_U18",
        "RESEARCH_COMPETITION_YOUTH",
        "International Youth",
        datetime(2026, 10, day, 16, tzinfo=UTC),
        "STRUCTURED_TEST",
        f"evidence-{day}",
    )


def _coverage_profile() -> ProviderCoverageProfile:
    return ProviderCoverageProfile(
        "TEST_FIXTURE",
        2,
        frozenset({"*"}),
        frozenset({"*"}),
        frozenset({"*"}),
        frozenset({"MEN"}),
        frozenset({"U18"}),
        frozenset({"NATIONAL"}),
        frozenset({"YOUTH"}),
        frozenset({"FRIENDLY"}),
        frozenset({"FIXTURE"}),
        0,
        True,
        False,
        True,
        "test coverage",
        datetime(2026, 10, 1, tzinfo=UTC),
        (("FIXTURE", CoverageStatus.VERIFIED),),
    )


class _StructuredProvider:
    def __init__(self, candidates: tuple[FixtureCandidate, ...]) -> None:
        self.candidates = candidates

    def find_fixtures(
        self, request: FixtureDiscoveryRequest
    ) -> tuple[FixtureCandidate, ...]:
        return self.candidates


def _structured_registry(
    candidates: tuple[FixtureCandidate, ...],
) -> GlobalProviderRegistry:
    return GlobalProviderRegistry(
        (
            RegisteredProvider(
                _coverage_profile(),
                frozenset({"fixtures.example.test"}),
                True,
                provider=_StructuredProvider(candidates),
            ),
        )
    )


class _ResearchProvider:
    def __init__(
        self, *, tier: int = 3, count: int = 1, fail: bool = False, empty: bool = False
    ) -> None:
        self.provider_id = "PRO_RESEARCH" if tier == 2 else "OFFICIAL_RESEARCH"
        self.source_tier = tier
        self.allowed_domains = (
            frozenset({"pro.example.org"})
            if tier == 2
            else frozenset({"fai.ie", "onsoranje.nl", "knvb.nl"})
        )
        self.count = count
        self.fail = fail
        self.empty = empty
        self.calls: list[tuple[str, str | None]] = []

    def search(
        self, query: str, *, source, as_of_time: datetime
    ) -> tuple[FixtureResearchEvidence, ...]:
        self.calls.append((query, source.association_id if source else None))
        if self.fail:
            raise OSError("provider unavailable")
        if self.empty:
            return ()
        if self.source_tier == 3 and source is None:
            return ()
        if self.source_tier == 3:
            source_id = source.association_id
            domain = source.official_domains[0]
        else:
            source_id = self.provider_id
            domain = "pro.example.org"
        kickoff = as_of_time + timedelta(days=2)
        return tuple(
            FixtureResearchEvidence(
                source_id=source_id,
                source_url=f"https://{domain}/teams/u18/fixtures",
                source_name=source_id,
                source_tier=self.source_tier,
                retrieved_at=as_of_time - timedelta(seconds=1),
                home_entity_id="NATIONAL_IRL_M_U18",
                away_entity_id="NATIONAL_NLD_M_U18",
                competition_name=(
                    "4-landentoernooi" if index == 0 else "International Friendly"
                ),
                confidence="HIGH",
                kickoff_utc=kickoff,
                venue="Marbella",
                fixture_id=f"research-fixture-{index}",
            )
            for index in range(self.count)
        )


def _research(
    provider: _ResearchProvider | None = None, store: EvidenceStore | None = None
) -> GlobalFixtureResearch:
    return GlobalFixtureResearch(
        associations=AssociationSourceRegistry(),
        providers=(provider,) if provider else (),
        evidence_store=store,
    )


def test_a_structured_provider_success_does_not_call_research() -> None:
    provider = _ResearchProvider()
    result = FixtureDiscoveryRouter(
        _structured_registry((_fixture(4),)), global_research=_research(provider)
    ).discover(_request())
    assert result.status == "FOUND"
    assert result.research_called is False
    assert provider.calls == []


def test_b_no_structured_coverage_calls_official_research() -> None:
    provider = _ResearchProvider()
    result = FixtureDiscoveryRouter(
        GlobalProviderRegistry(), global_research=_research(provider)
    ).discover(_request())
    assert result.research_called is True
    assert result.structured_status == "NO_COVERAGE"
    assert result.status == "RESEARCH_FOUND"
    assert any(source == "FAI" for _, source in provider.calls)
    assert any(source == "KNVB" for _, source in provider.calls)
    assert result.candidates[0].competition_name == "4-landentoernooi"


def test_c_covered_structured_provider_empty_falls_back_to_research() -> None:
    provider = _ResearchProvider()
    result = FixtureDiscoveryRouter(
        _structured_registry(()), global_research=_research(provider)
    ).discover(_request())
    assert result.structured_status == "NO_VERIFIED_FIXTURE"
    assert result.research_called is True
    assert result.status == "RESEARCH_FOUND"


def test_d_official_failure_does_not_fall_back_to_nonofficial_source() -> None:
    official = _ResearchProvider(fail=True)
    professional = _ResearchProvider(tier=2)
    research = GlobalFixtureResearch(
        associations=AssociationSourceRegistry(), providers=(official, professional)
    )
    assert professional not in research.providers
    result = FixtureDiscoveryRouter(
        GlobalProviderRegistry(),
        global_research=research,
    ).discover(_request())
    assert result.status == "OFFICIAL_SOURCE_UNREACHABLE"
    assert any("FAILED" in attempt for attempt in result.research_sources_attempted)
    assert result.candidates == ()


def test_research_allowlist_rejects_nonofficial_provider() -> None:
    professional = _ResearchProvider(tier=2)
    with pytest.raises(ValueError, match="OFFICIAL_SOURCE_ONLY_PROVIDER_REQUIRED"):
        build_research_source_registry(
            AssociationSourceRegistry(), providers=(professional,)
        )


def test_e_empty_sources_report_no_verified_fixture_not_coverage_blocker() -> None:
    result = FixtureDiscoveryRouter(
        GlobalProviderRegistry(),
        global_research=_research(_ResearchProvider(empty=True)),
    ).discover(_request())
    assert result.status == "OFFICIAL_FIXTURE_NOT_FOUND"
    assert result.reason == "OFFICIAL_PAGES_HAD_NO_EXACT_FIXTURE"


def test_f_multiple_researched_fixtures_are_ambiguous() -> None:
    result = FixtureDiscoveryRouter(
        GlobalProviderRegistry(), global_research=_research(_ResearchProvider(count=2))
    ).discover(_request())
    assert result.status == "OFFICIAL_FIXTURE_AMBIGUOUS"
    assert len(result.candidates) == 2


def test_g_post_match_evidence_is_classified_and_not_available_pre_match(
    tmp_path,
) -> None:
    request = _request()
    associations = AssociationSourceRegistry()
    store = EvidenceStore(
        tmp_path / "fixture-evidence.sqlite",
        build_research_source_registry(associations),
    )
    provider = _ResearchProvider()
    kickoff = request.prediction_time - timedelta(hours=1)

    def post_match_search(query: str, *, source, as_of_time: datetime):
        if source.association_id != "FAI":
            return ()
        return (
            FixtureResearchEvidence(
                source_id="FAI",
                source_url="https://fai.ie/teams/u18/fixtures",
                source_name="FAI",
                source_tier=3,
                retrieved_at=as_of_time - timedelta(seconds=1),
                home_entity_id=request.home_team.team_id,
                away_entity_id=request.away_team.team_id,
                competition_name="4-landentoernooi",
                confidence="HIGH",
                kickoff_utc=kickoff,
                fixture_id="played-fixture",
            ),
        )

    provider.search = post_match_search  # type: ignore[method-assign]
    result = _research(provider, store).discover(
        request, structured_result="NO_COVERAGE"
    )
    assert "POST_MATCH_EVIDENCE" in result.pit_classifications
    assert store.available_at(kickoff - timedelta(minutes=1)) == ()
    assert store.available_at(request.prediction_time)
    store.close()


def test_h_auto_research_does_not_use_jc_provider_state() -> None:
    provider = _ResearchProvider()
    result = FixtureDiscoveryRouter(
        GlobalProviderRegistry(), global_research=_research(provider)
    ).discover(_request())
    assert result.status == "RESEARCH_FOUND"
    assert all("JC" not in attempt for attempt in result.research_sources_attempted)


def test_query_builder_uses_local_aliases_and_age_conventions() -> None:
    request = _request()
    registry = AssociationSourceRegistry()
    sources = registry.for_countries("IRL", "NLD")
    queries = localized_fixture_queries(
        request.home_team,
        request.away_team,
        sources,
        competition_hint="4-landentoernooi",
        date_hint=date(2026, 10, 3),
    )
    assert any("Republic of Ireland U18" in query for query in queries)
    assert any("Ierland O18" in query and "Nederland O18" in query for query in queries)
    assert any(
        "men's international" in query
        and "2026-10-03" in query
        and "4-landentoernooi" in query
        for query in queries
    )


def test_missing_official_site_adapter_is_reported_as_coverage_missing() -> None:
    result = FixtureDiscoveryRouter(
        GlobalProviderRegistry(), global_research=_research()
    ).discover(_request())
    assert result.status == "OFFICIAL_SOURCE_COVERAGE_MISSING"
    assert result.research_status == "OFFICIAL_SOURCE_COVERAGE_MISSING"
