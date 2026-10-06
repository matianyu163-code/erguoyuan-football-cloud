"""Synthetic contract tests for global provider routing and merge governance."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from erguoyuan_football.research.coverage_matrix import build_coverage_matrix
from erguoyuan_football.research.coverage_resolver import (
    CompetitionDataProfile,
    CoverageResolver,
    parse_competition_data_profile,
)
from erguoyuan_football.research.global_provider_registry import (
    GlobalProviderRegistry,
    RegisteredProvider,
)
from erguoyuan_football.research.global_registry_factory import (
    build_global_provider_registry,
)
from erguoyuan_football.research.live_data.football_data_org_directory import (
    FootballDataOrgDirectoryProvider,
)
from erguoyuan_football.research.live_data.openligadb_directory import (
    OpenLigaDBDirectoryProvider,
)
from erguoyuan_football.research.multi_provider_fetcher import MultiProviderFetcher
from erguoyuan_football.research.provider_coverage_profile import (
    CoverageStatus,
    ProviderCoverageProfile,
)
from erguoyuan_football.research.provider_merge import (
    ProviderMatchRecord,
    merge_historical_results,
)

ROOT = Path(__file__).resolve().parents[2]


def _profile(provider_id: str, *, status: CoverageStatus,
             tier: int = 2, capability: str = "TEAM_DIRECTORY",
             gender: str = "WOMEN") -> ProviderCoverageProfile:
    now = datetime.now(UTC)
    return ProviderCoverageProfile(
        provider_id, tier, frozenset({"EUROPE"}), frozenset({"GER"}),
        frozenset({"UEFA"}), frozenset({gender}), frozenset({"SENIOR"}),
        frozenset({"CLUB"}), frozenset({"FIRST_TEAM"}), frozenset({"LEAGUE"}),
        frozenset({capability}), 10, True, False, True, "ODbL test profile",
        now, ((capability, status),),
    )


def test_declared_capability_is_not_verified() -> None:
    profile = _profile("P", status=CoverageStatus.UNVERIFIED)
    assert profile.status_for("TEAM_DIRECTORY") == CoverageStatus.UNVERIFIED
    assert profile.status_for("XG") == CoverageStatus.UNSUPPORTED


def test_global_registry_factory_connects_configured_sources_without_promotion() -> None:
    registry = build_global_provider_registry(ROOT / "config/phase13_7_openligadb.yaml")
    try:
        ids = {row.profile.provider_id for row in registry.list()}
        assert ids == {"OPENLIGADB_GLOBAL_DIRECTORY", "FOOTBALL_DATA_ORG_DIRECTORY"}
        openligadb = registry.get("OPENLIGADB_GLOBAL_DIRECTORY")
        assert openligadb.profile.status_for("TEAM_DIRECTORY") == CoverageStatus.UNVERIFIED
        assert registry.get("FOOTBALL_DATA_ORG_DIRECTORY").profile.status_for(
            "COMPETITION_DIRECTORY") == CoverageStatus.UNVERIFIED
    finally:
        for row in registry.list():
            row.provider.close()


def test_coverage_resolver_hard_filters_gender_and_ranks_verified() -> None:
    verified = RegisteredProvider(_profile("verified", status=CoverageStatus.VERIFIED),
                                  frozenset({"api.example.test"}), True, "HEALTHY")
    partial = RegisteredProvider(_profile("partial", status=CoverageStatus.PARTIAL, tier=2),
                                 frozenset({"other.example.test"}), True)
    men = RegisteredProvider(_profile("men", status=CoverageStatus.VERIFIED, gender="MEN"),
                             frozenset({"men.example.test"}), True)
    resolver = CoverageResolver(GlobalProviderRegistry((partial, men, verified)))
    result = resolver.resolve(CompetitionDataProfile(
        "WOMEN_GER", "GER", "UEFA", "WOMEN", "SENIOR", "CLUB",
        "FIRST_TEAM", "LEAGUE", "EUROPE"), "TEAM_DIRECTORY")
    assert [row.profile.provider_id for row in result] == ["verified", "partial"]


def test_competition_profile_retains_unknown_dimensions() -> None:
    profile = parse_competition_data_profile("YOUTH_FRIENDLY", "国际友谊赛")
    assert profile.competition_type == "FRIENDLY"
    assert profile.country == "UNKNOWN"
    assert profile.gender == "UNKNOWN"


def test_coverage_matrix_does_not_promote_declarations() -> None:
    row = RegisteredProvider(_profile("declared", status=CoverageStatus.UNVERIFIED),
                             frozenset({"api.example.test"}), True)
    matrix = build_coverage_matrix(GlobalProviderRegistry((row,)),
        (CompetitionDataProfile("WOMEN_GER", country="GER", gender="WOMEN",
                                age_group="SENIOR", entity_type="CLUB",
                                squad_level="FIRST_TEAM", competition_type="LEAGUE",
                                region="EUROPE"),), ("TEAM_DIRECTORY", "XG"))
    assert matrix[0].status == CoverageStatus.UNVERIFIED
    assert matrix[1].status == CoverageStatus.UNSUPPORTED


def test_multi_provider_fetch_is_bounded_and_audited() -> None:
    providers = tuple(RegisteredProvider(_profile(name, status=CoverageStatus.VERIFIED),
        frozenset({f"{name}.example.test"}), True) for name in ("one", "two", "three", "four"))
    calls: list[str] = []

    def fail() -> object:
        calls.append("one")
        raise TimeoutError("PROVIDER_TIMEOUT")

    def good() -> SimpleNamespace:
        calls.append("two")
        return SimpleNamespace(evidence_ids=("E2",))

    result = MultiProviderFetcher(max_providers_per_task=2).fetch(providers,
        {"one": fail, "two": good}, {"one": bool, "two": bool})
    assert calls == ["one", "two"]
    assert result.status == "AVAILABLE" and result.provider_id == "two"
    assert result.evidence_ids == ("E2",)
    assert tuple(row.status for row in result.attempts) == ("FAILED", "SUCCESS")


def _match(provider: str, match_id: str, home: str, away: str, kickoff: datetime,
           hg: int, ag: int) -> ProviderMatchRecord:
    return ProviderMatchRecord(match_id, home, away, kickoff, "COMP", hg, ag,
        "FINISHED", provider, f"https://{provider.lower()}.example.test/match/{match_id}",
        kickoff + timedelta(days=1), (f"E_{match_id}",))


def test_merge_deduplicates_agreement_and_excludes_score_conflict() -> None:
    kickoff = datetime(2025, 1, 1, tzinfo=UTC)
    result = merge_historical_results((
        _match("A", "a1", "H", "A", kickoff, 2, 0),
        _match("B", "b1", "H", "A", kickoff, 2, 0),
        _match("C", "c1", "X", "Y", kickoff, 1, 0),
        _match("D", "d1", "X", "Y", kickoff, 0, 0),
    ))
    assert len(result.records) == 1
    assert result.records[0].evidence_ids == ("E_a1", "E_b1")
    assert len(result.conflicts) == 1
    assert result.conflicts[0].conflict_type == "SCORE_OR_STATUS_CONFLICT"


def test_merge_rejects_reversed_orientation_and_kickoff_conflict() -> None:
    kickoff = datetime(2025, 1, 1, tzinfo=UTC)
    result = merge_historical_results((
        _match("A", "a1", "H", "A", kickoff, 2, 0),
        _match("B", "b1", "A", "H", kickoff, 0, 2),
        _match("C", "c1", "X", "Y", kickoff, 1, 0),
        _match("D", "d1", "X", "Y", kickoff + timedelta(hours=2), 1, 0),
    ))
    assert not result.records
    assert {row.conflict_type for row in result.conflicts} == {
        "ORIENTATION_OR_KICKOFF_CONFLICT"}


class _FakeClient:
    """Synthetic test transport returning public API shaped JSON only."""

    def __init__(self, bodies: dict[str, object]) -> None:
        self.bodies = bodies
        self.calls: list[str] = []

    def fetch_json(self, source_id: str, endpoint_id: str, **kwargs: object) -> SimpleNamespace:
        self.calls.append(endpoint_id)
        body = self.bodies[endpoint_id]
        stamp = datetime(2026, 10, 1, tzinfo=UTC)
        return SimpleNamespace(body=body, retrieved_at=stamp,
            content_hash="abc123", final_url=f"https://api.openligadb.de/{endpoint_id}")


class _FootballDataFakeClient:
    """Synthetic test transport for the public competition list shape."""

    def fetch_json(self, source_id: str, endpoint_id: str, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(body={"competitions": [{"id": 1, "code": "PL",
            "name": "Premier League", "area": {"code": "ENG"}}]},
            retrieved_at=datetime(2026, 10, 1, tzinfo=UTC), content_hash="fd123",
            final_url="https://api.football-data.org/v4/competitions")


def test_football_data_directory_persists_source_evidence() -> None:
    provider = FootballDataOrgDirectoryProvider(client=_FootballDataFakeClient())
    try:
        result = provider.fetch_competitions()
        assert result.status == CoverageStatus.VERIFIED
        assert result.competitions[0]["code"] == "PL"
        assert result.evidence_id is not None
        assert provider.evidence_store.get(result.evidence_id) is not None
        profile = provider.profile_after_live_test(result)
        assert profile.status_for("COMPETITION_DIRECTORY") == CoverageStatus.VERIFIED
        assert profile.status_for("TEAM_DIRECTORY") == CoverageStatus.UNSUPPORTED
    finally:
        provider.close()


def test_openligadb_directory_uses_only_registered_scopes_and_saves_evidence() -> None:
    client = _FakeClient({"competitions_2026": [{"leagueShortcut": "ffb1"}],
        "teams_germany_women_top_division_2026": [
            {"teamId": 3, "teamName": "Test Frauen"}]})
    provider = OpenLigaDBDirectoryProvider(ROOT / "config/phase13_7_openligadb.yaml",
                                          client=client)  # SYNTHETIC_TEST transport
    try:
        leagues = provider.fetch_competitions(2026)
        women = provider.fetch_teams("germany_women_top_division_2026")
        assert leagues.status == CoverageStatus.VERIFIED
        assert women.status == CoverageStatus.VERIFIED
        assert women.rows[0].competition.gender == "WOMEN"
        assert women.rows[0].competition.age_group == "SENIOR"
        assert provider.evidence_store.get(women.rows[0].evidence_id) is not None
        assert provider.fetch_teams("germany_u16").reason == "PROVIDER_COVERAGE_MISSING"
        assert client.calls == ["competitions_2026", "teams_germany_women_top_division_2026"]
    finally:
        provider.close()
