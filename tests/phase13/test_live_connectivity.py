"""Offline, synthetic Phase 13.4 boundary tests; no prediction is executed."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from erguoyuan_football.app.input.match_input import MatchInputParserV2
from erguoyuan_football.network.errors import RetryExhausted
from erguoyuan_football.network.schemas import SourceDefinition
from erguoyuan_football.network.source_registry import ExternalSourceRegistry
from erguoyuan_football.research.orchestrator import MatchResearchOrchestrator
from erguoyuan_football.web_research.cache import ResearchCache
from erguoyuan_football.web_research.evidence.evidence_store import EvidenceStore
from erguoyuan_football.web_research.fetch.fixture_identity import (
    FixtureExpectation,
    verify_fixture,
)
from erguoyuan_football.web_research.fetch.live_fetcher import LiveResearchFetcher
from erguoyuan_football.web_research.network.readiness_gate import NetworkReadinessGate
from erguoyuan_football.web_research.policies.conflict_policy import (
    SourcedValue,
    resolve_conflicts,
)
from erguoyuan_football.web_research.policies.freshness_policy import (
    FreshnessPolicy,
    is_valid_for_cutoff,
)
from erguoyuan_football.web_research.providers.base_provider import BaseProvider
from erguoyuan_football.web_research.providers.capabilities import ProviderCapability
from erguoyuan_football.web_research.providers.provider_config import (
    build_live_provider_environment,
)
from erguoyuan_football.web_research.providers.provider_result import ProviderResult
from erguoyuan_football.web_research.providers.provider_router import ProviderRouter
from erguoyuan_football.web_research.search.query_builder import QueryBuilder
from erguoyuan_football.web_research.search.search_task import SearchTask
from erguoyuan_football.web_research.sources.source_registry import SourceRegistry
from erguoyuan_football.web_research.sources.source_schema import SourceRecord
from erguoyuan_football.web_research.time_utils import utc_iso

_NOW = datetime(2026, 10, 1, tzinfo=UTC)
_EXPECT = FixtureExpectation("ENG_ARS", "ENG_LIV")


class _SyntheticProvider(BaseProvider):
    """Deterministic fixture data solely for offline adapter tests."""

    configured = True

    def __init__(self, provider_id: str, tier: int,
                 capabilities: frozenset[ProviderCapability],
                 data: dict[str, Any] | None = None,
                 error: Exception | None = None) -> None:
        self.provider_id = provider_id
        self.provider_name = provider_id
        self.source_tier = tier
        self.capabilities = capabilities
        self.data = data
        self.error = error
        self.calls = 0

    def fetch(self, task: SearchTask) -> ProviderResult:
        """Return a typed synthetic response or a configured transport failure."""
        self.calls += 1
        if self.error is not None:
            raise self.error
        data = self.data or {
            "match_id": "SYNTHETIC_FIXTURE_ID", "competition_id": "ENG_PL",
            "home_team_id": "ENG_ARS", "away_team_id": "ENG_LIV",
            "kickoff_time": "2026-10-02T18:00:00Z", "observed_at": utc_iso(_NOW),
        }
        return ProviderResult(
            True, data, self.provider_name,
            f"https://{self.provider_id.lower()}.example.test/fixture",
            utc_iso(_NOW), "LOW", utc_iso(_NOW),
            provider_id=self.provider_id, source_tier=self.source_tier,
            task_type=task.task_type, observed_time=utc_iso(_NOW),
            http_status=200, endpoint_id="fixture", cache_status="LIVE",
        )


def _runtime(*providers: _SyntheticProvider, dns_ok: bool = True,
             now: datetime = _NOW) -> tuple[LiveResearchFetcher, EvidenceStore, ResearchCache,
                                             SourceRegistry]:
    definitions = tuple(SourceDefinition(
        source_id=p.provider_id, display_name=p.provider_name,
        category="WEB_RESEARCH", base_url=f"https://{p.provider_id.lower()}.example.test",
        endpoints={"fixture": f"https://{p.provider_id.lower()}.example.test/fixture"},
        schema_version="SYNTHETIC_TEST",
    ) for p in providers)
    registry = SourceRegistry(ExternalSourceRegistry(definitions))
    types = {3: "OFFICIAL", 2: "STRUCTURED", 1: "MEDIA"}
    for p in providers:
        registry.add(SourceRecord(
            p.provider_id, p.provider_name, types[p.source_tier], p.source_tier,
            f"https://{p.provider_id.lower()}.example.test", True,
            (f"{p.provider_id.lower()}.example.test",),
        ))
    evidence = EvidenceStore(":memory:", registry, clock=lambda: now)
    cache = ResearchCache(":memory:")
    router = ProviderRouter(registry, providers)
    gate = NetworkReadinessGate(registry, providers, dns_probe=lambda _host: dns_ok)
    return LiveResearchFetcher(router, gate, registry, cache, evidence,
                               clock=lambda: now), evidence, cache, registry


def _fixture_task() -> SearchTask:
    return QueryBuilder.build("Arsenal FC", "Liverpool FC", fixture_key="SYNTHETIC") [0]


def test_provider_capabilities_router_and_priority() -> None:
    official = _SyntheticProvider("OFFICIAL", 3, frozenset({ProviderCapability.FIXTURE}))
    structured = _SyntheticProvider("STRUCTURED", 2, frozenset({ProviderCapability.FIXTURE,
                                                                 ProviderCapability.ODDS}))
    media = _SyntheticProvider("MEDIA", 1, frozenset({ProviderCapability.NEWS}))
    fetcher, _, _, _ = _runtime(structured, media, official)
    assert fetcher.router.providers_for(_fixture_task()) == (official, structured)
    odds_task = next(task for task in QueryBuilder.build("Arsenal FC", "Liverpool FC")
                     if task.task_type == "ODDS")
    assert fetcher.router.providers_for(odds_task) == (structured,)
    assert not official.supports(ProviderCapability.ODDS)


def test_provider_registration_requires_explicit_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    for tier in ("OFFICIAL", "STRUCTURED", "MEDIA"):
        monkeypatch.delenv(f"CORE_{tier}_PROVIDER_URL", raising=False)
    environment = build_live_provider_environment()
    try:
        assert environment.providers == ()
    finally:
        environment.close()
    monkeypatch.setenv("CORE_STRUCTURED_PROVIDER_URL",
                       "https://structured.example.test/api")
    monkeypatch.setenv("CORE_STRUCTURED_PROVIDER_CAPABILITIES", "ODDS")
    environment = build_live_provider_environment()
    try:
        assert len(environment.providers) == 1
        assert environment.providers[0].supports(ProviderCapability.ODDS)
        assert environment.client.registry.get("CORE_STRUCTURED").endpoints["odds"]
    finally:
        environment.close()


def test_network_unavailable_fails_closed() -> None:
    provider = _SyntheticProvider("OFFICIAL", 3, frozenset({ProviderCapability.FIXTURE}))
    fetcher, _, cache, _ = _runtime(provider, dns_ok=False)
    outcome = fetcher.fetch_task(_fixture_task(), match_key="MATCH:SYNTHETIC",
                                 fixture_expectation=_EXPECT)
    assert outcome.status == "UNAVAILABLE"
    assert outcome.error_code == "NETWORK_UNAVAILABLE"
    assert provider.calls == 0
    assert cache.audit_records()[0].error_code == "NETWORK_UNAVAILABLE"


def test_timeout_classified_without_evidence() -> None:
    provider = _SyntheticProvider("OFFICIAL", 3, frozenset({ProviderCapability.FIXTURE}),
                                  error=TimeoutError("synthetic timeout"))
    fetcher, evidence, _, _ = _runtime(provider)
    outcome = fetcher.fetch_task(_fixture_task(), match_key="MATCH:SYNTHETIC",
                                 fixture_expectation=_EXPECT)
    assert outcome.error_code == "PROVIDER_TIMEOUT"
    assert evidence.available_at(_NOW) == ()


def test_rate_limit_classified() -> None:
    provider = _SyntheticProvider("OFFICIAL", 3, frozenset({ProviderCapability.FIXTURE}),
                                  error=RetryExhausted("RATE_LIMITED: synthetic"))
    fetcher, _, cache, _ = _runtime(provider)
    outcome = fetcher.fetch_task(_fixture_task(), match_key="MATCH:SYNTHETIC",
                                 fixture_expectation=_EXPECT)
    assert outcome.error_code == "PROVIDER_RATE_LIMIT"
    assert cache.audit_records()[0].error_code == "PROVIDER_RATE_LIMIT"


def test_fresh_cache_works_while_network_down() -> None:
    provider = _SyntheticProvider("OFFICIAL", 3, frozenset({ProviderCapability.FIXTURE}))
    fetcher, evidence, cache, sources = _runtime(provider)
    task = _fixture_task()
    first = fetcher.fetch_task(task, match_key="MATCH:SYNTHETIC",
                               fixture_expectation=_EXPECT)
    assert first.status == "LIVE" and first.verified_match_id == "SYNTHETIC_FIXTURE_ID"
    offline = LiveResearchFetcher(fetcher.router, NetworkReadinessGate(
        sources, (provider,), dns_probe=lambda _host: False), sources, cache, evidence,
        clock=lambda: _NOW + timedelta(minutes=1))
    second = offline.fetch_task(task, match_key="MATCH:SYNTHETIC",
                                fixture_expectation=_EXPECT)
    assert second.status == "CACHE_FRESH"
    assert second.verified_match_id == first.verified_match_id
    assert provider.calls == 1
    assert cache.audit_records()[-1].cache_hit


def test_stale_cache_not_live_and_attempts_refresh() -> None:
    provider = _SyntheticProvider("OFFICIAL", 3, frozenset({ProviderCapability.FIXTURE}))
    fetcher, evidence, cache, sources = _runtime(provider)
    task = _fixture_task()
    fetcher.fetch_task(task, match_key="MATCH:SYNTHETIC", fixture_expectation=_EXPECT)
    late = _NOW + timedelta(days=2)
    offline = LiveResearchFetcher(fetcher.router, NetworkReadinessGate(
        sources, (provider,), dns_probe=lambda _host: False), sources, cache, evidence,
        clock=lambda: late)
    outcome = offline.fetch_task(task, match_key="MATCH:SYNTHETIC",
                                 fixture_expectation=_EXPECT)
    assert outcome.status == "CACHE_STALE"
    assert outcome.evidence == ()
    assert provider.calls == 1


def test_match_key_prevents_cross_match_cache_use() -> None:
    provider = _SyntheticProvider("OFFICIAL", 3, frozenset({ProviderCapability.FIXTURE}))
    fetcher, evidence, cache, sources = _runtime(provider)
    task = _fixture_task()
    fetcher.fetch_task(task, match_key="MATCH:A", fixture_expectation=_EXPECT)
    offline = LiveResearchFetcher(fetcher.router, NetworkReadinessGate(
        sources, (provider,), dns_probe=lambda _host: False), sources, cache, evidence,
        clock=lambda: _NOW + timedelta(minutes=1))
    outcome = offline.fetch_task(task, match_key="MATCH:B", fixture_expectation=_EXPECT)
    assert outcome.status == "UNAVAILABLE"
    assert outcome.evidence == ()


def test_invalid_odds_never_written() -> None:
    data = {"market_type": "1X2", "selection": "HOME", "bookmaker": "SYNTHETIC",
            "odds": "2.1abc", "observed_at": utc_iso(_NOW)}
    provider = _SyntheticProvider("STRUCTURED", 2, frozenset({ProviderCapability.ODDS}),
                                  data=data)
    fetcher, evidence, _, _ = _runtime(provider)
    task = next(task for task in QueryBuilder.build("Arsenal FC", "Liverpool FC")
                if task.task_type == "ODDS")
    outcome = fetcher.fetch_task(task, match_key="MATCH:SYNTHETIC")
    assert outcome.error_code == "INVALID_PROVIDER_RESPONSE"
    assert evidence.available_at(_NOW) == ()


def test_fixture_reversal_and_ambiguity_rejected() -> None:
    candidate = {"match_id": "SYNTHETIC_A", "competition_id": "ENG_PL",
                 "home_team_id": "ENG_ARS", "away_team_id": "ENG_LIV",
                 "kickoff_time": "2026-10-02T18:00:00Z"}
    reversed_candidate = {**candidate, "home_team_id": "ENG_LIV",
                          "away_team_id": "ENG_ARS"}
    assert verify_fixture(reversed_candidate, _EXPECT).status == "REVERSE_FIXTURE"
    assert verify_fixture({"fixtures": [candidate, {**candidate, "match_id": "SYNTHETIC_B"}]},
                          _EXPECT).status == "FIXTURE_AMBIGUOUS"
    provider = _SyntheticProvider(
        "OFFICIAL", 3, frozenset({ProviderCapability.FIXTURE}),
        data={"fixtures": [candidate, {**candidate, "match_id": "SYNTHETIC_B"}],
              "observed_at": utc_iso(_NOW)},
    )
    fetcher, evidence, _, _ = _runtime(provider)
    outcome = fetcher.fetch_task(_fixture_task(), match_key="RESEARCH_SESSION:SYNTHETIC",
                                 fixture_expectation=_EXPECT)
    assert outcome.error_code == "FIXTURE_AMBIGUOUS"
    assert len(outcome.fixture_candidates) == 2
    assert evidence.available_at(_NOW) == ()


def test_source_conflict_and_same_tier_unresolved() -> None:
    official = SourcedValue("20:00", "OFFICIAL", 3)
    structured = SourcedValue("19:30", "STRUCTURED", 2)
    decision = resolve_conflicts((official, structured))
    assert decision.selected == official
    assert len(decision.conflicts) == 1
    same_tier = resolve_conflicts((structured, SourcedValue("20:00", "STRUCTURED_2", 2)))
    assert same_tier.status == "CONFLICT_UNRESOLVED"
    assert same_tier.selected is None


def test_live_fixture_conflict_preserves_official_selection() -> None:
    official = _SyntheticProvider("OFFICIAL", 3, frozenset({ProviderCapability.FIXTURE}))
    alternate = _SyntheticProvider("OFFICIAL_2", 3, frozenset({ProviderCapability.FIXTURE}),
                                   data={"match_id": "SYNTHETIC_ALT",
                                         "competition_id": "ENG_PL",
                                         "home_team_id": "ENG_ARS",
                                         "away_team_id": "ENG_LIV",
                                         "kickoff_time": "2026-10-02T19:00:00Z",
                                         "observed_at": utc_iso(_NOW)})
    fetcher, _, _, _ = _runtime(official, alternate)
    result = fetcher.fetch_task(_fixture_task(), match_key="MATCH:SYNTHETIC",
                                fixture_expectation=_EXPECT)
    assert result.status == "CONFLICT_UNRESOLVED"
    assert result.conflicts


def test_official_kickoff_wins_over_structured_with_conflict_record() -> None:
    official = _SyntheticProvider("OFFICIAL", 3, frozenset({ProviderCapability.FIXTURE}))
    structured = _SyntheticProvider("STRUCTURED", 2,
                                    frozenset({ProviderCapability.FIXTURE}),
                                    data={"match_id": "SYNTHETIC_FIXTURE_ID",
                                          "competition_id": "ENG_PL",
                                          "home_team_id": "ENG_ARS",
                                          "away_team_id": "ENG_LIV",
                                          "kickoff_time": "2026-10-02T19:00:00Z",
                                          "observed_at": utc_iso(_NOW)})
    fetcher, _, _, _ = _runtime(official, structured)
    result = fetcher.fetch_task(_fixture_task(), match_key="MATCH:SYNTHETIC",
                                fixture_expectation=_EXPECT)
    assert result.verified_match_id == "SYNTHETIC_FIXTURE_ID"
    assert result.status == "LIVE"
    assert len(result.conflicts) == 1
    assert result.conflicts[0].first.tier == 3
    assert result.selected_value is not None and result.selected_value.tier == 3


def test_freshness_and_point_in_time() -> None:
    policy = FreshnessPolicy()
    assert not policy.is_fresh("ODDS", _NOW, _NOW + timedelta(minutes=16))
    provider = _SyntheticProvider("OFFICIAL", 3, frozenset({ProviderCapability.FIXTURE}))
    fetcher, _, _, _ = _runtime(provider)
    evidence = fetcher.fetch_task(_fixture_task(), match_key="MATCH:A",
                                  fixture_expectation=_EXPECT).evidence[0]
    assert not is_valid_for_cutoff(evidence, _NOW - timedelta(seconds=1))
    assert is_valid_for_cutoff(evidence, _NOW)


def test_orchestrator_live_integration_stays_outside_prediction() -> None:
    provider = _SyntheticProvider("OFFICIAL", 3, frozenset({ProviderCapability.FIXTURE}))
    stats = _SyntheticProvider(
        "STRUCTURED", 2, frozenset({ProviderCapability.TEAM_STATS}),
        data={"team_id": "ENG_ARS", "competition": "Premier League",
              "season": "2026", "window": "last_10", "metric": "xG",
              "value": 1.5, "observed_at": utc_iso(_NOW)},
    )
    fetcher, evidence, _, sources = _runtime(provider, stats)
    request = MatchInputParserV2().parse("Arsenal VS Liverpool")
    request = replace(request, competition=None)
    package = MatchResearchOrchestrator(sources=sources, evidence_store=evidence,
                                        live_fetcher=fetcher).build(
        request, as_of_time=_NOW, fetch_live=True)
    assert package.match_id == "SYNTHETIC_FIXTURE_ID"
    assert package.research_session_id is not None
    assert package.available_data["fixture"]
    assert package.available_data["xg"]
    assert "odds" in package.missing_data
    assert {source.source_id for source in package.sources} == {"OFFICIAL", "STRUCTURED"}
    assert package.quality_score > 0
    assert package.network_status == "AVAILABLE"
    assert not package.prediction_executed


def test_audit_excludes_response_body_and_credentials() -> None:
    provider = _SyntheticProvider("OFFICIAL", 3, frozenset({ProviderCapability.FIXTURE}),
                                  error=ValueError("AUTH_FAILED"))
    fetcher, _, cache, _ = _runtime(provider)
    task = replace(_fixture_task(), query="Arsenal token=secret-token password=secret-password")
    fetcher.fetch_task(task, match_key="MATCH:SYNTHETIC",
                       fixture_expectation=_EXPECT)
    audit = repr(cache.audit_records())
    assert "secret-token" not in audit
    assert "secret-password" not in audit
    assert "sha256:" in audit
