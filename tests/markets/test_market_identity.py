"""Provider event identity must be established before quotes enter production storage."""

from datetime import timedelta

import pytest

from erguoyuan_football.data.schemas import Fixture, TeamAlias
from erguoyuan_football.external.market import MarketProviderResult
from erguoyuan_football.markets.identity import (
    EventOrientation,
    MarketEventBinder,
    ProviderEventIdentity,
)
from erguoyuan_football.markets.pipeline import MarketDataPipeline
from erguoyuan_football.markets.schemas import MarketType, Selection
from erguoyuan_football.network.client import CoreNetworkClient
from erguoyuan_football.network.schemas import (
    NetworkHealthReport,
    RequiredLevel,
    SourceDefinition,
    SourceHealth,
    SourceHealthStatus,
)
from erguoyuan_football.network.source_registry import ExternalSourceRegistry

pytestmark = pytest.mark.fast


class NoopHTTPClient:
    def close(self) -> None:
        pass


def _event(fixture: Fixture, **updates: object) -> ProviderEventIdentity:
    return ProviderEventIdentity.model_validate({
        "source_event_id": "event-1", "competition_name": "Premier League",
        "home_team_name": "Arsenal", "away_team_name": "Chelsea",
        "kickoff_time": fixture.kickoff_time,
        **updates,
    })


def _quote(quote_factory, fixture: Fixture):
    return quote_factory()[0].model_copy(update={
        "match_id": fixture.match_id, "source_event_id": "event-1",
    })


def test_provider_event_binds_via_exact_fixture_aliases(store, at, quote_factory) -> None:
    fixture = next(item for item in store.fixtures_at(at) if item.match_id == "test_match_1")
    quote = _quote(quote_factory, fixture)
    binding, quotes = MarketEventBinder(store).bind(fixture, _event(fixture), (quote,), prediction_time=at)
    assert binding.orientation == EventOrientation.NORMAL
    assert binding.match_id == fixture.match_id
    assert quotes == (quote,)


def test_swapped_provider_home_away_reverses_1x2_and_handicap(store, at, quote_factory) -> None:
    fixture = next(item for item in store.fixtures_at(at) if item.match_id == "test_match_1")
    home = _quote(quote_factory, fixture).model_copy(update={"market_type": MarketType.MATCH_1X2,
        "selection": Selection.HOME, "line_quarters": None})
    handicap = home.model_copy(update={"quote_id": "provider-ah", "market_type": MarketType.ASIAN_HANDICAP,
        "line_quarters": -2})
    event = _event(fixture, home_team_name="Chelsea", away_team_name="Arsenal")
    binding, quotes = MarketEventBinder(store).bind(fixture, event, (home, handicap), prediction_time=at)
    assert binding.orientation == EventOrientation.SWAPPED
    assert quotes[0].selection == Selection.AWAY
    assert quotes[1].selection == Selection.AWAY
    assert quotes[1].line_quarters == 2
    assert all(item.quote_id.startswith("swap:") for item in quotes)


@pytest.mark.parametrize(("changes", "error"), [
    ({"competition_name": "Champions League"}, "MARKET_EVENT_AMBIGUOUS_OR_UNRESOLVED"),
    ({"home_team_name": "Real Madrid"}, "MARKET_EVENT_AMBIGUOUS_OR_UNRESOLVED"),
    ({"kickoff_time": "2026-09-30T22:00:00Z"}, "MARKET_EVENT_KICKOFF_MISMATCH"),
])
def test_event_mismatch_is_rejected(store, at, quote_factory, changes, error) -> None:
    fixture = next(item for item in store.fixtures_at(at) if item.match_id == "test_match_1")
    with pytest.raises(ValueError, match=error):
        MarketEventBinder(store).bind(fixture, _event(fixture, **changes),
                                      (_quote(quote_factory, fixture),), prediction_time=at)


def test_same_teams_week_later_cannot_match_by_names_only(store, at, quote_factory) -> None:
    fixture = next(item for item in store.fixtures_at(at) if item.match_id == "test_match_1")
    later = fixture.model_copy(update={"match_id": "rematch", "kickoff_time": fixture.kickoff_time + timedelta(days=7)})
    store.add_fixture(later)
    with pytest.raises(ValueError, match="MARKET_EVENT_KICKOFF_MISMATCH"):
        MarketEventBinder(store).bind(fixture, _event(later),
                                      (_quote(quote_factory, fixture),), prediction_time=at)
    binding, _ = MarketEventBinder(store).bind(fixture, _event(fixture),
                                               (_quote(quote_factory, fixture),), prediction_time=at)
    assert binding.match_id == fixture.match_id


def test_stale_caller_fixture_cannot_override_catalog(store, at, quote_factory) -> None:
    fixture = next(item for item in store.fixtures_at(at) if item.match_id == "test_match_1")
    stale = fixture.model_copy(update={"competition_id": "test_ucl"})
    with pytest.raises(ValueError, match="MARKET_FIXTURE_NOT_CURRENT_CATALOG_VERSION"):
        MarketEventBinder(store).bind(stale, _event(fixture),
                                      (_quote(quote_factory, fixture),), prediction_time=at)


def test_duplicate_fixture_at_same_kickoff_is_ambiguous(store, at, quote_factory) -> None:
    fixture = next(item for item in store.fixtures_at(at) if item.match_id == "test_match_1")
    store.add_fixture(fixture.model_copy(update={"match_id": "duplicate-event"}))
    with pytest.raises(ValueError, match="MARKET_EVENT_AMBIGUOUS_OR_UNRESOLVED"):
        MarketEventBinder(store).bind(fixture, _event(fixture),
                                      (_quote(quote_factory, fixture),), prediction_time=at)


def test_reverse_fixture_at_same_kickoff_is_ambiguous(store, at, quote_factory) -> None:
    fixture = next(item for item in store.fixtures_at(at) if item.match_id == "test_match_1")
    store.add_fixture(fixture.model_copy(update={
        "match_id": "reverse-fixture", "home_team_id": fixture.away_team_id,
        "away_team_id": fixture.home_team_id,
    }))
    with pytest.raises(ValueError, match="MARKET_EVENT_AMBIGUOUS_OR_UNRESOLVED"):
        MarketEventBinder(store).bind(fixture, _event(fixture),
                                      (_quote(quote_factory, fixture),), prediction_time=at)


def test_ambiguous_provider_team_alias_is_rejected(store, at, quote_factory) -> None:
    fixture = next(item for item in store.fixtures_at(at) if item.match_id == "test_match_1")
    store.add_team_alias(TeamAlias(alias="Arsenal", team_id="test_chelsea", language="en",
        source="SYNTHETIC_TEST", confidence=1, created_at=at - timedelta(days=1)))
    with pytest.raises(ValueError, match="MARKET_EVENT_AMBIGUOUS_OR_UNRESOLVED"):
        MarketEventBinder(store).bind(fixture, _event(fixture),
                                      (_quote(quote_factory, fixture),), prediction_time=at)


def test_live_pipeline_binds_event_before_quote_persistence(
    store, at, quote_factory, provisional_market_engine, monkeypatch,
) -> None:
    fixture = next(item for item in store.fixtures_at(at) if item.match_id == "test_match_1")
    source_id = "synthetic-authorized-provider"
    quote = _quote(quote_factory, fixture).model_copy(update={"provider_id": source_id})
    report = NetworkHealthReport(status="PASS", checked_at=at, sources=(SourceHealth(
        source_id=source_id, status=SourceHealthStatus.HEALTHY, checked_at=at,
        authentication_status="NOT_REQUIRED", schema_valid=True, response_non_empty=True,
    ),))
    monkeypatch.setattr("erguoyuan_football.markets.pipeline.run_network_preflight",
                        lambda *args, **kwargs: report)
    registry = ExternalSourceRegistry((SourceDefinition(source_id=source_id, display_name="Synthetic test",
        category="MARKET", required_level=RequiredLevel.REQUIRED, schema_version="test-v1"),))
    client = CoreNetworkClient(registry, http_client=NoopHTTPClient())

    class SyntheticProvider:
        source_id = "synthetic-authorized-provider"

        def fetch_quotes(self, match_id, source_event_id, prediction_time):
            return MarketProviderResult(status="AVAILABLE", source_id=self.source_id,
                retrieved_at=at, event_identity=_event(fixture, home_team_name="Chelsea",
                    away_team_name="Arsenal"), quotes=(quote,))

    pipeline = MarketDataPipeline(provisional_market_engine, client, SyntheticProvider(),
        schema_validators={}, store=store, clock=lambda: at)
    unverified_opening = quote_factory(opening=True)[0].model_copy(update={
        "match_id": fixture.match_id, "source_event_id": "event-1", "provider_id": source_id,
    })
    with pytest.raises(ValueError, match="UNVERIFIED_OPENING_QUOTES:NOT_PERSISTED"):
        pipeline.execute_live(fixture, "event-1", opening_quotes=(unverified_opening,))
    assert store.connection.execute("SELECT COUNT(*) FROM odds_quotes").fetchone()[0] == 0
    result = pipeline.execute_live(fixture, "event-1")
    client.close()
    assert result.event_binding is not None
    assert result.event_binding.orientation == EventOrientation.SWAPPED
    assert result.engine_result.market_snapshot.quotes[0].selection == Selection.AWAY
    assert "MARKET_EVENT_BINDING_MISSING" not in (result.engine_result.reason or "")
    assert store.connection.execute("SELECT COUNT(*) FROM odds_quotes WHERE match_id=?",
                                    [fixture.match_id]).fetchone()[0] == 1
    assert store.connection.execute("SELECT COUNT(*) FROM market_event_bindings WHERE match_id=?",
                                    [fixture.match_id]).fetchone()[0] == 1


def test_live_pipeline_rejects_unresolved_provider_event(
    store, at, quote_factory, provisional_market_engine, monkeypatch,
) -> None:
    fixture = next(item for item in store.fixtures_at(at) if item.match_id == "test_match_1")
    source_id = "synthetic-authorized-provider"
    quote = _quote(quote_factory, fixture).model_copy(update={"provider_id": source_id})
    report = NetworkHealthReport(status="PASS", checked_at=at, sources=(SourceHealth(
        source_id=source_id, status=SourceHealthStatus.HEALTHY, checked_at=at,
        authentication_status="NOT_REQUIRED", schema_valid=True, response_non_empty=True,
    ),))
    monkeypatch.setattr("erguoyuan_football.markets.pipeline.run_network_preflight",
                        lambda *args, **kwargs: report)
    registry = ExternalSourceRegistry((SourceDefinition(source_id=source_id, display_name="Synthetic test",
        category="MARKET", required_level=RequiredLevel.REQUIRED, schema_version="test-v1"),))
    client = CoreNetworkClient(registry, http_client=NoopHTTPClient())

    class WrongEventProvider:
        source_id = "synthetic-authorized-provider"

        def fetch_quotes(self, match_id, source_event_id, prediction_time):
            return MarketProviderResult(status="AVAILABLE", source_id=self.source_id,
                retrieved_at=at, event_identity=_event(fixture, competition_name="Champions League"),
                quotes=(quote,))

    result = MarketDataPipeline(provisional_market_engine, client, WrongEventProvider(),
        schema_validators={}, store=store, clock=lambda: at).execute_live(fixture, "event-1")
    client.close()
    assert result.event_binding is None
    assert result.provider_result is not None and result.provider_result.status == "DATA_UNAVAILABLE"
    assert result.engine_result.production_gate_status == "PREDICTION_BLOCKED"
    assert "MARKET_EVENT_BINDING_MISSING" in (result.engine_result.reason or "")
    assert store.connection.execute("SELECT COUNT(*) FROM odds_quotes").fetchone()[0] == 0
    assert store.connection.execute("SELECT COUNT(*) FROM market_event_bindings").fetchone()[0] == 0


def test_provider_event_id_cannot_be_rebound_to_another_fixture(store, at, quote_factory) -> None:
    fixture = next(item for item in store.fixtures_at(at) if item.match_id == "test_match_1")
    binding, _ = MarketEventBinder(store).bind(fixture, _event(fixture),
                                               (_quote(quote_factory, fixture),), prediction_time=at)
    store.save_market_event_binding(binding)
    store.save_market_event_binding(binding)
    assert store.connection.execute("SELECT COUNT(*) FROM market_event_bindings").fetchone()[0] == 1
    another_match = binding.model_copy(update={"binding_id": "rebinding", "match_id": "test_match_2"})
    with pytest.raises(ValueError, match="PROVIDER_EVENT_REBOUND_TO_DIFFERENT_FIXTURE_OR_DIRECTION"):
        store.save_market_event_binding(another_match)


def test_preexisting_quote_for_other_match_blocks_event_binding(store, at, quote_factory) -> None:
    fixture = next(item for item in store.fixtures_at(at) if item.match_id == "test_match_1")
    binding, _ = MarketEventBinder(store).bind(fixture, _event(fixture),
                                               (_quote(quote_factory, fixture),), prediction_time=at)
    conflicting_quote = _quote(quote_factory, fixture).model_copy(update={
        "match_id": "test_match_2", "quote_id": "wrong-fixture-quote",
    })
    store.add_odds_quote(conflicting_quote)
    with pytest.raises(ValueError, match="PROVIDER_EVENT_QUOTE_ALREADY_BOUND_TO_OTHER_FIXTURE"):
        store.save_market_event_binding(binding)
