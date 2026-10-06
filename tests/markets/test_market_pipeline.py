"""Offline tests for the network-gated market provider boundary."""

from datetime import timedelta
from types import SimpleNamespace

import pytest

from erguoyuan_football.contracts.common import now
from erguoyuan_football.external.market import CoreNetworkMarketProvider
from erguoyuan_football.markets.normalizer import BookmakerRegistry
from erguoyuan_football.markets.pipeline import MarketDataPipeline
from erguoyuan_football.markets.schemas import Bookmaker, SourceType
from erguoyuan_football.network.client import CoreNetworkClient
from erguoyuan_football.network.errors import RetryExhausted
from erguoyuan_football.network.schemas import NetworkResponse
from erguoyuan_football.network.source_registry import ExternalSourceRegistry

pytestmark = pytest.mark.fast


class NoopHTTPClient:
    """Transport is injected because these tests must not access the Internet."""

    def close(self) -> None:
        pass


def _market_provider(client: CoreNetworkClient) -> CoreNetworkMarketProvider:
    bookmaker = Bookmaker(bookmaker_id="book-a", canonical_name="Book A",
                          provider_mapping={"authorized-market": "Book A"})
    return CoreNetworkMarketProvider(client, source_id="authorized-market", endpoint_id="odds",
        source_type=SourceType.AUTHORIZED_PROVIDER, bookmaker_registry=BookmakerRegistry((bookmaker,)),
        schema_version="market-v1")


def test_market_provider_uses_core_network_client(monkeypatch, market_fixture) -> None:
    client = CoreNetworkClient(ExternalSourceRegistry(), http_client=NoopHTTPClient())
    provider = _market_provider(client)
    called = {}
    as_of = market_fixture.kickoff_time - timedelta(hours=2)
    retrieved = as_of + timedelta(seconds=5)
    response = NetworkResponse(source_id="authorized-market", endpoint_id="odds", status_code=200,
        body={"event": {"source_event_id": "event-1", "competition_name": "Test League",
            "home_team_name": "Test Home", "away_team_name": "Test Away",
            "kickoff_time": market_fixture.kickoff_time.isoformat()},
            "quotes": [{"source_event_id": "event-1",
            "bookmaker": "Book A", "market_type": "1X2", "selection": "HOME", "odds": "2.125",
            "odds_format": "DECIMAL", "source_time": as_of.isoformat(),
            "timestamp_quality": "SOURCE_NATIVE"}]}, retrieved_at=retrieved, as_of_time=as_of,
        content_hash="synthetic-response-hash", schema_version="market-v1", latency_ms=1.0)

    def fetch_json(source_id, endpoint_id, *, params, prediction_time):
        called.update(source_id=source_id, endpoint_id=endpoint_id, params=params, prediction_time=prediction_time)
        return response

    monkeypatch.setattr(client, "fetch_json", fetch_json)
    result = provider.fetch_quotes(market_fixture.match_id, "event-1", retrieved)
    client.close()
    assert called["source_id"] == "authorized-market"
    assert called["endpoint_id"] == "odds"
    assert called["params"] == {"event_id": "event-1"}
    assert result.status == "AVAILABLE"
    assert result.event_identity is not None
    assert result.quotes[0].match_id == market_fixture.match_id
    assert str(result.quotes[0].odds_decimal) == "2.125"
    assert result.quotes[0].retrieved_at == retrieved


@pytest.mark.parametrize(("error", "expected"), [
    (ValueError("AUTH_FAILED:401"), "AUTH_FAILED"),
    (RetryExhausted("RATE_LIMITED: retry budget exhausted"), "RATE_LIMITED"),
    (KeyError("SOURCE_NOT_CONFIGURED:market"), "NETWORK_UNAVAILABLE"),
])
def test_market_provider_maps_network_failures(monkeypatch, market_fixture, error, expected) -> None:
    client = CoreNetworkClient(ExternalSourceRegistry(), http_client=NoopHTTPClient())
    monkeypatch.setattr(client, "fetch_json", lambda *args, **kwargs: (_ for _ in ()).throw(error))
    result = _market_provider(client).fetch_quotes(market_fixture.match_id, "event-1", now())
    client.close()
    assert result.status == expected
    assert result.reason


def test_empty_provider_quote_list_is_unavailable(monkeypatch, market_fixture) -> None:
    client = CoreNetworkClient(ExternalSourceRegistry(), http_client=NoopHTTPClient())
    provider = _market_provider(client)
    at = market_fixture.kickoff_time - timedelta(hours=1)
    response = SimpleNamespace(body={"quotes": []}, retrieved_at=at, as_of_time=at)
    monkeypatch.setattr(client, "fetch_json", lambda *args, **kwargs: response)
    result = provider.fetch_quotes(market_fixture.match_id, "event-1", at)
    client.close()
    assert result.status == "DATA_UNAVAILABLE"
    assert result.reason == "EMPTY_MARKET_QUOTE_SET"


def test_provider_rejects_quotes_without_event_identity(monkeypatch, market_fixture) -> None:
    client = CoreNetworkClient(ExternalSourceRegistry(), http_client=NoopHTTPClient())
    provider = _market_provider(client)
    at = market_fixture.kickoff_time - timedelta(hours=1)
    response = SimpleNamespace(body={"quotes": [{"source_event_id": "event-1"}]},
                               retrieved_at=at, as_of_time=at)
    monkeypatch.setattr(client, "fetch_json", lambda *args, **kwargs: response)
    result = provider.fetch_quotes(market_fixture.match_id, "event-1", at)
    client.close()
    assert result.status == "DATA_UNAVAILABLE"
    assert result.quotes == ()
    assert result.reason is not None and result.reason.startswith("PROVIDER_EVENT_IDENTITY_INVALID")


def test_enhanced_only_blocks_market_run_without_registered_legal_source(
    market_fixture, provisional_market_engine,
) -> None:
    client = CoreNetworkClient(ExternalSourceRegistry(), http_client=NoopHTTPClient())

    class ProviderThatMustNotRun:
        source_id = "missing-provider"

        def fetch_quotes(self, *args, **kwargs):
            raise AssertionError("unregistered provider must not be called")

    at = market_fixture.kickoff_time - timedelta(hours=24)
    pipeline = MarketDataPipeline(provisional_market_engine, client, ProviderThatMustNotRun(),
                                  schema_validators={}, clock=lambda: at)
    result = pipeline.execute_live(market_fixture, "event-1")
    client.close()
    assert result.network_gate.network_policy == "ENHANCED_ONLY"
    assert result.network_gate.status == "PREDICTION_BLOCKED"
    assert result.engine_result.production_gate_status == "PREDICTION_BLOCKED"
    assert result.provider_result is None
