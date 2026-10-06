"""SYNTHETIC_TEST HTTP fixtures; no real API quota is used by this module."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from erguoyuan_football.blind_test_r3.odds_api_cache import OddsApiResponseCache
from erguoyuan_football.blind_test_r3.odds_api_provider import (
    TheOddsApiV4Provider,
    match_fixture,
    parse_complete_h2h,
    resolve_sport_key,
)
from erguoyuan_football.blind_test_r3.the_odds_api_v4 import (
    TheOddsApiV4Client,
    load_odds_api_config,
    load_sport_map,
)
from erguoyuan_football.network.errors import RetryExhausted

ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_odds_api_config(ROOT / "config/r3_the_odds_api_p2a.yaml")
SPORT_MAP = load_sport_map(ROOT / "config/the_odds_api_sport_map.yaml")
SYNTHETIC_MAP = {"version": "THE_ODDS_API_SPORT_MAP_V1",
                 "mappings": {"SYNTHETIC_TEST": {
                     "sport_key": "soccer_synthetic", "verified": True,
                     "source": "THE_ODDS_API_SPORTS_ENDPOINT"}}, "candidates": []}


class FakeResponse:
    def __init__(self, body, status=200, headers=None):
        self.body = body
        self.status_code = status
        self.headers = headers or {"x-requests-remaining": "25",
            "x-requests-used": "5", "x-requests-last": "2"}

    def json(self):
        return self.body


class FakeHTTP:
    def __init__(self, sports, events, *, status=200):
        self.sports = sports
        self.events = events
        self.status = status
        self.calls = []

    def get(self, url, *, params, headers):
        self.calls.append((url, dict(params), dict(headers)))
        body = self.sports if url.endswith("/sports/") else self.events
        return FakeResponse(body, self.status)


def _event(kickoff, *, home="France", away="Belgium", bookmakers=None):
    return {"id": "SYNTHETIC_EVENT", "sport_key": "soccer_synthetic",
            "commence_time": kickoff.isoformat(), "home_team": home,
            "away_team": away, "bookmakers": bookmakers or []}


def _book(now, *, key="book_a", outcomes=None, minutes_old=1):
    return {"key": key, "title": "SYNTHETIC_" + key,
            "markets": [{"key": "h2h",
                "last_update": (now - timedelta(minutes=minutes_old)).isoformat(),
                "outcomes": outcomes or [
                    {"name": "Belgium", "price": 4.2},
                    {"name": "Draw", "price": 3.5},
                    {"name": "France", "price": 2.0}]}]}


def test_no_api_key_is_not_configured(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("YYCORE_THE_ODDS_API_KEY", raising=False)
    cache = OddsApiResponseCache(tmp_path / "cache.sqlite")
    client = TheOddsApiV4Client(CONFIG, SYNTHETIC_MAP, cache,
        http_client=FakeHTTP([], []))
    try:
        provider = TheOddsApiV4Provider(client, competition="SYNTHETIC_TEST")
        assert provider.prefetch() == "NOT_CONFIGURED"
        result = provider.fetch_1x2("SYNTHETIC_FIXTURE", "France", "Belgium",
            datetime.now(UTC) + timedelta(days=1), datetime.now(UTC))
        assert result.status == "DATA_UNAVAILABLE"
        assert result.reason == "NOT_CONFIGURED"
    finally:
        client.close()
        cache.close()


def test_catalog_exact_mapping_fixture_identity_and_ambiguity() -> None:
    catalog = [{"key": "soccer_uefa_nations_league",
                "title": "UEFA Nations League", "active": True}]
    assert resolve_sport_key("欧国联", SPORT_MAP, catalog) == "soccer_uefa_nations_league"
    assert resolve_sport_key("欧国联", SPORT_MAP, []) is None
    kickoff = datetime.now(UTC) + timedelta(days=1)
    event = _event(kickoff)
    assert match_fixture([event], home_team="法国", away_team="比利时",
        kickoff_utc=kickoff, tolerance_minutes=180,
        normalized_tolerance_minutes=15)[1] == "CANONICAL_IDS"
    assert match_fixture([_event(kickoff, home="Belgium", away="France")],
        home_team="法国", away_team="比利时", kickoff_utc=kickoff,
        tolerance_minutes=180, normalized_tolerance_minutes=15) is None
    for wrong in ("France U21", "France Women", "France B"):
        assert match_fixture([_event(kickoff, home=wrong)], home_team="France",
            away_team="Belgium", kickoff_utc=kickoff,
            tolerance_minutes=180, normalized_tolerance_minutes=15) is None
    with pytest.raises(ValueError, match="AMBIGUOUS_FIXTURE_MATCH"):
        match_fixture([event, dict(event, id="OTHER")], home_team="France",
            away_team="Belgium", kickoff_utc=kickoff,
            tolerance_minutes=180, normalized_tolerance_minutes=15)


def test_complete_quote_quality_and_source_times() -> None:
    now = datetime.now(UTC)
    kickoff = now + timedelta(days=1)
    event = _event(kickoff, bookmakers=[_book(now), _book(now, key="book_b")])
    quotes, details = parse_complete_h2h(event, fixture_id="SYNTHETIC_FIXTURE",
        fetched_at=now, raw_payload_sha256="a" * 64,
        maximum_age_seconds=1800, maximum_overround=.20,
        minimum_decimal_odds=1.01)
    assert len(quotes) == 6
    assert len(details["bookmakers"]) == 2
    assert {quote.selection.value for quote in quotes} == {"HOME", "DRAW", "AWAY"}
    assert all(quote.retrieved_at == now for quote in quotes)
    assert all(quote.source_time == now - timedelta(minutes=1) for quote in quotes)
    event["bookmakers"].append(_book(now, key="incomplete", outcomes=[
        {"name": "France", "price": 2.0}, {"name": "Belgium", "price": 3.0}]))
    event["bookmakers"].append(_book(now, key="stale", minutes_old=120))
    event["bookmakers"].append(_book(now, key="bad_odds", outcomes=[
        {"name": "France", "price": 1.0}, {"name": "Draw", "price": 3.5},
        {"name": "Belgium", "price": 4.2}]))
    event["bookmakers"].append(_book(now, key="high_margin", outcomes=[
        {"name": "France", "price": 1.2}, {"name": "Draw", "price": 2.0},
        {"name": "Belgium", "price": 2.0}]))
    quotes, details = parse_complete_h2h(event, fixture_id="SYNTHETIC_FIXTURE",
        fetched_at=now, raw_payload_sha256="a" * 64,
        maximum_age_seconds=1800, maximum_overround=.20,
        minimum_decimal_odds=1.01)
    assert len(quotes) == 6
    assert details["rejected"] == {"NO_COMPLETE_1X2": 2,
                                    "STALE_MARKET": 1, "DATA_QUALITY_WARNING": 1}


def test_client_query_secret_quota_and_cache_reuse(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("YYCORE_THE_ODDS_API_KEY", "SYNTHETIC_SECRET_DO_NOT_LOG")
    now = datetime.now(UTC)
    fixture = _event(now + timedelta(days=1), bookmakers=[_book(now)])
    http = FakeHTTP([{"key": "soccer_synthetic", "title": "SYNTHETIC_TEST",
                      "active": True}], [fixture])
    cache = OddsApiResponseCache(tmp_path / "cache.sqlite")
    client = TheOddsApiV4Client(CONFIG, SYNTHETIC_MAP, cache, http_client=http)
    try:
        catalog = client.refresh_sport_catalog()
        first = client.get_h2h("soccer_synthetic")
        second = client.get_h2h("soccer_synthetic")
        assert len(http.calls) == 2
        assert second["cache_hit"] is True
        assert first["quota_remaining"] == 25
        assert first["quota_used"] == 5 and first["quota_last_cost"] == 2
        assert all(call[1]["apiKey"] == "SYNTHETIC_SECRET_DO_NOT_LOG"
                   for call in http.calls)
        assert all("apiKey" not in call[0] for call in http.calls)
        assert "SYNTHETIC_SECRET_DO_NOT_LOG" not in json.dumps(
            [catalog, first, second], default=str)
        assert "SYNTHETIC_SECRET_DO_NOT_LOG" not in json.dumps([
            record.model_dump(mode="json") for record in client.network.audit.records()])
        assert "SYNTHETIC_SECRET_DO_NOT_LOG" not in str(cache.path.read_bytes())
        with pytest.raises(sqlite3.IntegrityError, match="APPEND_ONLY"):
            cache.connection.execute("DELETE FROM odds_api_responses")
    finally:
        client.close()
        cache.close()


@pytest.mark.parametrize("status,expected", [(429, "RATE_LIMITED"), (401, "AUTH_FAILED")])
def test_http_failures_do_not_expose_key(tmp_path, monkeypatch, status, expected) -> None:
    monkeypatch.setenv("YYCORE_THE_ODDS_API_KEY", "SYNTHETIC_SECRET_DO_NOT_LOG")
    http = FakeHTTP([], [], status=status)
    cache = OddsApiResponseCache(tmp_path / "cache.sqlite")
    client = TheOddsApiV4Client(CONFIG, SYNTHETIC_MAP, cache, http_client=http)
    try:
        if status == 429:
            with pytest.raises(RetryExhausted, match=expected) as caught:
                client.refresh_sport_catalog()
        else:
            with pytest.raises(ValueError, match=expected) as caught:
                client.refresh_sport_catalog()
        assert "SYNTHETIC_SECRET_DO_NOT_LOG" not in str(caught.value)
    finally:
        client.close()
        cache.close()


def test_transport_error_suppresses_secret_bearing_cause(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("YYCORE_THE_ODDS_API_KEY", "SYNTHETIC_SECRET_DO_NOT_LOG")
    class BrokenHTTP:
        def get(self, _url, *, params, headers):
            raise OSError("request failed apiKey=SYNTHETIC_SECRET_DO_NOT_LOG")
    cache = OddsApiResponseCache(tmp_path / "cache.sqlite")
    client = TheOddsApiV4Client(CONFIG, SYNTHETIC_MAP, cache,
        http_client=BrokenHTTP())
    try:
        with pytest.raises(RetryExhausted) as caught:
            client.refresh_sport_catalog()
        assert "SYNTHETIC_SECRET_DO_NOT_LOG" not in str(caught.value)
        assert caught.value.__cause__ is None
        assert caught.value.__suppress_context__ is True
    finally:
        client.close()
        cache.close()


def test_provider_uses_cached_real_shape_and_r3_r5_separation(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("YYCORE_THE_ODDS_API_KEY", "SYNTHETIC_SECRET_DO_NOT_LOG")
    now = datetime.now(UTC)
    kickoff = now + timedelta(days=1)
    cache = OddsApiResponseCache(tmp_path / "cache.sqlite")
    client = TheOddsApiV4Client(CONFIG, SYNTHETIC_MAP, cache,
        http_client=FakeHTTP([], []))
    try:
        cache.put(endpoint="sports", sport_key=None, regions=None,
            fetched_at=now - timedelta(seconds=2), ttl_seconds=21600,
            payload=[{"key": "soccer_synthetic", "title": "SYNTHETIC_TEST",
                      "active": True}], quota_remaining=25, quota_used=5, quota_last_cost=0)
        cache.put(endpoint="odds_soccer_synthetic", sport_key="soccer_synthetic",
            regions="eu,uk", fetched_at=now - timedelta(seconds=1), ttl_seconds=60,
            payload=[_event(kickoff, bookmakers=[_book(now - timedelta(seconds=1))])],
            quota_remaining=23, quota_used=7, quota_last_cost=2)
        provider = TheOddsApiV4Provider(client, competition="SYNTHETIC_TEST")
        result = provider.fetch_1x2("SYNTHETIC_FIXTURE", "France", "Belgium",
            kickoff, now)
        assert result.status == "AVAILABLE"
        assert len(result.quotes) == 3
        assert provider.last_evidence["source_quality"] == "TRUSTED_API"
        assert provider.last_evidence["quota_remaining"] == 23
    finally:
        client.close()
        cache.close()
