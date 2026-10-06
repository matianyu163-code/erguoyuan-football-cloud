"""Explicit live integration; default pytest never consumes real API quota."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from erguoyuan_football.blind_test_r3.odds_api_cache import OddsApiResponseCache
from erguoyuan_football.blind_test_r3.the_odds_api_v4 import (
    TheOddsApiV4Client,
    load_odds_api_config,
    load_sport_map,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.live_network
@pytest.mark.skipif(not os.getenv("YYCORE_THE_ODDS_API_KEY"),
                    reason="YYCORE_THE_ODDS_API_KEY_MISSING")
def test_live_sport_catalog_http_200(tmp_path) -> None:
    cache = OddsApiResponseCache(tmp_path / "the_odds_api_live.sqlite")
    client = TheOddsApiV4Client(
        load_odds_api_config(ROOT / "config/r3_the_odds_api_p2a.yaml"),
        load_sport_map(ROOT / "config/the_odds_api_sport_map.yaml"), cache)
    try:
        result = client.refresh_sport_catalog()
        assert result["http_status"] == 200
        assert result["cache_hit"] is False
        assert isinstance(result["payload"], list)
        assert result["payload_sha256"]
    finally:
        client.close()
        cache.close()
