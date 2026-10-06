"""Synthetic market evidence; no test quote is a real R3 market observation."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from erguoyuan_football.blind_test_r3.market_intelligence import (
    build_market_intelligence,
    fuse_model_market,
    load_r3_market_config,
    r5_market_only,
)
from erguoyuan_football.blind_test_r3.market_providers import (
    ManualMarketProvider,
    MarketProviderRegistry,
    NotConfiguredMarketProvider,
)
from erguoyuan_football.blind_test_r3.market_store import R3MarketStore
from erguoyuan_football.markets.devig import calculate_devig
from erguoyuan_football.markets.schemas import DeVigMethod

CONFIG = load_r3_market_config(__import__("pathlib").Path(__file__).resolve().parents[2] /
                               "config/r3_market_intelligence_v1.yaml")
MODEL = {"HOME": 0.58, "DRAW": 0.24, "AWAY": 0.18}


def _manual_file(directory, *, kickoff: datetime, confirmed: datetime,
                 opening: bool = False) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    quotes = []
    if opening:
        quotes.append({"bookmaker": "SYNTHETIC_BOOK_A", "home_odds": 2.0,
                       "draw_odds": 3.5, "away_odds": 4.0,
                       "fetched_at": (confirmed - timedelta(hours=2)).isoformat(),
                       "is_opening_confirmed": True})
    quotes.extend([
        {"bookmaker": "SYNTHETIC_BOOK_A", "home_odds": 1.65,
         "draw_odds": 3.8, "away_odds": 5.2,
         "fetched_at": (confirmed - timedelta(minutes=3)).isoformat()},
        {"bookmaker": "SYNTHETIC_BOOK_B", "home_odds": 1.68,
         "draw_odds": 3.7, "away_odds": 5.1,
         "fetched_at": (confirmed - timedelta(minutes=3)).isoformat()},
    ])
    payload = {"fixture_id": "SYNTHETIC_FIXTURE", "competition": "SYNTHETIC_TEST",
               "home_team": "SYNTHETIC_HOME", "away_team": "SYNTHETIC_AWAY",
               "kickoff_utc": kickoff.isoformat(),
               "confirmation": {"status": "USER_CONFIRMED",
                   "confirmation_id": "SYNTHETIC_CONFIRMATION",
                   "confirmed_at": confirmed.isoformat(),
                   "source_artifact_sha256": "a" * 64},
               "quotes": quotes}
    (directory / "synthetic.json").write_text(json.dumps(payload), encoding="utf-8")


def _registry(directory, *, broken: bool = False) -> MarketProviderRegistry:
    registry = MarketProviderRegistry()
    registry.register(NotConfiguredMarketProvider("JC_OFFICIAL_PROVIDER"))
    if broken:
        class Broken:
            provider_name = "SYNTHETIC_BROKEN_PROVIDER"
            status = "ERROR"

            def fetch_1x2(self, *_args):
                raise RuntimeError("SYNTHETIC_PROVIDER_FAILURE")

        registry.register(Broken())
    registry.register(ManualMarketProvider(directory))
    return registry


def _run(directory, at, kickoff, *, broken=False, model=MODEL):
    return build_market_intelligence(fixture_id="SYNTHETIC_FIXTURE",
        home_team="SYNTHETIC_HOME", away_team="SYNTHETIC_AWAY",
        kickoff_utc=kickoff, prediction_time=at, model_probabilities=model,
        registry=_registry(directory, broken=broken), config=CONFIG)


def test_existing_devig_and_consensus_provider_isolation_and_movement(tmp_path) -> None:
    at = datetime.now(UTC)
    kickoff = at + timedelta(days=1)
    directory = tmp_path / "manual"
    _manual_file(directory, kickoff=kickoff, confirmed=at - timedelta(minutes=1),
                 opening=True)
    result = _run(directory, at, kickoff, broken=True)
    assert result["market_status"] == "AVAILABLE"
    assert result["market_source_quality"] == "USER_CONFIRMED"
    assert result["bookmaker_count"] == 2
    assert result["provider_status"]["SYNTHETIC_BROKEN_PROVIDER"]["status"] == "ERROR"
    assert result["provider_status"]["MANUAL_MARKET_PROVIDER"]["status"] == "AVAILABLE"
    assert sum(result["market_no_vig"].values()) == pytest.approx(1.0)
    assert result["market_movement_pp"] is not None
    assert result["market_movement_pp"]["HOME_MOVE_PP"] > 0
    assert result["fusion"]["mode"] == "MODEL_MARKET"
    assert sum(result["fusion"]["probabilities"].values()) == pytest.approx(1.0)
    assert result["r5_market_only"]["mode"] == "MARKET_ONLY"
    assert result["r5_market_only"]["probabilities"] == result["market_no_vig"]
    store = R3MarketStore(tmp_path / "market.sqlite")
    try:
        store.append(result, prediction_id="SYNTHETIC_PREDICTION",
                     model_snapshot_id="SYNTHETIC_MODEL")
        assert len(store.snapshots_at("SYNTHETIC_FIXTURE", at)) == 1
        assert store.snapshots_at("SYNTHETIC_FIXTURE", at - timedelta(days=1)) == []
        with pytest.raises(sqlite3.IntegrityError, match="APPEND_ONLY"):
            store.connection.execute("UPDATE market_snapshots SET status='CHANGED'")
        with pytest.raises(sqlite3.IntegrityError):
            store.append(result, prediction_id="SYNTHETIC_PREDICTION",
                         model_snapshot_id="SYNTHETIC_MODEL")
    finally:
        store.close()


def test_market_unavailable_is_model_only_and_not_a_global_block(tmp_path) -> None:
    at = datetime.now(UTC)
    result = _run(tmp_path / "missing", at, at + timedelta(days=1), broken=True)
    assert result["market_status"] == "UNAVAILABLE"
    assert result["market_no_vig"] is None
    assert result["market_divergence"] == "UNAVAILABLE"
    assert result["fusion"]["mode"] == "MODEL_ONLY"
    assert result["fusion"]["probabilities"] == MODEL
    assert r5_market_only(None)["status"] == "UNAVAILABLE"


def test_devig_invalid_odds_divergence_and_fixed_fusion(tmp_path) -> None:
    devig = calculate_devig("SYNTHETIC_TEST", {"HOME": 2.0, "DRAW": 3.5,
        "AWAY": 4.0}, DeVigMethod.MULTIPLICATIVE)
    assert sum(devig.devig_probabilities.values()) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="INVALID_DECIMAL_ODDS"):
        calculate_devig("SYNTHETIC_TEST", {"HOME": 1.0, "DRAW": 3.5,
            "AWAY": 4.0}, DeVigMethod.MULTIPLICATIVE)
    market = {"HOME": 0.30, "DRAW": 0.30, "AWAY": 0.40}
    fused = fuse_model_market(MODEL, market, CONFIG)
    assert fused["probabilities"]["HOME"] == pytest.approx(0.496)
    assert fused["model_weight"] == pytest.approx(0.7)
    assert fused["market_weight"] == pytest.approx(0.3)
    at = datetime.now(UTC)
    kickoff = at + timedelta(days=1)
    _manual_file(tmp_path / "manual", kickoff=kickoff,
                 confirmed=at - timedelta(minutes=1))
    result = _run(tmp_path / "manual", at, kickoff,
                  model={"HOME": 0.90, "DRAW": 0.05, "AWAY": 0.05})
    assert result["market_divergence"] == "SEVERE"
    assert result["research_recommended"] is True
    with pytest.raises(ValueError, match="R3_MARKET_NOT_PREMATCH"):
        _run(tmp_path / "manual", kickoff, kickoff)


def test_unconfirmed_manual_odds_never_become_market(tmp_path) -> None:
    at = datetime.now(UTC)
    kickoff = at + timedelta(days=1)
    directory = tmp_path / "manual"
    _manual_file(directory, kickoff=kickoff, confirmed=at - timedelta(minutes=1))
    path = directory / "synthetic.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["confirmation"]["status"] = "OCR_ONLY"
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = _run(directory, at, kickoff)
    assert result["market_status"] == "UNAVAILABLE"
