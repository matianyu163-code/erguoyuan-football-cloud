"""Append a V1-fusion market supplement using a real V4 provider when configured."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from erguoyuan_football.blind_test_r3.market_intelligence import (
    build_market_intelligence,
    load_r3_market_config,
)
from erguoyuan_football.blind_test_r3.market_providers import (
    ManualMarketProvider,
    MarketProviderRegistry,
    NotConfiguredMarketProvider,
)
from erguoyuan_football.blind_test_r3.market_store import R3MarketStore
from erguoyuan_football.blind_test_r3.odds_api_cache import OddsApiResponseCache
from erguoyuan_football.blind_test_r3.odds_api_provider import TheOddsApiV4Provider
from erguoyuan_football.blind_test_r3.runner import PROJECT_ROOT, R3_ROOT
from erguoyuan_football.blind_test_r3.settlement import _verified_publication
from erguoyuan_football.blind_test_r3.store import (
    R3Store,
    canonical_bytes,
    sha256,
    utc_time,
    write_once,
)
from erguoyuan_football.blind_test_r3.the_odds_api_v4 import (
    TheOddsApiV4Client,
    load_odds_api_config,
    load_sport_map,
)

RELEASE_PATH = R3_ROOT / "manifest" / "R3_THE_ODDS_API_P2A_V1_1.json"
PROVIDER_CONFIG_PATH = PROJECT_ROOT / "config" / "r3_the_odds_api_p2a.yaml"
SPORT_MAP_PATH = PROJECT_ROOT / "config" / "the_odds_api_sport_map.yaml"
FUSION_CONFIG_PATH = PROJECT_ROOT / "config" / "r3_market_intelligence_v1.yaml"


def verify_p2a_release() -> dict[str, Any]:
    release = json.loads(RELEASE_PATH.read_text(encoding="utf-8"))
    if release.get("version") != "R3_THE_ODDS_API_P2A_V1_1":
        raise ValueError("R3_P2A_RELEASE_INVALID")
    paths = {"provider_config_sha256": PROVIDER_CONFIG_PATH,
             "sport_map_sha256": SPORT_MAP_PATH,
             "fusion_config_sha256": FUSION_CONFIG_PATH,
             "p1_market_release_sha256": R3_ROOT / "manifest/R3_MARKET_INTELLIGENCE_V1.json",
             "client_sha256": Path(__file__).with_name("the_odds_api_v4.py"),
             "provider_sha256": Path(__file__).with_name("odds_api_provider.py"),
             "cache_sha256": Path(__file__).with_name("odds_api_cache.py"),
             "market_p2a_sha256": Path(__file__),
             "market_intelligence_sha256": Path(__file__).with_name("market_intelligence.py"),
             "market_store_sha256": Path(__file__).with_name("market_store.py"),
             "market_providers_sha256": Path(__file__).with_name("market_providers.py"),
             "market_schemas_sha256": PROJECT_ROOT / "src/erguoyuan_football/markets/schemas.py",
             "market_normalizer_sha256": PROJECT_ROOT / "src/erguoyuan_football/markets/normalizer.py",
             "market_snapshot_sha256": PROJECT_ROOT / "src/erguoyuan_football/markets/snapshot.py",
             "market_devig_sha256": PROJECT_ROOT / "src/erguoyuan_football/markets/devig.py",
             "market_consensus_sha256": PROJECT_ROOT / "src/erguoyuan_football/markets/consensus.py",
             "market_movement_sha256": PROJECT_ROOT / "src/erguoyuan_football/markets/movement.py",
             "network_client_sha256": PROJECT_ROOT / "src/erguoyuan_football/network/client.py",
             "network_schema_sha256": PROJECT_ROOT / "src/erguoyuan_football/network/schemas.py"}
    for field, path in paths.items():
        if sha256(path.read_bytes()) != release.get(field):
            raise ValueError(f"R3_P2A_RELEASE_CHANGED:{field}")
    return release


def _previous_observed_movement(database_path: Path, result: dict[str, Any]) -> None:
    """Compare to an earlier persisted consensus, never invent an opening price."""
    if result["market_no_vig"] is None:
        result["market_movement_status"] = "UNAVAILABLE"
        return
    database = R3MarketStore(database_path)
    try:
        row = database.connection.execute(
            "SELECT s.snapshot_id,c.payload FROM market_snapshots s "
            "JOIN model_market_comparison c ON c.snapshot_id=s.snapshot_id "
            "WHERE s.fixture_id=? AND s.captured_at<? AND s.status='AVAILABLE' "
            "ORDER BY s.captured_at DESC,s.snapshot_id DESC LIMIT 1",
            (result["fixture_id"], result["prediction_time"])).fetchone()
    finally:
        database.close()
    if row is None:
        result["market_movement_status"] = "INSUFFICIENT_HISTORY"
        return
    previous = json.loads(row[1]).get("market_no_vig")
    if previous is None:
        result["market_movement_status"] = "INSUFFICIENT_HISTORY"
        return
    current = result["market_no_vig"]
    movement = {key + "_MOVE_PP": (current[key] - previous[key]) * 100
                for key in ("HOME", "DRAW", "AWAY")}
    movement["MAX_ABS_MOVE_PP"] = max(abs(value) for value in movement.values())
    result["market_movement_pp"] = movement
    result["market_movement_status"] = "PREVIOUS_OBSERVED_TO_CURRENT"
    result["previous_market_snapshot_id"] = row[0]
    result["movement_payloads"].append({
        "movement_id": sha256(canonical_bytes({"previous_snapshot_id": row[0],
            "current_snapshot_id": result["market_snapshot_id"], "movement": movement})),
        "previous_snapshot_id": row[0],
        "current_snapshot_id": result["market_snapshot_id"],
        "method": "PREVIOUS_OBSERVED_TO_CURRENT", "movement_pp": movement})


def append_p2a_market_lock(store: R3Store, prediction_id: str, *,
                           manual_directory: Path | None = None,
                           database_path: Path | None = None,
                           cache_path: Path | None = None,
                           http_client: Any | None = None) -> tuple[Path, dict[str, Any]]:
    """Fetch once, freeze at actual post-fetch time, and append a separate lock."""
    publication = _verified_publication(store, prediction_id)
    release = verify_p2a_release()
    if release.get("model_snapshot_id") != publication["model_snapshot_id"]:
        raise ValueError("R3_P2A_MODEL_SNAPSHOT_MISMATCH")
    config = load_odds_api_config(PROVIDER_CONFIG_PATH)
    sport_map = load_sport_map(SPORT_MAP_PATH)
    cache = OddsApiResponseCache(cache_path or store.root / "market/the_odds_api_cache.sqlite")
    client = TheOddsApiV4Client(config, sport_map, cache, http_client=http_client)
    try:
        provider = TheOddsApiV4Provider(client, competition=publication["competition"])
        prefetch_status = provider.prefetch()
        captured_at = datetime.now(UTC)
        if captured_at >= utc_time(publication["kickoff_time"]) or (
            captured_at < utc_time(publication["prediction_time"])
        ):
            raise ValueError("R3_P2A_NOT_PREMATCH")
        registry = MarketProviderRegistry()
        registry.register(NotConfiguredMarketProvider("JC_OFFICIAL_PROVIDER"))
        registry.register(ManualMarketProvider(manual_directory or store.root / "market/manual"))
        registry.register(provider)
        result = build_market_intelligence(fixture_id=publication["fixture_id"],
            home_team=publication["home_team"], away_team=publication["away_team"],
            kickoff_utc=utc_time(publication["kickoff_time"]), prediction_time=captured_at,
            model_probabilities=publication["model_probabilities"],
            registry=registry, config=load_r3_market_config(FUSION_CONFIG_PATH))
        result["provider_prefetch_status"] = prefetch_status
        if provider.last_evidence is not None:
            result["source_evidence"][provider.provider_name] = provider.last_evidence
            result["market_source_quality"] = (
                "TRUSTED_API" if result["sources"] == [provider.provider_name] else
                "MULTI_BOOK_CONSENSUS")
        if not client.configured:
            result["provider_status"][provider.provider_name]["status"] = "NOT_CONFIGURED"
        market_db_path = database_path or store.root / "market/market.sqlite"
        _previous_observed_movement(market_db_path, result)
        market_id = "R3MKT-" + sha256(canonical_bytes({
            "prediction_id": prediction_id, "snapshot_id": result["market_snapshot_id"],
            "snapshot_sha256": result["market_snapshot_sha256"],
            "provider_release": release["version"]}))[:32]
        lock_path = store.root / "official_locks" / prediction_id / (
            f"prediction_lock.market_v1.{market_id}.json")
        if lock_path.exists():
            raise FileExistsError("R3_P2A_MARKET_LOCK_ALREADY_EXISTS")
        database = R3MarketStore(market_db_path)
        try:
            comparison_id = database.append(result, prediction_id=prediction_id,
                model_snapshot_id=publication["model_snapshot_id"])
        finally:
            database.close()
        base_lock_path = store.root / "official_locks" / prediction_id / "prediction_lock.json"
        evidence = provider.last_evidence or {}
        market = {"status": result["market_status"],
                  "source_quality": result["market_source_quality"],
                  "snapshot_id": result["market_snapshot_id"],
                  "snapshot_sha256": result["market_snapshot_sha256"],
                  "captured_at": captured_at.isoformat(),
                  "sources": result["sources"], "source_evidence": result["source_evidence"],
                  "provider": provider.provider_name, "provider_category": "GLOBAL_ODDS_PROVIDER",
                  "provider_event_id": evidence.get("provider_event_id"),
                  "sport_key": evidence.get("sport_key"), "regions": evidence.get("regions"),
                  "bookmakers": evidence.get("bookmakers", []),
                  "bookmaker_last_updates": {item["bookmaker_key"]:
                      item["market_last_update"] for item in evidence.get("bookmakers", [])},
                  "market_fetched_at": evidence.get("market_fetched_at"),
                  "quota_remaining": evidence.get("quota_remaining"),
                  "odds_api_quota_remaining": evidence.get("quota_remaining"),
                  "raw_payload_sha256": evidence.get("raw_payload_sha256"),
                  "bookmaker_count": result["bookmaker_count"],
                  "no_vig": result["market_no_vig"],
                  "movement": result["market_movement_pp"] or result["market_movement_status"],
                  "movement_status": result["market_movement_status"],
                  "previous_market_snapshot_id": result.get("previous_market_snapshot_id"),
                  "model_market_edge": result["model_market_edge_pp"],
                  "divergence": result["market_divergence"],
                  "research_recommended": result["research_recommended"],
                  "fusion": result["fusion"], "r5_market_only": result["r5_market_only"],
                  "provider_status": result["provider_status"],
                  "provider_prefetch_status": prefetch_status}
        lock = {"event": "PREDICTION_LOCK_MARKET_V1",
                "market_lock_id": market_id, "publish_level": "FROZEN_BLIND_TEST",
                "result_status": "OFFICIAL_BLIND_TEST", "prediction_id": prediction_id,
                "fixture_id": publication["fixture_id"],
                "model_snapshot_id": publication["model_snapshot_id"],
                "model_version": publication["model_version"],
                "code_version": publication["code_version"],
                "base_official_lock_sha256": sha256(base_lock_path.read_bytes()),
                "official_publication_sha256": sha256((store.root / "official" /
                    f"{prediction_id}.json").read_bytes()),
                "market_database_comparison_id": comparison_id,
                "market_release_version": release["version"],
                "market_release_sha256": sha256(RELEASE_PATH.read_bytes()),
                "model_probability": result["model_probabilities"],
                "market_probability": result["market_no_vig"],
                "final_fusion_probability": result["fusion"]["probabilities"],
                "market": market, "locked_at": datetime.now(UTC).isoformat()}
        write_once(lock_path, canonical_bytes(lock))
        return lock_path, lock
    finally:
        client.close()
        cache.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_id")
    parser.add_argument("--root", type=Path, default=R3_ROOT)
    args = parser.parse_args(argv)
    path, lock = append_p2a_market_lock(R3Store(args.root), args.prediction_id)
    print(json.dumps({"path": str(path), **lock}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
