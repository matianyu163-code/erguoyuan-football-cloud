"""Attach a new market-aware lock without rewriting an earlier R3 prediction."""

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
from erguoyuan_football.blind_test_r3.market_providers import default_registry
from erguoyuan_football.blind_test_r3.market_store import R3MarketStore
from erguoyuan_football.blind_test_r3.runner import PROJECT_ROOT, R3_ROOT
from erguoyuan_football.blind_test_r3.settlement import _verified_publication
from erguoyuan_football.blind_test_r3.store import (
    R3Store,
    canonical_bytes,
    sha256,
    utc_time,
    write_once,
)

CONFIG_PATH = PROJECT_ROOT / "config" / "r3_market_intelligence_v1.yaml"
RELEASE_PATH = R3_ROOT / "manifest" / "R3_MARKET_INTELLIGENCE_V1.json"


def verify_market_release() -> dict[str, Any]:
    """Reject any silent edit to the frozen R3 market V1 implementation."""
    release = json.loads(RELEASE_PATH.read_text(encoding="utf-8"))
    if release.get("version") != "R3_MARKET_INTELLIGENCE_V1":
        raise ValueError("R3_MARKET_RELEASE_INVALID")
    paths = {"config_sha256": CONFIG_PATH,
             "providers_sha256": Path(__file__).with_name("market_providers.py"),
             "intelligence_sha256": Path(__file__).with_name("market_intelligence.py"),
             "store_sha256": Path(__file__).with_name("market_store.py"),
             "lock_sha256": Path(__file__)}
    for field, path in paths.items():
        if sha256(path.read_bytes()) != release.get(field):
            raise ValueError(f"R3_MARKET_RELEASE_CHANGED:{field}")
    return release


def append_market_lock(store: R3Store, prediction_id: str, *, at: datetime,
                       manual_directory: Path | None = None,
                       database_path: Path | None = None) -> tuple[Path, dict[str, Any]]:
    """Build a true current PIT snapshot and a separate richer lock artifact."""
    publication = _verified_publication(store, prediction_id)
    now = datetime.now(UTC)
    if at.tzinfo is None or at > now or (now - at).total_seconds() > 30 or (
        at >= utc_time(publication["kickoff_time"]) or
        at < utc_time(publication["prediction_time"])
    ):
        raise ValueError("R3_MARKET_LOCK_NOT_PREMATCH")
    release = verify_market_release()
    config = load_r3_market_config(CONFIG_PATH)
    registry = default_registry(manual_directory or store.root / "market" / "manual")
    result = build_market_intelligence(fixture_id=publication["fixture_id"],
        home_team=publication["home_team"], away_team=publication["away_team"],
        kickoff_utc=utc_time(publication["kickoff_time"]), prediction_time=at,
        model_probabilities=publication["model_probabilities"],
        registry=registry, config=config)
    market_id = "R3MKT-" + sha256(canonical_bytes({
        "prediction_id": prediction_id, "market_snapshot_id": result["market_snapshot_id"],
        "market_snapshot_sha256": result["market_snapshot_sha256"],
        "config_sha256": result["config_sha256"]}))[:32]
    lock_path = store.root / "official_locks" / prediction_id / (
        f"prediction_lock.market_v1.{market_id}.json")
    if lock_path.exists():
        raise FileExistsError("R3_MARKET_LOCK_ALREADY_EXISTS")
    database = R3MarketStore(database_path or store.root / "market" / "market.sqlite")
    try:
        comparison_id = database.append(result, prediction_id=prediction_id,
                                        model_snapshot_id=publication["model_snapshot_id"])
    finally:
        database.close()
    base_lock_path = store.root / "official_locks" / prediction_id / "prediction_lock.json"
    market = {"status": result["market_status"],
              "source_quality": result["market_source_quality"],
              "snapshot_id": result["market_snapshot_id"],
              "snapshot_sha256": result["market_snapshot_sha256"],
              "captured_at": at.isoformat(), "sources": result["sources"],
              "source_evidence": result["source_evidence"],
              "bookmaker_count": result["bookmaker_count"],
              "no_vig": result["market_no_vig"],
              "movement": result["market_movement_pp"],
              "model_market_edge": result["model_market_edge_pp"],
              "divergence": result["market_divergence"],
              "research_recommended": result["research_recommended"],
              "fusion": result["fusion"],
              "r5_market_only": result["r5_market_only"],
              "provider_status": result["provider_status"]}
    lock = {"event": "PREDICTION_LOCK_MARKET_V1", "market_lock_id": market_id,
            "publish_level": "FROZEN_BLIND_TEST", "result_status": "OFFICIAL_BLIND_TEST",
            "prediction_id": prediction_id,
            "fixture_id": publication["fixture_id"],
            "model_snapshot_id": publication["model_snapshot_id"],
            "model_version": publication["model_version"],
            "code_version": publication["code_version"],
            "base_official_lock_sha256": sha256(base_lock_path.read_bytes()),
            "official_publication_sha256": sha256((store.root / "official" /
                f"{prediction_id}.json").read_bytes()),
            "market_database_comparison_id": comparison_id,
            "market_config_version": result["config_version"],
            "market_config_sha256": result["config_sha256"],
            "market_release_version": release["version"],
            "market_release_sha256": sha256(RELEASE_PATH.read_bytes()),
            "model_probability": result["model_probabilities"],
            "market_probability": result["market_no_vig"],
            "final_fusion_probability": result["fusion"]["probabilities"],
            "market": market, "locked_at": datetime.now(UTC).isoformat()}
    write_once(lock_path, canonical_bytes(lock))
    return lock_path, lock


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_id")
    parser.add_argument("--root", type=Path, default=R3_ROOT)
    args = parser.parse_args(argv)
    path, result = append_market_lock(R3Store(args.root), args.prediction_id,
                                      at=datetime.now(UTC))
    print(json.dumps({"path": str(path), **result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
