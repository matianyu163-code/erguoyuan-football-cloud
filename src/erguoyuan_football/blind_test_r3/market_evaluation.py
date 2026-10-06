"""Score frozen R3 model, market and fusion tracks against verified results."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from erguoyuan_football.blind_test_r3.market_intelligence import _triple
from erguoyuan_football.blind_test_r3.settlement import _verified_publication
from erguoyuan_football.blind_test_r3.store import (
    R3Store,
    canonical_bytes,
    sha256,
    write_once,
)

OUTCOMES = ("HOME", "DRAW", "AWAY")


def _score(probabilities: dict[str, Any] | None, outcome: str) -> dict[str, Any]:
    if probabilities is None:
        return {"status": "UNAVAILABLE"}
    triple = _triple(probabilities)
    top1 = max(OUTCOMES, key=lambda key: triple[key])
    actual_probability = triple[outcome]
    if actual_probability <= 0:
        return {"status": "UNAVAILABLE", "reason": "ZERO_ACTUAL_OUTCOME_PROBABILITY"}
    return {"status": "AVAILABLE", "top1": top1,
            "accuracy": int(top1 == outcome), "log_loss": -math.log(actual_probability),
            "multiclass_brier_score": sum((triple[key] - int(key == outcome)) ** 2
                                          for key in OUTCOMES),
            "calibration_observation": {"forecast_confidence": triple[top1],
                                         "top1_hit": int(top1 == outcome),
                                         "actual_outcome_probability": actual_probability}}


def append_market_evaluation(store: R3Store, prediction_id: str,
                             market_lock_path: Path) -> Path:
    """Append one comparison event; a single match supplies calibration observations."""
    publication = _verified_publication(store, prediction_id)
    result_path = store.root / "results" / f"{prediction_id}.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("verification_status") != "VERIFIED" or (
        result.get("prediction_id") != prediction_id or
        result.get("fixture_id") != publication["fixture_id"] or
        result.get("official_publication_sha256") != sha256((store.root / "official" /
            f"{prediction_id}.json").read_bytes()) or
        result.get("outcome") not in OUTCOMES
    ):
        raise ValueError("R3_MARKET_EVALUATION_RESULT_INVALID")
    lock = json.loads(market_lock_path.read_text(encoding="utf-8"))
    base_lock_path = store.root / "official_locks" / prediction_id / "prediction_lock.json"
    if market_lock_path.parent.resolve() != base_lock_path.parent.resolve() or (
        lock.get("event") != "PREDICTION_LOCK_MARKET_V1" or
        lock.get("prediction_id") != prediction_id or
        lock.get("fixture_id") != publication["fixture_id"] or
        lock.get("base_official_lock_sha256") != sha256(base_lock_path.read_bytes()) or
        lock.get("official_publication_sha256") != result["official_publication_sha256"]
    ):
        raise ValueError("R3_MARKET_EVALUATION_LOCK_INVALID")
    outcome = result["outcome"]
    market_id = lock["market_lock_id"]
    event = {"event": "R3_MODEL_MARKET_FUSION_EVALUATION_V1",
             "prediction_id": prediction_id, "fixture_id": publication["fixture_id"],
             "market_lock_id": market_id, "market_lock_sha256": sha256(market_lock_path.read_bytes()),
             "result_sha256": sha256(result_path.read_bytes()), "outcome": outcome,
             "model": _score(lock["model_probability"], outcome),
             "market_no_vig": _score(lock["market_probability"], outcome),
             "final_fusion": _score(lock["final_fusion_probability"], outcome),
             "scoring_version": "R3_MODEL_MARKET_FUSION_EVALUATION_V1",
             "calibration_status": "OBSERVATION_ONLY_AGGREGATE_AFTER_MULTIPLE_MATCHES",
             "evaluated_at": datetime.now(UTC).isoformat()}
    path = store.root / "evaluation" / f"{prediction_id}.market.{market_id}.json"
    write_once(path, canonical_bytes(event))
    return path


def aggregate_market_evaluations(paths: list[Path]) -> dict[str, dict[str, Any]]:
    """Aggregate separately by track; top-one ECE uses fixed decile bins."""
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if any(record.get("event") != "R3_MODEL_MARKET_FUSION_EVALUATION_V1"
           for record in records):
        raise ValueError("R3_MARKET_EVALUATION_EVENT_INVALID")
    output: dict[str, dict[str, Any]] = {}
    for track in ("model", "market_no_vig", "final_fusion"):
        scored = [record[track] for record in records
                  if record[track]["status"] == "AVAILABLE"]
        if not scored:
            output[track] = {"status": "UNAVAILABLE", "match_count": 0}
            continue
        bins: dict[int, list[dict[str, float]]] = {}
        for row in scored:
            observation = row["calibration_observation"]
            confidence = float(observation["forecast_confidence"])
            bins.setdefault(min(9, int(confidence * 10)), []).append(observation)
        ece = sum(len(items) / len(scored) * abs(
            sum(float(item["forecast_confidence"]) for item in items) / len(items) -
            sum(float(item["top1_hit"]) for item in items) / len(items))
            for items in bins.values())
        output[track] = {"status": "AVAILABLE", "match_count": len(scored),
                         "accuracy": sum(row["accuracy"] for row in scored) / len(scored),
                         "mean_log_loss": sum(row["log_loss"] for row in scored) / len(scored),
                         "mean_multiclass_brier_score": sum(
                             row["multiclass_brier_score"] for row in scored) / len(scored),
                         "top1_ece_10_bins": ece}
    return output
