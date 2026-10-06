"""Post-match verified result and R3-only OOS evaluation ledgers."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from erguoyuan_football.blind_test_r3.official import oos_eligibility
from erguoyuan_football.blind_test_r3.store import (
    R3Store,
    canonical_bytes,
    sha256,
    utc_time,
    write_once,
)


def _now() -> datetime:
    return datetime.now(UTC)


def _verified_publication(store: R3Store, prediction_id: str) -> dict[str, Any]:
    if not store.verify_lock(prediction_id):
        raise ValueError("R3_BASE_LOCK_INVALID")
    path = store.root / "official" / f"{prediction_id}.json"
    lock_path = store.root / "official_locks" / prediction_id / "prediction_lock.json"
    publication = json.loads(path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if publication.get("prediction_id") != prediction_id or (
        publication.get("result_status") != "OFFICIAL_BLIND_TEST" or
        publication.get("publish_level") != "FROZEN_BLIND_TEST" or
        publication.get("r3_production_signal") is not False or
        lock.get("prediction_id") != prediction_id or
        lock.get("publication_sha256") != sha256(path.read_bytes())
    ):
        raise ValueError("R3_OFFICIAL_PUBLICATION_LOCK_INVALID")
    return publication


def append_verified_result(store: R3Store, prediction_id: str,
                           evidence: dict[str, Any]) -> Path:
    """Append a sourced result only after kickoff, without touching prediction."""
    publication = _verified_publication(store, prediction_id)
    now = _now()
    kickoff = utc_time(publication["kickoff_time"])
    if now <= kickoff or utc_time(publication["prediction_time"]) >= kickoff:
        raise ValueError("R3_RESULT_BEFORE_KICKOFF_OR_PREDICTION_LATE")
    required = ("fixture_id", "home_score", "away_score", "source",
                "source_url", "source_timestamp", "retrieved_at",
                "verification_status")
    if any(key not in evidence for key in required) or (
        evidence["fixture_id"] != publication["fixture_id"] or
        evidence["verification_status"] != "VERIFIED" or
        not evidence["source"] or not evidence["source_url"]
    ):
        raise ValueError("R3_RESULT_EVIDENCE_INVALID")
    home, away = evidence["home_score"], evidence["away_score"]
    if any(isinstance(score, bool) or not isinstance(score, int) or score < 0
           for score in (home, away)):
        raise ValueError("R3_RESULT_SCORE_INVALID")
    source_time = utc_time(evidence["source_timestamp"])
    retrieved_at = utc_time(evidence["retrieved_at"])
    if source_time < kickoff or retrieved_at < source_time or retrieved_at > now:
        raise ValueError("R3_RESULT_SOURCE_TIME_INVALID")
    outcome = "HOME" if home > away else "DRAW" if home == away else "AWAY"
    event = {"event": "RESULT", "prediction_id": prediction_id,
             "publish_level": "FROZEN_BLIND_TEST", "fixture_id": publication["fixture_id"],
             "outcome": outcome, "home_score": home, "away_score": away,
             "source": evidence["source"], "source_url": evidence["source_url"],
             "source_timestamp": evidence["source_timestamp"],
             "retrieved_at": evidence["retrieved_at"],
             "verification_status": "VERIFIED", "recorded_at": now.isoformat(),
             "official_publication_sha256": sha256((store.root / "official" /
                f"{prediction_id}.json").read_bytes())}
    path = store.root / "results" / f"{prediction_id}.json"
    write_once(path, canonical_bytes(event))
    return path


def append_r3_oos_evaluation(store: R3Store, prediction_id: str) -> Path:
    """Score a verified locked blind-test result in the separate R3 OOS ledger."""
    publication = _verified_publication(store, prediction_id)
    result_path = store.root / "results" / f"{prediction_id}.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("event") != "RESULT" or (
        result.get("prediction_id") != prediction_id or
        result.get("fixture_id") != publication["fixture_id"] or
        result.get("verification_status") != "VERIFIED" or
        result.get("official_publication_sha256") != sha256((store.root /
            "official" / f"{prediction_id}.json").read_bytes())
    ):
        raise ValueError("R3_VERIFIED_RESULT_LINK_INVALID")
    eligibility = oos_eligibility(publication, result_verified=True)
    if not eligibility["r3_oos_eligible"]:
        raise ValueError("R3_OOS_NOT_ELIGIBLE")
    probabilities = publication["model_probabilities"]
    actual = result["outcome"]
    if actual not in ("HOME", "DRAW", "AWAY"):
        raise ValueError("R3_RESULT_OUTCOME_INVALID")
    probability = probabilities[actual]
    log_loss = -math.log(probability) if probability > 0 else math.inf
    if not math.isfinite(log_loss):
        raise ValueError("R3_ZERO_ACTUAL_OUTCOME_PROBABILITY")
    brier = sum((probabilities[key] - (1.0 if key == actual else 0.0)) ** 2
                for key in ("HOME", "DRAW", "AWAY"))
    event = {"event": "R3_OOS_EVALUATION", "prediction_id": prediction_id,
             "fixture_id": publication["fixture_id"],
             "model_snapshot_id": publication["model_snapshot_id"],
             "publish_level": "FROZEN_BLIND_TEST", "production_signal": False,
             "result_sha256": sha256(result_path.read_bytes()),
             "official_publication_sha256": result["official_publication_sha256"],
             "outcome": actual, "log_loss": log_loss,
             "multiclass_brier_score": brier,
             "scoring_version": "R3_MULTICLASS_1X2_V1",
             "r3_oos_eligible": True,
             "phase9_golden_holdout_promoted": False,
             "evaluated_at": _now().isoformat()}
    path = store.root / "evaluation" / f"{prediction_id}.json"
    write_once(path, canonical_bytes(event))
    return path
