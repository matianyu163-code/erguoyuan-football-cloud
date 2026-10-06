"""Official frozen blind-test publisher, independent of formal Production."""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from erguoyuan_football.blind_test_r3.capability import execute_available
from erguoyuan_football.blind_test_r3.production_gate import evaluate_production_gate
from erguoyuan_football.blind_test_r3.runner import R3_ROOT, run_slate
from erguoyuan_football.blind_test_r3.selection import run_locked
from erguoyuan_football.blind_test_r3.store import (
    R3Store,
    canonical_bytes,
    sha256,
    utc_time,
    write_once,
)

PUBLISH_LEVEL = "FROZEN_BLIND_TEST"
RESULT_STATUS = "OFFICIAL_BLIND_TEST"
RELEASE_PATH = R3_ROOT / "manifest" / "R3_OFFICIAL_PUBLISH_V1.json"


def _release(snapshot_id: str) -> dict[str, Any]:
    release = json.loads(RELEASE_PATH.read_text(encoding="utf-8"))
    if release.get("publish_level") != PUBLISH_LEVEL or (
        release.get("model_snapshot_id") != snapshot_id
    ):
        raise ValueError("R3_OFFICIAL_RELEASE_MISMATCH")
    paths = {"publisher_source_sha256": Path(__file__),
             "capability_source_sha256": Path(__file__).with_name("capability.py"),
             "production_gate_source_sha256": Path(__file__).with_name("production_gate.py")}
    for field, path in paths.items():
        if sha256(path.read_bytes()) != release.get(field):
            raise ValueError(f"R3_OFFICIAL_CODE_CHANGED:{field}")
    return release


def blind_test_gate(record: dict[str, Any], frozen: dict[str, Any],
                    fixture: dict[str, Any], slate_match: dict[str, Any]) -> dict[str, Any]:
    """Validate minimum official blind-test conditions without Production gate."""
    if record.get("mode") != "BLIND_TEST_R3" or (
        record.get("model_snapshot_id") != frozen["snapshot_id"]
    ):
        raise ValueError("BLIND_TEST_MODEL_SNAPSHOT_MISMATCH")
    if str(record.get("fixture_id")) != str(fixture["fixture_id"]) or (
        record.get("home_team") != slate_match["home"] or
        record.get("away_team") != slate_match["away"]
    ):
        raise ValueError("BLIND_TEST_FIXTURE_IDENTITY_MISMATCH")
    run_time = utc_time(record["run_time"])
    kickoff = utc_time(record["kickoff_time"])
    if run_time >= kickoff or utc_time(record["data_as_of"]) > run_time or (
        utc_time(record["fixture_evidence_retrieved_at"]) > run_time
    ):
        raise ValueError("BLIND_TEST_NOT_PREMATCH_OR_AS_OF_INVALID")
    if record.get("prediction_snapshot_id") is None or (
        record.get("market_verified") is not False
    ):
        raise ValueError("BLIND_TEST_INPUT_SNAPSHOT_OR_MARKET_STATUS_INVALID")
    raw = record.get("raw_model_output")
    standard = record.get("standardized_output")
    if not isinstance(raw, dict) or raw.get("execution_status") != "SUCCESS" or (
        not isinstance(standard, dict)
    ):
        raise ValueError("BLIND_TEST_MODEL_EXECUTION_FAILED")
    one_x_two = standard.get("one_x_two", {}).get("probabilities", {})
    values = [one_x_two.get(key) for key in ("HOME", "DRAW", "AWAY")]
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or
           not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("BLIND_TEST_1X2_MISSING")
    if not math.isclose(sum(values), 1.0, abs_tol=1e-9):
        raise ValueError("BLIND_TEST_1X2_NOT_NORMALIZED")
    spec = frozen["specification"]
    if raw.get("model_id") != spec["model_id"] or (
        raw.get("model_version") != spec["model_version"]
    ):
        raise ValueError("BLIND_TEST_MODEL_VERSION_MISMATCH")
    if not frozen["file_sha256"].get("runner_source") or (
        not frozen["file_sha256"].get("model_artifact")
    ):
        raise ValueError("BLIND_TEST_CODE_OR_ARTIFACT_NOT_FROZEN")
    return {"status": "PASS", "production_gate_required": False,
            "minimum_1x2_probability": dict(zip(("HOME", "DRAW", "AWAY"), values))}


def oos_eligibility(publication: dict[str, Any], *, result_verified: bool) -> dict[str, Any]:
    """Permit independently verified R3 blind tests in the R3 OOS ledger."""
    level = publication.get("publish_level")
    status = publication.get("result_status")
    eligible = level in ("FROZEN_BLIND_TEST", "PRODUCTION") and (
        status in ("OFFICIAL_BLIND_TEST", "OFFICIAL_PRODUCTION") and
        publication.get("lock_status") == "LOCKED" and result_verified and
        publication.get("input_snapshot_sha256") is not None)
    return {"r3_oos_eligible": bool(eligible),
            "phase9_golden_holdout_promoted": False,
            "reason": "ELIGIBLE_VERIFIED_RESULT" if eligible else
                      "PUBLISH_LOCK_SNAPSHOT_OR_RESULT_NOT_ELIGIBLE"}


def publish_locked(store: R3Store, prediction_id: str, frozen: dict[str, Any],
                   fixture: dict[str, Any], slate_match: dict[str, Any],
                   release: dict[str, Any]) -> tuple[Path, Path, dict[str, Any]]:
    """Append an official publication and rich lock; retain the original pair."""
    if not store.verify_lock(prediction_id):
        raise ValueError("R3_BASE_PREDICTION_LOCK_INVALID")
    prediction_path = store.root / "predictions" / f"{prediction_id}.json"
    original_lock_path = store.root / "locks" / f"{prediction_id}.json"
    record = json.loads(prediction_path.read_text(encoding="utf-8"))
    gate = blind_test_gate(record, frozen, fixture, slate_match)
    snapshot = {"prediction_snapshot_id": record["prediction_snapshot_id"],
                "fixture_evidence": fixture, "user_slate_match": slate_match,
                "model_data_as_of": record["data_as_of"],
                "fixture_evidence_retrieved_at": record["fixture_evidence_retrieved_at"]}
    snapshot_sha = sha256(canonical_bytes(snapshot))
    probabilities = gate["minimum_1x2_probability"]
    capability = execute_available({f"{frozen['specification']['model_id']}@"
                                    f"{frozen['specification']['model_version']}":
                                    lambda: probabilities})
    unavailable = {name: value.get("reason", "UNAVAILABLE") for name, value in
                   record["standardized_output"].items() if isinstance(value, dict)
                   and value.get("status") == "UNAVAILABLE"}
    unavailable["market"] = "MARKET_VERIFIED_FALSE"
    publication = {"mode": "BLIND_TEST_R3", "publish_level": PUBLISH_LEVEL,
                   "result_status": RESULT_STATUS, "r3_official_blind_test": True,
                   "r3_production_signal": False, "production_eligible": False,
                   "production_gate": evaluate_production_gate({}),
                   "prediction_id": prediction_id,
                   "fixture_id": record["fixture_id"],
                   "competition": record["competition"],
                   "home_team": record["home_team"],
                   "away_team": record["away_team"],
                   "kickoff_time": record["kickoff_time"],
                   "prediction_time": record["run_time"],
                   "prediction_snapshot_id": record["prediction_snapshot_id"],
                   "input_snapshot": snapshot,
                   "input_snapshot_sha256": snapshot_sha,
                   "model_snapshot_id": record["model_snapshot_id"],
                   "model_snapshot_sha256": sha256((store.root / "snapshots" /
                        f"{record['model_snapshot_id']}.json").read_bytes()),
                   "model_id": frozen["specification"]["model_id"],
                   "model_version": frozen["specification"]["model_version"],
                   "code_version": frozen["specification"]["code_hash"],
                   "code_commit": frozen["specification"]["code_commit"],
                   "publisher_code_sha256": release["publisher_source_sha256"],
                   "release_version": release["release_version"],
                   "model_probabilities": probabilities,
                   "derived_outputs": record["standardized_output"],
                   "available_models": capability["available_models"],
                   "unavailable_models": capability["unavailable_models"],
                   "available_model_count": capability["available_model_count"],
                   "registered_model_count": capability["registered_model_count"],
                   "available_model_ensemble": capability["available_model_ensemble"],
                   "ensemble_method": capability["ensemble_method"],
                   "unavailable_modules": unavailable,
                   "market_verified": False,
                   "lock_status": "LOCKED", "blind_test_gate": gate,
                   "base_prediction_sha256": sha256(prediction_path.read_bytes()),
                   "base_lock_sha256": sha256(original_lock_path.read_bytes())}
    official_path = store.root / "official" / f"{prediction_id}.json"
    lock_path = store.root / "official_locks" / prediction_id / "prediction_lock.json"
    if official_path.exists() or lock_path.exists():
        raise FileExistsError("R3_OFFICIAL_PUBLICATION_ALREADY_EXISTS")
    write_once(official_path, canonical_bytes(publication))
    rich_lock = {"event": "PREDICTION_LOCK", "publish_level": PUBLISH_LEVEL,
                 "result_status": RESULT_STATUS, "prediction_id": prediction_id,
                 "fixture_id": record["fixture_id"],
                 "prediction_timestamp": record["run_time"],
                 "kickoff_timestamp": record["kickoff_time"],
                 "snapshot_hash": snapshot_sha,
                 "model_snapshot_id": record["model_snapshot_id"],
                 "model_version": frozen["specification"]["model_version"],
                 "code_version": frozen["specification"]["code_hash"],
                 "code_commit": frozen["specification"]["code_commit"],
                 "model_probabilities": probabilities,
                 "derived_outputs": record["standardized_output"],
                 "available_models": capability["available_models"],
                 "unavailable_modules": unavailable,
                 "publication_sha256": sha256(official_path.read_bytes()),
                 "prediction_sha256": sha256(prediction_path.read_bytes()),
                 "locked_at": datetime.now(UTC).isoformat()}
    write_once(lock_path, canonical_bytes(rich_lock))
    return official_path, lock_path, publication


def run_official(store: R3Store, snapshot_id: str, slate_path: Path,
                 fixtures_path: Path) -> dict[str, Any]:
    """Preflight a frozen release, run new IDs, then publish successful games."""
    frozen = store.verify_snapshot(snapshot_id)
    release = _release(snapshot_id)
    if evaluate_production_gate({})["production_eligible"]:
        raise ValueError("R3_PRODUCTION_GATE_UNEXPECTEDLY_OPEN")
    slate = json.loads(slate_path.read_text(encoding="utf-8"))
    fixtures = json.loads(fixtures_path.read_text(encoding="utf-8"))
    fixture_by_code = {row["jc_code"]: row for row in fixtures["fixtures"]}
    match_by_code = {row["jc_code"]: row for row in slate["matches"]}
    rows = run_slate(store, snapshot_id, slate_path, fixtures_path)
    output = []
    official_ids = []
    for row in rows:
        if row["status"] != "LOCKED":
            output.append(row)
            continue
        try:
            prediction_id = row["prediction_id"]
            official_path, lock_path, publication = publish_locked(
                store, prediction_id, frozen, fixture_by_code[row["jc_code"]],
                match_by_code[row["jc_code"]], release)
            official_ids.append(prediction_id)
            output.append({"jc_code": row["jc_code"], "prediction_id": prediction_id,
                           "publish_level": publication["publish_level"],
                           "result_status": publication["result_status"],
                           "official_path": str(official_path),
                           "official_lock_path": str(lock_path)})
        except (OSError, ValueError, KeyError, TypeError) as error:
            output.append({"jc_code": row["jc_code"],
                           "prediction_id": row["prediction_id"],
                           "result_status": "BLOCKED_BEFORE_OFFICIAL_PUBLISH",
                           "reason": f"{type(error).__name__}:{error}"})
    selection = None
    if official_ids:
        selection_path, selection_record = run_locked(store, official_ids)
        selection = {"id": selection_record["selection_id"],
                     "path": str(selection_path)}
    return {"publish_level": PUBLISH_LEVEL, "production_eligible": False,
            "r3_production_signal": False,
            "official_count": len(official_ids), "results": output,
            "selection": selection}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--slate", type=Path,
                        default=R3_ROOT / "manifest/2026-10-05_user_slate.json")
    parser.add_argument("--fixtures", type=Path,
                        default=R3_ROOT / "manifest/2026-10-05_uefa_fixtures.json")
    args = parser.parse_args(argv)
    result = run_official(R3Store(R3_ROOT), args.snapshot_id, args.slate, args.fixtures)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["official_count"] == len(result["results"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
