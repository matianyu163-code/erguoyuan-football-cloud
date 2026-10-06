"""Append an auditable final R3 view over immutable per-match and slate records."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from erguoyuan_football.blind_test_r3.settlement import _verified_publication
from erguoyuan_football.blind_test_r3.store import (
    R3Store,
    canonical_bytes,
    sha256,
    write_once,
)

SELECTION_MODULES = ("one_x_two_pair", "handicap_pair", "total_goals_duplex",
                     "score_pair_duplex", "standard_plan_400", "free_plan_100")


def finalize(store: R3Store, selection_id: str) -> tuple[Path, dict[str, Any]]:
    """Keep legacy locked fields intact and publish the selected final statuses."""
    selection_path = store.root / "selections" / f"{selection_id}.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("selection_id") != selection_id or (
        selection.get("event") != "DERIVED_SELECTION"):
        raise ValueError("R3_SELECTION_RECORD_INVALID")
    ids = selection["identity"]["prediction_ids"]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("R3_SELECTION_INPUT_IDS_INVALID")
    for prediction_id in ids:
        lock_path = store.root / "locks" / f"{prediction_id}.json"
        if sha256(lock_path.read_bytes()) != selection["identity"]["lock_sha256"].get(
                prediction_id):
            raise ValueError("R3_SELECTION_LOCK_LINEAGE_INVALID")
    publications = [_verified_publication(store, prediction_id)
                    for prediction_id in ids]
    if len({row["model_snapshot_id"] for row in publications}) != 1 or (
        {row["model_snapshot_id"] for row in publications} !=
        {selection["model_snapshot_id"]}):
        raise ValueError("R3_FINAL_SNAPSHOT_MISMATCH")
    outcomes = selection["results"]
    statuses = {key: outcomes[key]["status"] for key in SELECTION_MODULES}
    statuses["half_full_time"] = outcomes["half_full_time_duplex"]["status"]
    statuses["high_odds_20"] = outcomes["high_odds_20"]["status"]
    source_hashes = {prediction_id: sha256((store.root / "official" /
        f"{prediction_id}.json").read_bytes()) for prediction_id in ids}
    batch_id = "R3OFF-" + sha256(canonical_bytes({
        "selection_id": selection_id, "selection_sha256": sha256(selection_path.read_bytes()),
        "official_sha256": source_hashes}))[:32]
    batch_path = store.root / "official_batches" / f"{batch_id}.json"
    if batch_path.exists():
        existing = json.loads(batch_path.read_text(encoding="utf-8"))
        if existing["selection_id"] != selection_id or (
            existing["official_sha256"] != source_hashes or
            existing["final_module_statuses"] != statuses):
            raise ValueError("R3_FINAL_BATCH_CONFLICT")
        return batch_path, existing
    audit_paths = {}
    for prediction_id in ids:
        publication = next(row for row in publications
                           if row["prediction_id"] == prediction_id)
        legacy = publication["unavailable_modules"]
        if "daily_combinations" not in legacy and (
            "reference_plans_400_100_20" not in legacy):
            continue
        audit_id = "R3ERR-" + sha256(canonical_bytes({
            "prediction_id": prediction_id, "selection_id": selection_id,
            "official_sha256": source_hashes[prediction_id]}))[:32]
        audit_path = store.root / "official_locks" / prediction_id / (
            f"post_lock_error.{audit_id}.json")
        audit = {"event": "POST_LOCK_ERROR", "error_id": audit_id,
                 "prediction_id": prediction_id,
                 "reason": "LEGACY_PER_MATCH_SELECTION_STATUS_SUPERSEDED",
                 "unchanged_prediction_and_official_lock": True,
                 "legacy_unavailable": {key: legacy[key] for key in (
                     "daily_combinations", "reference_plans_400_100_20") if key in legacy},
                 "selection_id": selection_id,
                 "selection_sha256": sha256(selection_path.read_bytes()),
                 "corrected_slate_statuses": statuses,
                 "recorded_at": datetime.now(UTC).isoformat()}
        write_once(audit_path, canonical_bytes(audit))
        audit_paths[prediction_id] = str(audit_path)
    batch = {"event": "OFFICIAL_BLIND_TEST_BATCH", "batch_id": batch_id,
             "publish_level": "FROZEN_BLIND_TEST",
             "result_status": "OFFICIAL_BLIND_TEST",
             "r3_official_blind_test": True,
             "r3_production_signal": False,
             "production_eligible": False,
             "model_snapshot_id": selection["model_snapshot_id"],
             "prediction_date": selection["prediction_date"],
             "prediction_ids": ids, "official_sha256": source_hashes,
             "selection_id": selection_id,
             "selection_sha256": sha256(selection_path.read_bytes()),
             "final_module_statuses": statuses,
             "post_lock_error_paths": audit_paths,
             "created_at": datetime.now(UTC).isoformat()}
    write_once(batch_path, canonical_bytes(batch))
    return batch_path, batch
