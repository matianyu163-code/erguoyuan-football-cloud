"""R3 storage invariants with synthetic, never-real prediction data."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from erguoyuan_football.blind_test_r3.store import R3Store, sha256


def test_snapshot_prediction_lock_and_error_are_append_only(tmp_path):
    store = R3Store(tmp_path / "r3")
    store.initialize()
    artifact = tmp_path / "model.joblib"
    artifact.write_bytes(b"SYNTHETIC_TEST_MODEL_ARTIFACT")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"model_id": "SYNTHETIC_TEST", "model_version": "1",
        "payload_sha256": sha256(artifact.read_bytes()), "trained_until": "2026-01-01T00:00:00Z",
        "training_data_hash": "synthetic", "config_hash": "synthetic-config"}))
    config = tmp_path / "config.yaml"
    config.write_text("SYNTHETIC_TEST: true")
    source = tmp_path / "source.py"
    source.write_text("# SYNTHETIC_TEST\n")
    spec = {"model_id": "SYNTHETIC_TEST", "model_version": "1",
        "feature_version": "synthetic", "data_version": "synthetic",
        "score_matrix_method": "synthetic", "derivation_method": "synthetic",
        "calibrator_version": "UNAVAILABLE"}
    files = {"model_artifact": artifact, "artifact_manifest": manifest,
             "model_config": config, "model_source": source,
             "score_matrix_source": source, "derivation_source": source}
    snapshot_id = store.freeze_snapshot(spec, files)
    assert store.freeze_snapshot(spec, files) == snapshot_id
    now = datetime.now(UTC)
    payload = {"prediction_date": now.date().isoformat(), "competition": "SYNTHETIC_TEST",
        "fixture_id": "SYNTHETIC_TEST_1", "home_team": "TEST_A", "away_team": "TEST_B",
        "kickoff_time": (now + timedelta(days=1)).isoformat(), "handicap": -1,
        "model_snapshot_id": snapshot_id, "data_as_of": now.isoformat(),
        "run_time": now.isoformat(), "raw_model_output": {"status": "SYNTHETIC_TEST"},
        "standardized_output": {"status": "UNAVAILABLE"},
        "model_execution_record_id": "SYNTHETIC_TEST_RECORD"}
    prediction_id = store.append_prediction(payload)
    lock = store.lock_prediction(prediction_id)
    assert store.verify_lock(prediction_id)
    with pytest.raises(FileExistsError):
        store.lock_prediction(prediction_id)
    error = store.post_lock_error(prediction_id, "SYNTHETIC_TEST_ERROR")
    assert error.exists() and lock.exists()
    artifact.write_bytes(b"CHANGED")
    assert store.verify_snapshot(snapshot_id)
    frozen = tmp_path / "r3" / "snapshots" / snapshot_id / "artifact" / "model.joblib"
    frozen.write_bytes(b"CHANGED")
    with pytest.raises(ValueError, match="R3_SNAPSHOT_FILE_CHANGED"):
        store.verify_snapshot(snapshot_id)
