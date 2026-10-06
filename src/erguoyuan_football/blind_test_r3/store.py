"""Content-addressed snapshots and append-only prediction/lock records for R3."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

DIRECTORIES = ("snapshots", "predictions", "locks", "results", "evaluation", "manifest")


def canonical_bytes(value: dict[str, Any]) -> bytes:
    """Use one deterministic encoding for identities and checksums."""
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def sha256(data: bytes) -> str:
    """Return a lowercase SHA-256 digest."""
    return hashlib.sha256(data).hexdigest()


def utc_time(value: str) -> datetime:
    """Require an aware timestamp; naive times cannot satisfy a PIT check."""
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("R3_TIMEZONE_REQUIRED")
    return parsed.astimezone(UTC)


def write_once(path: Path, data: bytes) -> None:
    """Create a file exclusively and durably; never replace an existing record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())


class R3Store:
    """Independent R3 storage with no reference to the Production trial database."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def initialize(self) -> None:
        """Create the six dedicated directories without touching existing files."""
        for name in DIRECTORIES:
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def freeze_snapshot(self, specification: dict[str, Any], files: dict[str, Path]) -> str:
        """Freeze existing model inputs by hash; require a fitted artifact payload."""
        required = {"model_id", "model_version", "feature_version", "data_version",
                    "score_matrix_method", "derivation_method", "calibrator_version"}
        if not required <= specification.keys():
            raise ValueError("R3_SNAPSHOT_FIELDS_MISSING")
        source_files = {"model_source", "score_matrix_source", "derivation_source"}
        if not ({"model_artifact", "artifact_manifest", "model_config"} | source_files) <= files.keys():
            raise ValueError("R3_FITTED_ARTIFACT_CONFIG_AND_CODE_REQUIRED")
        if specification["calibrator_version"] != "UNAVAILABLE" and "calibrator_artifact" not in files:
            raise ValueError("R3_CALIBRATOR_ARTIFACT_REQUIRED")
        artifact = json.loads(files["artifact_manifest"].read_text(encoding="utf-8"))
        if (artifact.get("model_id") != specification["model_id"]
                or artifact.get("model_version") != specification["model_version"]
                or artifact.get("payload_sha256") != sha256(files["model_artifact"].read_bytes())
                or not artifact.get("trained_until")
                or not artifact.get("training_data_hash")
                or not artifact.get("config_hash")):
            raise ValueError("R3_FITTED_ARTIFACT_MANIFEST_INVALID")
        if specification["data_version"] != artifact["training_data_hash"]:
            raise ValueError("R3_TRAINING_DATA_VERSION_MISMATCH")
        hashes = {name: sha256(path.read_bytes()) for name, path in sorted(files.items())}
        identity = {"mode": "BLIND_TEST_R3", "specification": specification,
                    "file_sha256": hashes}
        snapshot_id = "R3-" + sha256(canonical_bytes(identity))[:32]
        snapshot_dir = self.root / "snapshots" / snapshot_id
        copied = {name: snapshot_dir / (
            "artifact/model.joblib" if name == "model_artifact" else
            "artifact/manifest.json" if name == "artifact_manifest" else
            f"inputs/{name}{path.suffix}") for name, path in sorted(files.items())}
        record = {**identity, "snapshot_id": snapshot_id,
                  "file_paths": {name: str(path.resolve()) for name, path in copied.items()}}
        target = self.root / "snapshots" / f"{snapshot_id}.json"
        if target.exists():
            if json.loads(target.read_text(encoding="utf-8")) != record:
                raise ValueError("R3_SNAPSHOT_ID_COLLISION")
            self.verify_snapshot(snapshot_id)
        else:
            for name, path in copied.items():
                write_once(path, files[name].read_bytes())
            write_once(target, canonical_bytes(record))
        return snapshot_id

    def verify_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        """Reject missing or changed files before any prediction is accepted."""
        path = self.root / "snapshots" / f"{snapshot_id}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        identity = {key: record[key] for key in ("mode", "specification", "file_sha256")}
        if record["snapshot_id"] != snapshot_id or (
            "R3-" + sha256(canonical_bytes(identity))[:32] != snapshot_id
        ):
            raise ValueError("R3_SNAPSHOT_ID_INVALID")
        for name, expected in record["file_sha256"].items():
            if sha256(Path(record["file_paths"][name]).read_bytes()) != expected:
                raise ValueError(f"R3_SNAPSHOT_FILE_CHANGED:{name}")
        return record

    def append_prediction(self, payload: dict[str, Any]) -> str:
        """Persist actual model output with explicit lineage and no generated probabilities."""
        required = {"prediction_date", "competition", "fixture_id", "home_team",
                    "away_team", "kickoff_time", "handicap", "model_snapshot_id",
                    "data_as_of", "run_time", "raw_model_output", "standardized_output",
                    "model_execution_record_id"}
        if not required <= payload.keys():
            raise ValueError("R3_PREDICTION_FIELDS_MISSING")
        if "prediction_id" in payload or "mode" in payload:
            raise ValueError("R3_RESERVED_PREDICTION_FIELD")
        self.verify_snapshot(str(payload["model_snapshot_id"]))
        if payload["raw_model_output"] is None or not payload["model_execution_record_id"]:
            raise ValueError("R3_EXECUTED_MODEL_OUTPUT_REQUIRED")
        now = datetime.now(UTC)
        if utc_time(str(payload["kickoff_time"])) <= now:
            raise ValueError("R3_KICKOFF_NOT_FUTURE")
        data_as_of = utc_time(str(payload["data_as_of"]))
        run_time = utc_time(str(payload["run_time"]))
        if data_as_of > run_time:
            raise ValueError("R3_DATA_AFTER_RUN")
        for field in ("data_as_of", "run_time"):
            if utc_time(str(payload[field])) > now:
                raise ValueError(f"R3_FUTURE_TIMESTAMP:{field}")
        prediction_id = str(uuid4())
        record = {"mode": "BLIND_TEST_R3", "prediction_id": prediction_id, **payload}
        write_once(self.root / "predictions" / f"{prediction_id}.json", canonical_bytes(record))
        return prediction_id

    def append_failure(self, payload: dict[str, Any]) -> Path:
        """Keep a failed real attempt as an append-only event with no probability."""
        if not payload.get("reason") or not payload.get("fixture_id"):
            raise ValueError("R3_FAILURE_IDENTITY_REQUIRED")
        event_id = str(uuid4())
        target = self.root / "predictions" / f"failed.{event_id}.json"
        write_once(target, canonical_bytes({"mode": "BLIND_TEST_R3",
            "event": "PREDICTION_FAILED", "event_id": event_id,
            "recorded_at": datetime.now(UTC).isoformat(), **payload}))
        return target

    def lock_prediction(self, prediction_id: str) -> Path:
        """Write a separate exclusive lock binding the exact prediction bytes."""
        source = self.root / "predictions" / f"{prediction_id}.json"
        data = source.read_bytes()
        record = json.loads(data)
        if record["prediction_id"] != prediction_id:
            raise ValueError("R3_PREDICTION_ID_MISMATCH")
        self.verify_snapshot(record["model_snapshot_id"])
        lock = {"event": "PREDICTION_LOCK", "prediction_id": prediction_id,
                "model_snapshot_id": record["model_snapshot_id"],
                "prediction_sha256": sha256(data), "locked_at": datetime.now(UTC).isoformat()}
        target = self.root / "locks" / f"{prediction_id}.json"
        write_once(target, canonical_bytes(lock))
        return target

    def verify_lock(self, prediction_id: str) -> bool:
        """Check that the frozen prediction still matches its lock."""
        source = self.root / "predictions" / f"{prediction_id}.json"
        lock = json.loads((self.root / "locks" / f"{prediction_id}.json").read_text(encoding="utf-8"))
        return lock["event"] == "PREDICTION_LOCK" and (
            lock["prediction_sha256"] == sha256(source.read_bytes()))

    def post_lock_error(self, prediction_id: str, reason: str) -> Path:
        """Append an error event without editing the prediction or lock."""
        self.verify_lock(prediction_id)
        event_id = str(uuid4())
        path = self.root / "locks" / f"{prediction_id}.error.{event_id}.json"
        write_once(path, canonical_bytes({"event": "POST_LOCK_ERROR", "event_id": event_id,
            "prediction_id": prediction_id, "reason": reason,
            "recorded_at": datetime.now(UTC).isoformat()}))
        return path
