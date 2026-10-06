"""Hash-bound Phase 9 development checkpoint for safe research resume."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import yaml

from erguoyuan_football.meta.artifact import MetaArtifact
from erguoyuan_football.meta.candidate import load_verified_candidate
from erguoyuan_football.meta.model import select_rows
from erguoyuan_football.meta.pipeline import Phase9Development
from erguoyuan_football.meta.runtime import ArtifactIndex, LoadedArtifactCache
from erguoyuan_football.ml.schemas import stable_hash


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_phase9_checkpoint(*, result: Phase9Development, artifact: MetaArtifact,
                            report_path: str | Path, checkpoint_path: str | Path,
                            artifact_index: ArtifactIndex) -> dict[str, Any]:
    """Publish a resume pointer only after report and artifact integrity checks pass."""
    report = Path(report_path).resolve()
    if not report.is_file() or artifact.manifest["candidate_data_hash"] != result.candidate.data_hash:
        raise ValueError("PHASE9_CHECKPOINT_SOURCE_INVALID")
    artifact_index.register(artifact.artifact_id, artifact.path)
    entry = {
        "schema_version": "PHASE9_RESEARCH_CHECKPOINT_V1",
        "candidate_dataset_id": result.candidate.dataset_id,
        "candidate_data_hash": result.candidate.data_hash,
        "config_hash": artifact.manifest["config_hash"],
        "artifact_id": artifact.artifact_id,
        "calibrator_id": artifact.manifest["calibrator_id"],
        "report_path": str(report),
        "report_sha256": _sha256(report),
        "created_at": datetime.now(UTC).isoformat(),
    }
    target = Path(checkpoint_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(json.dumps(entry, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, target)
    return entry


def adopt_existing_phase9_result(*, db_path: str | Path, config_path: str | Path,
                                 artifact_root: str | Path, report_path: str | Path,
                                 checkpoint_path: str | Path, artifact_index: ArtifactIndex,
                                 artifact_cache: LoadedArtifactCache
                                 ) -> dict[str, Any]:
    """Create a resume pointer from existing verified Phase 9 outputs without fitting."""
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    report_file = Path(report_path).resolve()
    report = json.loads(report_file.read_text(encoding="utf-8"))
    artifact_id = report.get("meta_artifact_id")
    if (report.get("candidate_audit", {}).get("dataset_id") != config.get("candidate_dataset_id") or
            report.get("candidate_audit", {}).get("dataset_hash") != config.get("candidate_data_hash") or
            report.get("final_holdout_rows") != 0 or not isinstance(artifact_id, str)):
        raise ValueError("PHASE9_EXISTING_RESULT_NOT_ADOPTABLE")
    candidate = load_verified_candidate(db_path, dataset_id=config["candidate_dataset_id"],
        expected_hash=config["candidate_data_hash"], expected_part_hashes=config["candidate_part_hashes"])
    artifact_path = Path(artifact_root) / artifact_id
    artifact_index.register(artifact_id, artifact_path)
    artifact, _ = artifact_cache.get(artifact_path, expected_candidate_hash=candidate.data_hash,
        expected_config_hash=stable_hash(config),
        allow_phase9_1_frozen_compatibility=True)
    if (artifact.manifest["calibrator_id"] != report.get("calibrator_id") or
            artifact.manifest["selected_calibration_method"] !=
            report.get("selected_calibration_method")):
        raise ValueError("PHASE9_EXISTING_ARTIFACT_REPORT_MISMATCH")
    eval_start, eval_end = (date.fromisoformat(value) for value in
                            report["development_windows"]["calibration_dev_eval"])
    rows = select_rows(candidate, eval_start, eval_end, min_available_models=8)
    result = Phase9Development(candidate, artifact.model, artifact.calibrator,
                               config, report, rows)
    return save_phase9_checkpoint(result=result, artifact=artifact,
        report_path=report_file, checkpoint_path=checkpoint_path,
        artifact_index=artifact_index)


def load_phase9_checkpoint(*, db_path: str | Path, config_path: str | Path,
                           checkpoint_path: str | Path, artifact_index: ArtifactIndex,
                           artifact_cache: LoadedArtifactCache
                           ) -> tuple[Phase9Development, MetaArtifact, bool]:
    """Resume only when frozen config, candidate, report, and artifact identities match."""
    checkpoint = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    config_hash = stable_hash(config)
    if (checkpoint.get("schema_version") != "PHASE9_RESEARCH_CHECKPOINT_V1" or
            checkpoint.get("config_hash") != config_hash or
            checkpoint.get("candidate_dataset_id") != config.get("candidate_dataset_id") or
            checkpoint.get("candidate_data_hash") != config.get("candidate_data_hash")):
        raise ValueError("PHASE9_CHECKPOINT_COMPATIBILITY_REJECTED")
    report_path = Path(checkpoint["report_path"])
    if _sha256(report_path) != checkpoint["report_sha256"]:
        raise ValueError("PHASE9_CHECKPOINT_REPORT_HASH_MISMATCH")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("meta_artifact_id") != checkpoint["artifact_id"]:
        raise ValueError("PHASE9_CHECKPOINT_REPORT_INCOMPATIBLE")
    candidate = load_verified_candidate(db_path, dataset_id=checkpoint["candidate_dataset_id"],
        expected_hash=checkpoint["candidate_data_hash"],
        expected_part_hashes=config["candidate_part_hashes"])
    artifact_path = artifact_index.resolve(checkpoint["artifact_id"])
    artifact, cache_hit = artifact_cache.get(artifact_path,
        expected_candidate_hash=candidate.data_hash, expected_config_hash=config_hash,
        allow_phase9_1_frozen_compatibility=True)
    if (artifact.manifest["calibrator_id"] != checkpoint["calibrator_id"] or
            artifact.manifest["selected_calibration_method"] !=
            report.get("selected_calibration_method")):
        raise ValueError("PHASE9_CHECKPOINT_CALIBRATOR_MISMATCH")
    windows = report["development_windows"]
    eval_start, eval_end = (date.fromisoformat(value) for value in windows["calibration_dev_eval"])
    rows = select_rows(candidate, eval_start, eval_end, min_available_models=8)
    if rows.count != report["common_sample_development"]["match_count"]:
        raise ValueError("PHASE9_CHECKPOINT_EVAL_ROW_MISMATCH")
    result = Phase9Development(candidate, artifact.model, artifact.calibrator, config, report, rows)
    return result, artifact, cache_hit
