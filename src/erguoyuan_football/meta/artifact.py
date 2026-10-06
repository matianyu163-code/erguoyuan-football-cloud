"""Content-verified local research artifacts for no-market META and calibration."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy
import pandas
import sklearn

from erguoyuan_football.meta.calibration import MulticlassCalibrator
from erguoyuan_football.meta.model import NoMarketMetaModel
from erguoyuan_football.meta.pipeline import Phase9Development
from erguoyuan_football.ml.schemas import stable_hash

# Phase 9.1 only changes the single-row entry point into a batch wrapper; the
# frozen weights and calibration are checked against the captured CORE V2
# reference before the compatibility path is accepted.
PHASE9_1_FROZEN_COMPATIBILITY = {
    "artifact_id": "13a49d400f358e9e8ea69962c07e64db",
    "code_version": "0d865041e0c252292d23067102a27266ecd80dec7f5242da4cb171d477e70902",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class MetaArtifact:
    """Loadable research artifact with immutable candidate and split provenance."""

    artifact_id: str
    path: Path
    manifest: dict[str, Any]
    model: NoMarketMetaModel
    calibrator: MulticlassCalibrator

    def predict_proba(self, rows: Any) -> tuple[Any, Any]:
        """Return raw and calibrated probabilities for a strictly later date."""
        return self.predict_many(rows)

    def predict_many(self, rows: Any) -> tuple[Any, Any]:
        """Apply the frozen META and calibrator once to a batch of rows."""
        raw = self.model.predict_many(rows)
        return raw, self.calibrator.calibrate_many(raw)


def save_artifact(result: Phase9Development, *, root: str | Path) -> MetaArtifact:
    """Write weights first, hash them, and publish manifest last."""
    calibrator_id = stable_hash({
        "candidate_data_hash": result.candidate.data_hash,
        "meta_training_data_hash": result.report["selected_training_data_hash"],
        "calibration_fit_data_hash": result.report["calibration_fit_data_hash"],
        "method": result.report["selected_calibration_method"],
    })[:32]
    identity = {
        "model_id": "META_NO_MARKET_V1", "model_version": result.model.version,
        "meta_model_id": "META_NO_MARKET_V1", "version": result.model.version,
        "candidate_dataset_id": result.candidate.dataset_id,
        "candidate_data_hash": result.candidate.data_hash,
        "candidate_dataset_schema_hash": stable_hash([
            (name, str(dtype)) for name, dtype in result.candidate.features.dtypes.items()]),
        "candidate_part_hashes": result.candidate.part_hashes,
        "base_model_schema_version": "REAL_OOS_HDA_V1",
        "split_config_hash": result.report["split_config_hash"],
        "config_hash": result.report["split_config_hash"],
        "data_hash": result.report["selected_training_data_hash"],
        "training_period": [str(result.config["max_coverage_train"][0])
                            if result.report["selected_research_dataset"] == "MAX_COVERAGE"
                            else str(result.config["common_model_era_train"][0]),
                            result.report["selected_meta_trained_until"]],
        "development_period": result.report["development_windows"]["meta_dev_validation"],
        "selected_dataset": result.report["selected_research_dataset"],
        "selected_calibration_method": result.report["selected_calibration_method"],
        "meta_trained_until": result.report["selected_meta_trained_until"],
        "meta_training_sample_count": result.model.training_sample_count,
        "calibration_fit_sample_count": result.calibrator.fit_sample_count,
        "calibration_fit_end": str(result.config["calibration_fit"][1]),
        "temporal_mode": "DATE_SAFE_BATCH",
        "production_compatibility": "RESEARCH_ONLY_NOT_LIVE_VALIDATED",
        "validation_status": "DEVELOPMENT_VALIDATED_DATE_SAFE",
        "final_holdout_status": "LOCKED_AWAITING_DATA",
        "market_dependency_count": 0,
        "base_model_versions": result.report["base_model_versions"],
        "feature_schema_version": result.report["feature_schema_version"],
        "dependency_schema_version": result.report["dependency_schema_version"],
        "log_ratio_epsilon": result.model.epsilon,
        "selected_training_data_hash": result.report["selected_training_data_hash"],
        "calibration_fit_data_hash": result.report["calibration_fit_data_hash"],
        "development_windows": result.report["development_windows"],
        "class_order": [0, 1, 2],
        "calibrator_id": calibrator_id,
        "calibrator_version": "1.0.0",
        "code_version": stable_hash({
            "meta_model": _sha256(Path(__file__).with_name("model.py")),
            "calibration": _sha256(Path(__file__).with_name("calibration.py")),
        }),
        "scikit_learn_version": sklearn.__version__,
        "numpy_version": numpy.__version__,
        "pandas_version": pandas.__version__,
        "library_versions": {"scikit-learn": sklearn.__version__, "numpy": numpy.__version__,
                             "pandas": pandas.__version__},
    }
    artifact_id = stable_hash(identity)[:32]
    path = Path(root) / artifact_id
    manifest_file = path / "manifest.json"
    if manifest_file.exists():
        return load_artifact(path, expected_candidate_hash=result.candidate.data_hash)
    if path.exists():
        raise ValueError("META_ARTIFACT_DESTINATION_COLLISION")
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f"{path.name}.staging-{uuid.uuid4().hex}")
    staging.mkdir()
    try:
        model_path = staging / "meta.joblib"
        calibrator_dir = staging / "calibrator"
        calibrator_dir.mkdir()
        calibration_path = calibrator_dir / "calibrator.joblib"
        joblib.dump(result.model, model_path)
        joblib.dump(result.calibrator, calibration_path)
        calibrator_manifest = {
            "calibrator_id": calibrator_id, "calibrator_version": "1.0.0",
            "meta_artifact_id": artifact_id,
            "method": result.report["selected_calibration_method"],
            "fit_sample_count": result.calibrator.fit_sample_count,
            "fit_data_hash": result.report["calibration_fit_data_hash"],
            "data_hash": result.report["calibration_fit_data_hash"],
            "config_hash": result.report["split_config_hash"],
            "calibration_period": result.report["development_windows"]["calibration_fit"],
            "fit_end": str(result.config["calibration_fit"][1]),
            "weight_sha256": _sha256(calibration_path),
            "created_at": datetime.now(UTC).isoformat(),
        }
        calibrator_manifest_path = calibrator_dir / "manifest.json"
        calibrator_manifest_path.write_text(json.dumps(calibrator_manifest, indent=2,
            sort_keys=True), encoding="utf-8")
        manifest = identity | {
            "artifact_id": artifact_id,
            "meta_sha256": _sha256(model_path),
            "calibrator_sha256": _sha256(calibration_path),
            "calibrator_manifest_sha256": _sha256(calibrator_manifest_path),
            "created_at": datetime.now(UTC).isoformat(),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        load_artifact(staging, expected_candidate_hash=result.candidate.data_hash,
                      allow_staging_path=True)
        os.replace(staging, path)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return load_artifact(path, expected_candidate_hash=result.candidate.data_hash)


def load_artifact(path: str | Path, *, expected_candidate_hash: str,
                  allow_staging_path: bool = False,
                  allow_phase9_1_frozen_compatibility: bool = False) -> MetaArtifact:
    """Refuse changed weights, wrong candidate, or live-production claims."""
    directory = Path(path)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    code_version_compatible = manifest["code_version"] == stable_hash({
        "meta_model": _sha256(Path(__file__).with_name("model.py")),
        "calibration": _sha256(Path(__file__).with_name("calibration.py")),
    })
    if (allow_phase9_1_frozen_compatibility and
            manifest["artifact_id"] == PHASE9_1_FROZEN_COMPATIBILITY["artifact_id"] and
            manifest["code_version"] == PHASE9_1_FROZEN_COMPATIBILITY["code_version"]):
        code_version_compatible = True
    if (manifest["candidate_data_hash"] != expected_candidate_hash or
            manifest["candidate_dataset_id"] != expected_candidate_hash[:32] or
            manifest["model_id"] != "META_NO_MARKET_V1" or
            manifest["temporal_mode"] != "DATE_SAFE_BATCH" or
            manifest["final_holdout_status"] != "LOCKED_AWAITING_DATA" or
            manifest["market_dependency_count"] != 0 or
            manifest["class_order"] != [0, 1, 2] or
            manifest["meta_model_id"] != manifest["model_id"] or
            manifest["version"] != manifest["model_version"] or
            manifest["config_hash"] != manifest["split_config_hash"] or
            manifest["data_hash"] != manifest["selected_training_data_hash"] or
            manifest["base_model_schema_version"] != "REAL_OOS_HDA_V1" or
            not isinstance(manifest["candidate_dataset_schema_hash"], str) or
            len(manifest["candidate_dataset_schema_hash"]) != 64 or
            manifest["feature_schema_version"] != "META_LOG_RATIO_MASK_V1" or
            manifest["dependency_schema_version"] != "META_DEPENDENCY_V1" or
            manifest["scikit_learn_version"] != sklearn.__version__ or
            manifest["numpy_version"] != numpy.__version__ or
            manifest["pandas_version"] != pandas.__version__ or
            manifest["library_versions"] != {"scikit-learn": sklearn.__version__,
                                             "numpy": numpy.__version__,
                                             "pandas": pandas.__version__} or
            not code_version_compatible or
            manifest["production_compatibility"] != "RESEARCH_ONLY_NOT_LIVE_VALIDATED"):
        raise ValueError("META_ARTIFACT_PROVENANCE_REJECTED")
    identity = {key: value for key, value in manifest.items() if key not in (
        "artifact_id", "meta_sha256", "calibrator_sha256", "calibrator_manifest_sha256", "created_at")}
    if (stable_hash(identity)[:32] != manifest["artifact_id"] or
            (not allow_staging_path and directory.name != manifest["artifact_id"])):
        raise ValueError("META_ARTIFACT_ID_MISMATCH")
    model_path = directory / "meta.joblib"
    calibration_path = directory / "calibrator" / "calibrator.joblib"
    calibration_manifest_path = directory / "calibrator" / "manifest.json"
    if _sha256(calibration_manifest_path) != manifest["calibrator_manifest_sha256"]:
        raise ValueError("CALIBRATOR_MANIFEST_HASH_MISMATCH")
    calibration_manifest = json.loads(calibration_manifest_path.read_text(encoding="utf-8"))
    if (calibration_manifest["calibrator_id"] != manifest["calibrator_id"] or
            calibration_manifest["meta_artifact_id"] != manifest["artifact_id"] or
            calibration_manifest["fit_data_hash"] != manifest["calibration_fit_data_hash"] or
            calibration_manifest["data_hash"] != manifest["calibration_fit_data_hash"] or
            calibration_manifest["config_hash"] != manifest["split_config_hash"] or
            calibration_manifest["calibration_period"] != manifest["development_windows"]["calibration_fit"] or
            calibration_manifest["method"] != manifest["selected_calibration_method"] or
            calibration_manifest["fit_end"] != manifest["calibration_fit_end"] or
            calibration_manifest["weight_sha256"] != manifest["calibrator_sha256"]):
        raise ValueError("CALIBRATOR_LINEAGE_MISMATCH")
    if _sha256(model_path) != manifest["meta_sha256"] or _sha256(calibration_path) != manifest["calibrator_sha256"]:
        raise ValueError("META_ARTIFACT_WEIGHT_HASH_MISMATCH")
    model = joblib.load(model_path)
    calibrator = joblib.load(calibration_path)
    if (not isinstance(model, NoMarketMetaModel) or
            not isinstance(calibrator, MulticlassCalibrator) or
            model.trained_until is None or model.trained_until.isoformat() != manifest["meta_trained_until"] or
            calibrator.method != manifest["selected_calibration_method"] or
            model.epsilon != manifest["log_ratio_epsilon"] or
            calibrator.fit_sample_count != manifest["calibration_fit_sample_count"]):
        raise ValueError("META_ARTIFACT_OBJECT_MISMATCH")
    return MetaArtifact(manifest["artifact_id"], directory, manifest, model, calibrator)
