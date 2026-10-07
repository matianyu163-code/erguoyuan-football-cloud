"""Load and verify the immutable multi-domain R3 blind-test model release."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from erguoyuan_football.blind_test_r3.club_brazil import (
    BrazilBivariatePoissonModel,
    BrazilDixonColesModel,
    BrazilEloModel,
)
from erguoyuan_football.blind_test_r3.store import sha256
from erguoyuan_football.models.dixon_coles import CoreDixonColesModel

MODEL_CLASSES = {
    "CLUB_ELO_BRAZIL_V1": BrazilEloModel,
    "CLUB_DIXON_COLES_BRAZIL_V1": BrazilDixonColesModel,
    "CLUB_BIVARIATE_POISSON_BRAZIL_V1": BrazilBivariatePoissonModel,
}


def load_verified_release(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify every registered artifact and code/data lineage before loading it."""
    manifest_path = root / "cloud_release/verified_multi_model_release.json"
    release = json.loads(manifest_path.read_text(encoding="utf-8"))
    if release.get("release_version") != "R3_RELEASE_V2":
        raise ValueError("VERIFIED_RELEASE_VERSION_INVALID")
    if release.get("production_ready") is not False or release.get("blind_test_ready") is not True:
        raise ValueError("VERIFIED_RELEASE_STATUS_INVALID")
    expected_manifest_sha = release.get("release_sha256")
    unsigned = {key: value for key, value in release.items() if key != "release_sha256"}
    actual_manifest_sha = hashlib.sha256(
        json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if expected_manifest_sha != actual_manifest_sha:
        raise ValueError("VERIFIED_RELEASE_MANIFEST_HASH_MISMATCH")
    for name, relative in release.get("code_paths", {}).items():
        if sha256((root / relative).read_bytes()) != release.get("code_sha256", {}).get(name):
            raise ValueError(f"VERIFIED_RELEASE_CODE_HASH_MISMATCH:{name}")
    if release.get("cloud_portable"):
        portability_path = root / release["cloud_portability_receipt"]
        portability = json.loads(portability_path.read_text(encoding="utf-8"))
        if (sha256(portability_path.read_bytes()) != release.get("cloud_portability_receipt_sha256")
                or portability.get("status") != "PASS"):
            raise ValueError("CLOUD_PORTABILITY_RECEIPT_INVALID")

    policy = json.loads((root / "cloud_release/model_policy.json").read_text(encoding="utf-8"))
    legacy = json.loads((root / "cloud_release/release.json").read_text(encoding="utf-8"))
    loaded: dict[str, Any] = {}
    national = next((row for row in release["models"] if row["model_id"] == "DIXON_COLES_V1"), None)
    if national is None:
        raise ValueError("VERIFIED_RELEASE_NATIONAL_MODEL_MISSING")
    national_dir = root / legacy["paths"]["model_directory"]
    national_native = json.loads((national_dir / "manifest.json").read_text(encoding="utf-8"))
    if sha256((national_dir / "manifest.json").read_bytes()) != national["artifact_manifest_sha256"]:
        raise ValueError("NATIONAL_MODEL_SCHEMA_HASH_MISMATCH")
    if (national_native.get("model_id") != national["model_id"]
            or national_native.get("model_version") != national["model_version"]
            or national_native.get("payload_sha256") != national["artifact_sha256"]):
        raise ValueError("NATIONAL_MODEL_NATIVE_MANIFEST_MISMATCH")
    national_model = CoreDixonColesModel.load(national_dir)
    if not national_model.fitted or national_model.trained_until is None:
        raise ValueError("NATIONAL_MODEL_ARTIFACT_NOT_FITTED")
    if national_model.model_version != national["model_version"]:
        raise ValueError("NATIONAL_MODEL_VERSION_MISMATCH")
    if sha256((root / legacy["paths"]["model_directory"] / "model.joblib").read_bytes()) != national["artifact_sha256"]:
        raise ValueError("NATIONAL_MODEL_ARTIFACT_HASH_MISMATCH")
    national_data_path = root / national["data_manifest_path"]
    national_data = json.loads(national_data_path.read_text(encoding="utf-8"))
    if sha256(national_data_path.read_bytes()) != national["data_manifest_sha256"]:
        raise ValueError("NATIONAL_DATA_MANIFEST_HASH_MISMATCH")
    if (national_data.get("training_data_hash") != national_model.training_data_hash
            or national_data.get("date_safe") is not True
            or national_data.get("temporal_mode") != "DATE_SAFE_BATCH"
            or national_data.get("artifact_sha256") != national["artifact_sha256"]):
        raise ValueError("NATIONAL_DATA_MANIFEST_LINEAGE_MISMATCH")
    if national.get("schema") != "ModelArtifact/1" or (
            national.get("golden_reference_status") != "PASS"):
        raise ValueError("NATIONAL_MODEL_GATE_METADATA_INVALID")
    loaded[national["model_id"]] = national_model

    brazil_manifest_path = root / release["brazil_snapshot_manifest"]
    brazil_manifest = json.loads(brazil_manifest_path.read_text(encoding="utf-8"))
    if brazil_manifest.get("snapshot_id") != release.get("brazil_snapshot_id"):
        raise ValueError("BRAZIL_SNAPSHOT_ID_MISMATCH")
    if not brazil_manifest.get("data", {}).get("date_safe") or brazil_manifest.get("runtime_fit") is not False:
        raise ValueError("BRAZIL_DATA_DATE_SAFETY_INVALID")
    if not brazil_manifest.get("golden_test_pass") or len(brazil_manifest.get("golden_references", [])) < 2:
        raise ValueError("BRAZIL_GOLDEN_REFERENCE_MISSING")
    if not release.get("models") or release.get("golden_reference_status") != "PASS":
        raise ValueError("VERIFIED_RELEASE_GOLDEN_GATE_FAILED")
    mapping_path = root / release["team_mapping_path"]
    if not mapping_path.is_file() or sha256(mapping_path.read_bytes()) != release["team_mapping_sha256"]:
        raise ValueError("BRAZIL_TEAM_MAPPING_MISSING")

    gate_by_id = {row["model_id"]: row for row in release["models"]}
    for model_id, model_class in MODEL_CLASSES.items():
        gate = gate_by_id.get(model_id)
        source = brazil_manifest.get("models", {}).get(model_id)
        if gate is None or source is None:
            raise ValueError(f"BRAZIL_MODEL_NOT_REGISTERED:{model_id}")
        if gate.get("domain") != "CLUB" or "BRAZIL_SERIE_A" not in gate.get("competition_scope", []):
            raise ValueError(f"BRAZIL_MODEL_DOMAIN_MISMATCH:{model_id}")
        if gate.get("model_version") != source.get("model_version"):
            raise ValueError(f"BRAZIL_MODEL_VERSION_MISMATCH:{model_id}")
        if gate.get("artifact_id") != f"{brazil_manifest['snapshot_id']}:{model_id}":
            raise ValueError(f"BRAZIL_ARTIFACT_ID_MISMATCH:{model_id}")
        if gate.get("artifact_sha256") != source.get("artifact_sha256"):
            raise ValueError(f"BRAZIL_ARTIFACT_HASH_MANIFEST_MISMATCH:{model_id}")
        if source.get("date_safe") is not True or source.get("runtime_fit") is not False:
            raise ValueError(f"BRAZIL_MODEL_DATE_SAFETY_INVALID:{model_id}")
        if not source.get("training_cutoff") or not source.get("training_data_hash"):
            raise ValueError(f"BRAZIL_MODEL_TRAINING_LINEAGE_MISSING:{model_id}")
        data_path = root / gate["data_manifest_path"]
        if (not data_path.is_file()
                or sha256(data_path.read_bytes()) != gate.get("data_manifest_sha256")
                or gate.get("date_safe") is not True
                or gate.get("schema") != "ModelArtifact/1"
                or gate.get("golden_reference_status") != "PASS"):
            raise ValueError(f"BRAZIL_MODEL_DATA_OR_GOLDEN_GATE_FAILED:{model_id}")
        data_manifest = json.loads(data_path.read_text(encoding="utf-8"))
        if (data_manifest.get("snapshot_id") != brazil_manifest["snapshot_id"]
                or data_manifest.get("data", {}).get("training_cutoff") != gate["training_cutoff"]
                or data_manifest.get("models", {}).get(model_id, {}).get("training_data_hash")
                   != gate["training_data_hash"]):
            raise ValueError(f"BRAZIL_MODEL_DATA_MANIFEST_LINEAGE_MISMATCH:{model_id}")
        model_manifest_path = root / "cloud_release/models/club_brazil" / source["artifact_path"] / ".." / "manifest.json"
        model_manifest_path = model_manifest_path.resolve()
        model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
        if model_manifest.get("artifact_sha256") != gate["artifact_sha256"]:
            raise ValueError(f"BRAZIL_MODEL_NATIVE_MANIFEST_MISMATCH:{model_id}")
        artifact_dir = (root / "cloud_release/models/club_brazil" / source["artifact_path"]).resolve()
        model_joblib = artifact_dir / "model.joblib"
        model_meta = artifact_dir / "manifest.json"
        if sha256(model_joblib.read_bytes()) != gate["artifact_sha256"]:
            raise ValueError(f"BRAZIL_MODEL_ARTIFACT_HASH_MISMATCH:{model_id}")
        if sha256(model_meta.read_bytes()) != gate.get("artifact_manifest_sha256"):
            raise ValueError(f"BRAZIL_MODEL_SCHEMA_HASH_MISMATCH:{model_id}")
        model = model_class.load(artifact_dir)
        if not model.fitted or model.model_id != model_id or model.model_version != gate["model_version"]:
            raise ValueError(f"BRAZIL_MODEL_LOAD_OR_IDENTITY_FAILED:{model_id}")
        if model.trained_until is None or model.trained_until.date().isoformat() != gate["training_cutoff"]:
            raise ValueError(f"BRAZIL_MODEL_CUTOFF_MISMATCH:{model_id}")
        if len(model.teams) != int(gate["team_count"]):
            raise ValueError(f"BRAZIL_MODEL_TEAM_COUNT_MISMATCH:{model_id}")
        loaded[model_id] = model
    router_path = root / "cloud_release/model_router.json"
    router = json.loads(router_path.read_text(encoding="utf-8"))
    release_ids = {row["model_id"] for row in release["models"]}
    if any(row["model_id"] not in release_ids for row in router.get("models", [])):
        raise ValueError("MODEL_ROUTER_ENTRY_NOT_IN_VERIFIED_RELEASE")
    if {row["model_id"] for row in router.get("models", [])} != release_ids:
        raise ValueError("VERIFIED_RELEASE_ROUTER_SET_MISMATCH")
    for row in router.get("models", []):
        bound = gate_by_id[row["model_id"]]
        if any(row.get(field) != bound.get(field) for field in (
                "model_version", "domain", "competition_scope", "artifact_id", "artifact_sha256")):
            raise ValueError(f"VERIFIED_RELEASE_ROUTER_BINDING_MISMATCH:{row['model_id']}")
    return {**release, "loaded_models": loaded,
            "release_manifest_path": str(manifest_path),
            "brazil_mapping": json.loads(mapping_path.read_text(encoding="utf-8"))}, policy
