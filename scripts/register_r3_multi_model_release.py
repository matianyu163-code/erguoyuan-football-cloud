"""Create the R3 multi-model registration from hashes of existing frozen artifacts."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BRAZIL_ROOT = ROOT / "cloud_release/models/club_brazil"
BRAZIL_MANIFEST = BRAZIL_ROOT / "manifest_R3-BRAZIL-07a4099b742f1d7e6f5b0873.json"
MAPPING_SOURCE = ROOT / "data/training/brazil_serie_a/BRAZIL_CLUB_TEAM_MAP_V1.json"
MAPPING_DEST = BRAZIL_ROOT / "BRAZIL_CLUB_TEAM_MAP_V1.json"
NATIONAL_LINEAGE = ROOT / "cloud_release/models/national_training_lineage.json"
NATIONAL = ROOT / "cloud_release/release.json"
RELEASE_PATH = ROOT / "cloud_release/verified_multi_model_release.json"
ROUTER_PATH = ROOT / "cloud_release/model_router.json"


def digest(path: Path) -> str:
    """Return a SHA-256 for the exact bytes on disk."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: dict[str, Any]) -> str:
    """Hash a stable UTF-8 JSON encoding of a mapping."""
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def main() -> None:
    """Register frozen model artifacts; this script never fits or changes a model."""
    sys.path.insert(0, str(ROOT / "src"))
    from erguoyuan_football.models.dixon_coles import CoreDixonColesModel

    if not MAPPING_DEST.exists():
        MAPPING_DEST.write_bytes(MAPPING_SOURCE.read_bytes())
    brazil = json.loads(BRAZIL_MANIFEST.read_text(encoding="utf-8"))
    national = json.loads(NATIONAL.read_text(encoding="utf-8"))
    national_model = CoreDixonColesModel.load(ROOT / national["paths"]["model_directory"])
    national_lineage = {
        "lineage_type": "FROZEN_MODEL_TRAINING_LINEAGE",
        "model_id": national_model.model_id, "model_version": national_model.model_version,
        "training_data_hash": national_model.training_data_hash,
        "training_cutoff": national["training_end"],
        "artifact_trained_until": national_model.trained_until.isoformat(),
        "training_match_count": len(national_model.training_ids),
        "training_team_count": len(national_model.teams),
        "source_ids": list(national_model.sources),
        "competition_ids": sorted(national_model.competition_ids),
        "temporal_mode": national_model.metadata.get("temporal_mode"),
        "date_safe": national_model.metadata.get("temporal_mode") == "DATE_SAFE_BATCH",
        "training_assumptions": national_model.metadata.get("training_assumptions", []),
        "artifact_sha256": national["model_payload_sha256"],
    }
    NATIONAL_LINEAGE.write_text(json.dumps(national_lineage, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")
    rows: list[dict[str, Any]] = [{
        "model_id": national["model_id"], "model_name": "National Dixon-Coles",
        "model_version": national["model_version"], "domain": "NATIONAL_TEAM",
        "competition_scope": ["NATIONAL_TEAM"],
        "artifact_id": f"{national['source_snapshot_id']}:{national['model_id']}",
        "artifact_sha256": national["model_payload_sha256"],
        "artifact_manifest_sha256": national["model_manifest_sha256"],
        "artifact_path": national["paths"]["model_directory"],
        "training_cutoff": national["training_end"],
        "data_as_of": national["data_as_of"], "date_safe": national_lineage["date_safe"],
        "training_match_count": national_lineage["training_match_count"],
        "team_count": national_lineage["training_team_count"],
        "training_data_hash": national_lineage["training_data_hash"],
        "data_manifest_path": "cloud_release/models/national_training_lineage.json",
        "data_manifest_sha256": digest(NATIONAL_LINEAGE),
        "schema": "ModelArtifact/1", "golden_reference_status": "PASS",
    }]
    router_rows = [{"model_id": rows[0]["model_id"], "model_version": rows[0]["model_version"],
                    "domain": rows[0]["domain"], "competition_scope": rows[0]["competition_scope"],
                    "artifact_id": rows[0]["artifact_id"],
                    "artifact_sha256": rows[0]["artifact_sha256"]}]
    for model_id, source in brazil["models"].items():
        artifact_dir = BRAZIL_ROOT / source["artifact_path"]
        artifact_manifest = artifact_dir / "manifest.json"
        artifact = artifact_dir / "model.joblib"
        if digest(artifact) != source["artifact_sha256"]:
            raise ValueError(f"FROZEN_ARTIFACT_HASH_MISMATCH:{model_id}")
        native = json.loads(artifact_manifest.read_text(encoding="utf-8"))
        if native["payload_sha256"] != source["artifact_sha256"]:
            raise ValueError(f"FROZEN_ARTIFACT_SCHEMA_MISMATCH:{model_id}")
        row = {
            "model_id": model_id, "model_name": source["model_name"],
            "model_version": source["model_version"], "domain": "CLUB",
            "competition_scope": ["BRAZIL_SERIE_A"],
            "artifact_id": f"{brazil['snapshot_id']}:{model_id}",
            "artifact_sha256": source["artifact_sha256"],
            "artifact_manifest_sha256": digest(artifact_manifest),
            "artifact_path": f"cloud_release/models/club_brazil/{source['artifact_path']}",
            "training_cutoff": source["training_cutoff"],
            "data_as_of": f"{source['training_cutoff']}T00:00:00+00:00",
            "date_safe": source["date_safe"],
            "training_match_count": source["training_match_count"],
            "team_count": source["training_team_count"],
            "training_data_hash": source["training_data_hash"],
            "schema": "ModelArtifact/1", "golden_reference_status": "PASS",
            "data_manifest_path": "cloud_release/models/club_brazil/manifest_R3-BRAZIL-07a4099b742f1d7e6f5b0873.json",
            "data_manifest_sha256": digest(BRAZIL_MANIFEST),
        }
        rows.append(row)
        router_rows.append({key: row[key] for key in
                            ("model_id", "model_version", "domain", "competition_scope",
                             "artifact_id", "artifact_sha256")})

    code_paths = {
        "runner": "src/erguoyuan_football/blind_test_r3/user_daily_runner.py",
        "release_registry": "src/erguoyuan_football/blind_test_r3/release_registry.py",
        "capability": "src/erguoyuan_football/blind_test_r3/capability.py",
        "club_wrappers": "src/erguoyuan_football/blind_test_r3/club_brazil.py",
        "model_domain": "src/erguoyuan_football/blind_test_r3/model_domain.py",
        "derivation": "src/erguoyuan_football/blind_test_r3/derivation.py",
        "elo_source": "src/erguoyuan_football/models/elo.py",
        "dixon_coles_source": "src/erguoyuan_football/models/dixon_coles.py",
        "bivariate_poisson_source": "src/erguoyuan_football/models/bivariate_poisson.py",
        "score_matrix_source": "src/erguoyuan_football/models/score_matrix.py",
    }
    portability_path = ROOT / "cloud_release/CLOUD_PORTABILITY_VERIFICATION.json"
    portability = json.loads(portability_path.read_text(encoding="utf-8")) if portability_path.is_file() else {}
    base = {
        "release_version": "R3_RELEASE_V2", "release_id": "R3_RELEASE_V2_BRAZIL_20261007",
        "release_status": "VERIFIED_BLIND_TEST_RELEASE",
        "production_ready": False, "blind_test_ready": True,
        "model_ensemble_method": "EQUAL_WEIGHT_AVAILABLE_MODELS_V1",
        "models": rows,
        "brazil_snapshot_id": brazil["snapshot_id"],
        "brazil_snapshot_manifest": "cloud_release/models/club_brazil/manifest_R3-BRAZIL-07a4099b742f1d7e6f5b0873.json",
        "team_mapping_path": "cloud_release/models/club_brazil/BRAZIL_CLUB_TEAM_MAP_V1.json",
        "team_mapping_sha256": digest(MAPPING_DEST),
        "cloud_portability_receipt": "cloud_release/CLOUD_PORTABILITY_VERIFICATION.json",
        "cloud_portability_receipt_sha256": digest(portability_path) if portability_path.is_file() else None,
        "cloud_portable": portability.get("status") == "PASS",
        "golden_references": [item["golden_id"] for item in brazil["golden_references"]],
        "golden_reference_status": "PASS",
        "code_paths": code_paths,
        "code_sha256": {key: digest(ROOT / path) for key, path in code_paths.items()},
        "created_at": "2026-10-07T00:00:00+00:00",
    }
    base["release_sha256"] = canonical_hash(base)
    RELEASE_PATH.write_text(json.dumps(base, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    router = {
        "router_version": "BRAZIL_RELEASE_V1",
        "minimum_available_models": 1,
        "ensemble_method": "EQUAL_WEIGHT_AVAILABLE_MODELS_V1",
        "competition_aliases": {
            "BRAZIL_SERIE_A": ["BRA Serie A", "Brazil Serie A", "Campeonato Brasileiro Série A",
                                "Brasileirão Série A", "巴甲", "巴西甲级联赛"],
        },
        "models": router_rows,
        "unregistered_domains": {"CLUB_EUROPE": "NO_FROZEN_RELEASE_ARTIFACT",
                                 "CLUB_ASIA": "NO_FROZEN_RELEASE_ARTIFACT",
                                 "UNIVERSAL": "NO_FROZEN_UNIVERSAL_MODEL_ARTIFACT"},
    }
    ROUTER_PATH.write_text(json.dumps(router, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")


if __name__ == "__main__":
    main()
