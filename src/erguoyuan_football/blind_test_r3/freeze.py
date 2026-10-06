"""Freeze one fitted national-team artifact into an immutable R3 model snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from erguoyuan_football.blind_test_r3.store import R3Store, sha256
from erguoyuan_football.models.dixon_coles import CoreDixonColesModel

PROJECT_ROOT = Path(__file__).resolve().parents[3]
R3_ROOT = PROJECT_ROOT / "blind_test" / "r3"


def freeze(artifact_dir: Path) -> str:
    """Copy and hash code, data, configuration and fitted parameters once."""
    artifact_dir = artifact_dir.resolve()
    model = CoreDixonColesModel.load(artifact_dir)
    report = json.loads((artifact_dir / "build_report.json").read_text(encoding="utf-8"))
    summary = report["summary"]
    source_files = {
        "model_source": PROJECT_ROOT / "src/erguoyuan_football/models/dixon_coles.py",
        "adapter_source": PROJECT_ROOT / "src/erguoyuan_football/models/penaltyblog_adapter.py",
        "score_matrix_source": PROJECT_ROOT / "src/erguoyuan_football/models/score_matrix.py",
        "derivation_source": PROJECT_ROOT / "src/erguoyuan_football/blind_test_r3/derivation.py",
        "runner_source": PROJECT_ROOT / "src/erguoyuan_football/blind_test_r3/runner.py",
        "provider_source": PROJECT_ROOT / "src/erguoyuan_football/blind_test_r3/national_history.py",
        "model_build_source": PROJECT_ROOT / "src/erguoyuan_football/blind_test_r3/model_build.py",
        "country_mapping_source": PROJECT_ROOT / "src/erguoyuan_football/knowledge/entities/country_alias_registry.py",
    }
    files = {**source_files,
        "model_artifact": artifact_dir / "model.joblib",
        "artifact_manifest": artifact_dir / "manifest.json",
        "model_config": PROJECT_ROOT / "configs/models.yaml",
        "model_policy": R3_ROOT / "manifest/model_policy_v1.json",
        "build_report": artifact_dir / "build_report.json",
    }
    code_hash = hashlib.sha256("|".join(
        f"{name}:{sha256(path.read_bytes())}" for name, path in sorted(source_files.items())
    ).encode()).hexdigest()
    config = model.config.model_dump(mode="json")
    specification = {
        "model_name": "Dixon-Coles national-team cohort",
        "model_id": model.model_id, "model_version": model.model_version,
        "feature_version": "NATIONAL_HISTORY_DATE_SAFE_BATCH_V1",
        "data_version": model.training_data_hash,
        "data_as_of": summary["data_as_of"],
        "training_start": summary["train_start"],
        "training_end": summary["train_end"],
        "training_cutoff": report["training_cutoff"],
        "match_count": report["training_rows_used"],
        "team_count": summary["team_count"],
        "competition_count": summary["competition_count"],
        "hyperparameters": config,
        "home_advantage_log": model.metadata["home_advantage_log"],
        "neutral_handling": "SOURCE_BOOL_IN_FIT_AND_PREDICT",
        "friendly_policy": report["friendly_policy"],
        "calibrator_version": "UNAVAILABLE",
        "score_matrix_method": "PENALTYBLOG_DC_TO_SCORE_MATRIX_V1",
        "derivation_method": "R3_SCORE_MATRIX_DERIVATION_V1",
        "code_commit": "UNVERSIONED_SOURCE_TREE",
        "code_hash": code_hash,
        "creation_time": json.loads((artifact_dir / "manifest.json").read_text(
            encoding="utf-8"))["created_at"],
    }
    store = R3Store(R3_ROOT)
    store.initialize()
    return store.freeze_snapshot(specification, files)


def main(argv: list[str] | None = None) -> int:
    """Freeze only an existing, validated MODEL_BUILD artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(freeze(args.artifact_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
