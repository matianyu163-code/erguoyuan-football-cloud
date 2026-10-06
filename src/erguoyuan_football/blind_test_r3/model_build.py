"""Development-only, one-time national-team Dixon-Coles artifact builder."""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path

from erguoyuan_football.blind_test_r3.national_history import (
    PINNED_COMMIT,
    NationalTeamHistoryProvider,
)
from erguoyuan_football.models.config import load_config
from erguoyuan_football.models.dixon_coles import CoreDixonColesModel
from erguoyuan_football.models.training import TrainingDatasetValidator

PROJECT_ROOT = Path(__file__).resolve().parents[3]
R3_ROOT = PROJECT_ROOT / "blind_test" / "r3"


def main(argv: list[str] | None = None) -> int:
    """Fit only in MODEL_BUILD and save a new, never-overwritten artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", required=True, help="UTC as-of timestamp before target kickoff")
    parser.add_argument("--train-start", type=date.fromisoformat, required=True)
    args = parser.parse_args(argv)
    cutoff = datetime.fromisoformat(args.cutoff)
    provider = NationalTeamHistoryProvider(
        R3_ROOT / "model_build" / "raw" / f"results-{PINNED_COMMIT}.csv",
        R3_ROOT / "model_build" / "raw" / f"results-{PINNED_COMMIT}.json")
    dataset = provider.training_dataset(cutoff, train_start=args.train_start)
    config = load_config(PROJECT_ROOT / "configs" / "models.yaml", profile="development")
    TrainingDatasetValidator().validate(dataset, cutoff, max_score=config.max_score)
    if len(dataset.matches) < config.min_matches:
        raise ValueError("NATIONAL_MODEL_TRAINING_SAMPLE_INSUFFICIENT")
    directory = R3_ROOT / "model_build" / "artifacts" / (
        "dixon_coles_national_" + dataset.data_hash[:24])
    if directory.exists():
        raise FileExistsError(f"FROZEN_BUILD_ARTIFACT_EXISTS:{directory}")
    model = CoreDixonColesModel().fit(dataset, cutoff, config)
    artifact = model.save(directory)
    summary = provider.summary(cutoff, args.train_start, training_only=True)
    record = {"mode": "MODEL_BUILD", "artifact_path": str(directory.resolve()),
              "model_id": model.model_id, "model_version": model.model_version,
              "artifact_sha256": artifact.payload_sha256,
              "training_data_hash": dataset.data_hash,
              "config_hash": config.config_hash, "train_start": args.train_start.isoformat(),
              "training_cutoff": cutoff.isoformat(), "training_rows_used": len(
                  dataset.window(cutoff, config.training_window).matches),
              "friendly_policy": "EXCLUDED", "cup_policy": "EXCLUDED_SCORE_INCLUDES_EXTRA_TIME",
              "neutral_policy": "SOURCE_BOOL",
              "summary": summary}
    with (directory / "build_report.json").open("x", encoding="utf-8") as output:
        json.dump(record, output, ensure_ascii=False, indent=2)
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
