"""Run only Dynamic Bayesian Poisson V1 against a real store snapshot."""

from __future__ import annotations

import argparse
import json
from datetime import datetime

from erguoyuan_football.data.snapshots import SnapshotService
from erguoyuan_football.data.store import Store
from erguoyuan_football.models.config import ModelConfig
from erguoyuan_football.models.runner import ModelRunner
from erguoyuan_football.models.training import InsufficientData, dataset_from_store


def _parse_iso8601(value: str) -> datetime:
    """Parse an ISO-8601 timestamp, including the conventional Z suffix."""
    return datetime.fromisoformat(value)


def main() -> int:
    """Create a frozen snapshot and execute the dynamic model only."""
    parser = argparse.ArgumentParser(description="二果园足球预测系统 — Dynamic Bayesian Poisson V1")
    parser.add_argument("--db", required=True)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--competition-id", required=True)
    parser.add_argument("--trained-until", required=True)
    parser.add_argument("--allow-test-data", action="store_true")
    args = parser.parse_args()
    prediction_time = _parse_iso8601(args.as_of)
    trained_until = _parse_iso8601(args.trained_until)
    with Store(args.db) as store:
        try:
            snapshot = SnapshotService(store).create(args.match_id, prediction_time)
            data = dataset_from_store(store, args.competition_id, trained_until)
            result = ModelRunner().run(
                snapshot.match_data_snapshot, snapshot, data, trained_until=trained_until,
                config=ModelConfig(allow_test_data=args.allow_test_data),
                model_ids=("DYNAMIC_BAYESIAN_POISSON_V1",),
            )
            print(json.dumps({
                "status": "OK",
                "match_id": snapshot.match_id,
                "prediction_snapshot_id": snapshot.prediction_snapshot_id,
                "model": result.predictions[0].model_dump(mode="json"),
            }, ensure_ascii=False, indent=2))
            return 0
        except (InsufficientData, ValueError) as error:
            print(json.dumps({"status": "INSUFFICIENT_DATA", "reason": f"{type(error).__name__}:{error}"},
                             ensure_ascii=False))
            return 2


if __name__ == "__main__":
    raise SystemExit(main())
