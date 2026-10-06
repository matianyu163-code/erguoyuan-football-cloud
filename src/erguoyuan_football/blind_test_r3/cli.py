"""Initialize and inspect the independent R3 store without running a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from erguoyuan_football.blind_test_r3.store import R3Store

ROOT = Path(__file__).resolve().parents[3] / "blind_test" / "r3"


def main(argv: list[str] | None = None) -> int:
    """Report R3 readiness from actual frozen snapshots and locked records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "status"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    store = R3Store(args.root)
    if args.command == "init":
        store.initialize()
    snapshots = []
    if (store.root / "snapshots").is_dir():
        for path in sorted((store.root / "snapshots").glob("R3-*.json")):
            try:
                store.verify_snapshot(path.stem)
                snapshots.append({"id": path.stem, "status": "VERIFIED"})
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
                snapshots.append({"id": path.stem, "status": f"INVALID:{type(error).__name__}"})
    locked = []
    if (store.root / "locks").is_dir():
        for path in sorted((store.root / "locks").glob("*.json")):
            if ".error." in path.name:
                continue
            try:
                if store.verify_lock(path.stem):
                    locked.append(path.stem)
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
    ready = bool(locked and any(item["status"] == "VERIFIED" for item in snapshots))
    print(json.dumps({"MODE": "BLIND_TEST_R3", "YY_CORE_PRODUCTION_READY": False,
                      "YY_CORE_BLIND_TEST_R3_READY": ready,
                      "snapshots": snapshots, "locked_prediction_count": len(locked),
                      "model_execution": "LOCKED_R3_PREDICTION_PRESENT" if ready else
                      "UNAVAILABLE_NO_VERIFIED_LOCKED_PREDICTION",
                      "root": str(store.root)},
                     ensure_ascii=False, indent=2))
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
