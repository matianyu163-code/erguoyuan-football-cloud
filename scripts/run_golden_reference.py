"""Replay Golden Reference 001 without writing predictions or fitting a model."""

import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from erguoyuan_football.blind_test_r3.market_intelligence import (  # noqa: I001
    fuse_model_market,
    load_r3_market_config,
)
from erguoyuan_football.blind_test_r3.user_daily import UserFixture
from erguoyuan_football.blind_test_r3.user_daily_runner import _model_output, _verify_release


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    reference = json.loads((root / "tests/reference/GOLDEN_REFERENCE_001.json")
                           .read_text(encoding="utf-8"))
    release, policy = _verify_release(root)
    model = release["loaded_models"]["DIXON_COLES_V1"]
    national_release = next(row for row in release["models"]
                            if row["model_id"] == "DIXON_COLES_V1")
    if not national_release["artifact_id"].startswith(reference["model_snapshot_id"]):
        raise ValueError("GOLDEN_MODEL_SNAPSHOT_CHANGED")
    fixture = UserFixture.model_validate(reference["fixture_input"])
    raw, standard = _model_output(model, fixture, reference["fixture_id"],
                                  datetime.fromisoformat(reference["prediction_time"]), policy)
    model_prob = standard["one_x_two"]["probabilities"]
    market = fixture.spf.no_vig()
    fusion = fuse_model_market(model_prob, market,
        load_r3_market_config(root / "config/r3_market_intelligence_v1.yaml"))
    for actual, expected in ((model_prob, reference["model_probabilities"]),
                             (market, reference["jc_no_vig"]),
                             (fusion["probabilities"], reference["fusion_probabilities"])):
        if any(not math.isclose(actual[key], expected[key], abs_tol=1e-10)
               for key in ("HOME", "DRAW", "AWAY")):
            raise ValueError("GOLDEN_REFERENCE_PROBABILITY_MISMATCH")
    if raw["execution_status"] != "SUCCESS":
        raise ValueError("GOLDEN_MODEL_EXECUTION_FAILED")
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/blind_test_r3/test_release_registry.py", "-q"],
        cwd=root, check=False, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(f"GOLDEN_BRAZIL_CLUB_FAILED:{completed.stdout[-1200:]}")
    print("GOLDEN_NATIONAL_PASS")
    print("GOLDEN_BRAZIL_CLUB_001_002_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
