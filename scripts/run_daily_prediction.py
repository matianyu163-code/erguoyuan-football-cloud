"""Run a validated user daily slate through frozen model, JC, fusion and locks."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from erguoyuan_football.blind_test_r3.user_daily_runner import run_daily

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(run_daily(arguments.input), ensure_ascii=False, indent=2))
