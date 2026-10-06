"""Run the independent Cloud Clean Build gate without touching live ledgers."""

from __future__ import annotations

import json
import re
import sqlite3
import subprocess
import sys
import tempfile
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from erguoyuan_football.blind_test_r3.user_daily_runner import init_state, run_daily

ROOT = Path(__file__).resolve().parents[1]
TEXT_EXTENSIONS = {".py", ".json", ".yaml", ".yml", ".toml", ".md", ".txt", ".ps1"}


def _absolute_path_hits() -> list[str]:
    hits: list[str] = []
    pattern = re.compile(r"(?i)\b[a-z]:[\\/]")
    for directory in ("src", "config", "configs", "tests", "docs", "inputs",
                      "scripts", "cloud_release", "blind_test", "archive"):
        folder = ROOT / directory
        if not folder.exists():
            continue
        for path in folder.rglob("*"):
            if path.suffix.lower() not in TEXT_EXTENSIONS or "__pycache__" in path.parts:
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeError:
                continue
            if pattern.search(content):
                hits.append(str(path.relative_to(ROOT)))
    return hits


def _command(*args: str) -> bool:
    completed = subprocess.run([sys.executable, *args], cwd=ROOT, check=False,
                               capture_output=True, text=True)
    print(completed.stdout[-1000:])
    if completed.returncode:
        print(completed.stderr[-1000:], file=sys.stderr)
    return completed.returncode == 0


def _synthetic_daily() -> bool:
    with tempfile.TemporaryDirectory(prefix="yycore-cloud-gate-") as temporary:
        directory = Path(temporary)
        data = json.loads((ROOT / "inputs/daily_market/2026-10-06_luxembourg_bulgaria.json")
                          .read_text(encoding="utf-8"))
        future = datetime.now(UTC) + timedelta(days=7)
        data["slate_date"] = future.date().isoformat()
        data["fixtures"][0]["kickoff"] = future.isoformat()
        data["fixtures"][0]["screenshot_path"] = None
        data["fixtures"][0]["metadata_sources"] = {"gate": "SYNTHETIC_TEST"}
        input_path = directory / "synthetic_input.json"
        input_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        state = directory / "state"
        result = run_daily(input_path, state_root=state)
        row = result["results"][0]
        return (row["model_status"] == "AVAILABLE"
                and (state / "predictions" / f"{row['prediction_id']}.json").exists()
                and (state / "locks" / f"{row['lock_id']}.json").exists()
                and (state / "locks" / f"{result['final_output_lock_id']}.json").exists())


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="yycore-cloud-init-") as temporary:
        state = Path(temporary)
        init_state(state)
        with closing(sqlite3.connect(state / "market.sqlite")) as connection:
            empty = connection.execute("SELECT COUNT(*) FROM market_snapshots").fetchone()
        init_pass = bool(empty and empty[0] == 0 and (state / "locks").exists())
    checks = {
        "NO_ABSOLUTE_WINDOWS_PATH": not _absolute_path_hits(),
        "INSTALL_PASS": _command("-m", "pip", "check"),
        "PYTEST_PASS": _command("-m", "pytest", "-q"),
        "GOLDEN_REFERENCE_PASS": _command("scripts/run_golden_reference.py"),
        "INIT_STATE_PASS": init_pass,
        "DAILY_RUN_PASS": _synthetic_daily(),
    }
    ready = all(checks.values())
    print(json.dumps({"YYCORE_CLOUD_READY": ready, "checks": checks,
                      "checked_at": datetime.now(UTC).isoformat()}, indent=2))
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
