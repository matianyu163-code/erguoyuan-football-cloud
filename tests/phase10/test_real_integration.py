"""Explicit integration checks against the repository's verified development data."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from erguoyuan_football.context.cli import run

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "football.duckdb"
PREVIEW = ROOT / "reports" / "phase9_development_core_preview.jsonl"


@pytest.mark.integration
def test_real_phase10_context_one_match_and_twenty_match_batch():
    """Use real CORE V2 development OOS and canonical completed result rows."""
    if not DB.is_file() or not PREVIEW.is_file():
        pytest.skip("verified local Phase 9 development artifacts are not present")
    with PREVIEW.open(encoding="utf-8") as handle:
        prediction_ids = [json.loads(line)["match_id"] for line in handle if line.strip()]
    one = run(["--db", str(DB), "--from-date", "2026-05-01", "--to-date", "2026-05-31",
               "--match-id", prediction_ids[0], "--dry-run"])
    twenty_args = ["--db", str(DB), "--from-date", "2026-05-01", "--to-date", "2026-05-31"]
    for match_id in prediction_ids[:20]:
        twenty_args.extend(("--match-id", match_id))
    twenty_args.append("--dry-run")
    twenty = run(twenty_args)
    assert one["target_count"] == 1
    assert twenty["target_count"] == 20
    assert one["audit_fail_count"] == twenty["audit_fail_count"] == 0
    assert one["phase9_final_holdout_rows_read"] == twenty["phase9_final_holdout_rows_read"] == 0
    assert one["production_promoted"] is twenty["production_promoted"] is False
    assert all(item["base_probability"] == item["final_probability"]
               for item in twenty["canonical_outputs"])


@pytest.mark.integration
def test_phase10_final_holdout_request_is_rejected():
    if not DB.is_file() or not PREVIEW.is_file():
        pytest.skip("verified local Phase 9 development artifacts are not present")
    with pytest.raises(ValueError, match="REQUEST_OUTSIDE_PHASE10_DEVELOPMENT_WINDOW"):
        run(["--db", str(DB), "--from-date", "2026-08-01", "--to-date", "2026-08-01", "--dry-run"])
