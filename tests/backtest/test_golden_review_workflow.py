"""Golden review workflow tests using fixtures marked SYNTHETIC_TEST."""

import csv
import json
import os
import stat
from datetime import UTC, datetime

import pytest

from erguoyuan_football.backtesting.golden_lock import lock_golden_selection
from erguoyuan_football.backtesting.golden_selector import (
    REVIEW_CSV_FIELDS,
    GoldenCandidate,
    GoldenSelection,
    candidate_id_for,
    write_review_csv,
    write_selection_template,
)


def _candidate(index: int) -> GoldenCandidate:
    return GoldenCandidate(
        match_id=f"SYNTHETIC_TEST_MATCH_{index:03}", competition="英超",
        competition_id="EPL", home_team=f"SYNTHETIC_TEST_HOME_{index:03}",
        away_team=f"SYNTHETIC_TEST_AWAY_{index:03}",
        kickoff_time=datetime(2026, 8, 1, tzinfo=UTC), result=(index % 4, (index + 1) % 3),
        source_type="JC_VERIFIED", evidence_id=f"SYNTHETIC_TEST_EVIDENCE_{index:03}",
        source_id="SYNTHETIC_TEST",
    )


def _selection(count: int = 100) -> GoldenSelection:
    candidates = tuple(_candidate(index) for index in range(count))
    return GoldenSelection(
        created_at=datetime(2026, 10, 2, tzinfo=UTC), status="READY",
        target_count=count, candidates=candidates, rejected=(), quota_counts={"英超": count},
    )


def _candidate_file(path, *, version: str = "GOLDEN_OOS_V1", count: int = 100) -> list[str]:
    items = [_candidate(index) for index in range(count)]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_CSV_FIELDS)
        writer.writeheader()
        for item in items:
            writer.writerow({
                "candidate_id": candidate_id_for(item, version),
                "match_id": item.match_id,
                "competition": item.competition,
                "home_team": item.home_team,
                "away_team": item.away_team,
                "kickoff_time": item.kickoff_time.isoformat(),
                "data_quality": "PASS: SYNTHETIC_TEST",
                "source_available": "AVAILABLE",
                "review_status": "PENDING",
            })
    return [candidate_id_for(item, version) for item in items]


def _approve(selection_path, candidate_ids: list[str], version: str) -> None:
    selection_path.write_text(json.dumps({
        "dataset_version": version,
        "selected_matches": [
            {"candidate_id": candidate_id, "approved": True}
            for candidate_id in candidate_ids
        ],
    }), encoding="utf-8")


def test_candidate_generation_has_fixed_columns_and_no_result(tmp_path) -> None:
    path = write_review_csv(_selection(2), tmp_path / "golden_candidates_review.csv")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)

    assert tuple(reader.fieldnames or ()) == REVIEW_CSV_FIELDS
    assert len(rows) == 2
    assert not {"result", "score", "home_goals", "away_goals"} & set(reader.fieldnames or ())
    assert all(row["review_status"] == "PENDING" for row in rows)


def test_selection_json_template_is_readable_and_preserves_manual_edits(tmp_path) -> None:
    candidates = tmp_path / "candidates.csv"
    selection = tmp_path / "golden_selection.json"
    ids = _candidate_file(candidates, count=2)
    write_selection_template(candidates, selection)
    payload = json.loads(selection.read_text(encoding="utf-8"))
    payload["selected_matches"][0]["approved"] = True
    selection.write_text(json.dumps(payload), encoding="utf-8")

    write_selection_template(candidates, selection)

    reread = json.loads(selection.read_text(encoding="utf-8"))
    assert reread["dataset_version"] == "GOLDEN_OOS_V1"
    assert reread["selected_matches"][0] == {"candidate_id": ids[0], "approved": True}


def test_exactly_100_approvals_create_locked_dataset_and_audit(tmp_path) -> None:
    candidates = tmp_path / "candidates.csv"
    approvals = tmp_path / "selection.json"
    ids = _candidate_file(candidates, count=100)
    _approve(approvals, ids, "GOLDEN_OOS_V1")

    locked, audit = lock_golden_selection(
        candidate_csv=candidates, selection_json=approvals, output_dir=tmp_path,
        dataset_version="GOLDEN_OOS_V1", operator="reviewer",
        selected_time=datetime(2026, 10, 2, tzinfo=UTC),
    )

    manifest = json.loads(locked.read_text(encoding="utf-8"))
    audit_payload = json.loads(audit.read_text(encoding="utf-8"))
    assert manifest["status"] == "LOCKED"
    assert manifest["selected_count"] == 100
    assert "result" not in manifest["matches"][0]
    assert audit_payload["records"][0]["operator"] == "reviewer"
    assert audit_payload["records"][0]["selected_count"] == 100
    assert not (locked.stat().st_mode & stat.S_IWRITE)
    os.chmod(locked, stat.S_IREAD | stat.S_IWRITE)


def test_locked_version_cannot_be_rewritten_but_v2_is_allowed(tmp_path) -> None:
    candidates_v1 = tmp_path / "v1.csv"
    selection_v1 = tmp_path / "v1.json"
    ids_v1 = _candidate_file(candidates_v1, version="GOLDEN_OOS_V1", count=100)
    _approve(selection_v1, ids_v1, "GOLDEN_OOS_V1")
    lock_golden_selection(
        candidate_csv=candidates_v1, selection_json=selection_v1, output_dir=tmp_path,
        dataset_version="GOLDEN_OOS_V1", operator="reviewer",
    )
    with pytest.raises(FileExistsError, match="CREATE_NEW_VERSION"):
        lock_golden_selection(
            candidate_csv=candidates_v1, selection_json=selection_v1, output_dir=tmp_path,
            dataset_version="GOLDEN_OOS_V1", operator="reviewer",
        )

    candidates_v2 = tmp_path / "v2.csv"
    selection_v2 = tmp_path / "v2.json"
    ids_v2 = _candidate_file(candidates_v2, version="GOLDEN_OOS_V2", count=100)
    _approve(selection_v2, ids_v2, "GOLDEN_OOS_V2")
    locked_v2, _ = lock_golden_selection(
        candidate_csv=candidates_v2, selection_json=selection_v2, output_dir=tmp_path,
        dataset_version="GOLDEN_OOS_V2", operator="reviewer",
    )
    assert locked_v2.exists()
    os.chmod(tmp_path / "GOLDEN_OOS_V1_LOCKED.json", stat.S_IREAD | stat.S_IWRITE)
    os.chmod(locked_v2, stat.S_IREAD | stat.S_IWRITE)
