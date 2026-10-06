"""Lock manually approved Golden OOS reviews without exposing match outcomes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from erguoyuan_football.backtesting.golden_selector import REVIEW_CSV_FIELDS

_VERSION_PATTERN = re.compile(r"^GOLDEN_OOS_V([1-9][0-9]*)$")
_FORBIDDEN_REVIEW_FIELDS = {"result", "score", "home_goals", "away_goals"}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically create a JSON file and never replace an existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"LOCKED_FILE_ALREADY_EXISTS:{path.name}")
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", suffix=".tmp",
        prefix=f".{path.name}.", dir=path.parent, delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        temporary.unlink(missing_ok=True)
        raise FileExistsError(f"LOCKED_FILE_ALREADY_EXISTS:{path.name}") from None
    finally:
        temporary.unlink(missing_ok=True)


def _load_candidates(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        if fields & _FORBIDDEN_REVIEW_FIELDS:
            raise ValueError("REVIEW_CSV_MUST_NOT_CONTAIN_RESULTS")
        if tuple(reader.fieldnames or ()) != REVIEW_CSV_FIELDS:
            raise ValueError("CANDIDATE_REVIEW_SCHEMA_INVALID")
        candidates: dict[str, dict[str, str]] = {}
        for row in reader:
            candidate_id = row["candidate_id"]
            if not candidate_id or candidate_id in candidates:
                raise ValueError("CANDIDATE_ID_MISSING_OR_DUPLICATED")
            if row["source_available"] != "AVAILABLE":
                raise ValueError("CANDIDATE_SOURCE_NOT_AVAILABLE")
            if not row["data_quality"].startswith("PASS:"):
                raise ValueError("CANDIDATE_DATA_QUALITY_NOT_PASSED")
            candidates[candidate_id] = row
        return candidates


def _load_approvals(path: Path, dataset_version: str,
                    candidate_ids: set[str]) -> list[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"SELECTION_JSON_UNREADABLE:{error}") from error
    if not isinstance(payload, dict) or payload.get("dataset_version") != dataset_version:
        raise ValueError("SELECTION_FILE_VERSION_MISMATCH")
    entries = payload.get("selected_matches")
    if not isinstance(entries, list):
        raise TypeError("SELECTION_FILE_SCHEMA_INVALID")
    seen: set[str] = set()
    approved: list[str] = []
    for entry in entries:
        if (not isinstance(entry, dict) or not isinstance(entry.get("candidate_id"), str)
                or type(entry.get("approved")) is not bool):
            raise ValueError("SELECTION_ENTRY_SCHEMA_INVALID")
        candidate_id = entry["candidate_id"]
        if candidate_id in seen:
            raise ValueError("DUPLICATE_SELECTION_ENTRY")
        if candidate_id not in candidate_ids:
            raise ValueError("SELECTION_REFERENCES_UNKNOWN_CANDIDATE")
        seen.add(candidate_id)
        if entry["approved"]:
            approved.append(candidate_id)
    if len(approved) != 100:
        raise ValueError(f"GOLDEN_SELECTION_REQUIRES_100_APPROVALS:{len(approved)}")
    return approved


def lock_golden_selection(
    *,
    candidate_csv: str | Path,
    selection_json: str | Path,
    output_dir: str | Path,
    dataset_version: str,
    operator: str,
    selected_time: datetime | None = None,
) -> tuple[Path, Path]:
    """Create the immutable versioned manifest and append its selection audit."""
    if not _VERSION_PATTERN.fullmatch(dataset_version):
        raise ValueError("GOLDEN_DATASET_VERSION_INVALID")
    if not operator.strip():
        raise ValueError("OPERATOR_REQUIRED")
    output_root = Path(output_dir)
    locked_path = output_root / f"{dataset_version}_LOCKED.json"
    audit_path = output_root / "golden_selection_audit.json"
    if locked_path.exists():
        raise FileExistsError("GOLDEN_VERSION_ALREADY_LOCKED_CREATE_NEW_VERSION")

    candidates = _load_candidates(Path(candidate_csv))
    approved_ids = _load_approvals(
        Path(selection_json), dataset_version, set(candidates))
    selected = [candidates[candidate_id] for candidate_id in approved_ids]
    timestamp = selected_time or datetime.now(UTC)
    if timestamp.tzinfo is None:
        raise ValueError("SELECTED_TIME_MUST_BE_TIMEZONE_AWARE")
    timestamp = timestamp.astimezone(UTC)
    locked_payload = {
        "dataset_version": dataset_version,
        "status": "LOCKED",
        "locked_at": timestamp.isoformat(),
        "selected_count": len(selected),
        "matches": [
            {key: row[key] for key in REVIEW_CSV_FIELDS if key != "review_status"}
            for row in selected
        ],
    }
    _atomic_json(locked_path, locked_payload)
    os.chmod(locked_path, stat.S_IREAD)

    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if not isinstance(audit, dict) or not isinstance(audit.get("records"), list):
            raise ValueError("SELECTION_AUDIT_SCHEMA_INVALID")
    else:
        audit = {"records": []}
    audit["records"].append({
        "selected_time": timestamp.isoformat(),
        "dataset_version": dataset_version,
        "selected_count": len(selected),
        "operator": operator.strip(),
        "locked_file": locked_path.name,
        "locked_sha256": hashlib.sha256(locked_path.read_bytes()).hexdigest(),
    })
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
    return locked_path, audit_path


def main() -> int:
    """Lock the user's CSV/JSON selection from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-csv", default="data/backtest/golden_candidates_review.csv")
    parser.add_argument("--selection", default="data/backtest/golden_selection.json")
    parser.add_argument("--output-dir", default="data/backtest")
    parser.add_argument("--dataset-version", default="GOLDEN_OOS_V1")
    parser.add_argument("--operator", required=True)
    args = parser.parse_args()
    try:
        locked, audit = lock_golden_selection(
            candidate_csv=args.candidate_csv,
            selection_json=args.selection,
            output_dir=args.output_dir,
            dataset_version=args.dataset_version,
            operator=args.operator,
        )
    except (FileExistsError, TypeError, ValueError) as error:
        print(json.dumps({"status": "NOT_LOCKED", "reason": str(error)},
                         ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps({"status": "LOCKED", "locked_file": str(locked),
                      "audit_file": str(audit)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
