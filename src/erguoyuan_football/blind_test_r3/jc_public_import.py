"""Explicitly user-confirmed JC fixed bonus input; never official-page evidence."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from erguoyuan_football.blind_test_r3.jc_public_parser import (
    MATCH_NUMBER,
    JCParsedMatch,
)
from erguoyuan_football.blind_test_r3.store import sha256


def _time(value: Any) -> datetime:
    result = datetime.fromisoformat(str(value))
    if result.tzinfo is None:
        raise ValueError("JC_USER_CONFIRMED_TIMEZONE_REQUIRED")
    return result.astimezone(UTC)


def _bonuses(value: Any) -> dict[str, float] | None:
    if value is None or value == "":
        return None
    if not isinstance(value, dict) or set(value) != {"home", "draw", "away"}:
        raise ValueError("JC_USER_CONFIRMED_BONUS_INCOMPLETE")
    return {key.upper(): float(value[key]) for key in ("home", "draw", "away")}


def load_user_confirmed(path: Path) -> tuple[JCParsedMatch, dict[str, Any]]:
    """Accept JSON/CSV only with explicit confirmation and source timestamp."""
    content = path.read_bytes()
    if path.suffix.casefold() == ".json":
        record = json.loads(content)
    elif path.suffix.casefold() == ".csv":
        rows = list(csv.DictReader(content.decode("utf-8-sig").splitlines()))
        if len(rows) != 1:
            raise ValueError("JC_USER_CONFIRMED_CSV_ONE_ROW_REQUIRED")
        record = dict(rows[0])
        for market in ("spf", "rqspf"):
            if record.get(f"{market}_home"):
                record[market] = {side: record[f"{market}_{side}"]
                                  for side in ("home", "draw", "away")}
                if market == "rqspf":
                    market_row = record[market]
                    assert isinstance(market_row, dict)
                    market_row["handicap"] = record["handicap"]
    else:
        raise ValueError("JC_USER_CONFIRMED_FORMAT_UNSUPPORTED")
    if not isinstance(record, dict) or record.get("source") != "USER_CONFIRMED_MARKET" or (
        record.get("confirmation_status") != "CONFIRMED" or
        not record.get("confirmed_by") or not record.get("evidence_description")
    ):
        raise ValueError("JC_USER_CONFIRMED_EVIDENCE_REQUIRED")
    number = str(record.get("jc_match_number", ""))
    if not MATCH_NUMBER.fullmatch(number):
        raise ValueError("JC_USER_CONFIRMED_MATCH_NUMBER_INVALID")
    captured = _time(record["captured_at"])
    kickoff = _time(record["kickoff_time"])
    confirmed = _time(record["confirmed_at"])
    if captured > confirmed or confirmed >= kickoff or datetime.now(UTC) < confirmed:
        raise ValueError("JC_USER_CONFIRMED_NOT_PREMATCH")
    spf = _bonuses(record.get("spf"))
    rq_raw = record.get("rqspf")
    rq = _bonuses({key: rq_raw[key] for key in ("home", "draw", "away")}
                   if isinstance(rq_raw, dict) else None)
    handicap = int(rq_raw["handicap"]) if rq is not None and isinstance(rq_raw, dict) else None
    if spf is None and rq is None:
        raise ValueError("JC_USER_CONFIRMED_NO_MARKET")
    parsed = JCParsedMatch(number, str(record["competition"]),
        str(record["home_team"]), str(record["away_team"]), kickoff,
        "ON_SALE" if spf is not None else "UNKNOWN",
        "ON_SALE" if rq is not None else "UNKNOWN", spf, rq, handicap, captured)
    evidence = {"source_type": "USER_CONFIRMED_MARKET", "source_quality": "USER_CONFIRMED",
        "captured_at": captured.isoformat(), "confirmed_at": confirmed.isoformat(),
        "confirmed_by": record["confirmed_by"],
        "evidence_description": record["evidence_description"],
        "input_sha256": sha256(content), "input_path": str(path.resolve())}
    return parsed, evidence
