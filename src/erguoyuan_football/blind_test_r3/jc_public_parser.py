"""Fail-closed parsing of visible, semantically labeled JC public DOM rows."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from bs4 import BeautifulSoup

PARSER_VERSION = "JC_PUBLIC_PARSER_V1"
MATCH_NUMBER = re.compile(r"^周[一二三四五六日天][0-9]{3}$")
SELECTIONS = ("HOME", "DRAW", "AWAY")


@dataclass(frozen=True)
class JCParsedMatch:
    """Fields actually present in one public rendered DOM row."""

    jc_match_number: str
    competition: str
    home_team: str
    away_team: str
    kickoff_utc: datetime
    spf_sale_status: str
    rqspf_sale_status: str
    spf_bonus: dict[str, float] | None = None
    rqspf_bonus: dict[str, float] | None = None
    handicap: int | None = None
    bonus_updated_at: datetime | None = None


@dataclass(frozen=True)
class JCParseResult:
    status: str
    matches: tuple[JCParsedMatch, ...]
    reason: str | None = None


def _timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.strip())
    if parsed.tzinfo is None:
        raise ValueError("JC_PAGE_TIMEZONE_REQUIRED")
    return parsed.astimezone(UTC)


def _bonus(row: Any, market: str) -> dict[str, float] | None:
    values: dict[str, float] = {}
    for item in row.select(f'[data-market="{market}"][data-selection]'):
        selection = str(item.get("data-selection", "")).upper()
        if selection not in SELECTIONS or selection in values:
            raise ValueError("JC_BONUS_SELECTION_INVALID")
        raw = item.get("data-bonus")
        if raw is None:
            raise ValueError("JC_BONUS_MISSING")
        price = float(raw)
        if not 1 < price < 1000:
            raise ValueError("JC_BONUS_INVALID")
        values[selection] = price
    if not values:
        return None
    if set(values) != set(SELECTIONS):
        raise ValueError("JC_BONUS_INCOMPLETE")
    return values


def _parse_rows(html: str, *, require_bonus: bool) -> JCParseResult:
    soup = BeautifulSoup(html, "html.parser")
    table_id = "mainTbl" if require_bonus else "scheduleTbl"
    table = soup.find("table", id=table_id)
    if table is None and not require_bonus:
        # The current official schedule page has a header table without this id.
        table = soup.select_one("table[data-jc-schedule]")
    if table is None:
        return JCParseResult("PAGE_STRUCTURE_CHANGED", (), f"{table_id}_MISSING")
    rows = table.select("tr[data-jc-match-number]")
    if not rows:
        if table.get("data-no-active-matches") == "true":
            return JCParseResult("NO_ACTIVE_MATCHES", ())
        return JCParseResult("PUBLIC_PAGE_AUTOMATION_UNAVAILABLE", (),
                             "NO_SEMANTIC_MATCH_ROWS_IN_PUBLIC_HTML")
    updated_node = soup.select_one("[data-bonus-updated-at]")
    updated_value = (str(updated_node.get("data-bonus-updated-at"))
                     if updated_node is not None else None)
    try:
        updated = _timestamp(updated_value)
        parsed: list[JCParsedMatch] = []
        seen: set[str] = set()
        for row in rows:
            number = str(row.get("data-jc-match-number", ""))
            if not MATCH_NUMBER.fullmatch(number) or number in seen:
                raise ValueError("JC_MATCH_NUMBER_INVALID_OR_DUPLICATE")
            seen.add(number)
            competition = str(row.get("data-competition", "")).strip()
            home = str(row.get("data-home-team", "")).strip()
            away = str(row.get("data-away-team", "")).strip()
            kickoff = _timestamp(str(row.get("data-kickoff-utc", "")))
            if not competition or not home or not away or home == away or kickoff is None:
                raise ValueError("JC_FIXTURE_FIELDS_INCOMPLETE")
            spf_status = str(row.get("data-spf-sale-status", "UNKNOWN")).upper()
            rq_status = str(row.get("data-rqspf-sale-status", "UNKNOWN")).upper()
            if spf_status not in {"ON_SALE", "NOT_ON_SALE", "UNKNOWN"} or (
                rq_status not in {"ON_SALE", "NOT_ON_SALE", "UNKNOWN"}
            ):
                raise ValueError("JC_SALE_STATUS_INVALID")
            spf = _bonus(row, "JC_SPF") if require_bonus and spf_status == "ON_SALE" else None
            rq = _bonus(row, "JC_RQSPF") if require_bonus and rq_status == "ON_SALE" else None
            if require_bonus and ((spf_status == "ON_SALE" and spf is None) or
                                  (rq_status == "ON_SALE" and rq is None)):
                raise ValueError("JC_ON_SALE_BONUS_MISSING")
            handicap_raw = row.get("data-handicap")
            handicap = int(str(handicap_raw)) if handicap_raw is not None else None
            if rq is not None and handicap is None:
                raise ValueError("JC_RQSPF_HANDICAP_REQUIRED")
            parsed.append(JCParsedMatch(number, competition, home, away, kickoff,
                spf_status, rq_status, spf, rq, handicap, updated))
    except (TypeError, ValueError) as error:
        return JCParseResult("PARSE_FAILED", (), f"{type(error).__name__}:{error}")
    return JCParseResult("AVAILABLE", tuple(parsed))


def parse_schedule(html: str) -> JCParseResult:
    """Read a rendered schedule row only when its fields are explicitly labeled."""
    return _parse_rows(html, require_bonus=False)


def parse_spf_rqspf(html: str) -> JCParseResult:
    """Read fixed bonuses by market and selection labels, never column order."""
    return _parse_rows(html, require_bonus=True)
