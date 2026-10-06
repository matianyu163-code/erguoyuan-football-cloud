"""Parse only Playwright-verified visible JC rows from the rendered public page."""

from __future__ import annotations

import math
import re
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from erguoyuan_football.blind_test_r3.jc_public_parser import (
    JCParsedMatch,
    JCParseResult,
    MATCH_NUMBER,
)
from erguoyuan_football.blind_test_r3.jc_rendered_public import RenderedPage, page_status

TZ = ZoneInfo("Asia/Shanghai")
DISPLAY_NUMBER = re.compile(r"^\s*(周[一二三四五六日天])\s*(\d{3})\s*$")
DISPLAY_KICKOFF = re.compile(r"^\s*(\d{2})-(\d{2})\s+(\d{2}):(\d{2})\s*$")
HANDICAP = re.compile(r"^[+-]?\d+$")


def _one_cell(cells: list[str], pattern: re.Pattern[str], field: str) -> re.Match[str]:
    found = [match for cell in cells if (match := pattern.fullmatch(cell.replace("\n", " ")))]
    if len(found) != 1:
        raise ValueError(f"JC_RENDERED_{field}_AMBIGUOUS_OR_MISSING")
    return found[0]


def _kickoff(match_date: str, display: re.Match[str]) -> datetime:
    if not re.fullmatch(r"\d{6}", match_date):
        raise ValueError("JC_RENDERED_MATCH_DATE_INVALID")
    year = 2000 + int(match_date[:2])
    day = datetime(year, int(match_date[2:4]), int(match_date[4:6]), tzinfo=TZ)
    month, date, hour, minute = map(int, display.groups())
    choices = []
    for candidate_year in (year - 1, year, year + 1):
        try:
            value = datetime(candidate_year, month, date, hour, minute, tzinfo=TZ)
        except ValueError:
            continue
        if -timedelta(hours=6) <= value - day <= timedelta(hours=72):
            choices.append(value)
    if len(choices) != 1:
        raise ValueError("JC_RENDERED_KICKOFF_AMBIGUOUS")
    return choices[0]


def _bonuses(values: Any, market: str) -> dict[str, float] | None:
    if not isinstance(values, list):
        raise ValueError(f"JC_RENDERED_{market}_DOM_INVALID")
    if not values:
        return None
    if len(values) != 3:
        raise ValueError(f"JC_RENDERED_{market}_INCOMPLETE")
    result = {}
    for side, raw in zip(("HOME", "DRAW", "AWAY"), values, strict=True):
        if not re.fullmatch(r"\d+(?:\.\d+)?", str(raw).strip()):
            raise ValueError(f"JC_RENDERED_{market}_PRICE_INVALID")
        price = float(raw)
        if not math.isfinite(price) or not 1 < price < 1000:
            raise ValueError(f"JC_RENDERED_{market}_PRICE_INVALID")
        result[side] = price
    return result


def parse_rendered_page(page: RenderedPage) -> JCParseResult:
    """A row is eligible only when browser locators confirmed its fields visible."""
    status = page_status(page)
    if status in {"ACCESS_CHALLENGE", "ACCESS_RESTRICTED", "UNAVAILABLE"}:
        return JCParseResult(status, ())
    if not page.visible_rows:
        return JCParseResult("NO_ACTIVE_MATCHES" if status == "NO_ACTIVE_MATCHES" else
                             "PAGE_STRUCTURE_CHANGED", ())
    matches: list[JCParsedMatch] = []
    errors: list[str] = []
    seen: set[str] = set()
    for row in page.visible_rows:
        try:
            cells = row["visible_cells"]
            if not isinstance(cells, list):
                raise ValueError("JC_RENDERED_CELLS_INVALID")
            number_match = _one_cell(cells, DISPLAY_NUMBER, "NUMBER")
            number = number_match.group(1) + number_match.group(2)
            if not MATCH_NUMBER.fullmatch(number) or number in seen:
                raise ValueError("JC_RENDERED_DUPLICATE_MATCH_NUMBER")
            seen.add(number)
            kickoff_display = _one_cell(cells, DISPLAY_KICKOFF, "KICKOFF")
            kickoff = _kickoff(str(row["matchnumdate"]), kickoff_display)
            competition = str(row.get("competition") or "").strip()
            home = str(row.get("home_team") or "").strip()
            away = str(row.get("away_team") or "").strip()
            if not competition or not home or not away or home == away:
                raise ValueError("JC_RENDERED_IDENTITY_INCOMPLETE")
            spf = _bonuses(row.get("spf"), "SPF")
            rqspf = _bonuses(row.get("rqspf"), "RQSPF")
            handicap_text = str(row.get("handicap") or "").strip()
            handicap = int(handicap_text) if HANDICAP.fullmatch(handicap_text) else None
            if rqspf is not None and (handicap is None or handicap == 0):
                raise ValueError("JC_RENDERED_RQSPF_HANDICAP_INVALID")
            matches.append(JCParsedMatch(number, competition, home, away, kickoff,
                "ON_SALE" if spf is not None else "UNKNOWN",
                "ON_SALE" if rqspf is not None else "UNKNOWN",
                spf, rqspf, handicap, None))
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"{type(error).__name__}:{error}")
    if not matches:
        return JCParseResult("PAGE_STRUCTURE_CHANGED", (), ";".join(errors[:3]))
    return JCParseResult("PARTIAL" if errors else "AVAILABLE", tuple(matches),
                         ";".join(errors[:3]) if errors else None)
