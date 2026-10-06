"""Exact official fixture verification with reversal and ambiguity protection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from erguoyuan_football.web_research.time_utils import parse_utc


@dataclass(frozen=True)
class FixtureExpectation:
    """Canonical teams and optional competition/date supplied before research."""

    home_team_id: str
    away_team_id: str
    competition_id: str | None = None
    match_date: date | None = None


@dataclass(frozen=True)
class FixtureVerification:
    """Verified candidate or explicit ambiguity/reversal evidence."""

    status: str
    match_id: str | None
    candidates: tuple[dict[str, Any], ...]


def verify_fixture(
    payload: dict[str, Any],
    expected: FixtureExpectation,
) -> FixtureVerification:
    """Accept exactly one oriented, competition/date-compatible candidate."""
    raw = payload.get("fixtures")
    candidates = raw if isinstance(raw, list) else [payload]
    if not all(isinstance(item, dict) for item in candidates):
        return FixtureVerification("INVALID_PROVIDER_RESPONSE", None, ())
    exact: list[dict[str, Any]] = []
    reversed_seen = False
    for item in candidates:
        kickoff_raw = item.get("kickoff_time")
        if not isinstance(kickoff_raw, str):
            continue
        try:
            kickoff = parse_utc(kickoff_raw)
        except ValueError:
            continue
        if expected.match_date is not None and kickoff.date() != expected.match_date:
            continue
        if (expected.competition_id is not None
                and item.get("competition_id") != expected.competition_id):
            continue
        if (item.get("home_team_id") == expected.away_team_id
                and item.get("away_team_id") == expected.home_team_id):
            reversed_seen = True
        if (item.get("home_team_id") == expected.home_team_id
                and item.get("away_team_id") == expected.away_team_id
                and isinstance(item.get("competition_id"), str)
                and isinstance(item.get("match_id"), str)
                and item["match_id"].strip()):
            exact.append(item)
    if len(exact) > 1:
        return FixtureVerification("FIXTURE_AMBIGUOUS", None, tuple(exact))
    if len(exact) == 1:
        return FixtureVerification("VERIFIED", str(exact[0]["match_id"]), tuple(exact))
    return FixtureVerification("REVERSE_FIXTURE" if reversed_seen else "FIXTURE_NOT_FOUND",
                               None, tuple(candidates))
