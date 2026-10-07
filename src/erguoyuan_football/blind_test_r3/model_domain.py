"""Deterministic fixture-domain classification for frozen-model routing."""

from __future__ import annotations

from collections.abc import Callable

from erguoyuan_football.blind_test_r3.capability import CLUB, NATIONAL_TEAM, UNKNOWN
from erguoyuan_football.blind_test_r3.user_daily import UserFixture

CLUB_COMPETITION_MARKERS = (
    "serie", "brasileir", "veikkaus", "premier league", "bundesliga",
    "la liga", "ligue 1", "eredivisie", "championship", "league cup",
)
NATIONAL_COMPETITION_MARKERS = (
    "nations league", "world cup", "european championship", "euro 202",
    "copa america", "african cup of nations", "asian cup",
    "international friendly", "national team",
)


def classify_fixture(item: UserFixture,
                     resolve_national_team: Callable[[str], str]) -> str:
    """Classify by canonical national-team identity, then competition label."""
    try:
        resolve_national_team(item.home_team)
        resolve_national_team(item.away_team)
        return NATIONAL_TEAM
    except (ValueError, KeyError):
        pass

    competition = item.competition.casefold().strip()
    if any(marker in competition for marker in NATIONAL_COMPETITION_MARKERS):
        return NATIONAL_TEAM
    if any(marker in competition for marker in CLUB_COMPETITION_MARKERS):
        return CLUB
    return UNKNOWN
