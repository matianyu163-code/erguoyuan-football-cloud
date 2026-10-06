"""Phase 10 isolated synthetic contexts; never used by production loaders."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from erguoyuan_football.context.schemas import HistoricalMatchEvent


@pytest.fixture
def phase10_events() -> tuple[HistoricalMatchEvent, ...]:
    """Return explicitly synthetic, date-ordered league events for unit tests."""
    values = (
        ("m1", "A", "B", date(2025, 1, 1), 2, 0),
        ("m2", "C", "A", date(2025, 1, 8), 1, 1),
        ("m3", "B", "C", date(2025, 1, 15), 0, 1),
        ("same-day", "A", "C", date(2025, 2, 1), 8, 0),
        ("future", "B", "A", date(2025, 2, 2), 5, 0),
    )
    return tuple(HistoricalMatchEvent(
        match_id=match_id, competition_id="TEST_LEAGUE", season_id="2024-25",
        home_team_id=home, away_team_id=away, match_date=match_date,
        home_goals=home_goals, away_goals=away_goals, source="SYNTHETIC_TEST",
        retrieved_at=datetime(2025, 3, 1, tzinfo=UTC),
        as_of_time=datetime(2025, 3, 1, tzinfo=UTC))
        for match_id, home, away, match_date, home_goals, away_goals in values)
