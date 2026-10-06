"""SYNTHETIC_TEST deterministic query planning without live web calls."""

import pytest

from erguoyuan_football.web_research.search.query_builder import QueryBuilder


def test_query_builder_required_topics_and_stable_ids() -> None:
    """Fixture, injury, odds and xG plans are present and deterministic."""
    first = QueryBuilder.build("Arsenal", "Liverpool", "Premier League")
    assert first == QueryBuilder.build("Arsenal", "Liverpool", "Premier League")
    assert {task.task_type for task in first} == {
        "FIXTURE", "INJURY", "LINEUP", "ODDS", "TEAM_STATS", "NEWS"
    }
    assert any("xG" in task.query for task in first)
    assert any("odds" in task.query for task in first)
    assert all("Arsenal" in task.query and "Liverpool" in task.query for task in first)
    assert len({task.task_id for task in first}) == len(first)


def test_query_builder_rejects_missing_or_same_team() -> None:
    """No inferred second team is inserted into a query plan."""
    with pytest.raises(ValueError, match="TWO_DISTINCT_TEAMS_REQUIRED"):
        QueryBuilder.build("Arsenal", "")
    with pytest.raises(ValueError, match="TWO_DISTINCT_TEAMS_REQUIRED"):
        QueryBuilder.build("Arsenal", "arsenal")
    with pytest.raises(ValueError, match="NONEMPTY_FIXTURE_KEY_REQUIRED"):
        QueryBuilder.build("Arsenal", "Liverpool", fixture_key=" ")


def test_repeated_pairings_have_distinct_fixture_task_ids() -> None:
    """SYNTHETIC_TEST: evidence keys do not collide across known fixtures."""
    first = QueryBuilder.build("Arsenal", "Liverpool", fixture_key="SYNTHETIC_TEST_1")
    second = QueryBuilder.build("Arsenal", "Liverpool", fixture_key="SYNTHETIC_TEST_2")
    assert [task.query for task in first] == [task.query for task in second]
    assert {task.task_id for task in first}.isdisjoint(task.task_id for task in second)
