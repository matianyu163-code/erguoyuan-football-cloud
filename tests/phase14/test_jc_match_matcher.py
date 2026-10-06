"""JC match identity tests; provider rows are explicitly SYNTHETIC_TEST."""

from datetime import UTC, datetime, timedelta

from erguoyuan_football.jc_verification.jc_match_matcher import JCMatchMatcher
from erguoyuan_football.jc_verification.jc_schema import JCMatch, JCMatchQuery
from erguoyuan_football.knowledge.competitions.competition_database import (
    CompetitionDatabase,
)
from erguoyuan_football.knowledge.competitions.competition_resolver import (
    CompetitionResolver,
)
from erguoyuan_football.knowledge.competitions.competition_schema import (
    CompetitionIdentity,
)
from erguoyuan_football.knowledge.teams.team_database import TeamDatabase
from erguoyuan_football.knowledge.teams.team_resolver import TeamResolver


def _teams() -> TeamResolver:
    database = TeamDatabase()
    return TeamResolver(database)


def _competitions() -> CompetitionResolver:
    identity = CompetitionIdentity("TEST_PREMIER", "Premier League", "FA", "GB", "CLUB")
    return CompetitionResolver(CompetitionDatabase(
        (identity,), {"TEST_PREMIER": ("Premier League", "英超")}))


def test_match_requires_team_competition_date_and_kickoff_identity() -> None:
    teams = _teams()
    home = teams.resolve("Arsenal")
    away = teams.resolve("Liverpool")
    assert home and away
    query = JCMatchQuery(home.team_id, "阿森纳", away.team_id, "利物浦",
        None, "Premier League", datetime(2026, 10, 4, 15, tzinfo=UTC))
    row = JCMatch("jc-1", "Arsenal", "Liverpool", "Premier League",
        datetime(2026, 10, 4, 15, 45, tzinfo=UTC), ("1X2",),
        "synthetic-test", "synthetic-evidence", competition_id="TEST_PREMIER")
    assert JCMatchMatcher(teams, _competitions()).match(query, (row,)) is None
    exact_query = JCMatchQuery(home.team_id, "Arsenal", away.team_id, "Liverpool",
        "TEST_PREMIER", "Premier League", row.kickoff_time)
    match = JCMatchMatcher(teams, _competitions()).match(exact_query, (row,))
    assert match is not None
    assert match.exact_competition and match.exact_kickoff


def test_ambiguous_or_string_only_team_names_are_rejected() -> None:
    teams = _teams()
    home = teams.resolve("Arsenal")
    away = teams.resolve("Liverpool")
    assert home and away
    query = JCMatchQuery(home.team_id, "Arsenal", away.team_id, "Liverpool",
        "UEFA_CL", "UEFA Champions League", datetime(2026, 10, 4, 15, tzinfo=UTC))
    unknown_row = JCMatch("jc-2", "Arsenal Women", "Liverpool Women", "Premier League",
        query.kickoff_time, (), "synthetic-test", "synthetic-evidence")
    assert JCMatchMatcher(teams).match(query, (unknown_row,)) is None


def test_duplicate_candidates_are_not_arbitrarily_selected() -> None:
    teams = _teams()
    home = teams.resolve("Arsenal")
    away = teams.resolve("Liverpool")
    assert home and away
    query = JCMatchQuery(home.team_id, "Arsenal", away.team_id, "Liverpool",
        "UEFA_CL", "UEFA Champions League", datetime(2026, 10, 4, 15, tzinfo=UTC))
    rows = tuple(JCMatch(f"jc-{i}", "Arsenal", "Liverpool", "Premier League",
        query.kickoff_time + timedelta(minutes=i), (), "synthetic-test", f"e-{i}",
        competition_id="UEFA_CL")
        for i in (0, 1))
    assert JCMatchMatcher(teams).match(query, rows) is None
