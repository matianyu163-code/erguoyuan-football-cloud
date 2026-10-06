"""Execution-readiness tests: a partial Golden set cannot enter model replay."""

from datetime import UTC, datetime

import duckdb

from erguoyuan_football.backtesting.golden_selector import GoldenCandidateSelector


def test_no_predictions_run_when_golden_selection_is_partial(tmp_path) -> None:
    database = tmp_path / "empty.duckdb"
    with duckdb.connect(str(database)) as connection:
        connection.execute("CREATE TABLE real_canonical_matches ("
            "match_id VARCHAR, competition_id VARCHAR, season_id VARCHAR, "
            "home_team_id VARCHAR, away_team_id VARCHAR, match_date DATE, "
            "kickoff_time_utc TIMESTAMPTZ, source_local_time VARCHAR, "
            "timestamp_precision VARCHAR, status VARCHAR, home_goals INTEGER, "
            "away_goals INTEGER, source_id VARCHAR, source_priority INTEGER, "
            "source_commit VARCHAR, raw_hash VARCHAR, retrieved_at TIMESTAMPTZ, "
            "as_of_time TIMESTAMPTZ, payload JSON)")
        connection.execute("CREATE TABLE real_canonical_competitions ("
                           "competition_id VARCHAR, competition_name VARCHAR, source VARCHAR)")
        connection.execute("CREATE TABLE real_canonical_teams ("
                           "team_id VARCHAR, competition_id VARCHAR, team_name VARCHAR, source VARCHAR)")
        connection.execute("CREATE TABLE kickoff_enrichment_records ("
            "match_id VARCHAR, enriched_kickoff_utc TIMESTAMPTZ, verified BOOLEAN, "
            "timestamp_precision VARCHAR)")

    result = GoldenCandidateSelector(database).select(
        as_of=datetime(2026, 10, 2, tzinfo=UTC))

    assert result.status == "PARTIAL"
    assert len(result.candidates) < 100
    assert not result.candidates


def test_targeted_replay_rejects_unlocked_dataset(tmp_path) -> None:
    database = tmp_path / "empty.duckdb"
    with duckdb.connect(str(database)) as connection:
        connection.execute("CREATE TABLE real_canonical_matches ("
            "match_id VARCHAR, competition_id VARCHAR, season_id VARCHAR, "
            "home_team_id VARCHAR, away_team_id VARCHAR, match_date DATE, "
            "kickoff_time_utc TIMESTAMPTZ, source_local_time VARCHAR, "
            "timestamp_precision VARCHAR, status VARCHAR, home_goals INTEGER, "
            "away_goals INTEGER, source_id VARCHAR, source_priority INTEGER, "
            "source_commit VARCHAR, raw_hash VARCHAR, retrieved_at TIMESTAMPTZ, "
            "as_of_time TIMESTAMPTZ, payload JSON)")
        connection.execute("CREATE TABLE real_canonical_competitions ("
                           "competition_id VARCHAR, competition_name VARCHAR, source VARCHAR)")
        connection.execute("CREATE TABLE real_canonical_teams ("
                           "team_id VARCHAR, competition_id VARCHAR, team_name VARCHAR, source VARCHAR)")
        connection.execute("CREATE TABLE kickoff_enrichment_records ("
            "match_id VARCHAR, enriched_kickoff_utc TIMESTAMPTZ, verified BOOLEAN, "
            "timestamp_precision VARCHAR)")

    result = GoldenCandidateSelector(database).select(
        as_of=datetime(2026, 10, 2, tzinfo=UTC))
    assert result.status == "PARTIAL"
