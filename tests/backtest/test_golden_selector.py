"""Explicitly synthetic fixtures for testing the Golden candidate selector."""

from datetime import UTC, datetime

import duckdb

from erguoyuan_football.backtesting.golden_selector import (
    GoldenCandidateSelector,
    JCSourceEvidence,
    write_candidate_file,
)


def _database(path) -> None:
    with duckdb.connect(str(path)) as connection:
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
        connection.execute("INSERT INTO real_canonical_competitions VALUES "
                           "('EPL','English Premier League','SYNTHETIC_TEST')")
        connection.execute("INSERT INTO real_canonical_teams VALUES "
            "('h','EPL','Synthetic Home','SYNTHETIC_TEST'),"
            "('a','EPL','Synthetic Away','SYNTHETIC_TEST')")


def _insert_match(path, *, source_id="SYNTHETIC_TEST", kickoff=None) -> None:
    with duckdb.connect(str(path)) as connection:
        connection.execute("INSERT INTO real_canonical_matches VALUES ("
            "'m1','EPL','2026-27','h','a','2026-08-10',?,NULL,'EXACT_UTC',"
            "'FINISHED',2,1,?,1,'test','test-hash',current_timestamp,current_timestamp,'{}')",
            [kickoff, source_id])


def test_selector_rejects_match_without_jc_source_evidence(tmp_path) -> None:
    database = tmp_path / "synthetic.duckdb"
    _database(database)
    _insert_match(database, source_id="OPENFOOTBALL",
                  kickoff=datetime(2026, 8, 10, 18, tzinfo=UTC))

    result = GoldenCandidateSelector(database).select(
        as_of=datetime(2026, 10, 2, tzinfo=UTC), target_count=1)

    assert result.status == "PARTIAL"
    assert result.candidates == ()
    assert "JC_SOURCE_EVIDENCE_UNAVAILABLE" in result.rejected[0].reasons


def test_selector_accepts_audited_pre_kickoff_jc_fixture(tmp_path) -> None:
    database = tmp_path / "synthetic.duckdb"
    _database(database)
    kickoff = datetime(2026, 8, 10, 18, tzinfo=UTC)
    _insert_match(database, source_id="OPENFOOTBALL", kickoff=kickoff)
    evidence = JCSourceEvidence("m1", "JC_VERIFIED", "SYNTHETIC_TEST_EVIDENCE",
                                datetime(2026, 8, 10, 10, tzinfo=UTC))

    result = GoldenCandidateSelector(database).select(
        as_of=datetime(2026, 10, 2, tzinfo=UTC), jc_evidence={"m1": evidence},
        target_count=1)

    assert result.status == "READY"
    assert len(result.candidates) == 1
    assert result.candidates[0].result == (2, 1)
    assert result.candidates[0].source_type == "JC_VERIFIED"


def test_candidate_artifact_stays_unlocked_when_partial(tmp_path) -> None:
    database = tmp_path / "synthetic.duckdb"
    _database(database)
    _insert_match(database, source_id="OPENFOOTBALL",
                  kickoff=datetime(2026, 8, 10, 18, tzinfo=UTC))
    result = GoldenCandidateSelector(database).select(
        as_of=datetime(2026, 10, 2, tzinfo=UTC), target_count=1)

    artifact = write_candidate_file(result, tmp_path / "golden_candidates.json")
    assert '"locked": false' in artifact.read_text(encoding="utf-8")

    assert '"selected_count": 0' in artifact.read_text(encoding="utf-8")


def test_synthetic_match_is_rejected_even_with_jc_claim(tmp_path) -> None:
    database = tmp_path / "synthetic.duckdb"
    _database(database)
    kickoff = datetime(2026, 8, 10, 18, tzinfo=UTC)
    _insert_match(database, source_id="SYNTHETIC_TEST", kickoff=kickoff)
    evidence = JCSourceEvidence("m1", "USER_JC_CONFIRMED", "SYNTHETIC_TEST_EVIDENCE",
                                datetime(2026, 8, 10, 10, tzinfo=UTC))

    result = GoldenCandidateSelector(database).select(
        as_of=datetime(2026, 10, 2, tzinfo=UTC), jc_evidence={"m1": evidence},
        target_count=1)

    assert result.candidates == ()
    assert "SYNTHETIC_TEST_DATA" in result.rejected[0].reasons
