"""Kickoff enrichment requires a unique canonical match and verified UTC time."""

from datetime import UTC, datetime, timedelta

import duckdb
import pytest

from erguoyuan_football.data.kickoff_enrichment import KickoffEnrichmentLayer


def make_database(path, fixture_dates: tuple[str, ...]) -> None:
    with duckdb.connect(str(path)) as connection:
        connection.execute("CREATE TABLE real_canonical_teams(team_id VARCHAR, team_name VARCHAR)")
        connection.execute("INSERT INTO real_canonical_teams VALUES ('h','Manchester United FC'),('a','Arsenal FC')")
        connection.execute("""CREATE TABLE real_canonical_matches(
            match_id VARCHAR, competition_id VARCHAR, season_id VARCHAR, match_date DATE,
            home_team_id VARCHAR, away_team_id VARCHAR)""")
        for index, day in enumerate(fixture_dates):
            connection.execute("INSERT INTO real_canonical_matches VALUES (?, 'EPL', '2025-26', ?, 'h', 'a')",
                               [f"match-{index}", day])


def event(utc_date: str) -> dict:
    return {"id": "provider-event", "competition": "PL", "season": 2025,
            "home": "Manchester United", "away": "Arsenal", "utcDate": utc_date}


def test_exact_match_enrichment(tmp_path) -> None:
    path = tmp_path / "exact.duckdb"
    make_database(path, ("2025-10-01",))
    with KickoffEnrichmentLayer(path) as layer:
        result = layer.resolve(event("2025-10-01T19:00:00Z"), provider_id="TEST_PROVIDER",
            retrieved_at=datetime(2026, 1, 1, tzinfo=UTC), competition_map={"PL": "EPL"})
        assert result is not None
        assert result.match_id == "match-0" and result.match_confidence == "EXACT"
        assert result.verified and result.enriched_kickoff_utc.utcoffset() == timedelta(0)


def test_high_confidence_enrichment(tmp_path) -> None:
    path = tmp_path / "neighbor.duckdb"
    make_database(path, ("2025-09-30",))
    with KickoffEnrichmentLayer(path) as layer:
        result = layer.resolve(event("2025-10-01T19:00:00Z"), provider_id="TEST_PROVIDER",
            retrieved_at=datetime(2026, 1, 1, tzinfo=UTC), competition_map={"PL": "EPL"})
        assert result is not None and result.match_confidence == "HIGH_CONFIDENCE"
        assert result.verified


def test_provider_team_alias_can_be_linked_by_canonical_id(tmp_path) -> None:
    path = tmp_path / "alias.duckdb"
    make_database(path, ("2025-10-01",))
    with KickoffEnrichmentLayer(path) as layer:
        layer.connection.execute("INSERT INTO real_canonical_team_aliases VALUES "
            "('alias-id','h','Man Utd','en','CURATED_TEST',1.0,?)",
            [datetime(2026, 1, 1, tzinfo=UTC)])
        provider_event = event("2025-10-01T19:00:00Z")
        provider_event["home"] = "Man Utd"
        result = layer.resolve(provider_event, provider_id="TEST_PROVIDER",
            retrieved_at=datetime(2026, 1, 1, tzinfo=UTC), competition_map={"PL": "EPL"})
        assert result is not None and result.match_id == "match-0"
        assert result.match_confidence == "HIGH_CONFIDENCE"


def test_ambiguous_not_auto_promoted(tmp_path) -> None:
    path = tmp_path / "ambiguous.duckdb"
    make_database(path, ("2025-09-30", "2025-10-02"))
    with KickoffEnrichmentLayer(path) as layer:
        result = layer.resolve(event("2025-10-01T19:00:00Z"), provider_id="TEST_PROVIDER",
            retrieved_at=datetime(2026, 1, 1, tzinfo=UTC), competition_map={"PL": "EPL"})
        assert result is None
        rows = layer.connection.execute("SELECT reason, candidates FROM kickoff_enrichment_review_queue").fetchone()
        assert rows[0] == "AMBIGUOUS_MATCH"
        assert len(__import__("json").loads(rows[1])) == 2


def test_utc_timezone_aware(tmp_path) -> None:
    path = tmp_path / "naive.duckdb"
    make_database(path, ("2025-10-01",))
    with KickoffEnrichmentLayer(path) as layer, pytest.raises(ValueError, match="TIMEZONE_AWARE"):
        layer.resolve(event("2025-10-01T19:00:00"), provider_id="TEST_PROVIDER",
            retrieved_at=datetime(2026, 1, 1, tzinfo=UTC), competition_map={"PL": "EPL"})
