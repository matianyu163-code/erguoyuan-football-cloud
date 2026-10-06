from datetime import timedelta

import duckdb
import pytest

from erguoyuan_football.data.schemas import (
    Fixture,
    LineupSnapshot,
    MatchResult,
    MetricSnapshot,
)
from erguoyuan_football.data.snapshots import PredictionSnapshot, SnapshotService


@pytest.mark.pit
def test_point_in_time_filter(store, at, quote_factory):
    valid = quote_factory()
    future = quote_factory(as_of_time=at + timedelta(seconds=1))
    late = quote_factory(retrieved_at=at + timedelta(seconds=1))
    for quote in (valid, future, late):
        store.add_snapshot("odds_snapshots", quote)
    result = store.snapshots_at("odds_snapshots", "test_match_1", at)
    assert [q.snapshot_id for q in result] == [valid.snapshot_id]


@pytest.mark.parametrize("field", ["as_of_time", "retrieved_at"])
def test_future_data_rejected(store, at, quote_factory, field):
    quote = quote_factory(**{field: at + timedelta(seconds=1)})
    with pytest.raises(ValueError, match="future input"):
        PredictionSnapshot(match_id="test_match_1", prediction_time=at,
                           match_data_snapshot=store.fixture_at("test_match_1", at), market_snapshot=(quote,))


def test_kickoff_boundary(store, at):
    kickoff = store.fixture_at("test_match_1", at).kickoff_time
    for time in (kickoff, kickoff + timedelta(seconds=1)):
        with pytest.raises(ValueError, match="precede kickoff"):
            SnapshotService(store).create("test_match_1", time)


def test_snapshot_creation(store, at, quote_factory):
    store.add_snapshot("odds_snapshots", quote_factory())
    early = SnapshotService(store).create("test_match_1", at)
    six = at + timedelta(hours=6)
    store.add_snapshot("odds_snapshots", quote_factory(as_of_time=six, retrieved_at=six, odds=2.2, data_version="test_v2"))
    later = SnapshotService(store).create("test_match_1", six)
    assert early.prediction_snapshot_id != later.prediction_snapshot_id
    assert early.input_data_version != later.input_data_version
    assert store.load_prediction_snapshot(early.prediction_snapshot_id).market_snapshot[0].odds == 2
    assert later.market_snapshot[0].odds == 2.2
    replay = SnapshotService(store).create("test_match_1", at)
    assert replay.input_data_version == early.input_data_version


def test_schedule_revision_does_not_leak(store, at):
    old = store.fixture_at("test_match_1", at)
    store.add_fixture(Fixture(**{**old.model_dump(), "kickoff_time": at + timedelta(hours=9),
                                "as_of_time": at + timedelta(hours=1), "retrieved_at": at + timedelta(hours=1),
                                "data_version": "test_postponed"}))
    assert store.fixture_at(old.match_id, at) == old
    assert store.fixture_at(old.match_id, at + timedelta(hours=2)).kickoff_time == at + timedelta(hours=9)


def test_market_snapshot(store, at, quote_factory):
    for selection in ("HOME", "DRAW"):
        store.add_snapshot("market_snapshots", quote_factory(selection=selection))
    report = SnapshotService(store).create("test_match_1", at).data_completeness
    assert report.items["current_odds"].availability == "UNAVAILABLE"
    store.add_snapshot("market_snapshots", quote_factory(selection="AWAY"))
    report = SnapshotService(store).create("test_match_1", at).data_completeness
    assert report.items["current_odds"].availability == "AVAILABLE"
    assert report.items["opening_odds"].availability == "UNAVAILABLE"


def test_mixed_market_timestamps_not_complete(store, at, quote_factory):
    for index, selection in enumerate(("HOME", "DRAW", "AWAY")):
        store.add_snapshot("odds_snapshots", quote_factory(selection=selection, as_of_time=at - timedelta(seconds=index)))
    assert SnapshotService(store).create("test_match_1", at).data_completeness.items["market_odds"].availability == "UNAVAILABLE"


@pytest.mark.parametrize("market,selections,line,feature", [
    ("ASIAN_HANDICAP", ("HOME", "AWAY"), -0.5, "asian_handicap"),
    ("OVER_UNDER", ("OVER", "UNDER"), 2.5, "over_under"),
    ("SPORTS_LOTTERY", ("HOME", "DRAW", "AWAY"), None, "sports_lottery"),
])
def test_market_types(store, at, quote_factory, market, selections, line, feature):
    for selection in selections:
        store.add_snapshot("odds_snapshots", quote_factory(market_type=market, selection=selection, line=line))
    assert SnapshotService(store).create("test_match_1", at).data_completeness.items[feature].availability == "AVAILABLE"


def test_invalid_market(quote_factory):
    for fields in ({"market_type": "ASIAN_HANDICAP"}, {"odds": float("nan")}, {"odds": 1},
                   {"raw_implied_probability": 0.1}, {"source": "USER_SCREENSHOT", "phase": "CLOSE"}):
        with pytest.raises(ValueError):
            quote_factory(**fields)


def test_history_excludes_future_results(store, at):
    fixture = store.fixture_at("test_match_1", at)
    past = Fixture(**{**fixture.model_dump(), "match_id": "test_past", "kickoff_time": at - timedelta(days=1)})
    store.add_fixture(past)
    fields = {"match_id": past.match_id, "home_goals": 1, "away_goals": 0,
              "completed_at": at - timedelta(hours=20), "source": "SYNTHETIC_TEST",
              "as_of_time": at - timedelta(hours=20), "retrieved_at": at - timedelta(hours=20),
              "data_version": "test_v1"}
    store.add_result(MatchResult(**fields))
    store.add_result(MatchResult(**{**fields, "home_goals": 2, "data_version": "test_corrected",
                                  "retrieved_at": at + timedelta(seconds=1)}))
    snapshot = SnapshotService(store).create("test_match_1", at)
    assert snapshot.historical_results[0].result.home_goals == 1
    assert snapshot.data_completeness.items["historical_goals"].availability == "AVAILABLE"


def test_stat_and_lineup_completeness(store, at):
    for team in ("test_arsenal", "test_chelsea"):
        common = {"match_id": "test_match_1", "team_id": team, "source": "SYNTHETIC_TEST", "retrieved_at": at,
                  "as_of_time": at, "data_version": "test_v1"}
        store.add_snapshot("xg_snapshots", MetricSnapshot(**common, metrics={"xg": 1.2, "xga": 1.1}))
        store.add_snapshot("lineup_snapshots", LineupSnapshot(**common, player_ids=("test_player",), confirmed=False))
    report = SnapshotService(store).create("test_match_1", at).data_completeness
    assert report.items["xg"].availability == "AVAILABLE"
    assert report.items["lineup"].availability == "UNAVAILABLE"


def test_naive_timestamp_rejected(quote_factory, at):
    with pytest.raises(ValueError, match="timezone-aware"):
        quote_factory(as_of_time=at.replace(tzinfo=None))


def test_append_only_and_parquet(store, quote_factory, tmp_path):
    quote = quote_factory()
    store.add_snapshot("odds_snapshots", quote)
    with pytest.raises(duckdb.ConstraintException):
        store.add_snapshot("odds_snapshots", quote)
    destination = store.export_parquet("odds_snapshots", tmp_path / "quoted ' path.parquet")
    row = store.connection.read_parquet(str(destination)).fetchone()
    assert row[0] == quote.snapshot_id
    with pytest.raises(FileExistsError):
        store.export_parquet("odds_snapshots", destination)
    with pytest.raises(ValueError):
        store.export_parquet("matches; DROP TABLE teams", tmp_path / "bad.parquet")


def test_database_schema(store):
    required = {"teams", "team_aliases", "competitions", "matches", "match_requests", "odds_snapshots",
                "market_snapshots", "team_stats_snapshots", "xg_snapshots", "lineup_snapshots", "injury_snapshots",
                "opta_snapshots", "model_predictions", "oos_predictions", "final_predictions", "match_results"}
    assert required <= set(store.tables)
    for table, columns in store.schema_description().items():
        if table.endswith("_snapshots") and table not in {"prediction_snapshots", "market_snapshots"}:
            assert {"snapshot_id", "match_id", "source", "retrieved_at", "as_of_time", "data_version"} <= {c[0] for c in columns}
    market_columns = {column[0] for column in store.schema_description()["market_snapshots"]}
    assert {"market_snapshot_id", "match_id", "prediction_time", "created_at", "payload"} <= market_columns


def test_wrong_match_rejected(store, at, quote_factory):
    with pytest.raises(ValueError, match="cross-match"):
        PredictionSnapshot(match_id="test_match_1", prediction_time=at,
                           match_data_snapshot=store.fixture_at("test_match_1", at),
                           market_snapshot=(quote_factory(match_id="test_match_2"),))


def test_conflicting_revision_rejected(store, at, quote_factory):
    store.add_snapshot("odds_snapshots", quote_factory())
    store.add_snapshot("odds_snapshots", quote_factory(odds=2.1))
    with pytest.raises(ValueError, match="conflicting"):
        SnapshotService(store).create("test_match_1", at)
    assert store.connection.execute("SELECT count(*) FROM prediction_snapshots").fetchone()[0] == 0
