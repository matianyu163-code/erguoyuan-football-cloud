"""DuckDB persistence and same-cutoff snapshot identity for market-engine outputs."""

import pytest

from erguoyuan_football.data.snapshots import SnapshotService
from erguoyuan_football.markets.schemas import MarketType


@pytest.mark.pit
def test_market_snapshot_and_goal_features_persist_as_immutable_cutoff(store, at, quote_factory,
                                                                        provisional_market_engine) -> None:
    fixture = store.fixture_at("test_match_1", at)
    quotes = tuple(item.model_copy(update={"match_id": fixture.match_id}) for item in quote_factory())
    openings = tuple(item.model_copy(update={"match_id": fixture.match_id}) for item in quote_factory(opening=True))
    for quote in (*quotes, *openings):
        store.add_odds_quote(quote)
    assert "odds_quotes" in store.tables and "market_snapshots" in store.tables
    run = provisional_market_engine.run(fixture, quotes, prediction_time=at, opening_quotes=openings)
    store.save_market_snapshot(run.market_snapshot)
    for consensus in run.consensuses:
        store.save_market_consensus(consensus)
    store.save_market_quality(run.quality_report)
    store.save_market_implied_goals(run.market_goal_features)
    loaded = store.market_goal_features_at(at, match_ids=(fixture.match_id,))
    assert len(loaded) == 1
    assert loaded[0] == run.market_goal_features
    snapshot = SnapshotService(store).create(fixture.match_id, at)
    assert snapshot.market_goal_features == run.market_goal_features
    assert snapshot.data_completeness.items["market_odds"].availability.value == "AVAILABLE"
    assert snapshot.data_completeness.items["current_odds"].availability.value == "AVAILABLE"
    assert snapshot.data_completeness.items["opening_odds"].availability.value == "AVAILABLE"


@pytest.mark.pit
def test_market_store_rejects_unpersisted_and_future_snapshot_quote(store, at, quote_factory,
                                                                     provisional_market_engine) -> None:
    fixture = store.fixture_at("test_match_1", at)
    quotes = tuple(item.model_copy(update={"match_id": fixture.match_id}) for item in quote_factory())
    run = provisional_market_engine.run(fixture, quotes, prediction_time=at)
    with pytest.raises(ValueError, match="MARKET_SNAPSHOT_REFERENCES_UNPERSISTED_OR_FUTURE_QUOTES"):
        store.save_market_snapshot(run.market_snapshot)
    assert MarketType.MATCH_1X2 in run.market_snapshot.markets_available
