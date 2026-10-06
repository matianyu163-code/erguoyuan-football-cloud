from datetime import timedelta, timezone

import pytest

from erguoyuan_football.daily import DailyPredictionRequest, DailyService
from erguoyuan_football.data.availability import build_report
from erguoyuan_football.data.schemas import InjurySnapshot, MetricSnapshot, OptaSnapshot
from erguoyuan_football.data.snapshots import PredictionSnapshot, SnapshotService
from erguoyuan_football.input.match_resolver import MatchResolver
from erguoyuan_football.input.schemas import MatchRequest
from erguoyuan_football.models.requirements import check_requirements


def test_timezone_conversion(quote_factory, at):
    quote = quote_factory(as_of_time=at.astimezone(timezone(timedelta(hours=8))))
    assert quote.as_of_time == at and quote.as_of_time.utcoffset() == timedelta(0)


def test_unavailable_odds_are_null(quote_factory):
    with pytest.raises(ValueError):
        quote_factory(availability="UNAVAILABLE", reason="MISSING")
    quote = quote_factory(availability="UNAVAILABLE", reason="MISSING", odds=None)
    assert quote.odds is None and quote.devig_probability is None


def test_unknown_table_or_match_rejected(store, quote_factory):
    with pytest.raises(ValueError):
        store.add_snapshot("unknown", quote_factory())
    with pytest.raises(ValueError):
        store.add_snapshot("odds_snapshots", quote_factory(match_id="unknown"))


def test_screenshot_confidence_not_bypassed(store, at):
    request = MatchRequest(input_type="SCREENSHOT", source="USER_SCREENSHOT", home_team_name="阿森纳",
                           away_team_name="切尔西", input_confidence=0.1, reason="VISION_VERIFIED")
    assert MatchResolver(store).resolve(request, prediction_time=at).resolution_status == "AMBIGUOUS"


def test_unknown_or_incomplete_external_does_not_become_available(store, at):
    unavailable = OptaSnapshot(match_id="test_match_1", source="SYNTHETIC_TEST", as_of_time=at,
                               retrieved_at=at, data_version="test", product="POWER_RANKING",
                               availability="UNAVAILABLE", reason="NO_AUTHORIZED_DATA")
    store.add_snapshot("opta_snapshots", unavailable)
    report = SnapshotService(store).create("test_match_1", at).data_completeness
    assert report.items["opta_power_ranking"].availability == "UNAVAILABLE"


def test_other_snapshot_tables(store, at):
    for team in ("test_arsenal", "test_chelsea"):
        common = {"match_id": "test_match_1", "team_id": team, "source": "SYNTHETIC_TEST", "as_of_time": at,
                  "retrieved_at": at, "data_version": "test"}
        store.add_snapshot("injury_snapshots", InjurySnapshot(**common, player_ids=()))
        store.add_snapshot("team_stats_snapshots", MetricSnapshot(**common, metrics={"shots": 10}))
    report = SnapshotService(store).create("test_match_1", at).data_completeness
    assert report.items["injuries"].availability == "AVAILABLE"
    assert report.items["team_stats"].availability == "AVAILABLE"
    assert report.items["league_hierarchy"].availability == "UNAVAILABLE"


def test_snapshot_cannot_forge_persisted_inputs(store, at, quote_factory):
    snapshot = PredictionSnapshot(match_id="test_match_1", prediction_time=at,
                                   match_data_snapshot=store.fixture_at("test_match_1", at),
                                   market_snapshot=(quote_factory(),))
    snapshot = PredictionSnapshot(**{**snapshot.model_dump(), "data_completeness": build_report(snapshot)})
    with pytest.raises(ValueError, match="not persisted"):
        store.save_prediction_snapshot(snapshot)


def test_two_line_list_routes_to_batch(store, at):
    result = DailyService(store).run(DailyPredictionRequest(input_type="TEXT", raw_text="阿森纳VS切尔西\n皇马VS巴萨", prediction_time=at))
    assert len(result.snapshots) == 2


def test_empty_catalog_does_not_claim_availability(store, at):
    report = SnapshotService(store).create("test_match_1", at).data_completeness
    assert all(item.availability == "UNAVAILABLE" for item in report.items.values())


def test_required_evidence_present_still_not_implemented(store, at):
    from erguoyuan_football.data.schemas import Fixture, MatchResult
    old = store.fixture_at("test_match_1", at)
    past = Fixture(**{**old.model_dump(), "match_id": "test_history", "kickoff_time": at - timedelta(days=1)})
    store.add_fixture(past)
    store.add_result(MatchResult(match_id=past.match_id, home_goals=0, away_goals=0, completed_at=at - timedelta(hours=20),
        as_of_time=at - timedelta(hours=20), retrieved_at=at - timedelta(hours=20), source="SYNTHETIC_TEST", data_version="test"))
    report = SnapshotService(store).create(old.match_id, at).data_completeness
    decision = check_requirements("DIXON_COLES", report)
    assert decision.execution_status == "SKIPPED" and decision.reason == "NOT_IMPLEMENTED"
    assert decision.missing_required == ()
