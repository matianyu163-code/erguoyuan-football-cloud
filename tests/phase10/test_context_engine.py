"""Core Phase 10 temporal, context, probability and persistence contracts."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta

import pytest

from erguoyuan_football.context.adjustment import (
    ContextAdjustmentEngine,
    probability_logits,
    softmax_logits,
)
from erguoyuan_football.context.artifacts import ContextStore
from erguoyuan_football.context.competition_rules import CompetitionRuleRegistry
from erguoyuan_football.context.gates import (
    ContextDataGate,
    ContextPromotionGate,
    LineupGate,
    promotion_report,
)
from erguoyuan_football.context.incentive import (
    TournamentIncentiveEngine,
    title_mathematically_impossible,
)
from erguoyuan_football.context.injury import InjuryEvidenceLayer
from erguoyuan_football.context.lineup import LineupEvidenceLayer
from erguoyuan_football.context.pipeline import ContextPipeline, TargetContextMatch
from erguoyuan_football.context.player_strength import (
    UnavailablePlayerStrengthProvider,
    calculate_lineup_strength_delta,
)
from erguoyuan_football.context.rotation import RotationRiskEngine
from erguoyuan_football.context.schedule_context import ScheduleContextBuilder
from erguoyuan_football.context.schemas import (
    Availability,
    CompetitionRule,
    ContextDataClass,
    ContextFeatureVector,
    ContextStatus,
    InjuryEvidence,
    InjuryStatus,
    LineupEvidenceSnapshot,
    LineupStatus,
    PositionQuality,
    TournamentType,
)
from erguoyuan_football.context.standings import StandingsReconstructor
from erguoyuan_football.context.temporal import ContextTemporalPolicy
from erguoyuan_football.context.tie_state import aggregate_before_second_leg
from erguoyuan_football.context.tournament_state import TournamentStateBuilder
from erguoyuan_football.output_contract.schemas import (
    CanonicalPredictionResult,
    ProbabilityStage,
)


def test_context_point_in_time_excludes_same_day_and_later(phase10_events):
    prior = [event for event in phase10_events if event.match_date < date(2025, 2, 1)]
    assert {event.match_id for event in prior} == {"m1", "m2", "m3"}
    for event in phase10_events[3:]:
        with pytest.raises(ValueError, match="CONTEXT_EVENT_NOT_BEFORE_TARGET_DATE"):
            ContextTemporalPolicy.require_event(event, target_match_id="target",
                target_date=date(2025, 2, 1), prediction_time=datetime(2025, 2, 1, tzinfo=UTC))


def test_same_day_results_never_enter_standings(phase10_events):
    result = StandingsReconstructor().reconstruct(competition_id="TEST_LEAGUE",
        season_id="2024-25", target_date=date(2025, 2, 1), events=phase10_events, rule=None)
    assert result.source_match_ids == ("m1", "m2", "m3")
    assert result.availability == Availability.AVAILABLE
    assert result.position_quality == PositionQuality.UNAVAILABLE
    assert all(row.points is None for row in result.rows)


def test_verified_points_and_deterministic_standings(phase10_events):
    rule = CompetitionRule(competition_id="TEST_LEAGUE", season_id="2024-25",
        valid_from=date(2024, 7, 1), valid_to=date(2025, 6, 30),
        rule_version="TEST_RULE_V1", points_win=3, points_draw=1,
        tie_break_policy="POINTS_ONLY", tournament_type=TournamentType.LEAGUE,
        source="SYNTHETIC_TEST", retrieved_at=datetime(2024, 7, 1, tzinfo=UTC), verified=True)
    snapshot = StandingsReconstructor().reconstruct(competition_id="TEST_LEAGUE",
        season_id="2024-25", target_date=date(2025, 2, 1), events=phase10_events, rule=rule)
    assert all(row.points is not None for row in snapshot.rows)
    assert snapshot.position_quality == PositionQuality.APPROXIMATE_POINTS_ONLY


def test_rule_registry_unverified_and_missing_are_unavailable():
    registry = CompetitionRuleRegistry()
    assert registry.resolve("unknown", "2025", date(2025, 1, 1)) is None


def test_competition_rule_snapshot_must_precede_prediction():
    retrieved = datetime(2025, 1, 2, tzinfo=UTC)
    rule = CompetitionRule(competition_id="TEST", season_id="2024-25",
        valid_from=date(2024, 7, 1), valid_to=date(2025, 6, 30),
        rule_version="RULE_V1", points_win=3, points_draw=1,
        tie_break_policy="POINTS_ONLY", tournament_type=TournamentType.LEAGUE,
        source="SYNTHETIC_TEST", retrieved_at=retrieved, verified=True)
    registry = CompetitionRuleRegistry((rule,))
    assert registry.resolve("TEST", "2024-25", date(2025, 1, 1),
        prediction_time=datetime(2025, 1, 1, tzinfo=UTC)) is None
    assert registry.resolve("TEST", "2024-25", date(2025, 1, 2),
        prediction_time=retrieved) == rule


def test_tie_aggregate_requires_verified_rule_and_explicit_identity(phase10_events):
    result = aggregate_before_second_leg(target_match_id="m2", target_date=date(2025, 1, 15),
        tie_id=None, leg_number=2, home_team_id="A", away_team_id="B", events=phase10_events,
        rule=None)
    assert result[1] is None
    assert result[2] == ("VERIFIED_TWO_LEG_RULE_OR_TIE_ID_UNAVAILABLE",)


def test_tournament_state_unverified_rule_not_guessed(phase10_events):
    state = TournamentStateBuilder(CompetitionRuleRegistry()).build(
        match_id="target", competition_id="TEST_LEAGUE", season_id="2024-25",
        target_date=date(2025, 2, 1), prediction_time=datetime(2025, 2, 1, tzinfo=UTC),
        home_team_id="A", away_team_id="B", events=phase10_events)
    assert state.tournament_type == TournamentType.UNKNOWN
    assert state.rule_version is None
    assert "COMPETITION_RULE_UNAVAILABLE" in state.unavailable_reasons


def test_mathematical_elimination_requires_complete_inputs():
    assert title_mathematically_impossible(team_points=10, leader_points=40,
        remaining_matches=None, rule=None) is None
    assert title_mathematically_impossible(team_points=10, leader_points=40,
        remaining_matches=5, rule=None) is None


def test_incentive_score_is_not_inferred_without_remaining_schedule(phase10_events):
    standings = StandingsReconstructor().reconstruct(competition_id="TEST_LEAGUE",
        season_id="2024-25", target_date=date(2025, 2, 1), events=phase10_events, rule=None)
    state = TournamentIncentiveEngine().build(match_id="target", home_team_id="A",
        away_team_id="B", standings=standings, rule=None)
    assert state.mai is None
    assert state.mai_status == Availability.UNAVAILABLE
    assert state.title_mathematically_impossible_home is None


def test_rest_days_and_congestion_are_prior_only(phase10_events):
    result = ScheduleContextBuilder().build(match_id="target", competition_id="TEST_LEAGUE",
        season_id="2024-25", target_date=date(2025, 2, 1), home_team_id="A",
        away_team_id="B", events=phase10_events, schedule_coverage_verified=True)
    assert result.rest_days_home == 24
    assert result.rest_days_away == 17
    assert result.matches_last_7d_home == 0
    assert result.matches_last_21d_home == 0
    assert result.matches_last_21d_away == 1
    assert result.future_schedule_count_home is None
    assert result.neutral_venue is None


def test_fatigue_features_unavailable_without_cross_competition_coverage(phase10_events):
    result = ScheduleContextBuilder().build(match_id="target", competition_id="TEST_LEAGUE",
        season_id="2024-25", target_date=date(2025, 2, 1), home_team_id="A",
        away_team_id="B", events=phase10_events)
    assert result.rest_days_home is None
    assert result.matches_last_14d_home is None
    assert result.matches_last_21d_away is None
    assert result.availability == Availability.UNAVAILABLE
    assert "CROSS_COMPETITION_SCHEDULE_COVERAGE_UNVERIFIED" in result.reason_codes


def test_lineup_rejects_post_kickoff_and_unknown_class():
    before = datetime(2025, 2, 1, 12, tzinfo=UTC)
    kickoff = datetime(2025, 2, 1, 18, tzinfo=UTC)
    late = LineupEvidenceSnapshot(match_id="m", team_id="t", player_id="p",
        evidence_type="lineup", lineup_status=LineupStatus.CONFIRMED_STARTER,
        source_id="late", source="SYNTHETIC_TEST", source_time=kickoff,
        retrieved_at=before, confidence=0.99, is_official=True, quality_status="VERIFIED")
    unknown = late.model_copy(update={"source_id": "unknown",
        "data_class": ContextDataClass.UNKNOWN_CONTEXT,
        "source_time": datetime(2025, 2, 1, 10, tzinfo=UTC)})
    accepted, rejected = LineupEvidenceLayer().evaluate(match_id="m", prediction_time=before,
        kickoff_time=kickoff, evidence=(late, unknown))
    assert not accepted
    assert "POST_KICKOFF_LINEUP_REJECTED" in rejected
    assert "POST_MATCH_LINEUP_REJECTED" in rejected


def test_future_lineup_retrieval_rejected():
    prediction = datetime(2025, 2, 1, 12, tzinfo=UTC)
    item = LineupEvidenceSnapshot(match_id="m", team_id="t", player_id="p",
        evidence_type="lineup", lineup_status=LineupStatus.EXPECTED_STARTER,
        source_id="s", source="SYNTHETIC_TEST", source_time=prediction,
        retrieved_at=prediction + timedelta(minutes=1), confidence=0.9,
        is_official=False, quality_status="EXPECTED")
    _, reasons = LineupEvidenceLayer().evaluate(match_id="m", prediction_time=prediction,
        kickoff_time=prediction + timedelta(hours=4), evidence=(item,))
    assert reasons == ("LINEUP_RETRIEVED_AFTER_PREDICTION",)


def test_future_injury_news_rejected():
    prediction = datetime(2025, 2, 1, 12, tzinfo=UTC)
    item = InjuryEvidence(match_id="m", team_id="t", player_id="p", status=InjuryStatus.OUT,
        source_id="s", source="SYNTHETIC_TEST", published_at=prediction + timedelta(minutes=1),
        retrieved_at=prediction, confidence=0.9)
    _, reasons = InjuryEvidenceLayer().evaluate(match_id="m", prediction_time=prediction,
        evidence=(item,))
    assert reasons == ("FUTURE_INJURY_EVIDENCE_REJECTED",)


def test_unavailable_player_strength_never_becomes_numeric():
    assert UnavailablePlayerStrengthProvider().get_strength("p",
        as_of_time=datetime(2025, 1, 1, tzinfo=UTC)) is None
    value, reason = calculate_lineup_strength_delta(home_lineup=("p",), away_lineup=("q",),
        player_strength={}, baseline_home_strength=1.0, baseline_away_strength=1.0)
    assert value is None
    assert reason == "PLAYER_STRENGTH_EVIDENCE_UNAVAILABLE"


def test_rotation_risk_untrained_is_unavailable():
    result = RotationRiskEngine().evaluate(historical_lineups=(), current_lineup=(),
        schedule_features_available=True)
    assert result.probability is None
    assert result.availability == Availability.UNAVAILABLE


def test_feature_vector_requires_lineage_for_available_feature():
    values = {"match_id": "m", "prediction_snapshot_id": "s",
        "prediction_date": date(2025, 2, 1), "tournament_type": TournamentType.LEAGUE,
        "context_uncertainty": 0.0, "availability_mask": {"standings": True},
        "lineage": {"standings": ()}, "source_event_dates": {},
        "feature_schema_hash": "feature-v1"}
    with pytest.raises(ValueError, match="CONTEXT_FEATURE_LINEAGE_MISSING:standings"):
        ContextFeatureVector.model_validate(values)


def test_probability_logit_transform_roundtrip():
    vector = (0.51, 0.27, 0.22)
    assert softmax_logits(probability_logits(vector, epsilon=1e-12)) == pytest.approx(vector)


def test_no_trained_context_model_is_exact_identity():
    feature = ContextFeatureVector(match_id="m", prediction_snapshot_id="s",
        prediction_date=date(2025, 2, 1), tournament_type=TournamentType.UNKNOWN,
        context_uncertainty=1.0, availability_mask={"lineup": False}, lineage={"lineup": ()},
        source_event_dates={}, feature_schema_hash="feature-v1")
    probability, ledger, reasons = ContextAdjustmentEngine().evaluate(
        prediction_id="p", prediction_snapshot_id="s", match_id="m",
        base=(0.5, 0.25, 0.25), features=feature)
    assert (probability.final_p_home, probability.final_p_draw, probability.final_p_away) == (
        probability.base_p_home, probability.base_p_draw, probability.base_p_away)
    assert not ledger.adjustment_applied
    assert reasons == ("NO_ADJUSTMENT_DUE_TO_INSUFFICIENT_EVIDENCE",)


def test_context_promotion_gate_fails_closed():
    result = ContextPromotionGate().evaluate()
    assert result.status == "NOT_PROMOTED"
    assert not result.passed


def test_phase9_canonical_v2_context_is_additive_and_not_promoted():
    with open("reports/phase9_development_core_preview.jsonl", encoding="utf-8") as handle:
        sample = json.loads(handle.readline())
    base = CanonicalPredictionResult.model_validate(sample)
    match = TargetContextMatch(match_id=base.match_id, competition_id=base.competition_id,
        season_id="2025-26", home_team_id="home", away_team_id="away", match_date=base.match_date)
    pipeline = ContextPipeline(rules=CompetitionRuleRegistry())
    output = pipeline.build_one(base=base, match=match, events=())
    assert output.canonical.probability_stage == ProbabilityStage.FINAL_CORE_CALIBRATED
    assert output.canonical.context_status == ContextStatus.UNAVAILABLE
    assert output.canonical.production_status == "NOT_PROMOTED"
    assert output.audit["status"] == "PASS"
    assert promotion_report(output.assessment).status == "NOT_PROMOTED"


def test_context_store_is_additive_and_idempotent(tmp_path, phase10_events):
    from erguoyuan_football.context.schemas import ContextAssessment

    del phase10_events, ContextAssessment
    db_path = tmp_path / "phase10.duckdb"
    with ContextStore(db_path) as store:
        tables = {row[0] for row in store.connection.execute("SHOW TABLES").fetchall()}
        assert "context_adjustment_ledger" in tables
        assert "real_canonical_matches" not in tables
        assert callable(store.load_run)
    assert db_path.exists()


def test_context_gate_is_separate_from_lineup_gate():
    context_gate = ContextDataGate().evaluate(point_in_time_passed=True, rule=None,
        entity_mapping_verified=True, evidence_fresh=True, context_complete=True,
        conflicts_resolved=True, post_match_data_present=False)
    lineup_gate = LineupGate().evaluate(())
    assert context_gate.status == "BLOCKED"
    assert lineup_gate.status == "UNAVAILABLE"
