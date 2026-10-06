"""World Cup metric tests use synthetic, explicitly labelled fixtures."""

from datetime import UTC, datetime, timedelta

import pytest

from erguoyuan_football.backtesting.worldcup_dataset import (
    WorldCupFavorite,
    WorldCupMatch,
    WorldCupPrediction,
    WorldCupResult,
    WorldCupSourceEvidence,
    WorldCupStage,
    evaluate_worldcup_predictions,
)


def _source(evidence_id: str, time: datetime) -> WorldCupSourceEvidence:
    return WorldCupSourceEvidence(
        evidence_id=evidence_id,
        source="SYNTHETIC_TEST",
        source_url="https://example.invalid/synthetic",
        retrieved_at=time,
        as_of_time=time,
    )


def test_wc_metrics_split_group_knockout_upsets_and_calibration() -> None:
    kickoff = datetime(2026, 6, 11, tzinfo=UTC)
    matches = (
        WorldCupMatch(
            match_id="SYNTHETIC_TEST_GROUP",
            match_number=1,
            stage=WorldCupStage.GROUP,
            group="A",
            home_team_id="H1",
            home_team="Home 1",
            away_team_id="A1",
            away_team="Away 1",
            kickoff_time=kickoff,
            venue="Venue",
            fixture_evidence=(_source("fixture-1", kickoff - timedelta(days=1)),),
            result=WorldCupResult(regulation_home_goals=1, regulation_away_goals=0,
                                  winner_team_id="H1"),
            result_evidence=(_source("result-1", kickoff + timedelta(days=1)),),
        ),
        WorldCupMatch(
            match_id="SYNTHETIC_TEST_KO",
            match_number=73,
            stage=WorldCupStage.ROUND32,
            home_team_id="H2",
            home_team="Home 2",
            away_team_id="A2",
            away_team="Away 2",
            kickoff_time=kickoff + timedelta(days=1),
            venue="Venue",
            fixture_evidence=(_source("fixture-2", kickoff - timedelta(days=1)),),
            result=WorldCupResult(regulation_home_goals=0, regulation_away_goals=1,
                                  winner_team_id="A2"),
            result_evidence=(_source("result-2", kickoff + timedelta(days=2)),),
        ),
    )
    predictions = tuple(
        WorldCupPrediction(
            match_id=match.match_id,
            model_id="SYNTHETIC_TEST_MODEL",
            model_version="test",
            prediction_time=match.kickoff_time - timedelta(hours=1),
            kickoff_time=match.kickoff_time,
            training_end_time=match.kickoff_time - timedelta(days=2),
            input_data_version="SYNTHETIC_TEST",
            input_evidence=(_source(f"input-{index}", match.kickoff_time - timedelta(hours=2)),),
            status="SUCCESS",
            p_home=0.7 if index == 0 else 0.1,
            p_draw=0.2,
            p_away=0.1 if index == 0 else 0.7,
        )
        for index, match in enumerate(matches)
    )
    report = evaluate_worldcup_predictions(
        matches,
        predictions,
        pre_match_favorites={
            "SYNTHETIC_TEST_GROUP": WorldCupFavorite(
                team_id="A1", evidence=_source("favorite-1", kickoff - timedelta(hours=3))),
            "SYNTHETIC_TEST_KO": WorldCupFavorite(
                team_id="H2", evidence=_source("favorite-2", kickoff - timedelta(hours=3))),
        },
        calibration_bins=5,
    )
    assert report.sample_size == 2
    assert report.group_stage_accuracy == 1
    assert report.knockout_regulation_accuracy == 1
    assert report.upset_sample_size == 2
    assert report.upset_match_accuracy == 1
    assert report.multiclass_brier >= 0
    assert report.log_loss >= 0
    assert 0 <= report.top_label_ece <= 1


def test_wc_metrics_reject_predictions_without_frozen_success_rows() -> None:
    with pytest.raises(ValueError, match="NO_SUCCESSFUL_WORLD_CUP_PREDICTIONS"):
        evaluate_worldcup_predictions((), (), pre_match_favorites={})
