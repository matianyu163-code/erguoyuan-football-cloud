"""SPI-like model uses labelled fixtures and actual chronological updates."""

from datetime import timedelta

import numpy as np
import pytest

from erguoyuan_football.contracts.common import ExecutionStatus
from erguoyuan_football.data.snapshots import PredictionSnapshot
from erguoyuan_football.models.spi import CoreSPILikeModel
from erguoyuan_football.models.spi.model import (
    OverallRatingMapper,
    SPIConfig,
    estimate_draw_adjustment,
)
from erguoyuan_football.models.spi.state import (
    SPIFeatureMode,
    SPIMatchPerformanceEstimator,
)
from erguoyuan_football.models.training import (
    TeamHierarchyMembership,
    TrainingXGObservation,
)
from tests.unit.test_phase3_models import (
    START,
    fit_config,
    make_dataset,
    make_match,
    make_snapshot,
)

pytestmark = pytest.mark.model


def fitted_spi(rows: int = 48, **config_changes) -> CoreSPILikeModel:
    data = make_dataset(rows=rows)
    options = SPIConfig(**fit_config().model_dump(), **config_changes)
    return CoreSPILikeModel().fit(data, data.matches[-1].kickoff_time + timedelta(days=1), options)


def test_spi_interface_and_identity() -> None:
    model = fitted_spi()
    assert model.model_id == "CORE_SPI_LIKE_V1"
    assert model.implementation_type == "LIKE_IMPLEMENTATION"
    assert model.metadata["official_model"] == "NOT_OFFICIAL_FIVETHIRTYEIGHT_SPI"


def test_spi_pre_match_state_is_before_kickoff() -> None:
    model = fitted_spi(rows=24)
    assert model.state_store.journal
    assert all(state.phase in {"PRE_MATCH", "POST_MATCH"} for state in model.state_store.journal)
    for state in model.state_store.journal:
        if state.phase == "PRE_MATCH":
            row = next(row for row in make_dataset(rows=24).matches if row.match_id == state.match_id)
            assert state.as_of_time < row.kickoff_time


def test_spi_offense_defense_semantics() -> None:
    estimator = SPIMatchPerformanceEstimator(baseline_goals=1.25, smoothing=0.5, learning_rate=0.1)
    attack_low_concede, defense_low_concede = estimator.estimate(3, 0, 0.0, 0.0)
    attack_high_concede, defense_high_concede = estimator.estimate(3, 3, 0.0, 0.0)
    assert attack_low_concede == pytest.approx(attack_high_concede)
    assert defense_high_concede > defense_low_concede  # higher means more expected goals conceded


def test_spi_overall_rating_is_expected_neutral_points_share() -> None:
    rating = OverallRatingMapper.map_rating(0, 0, league_log_goal=np.log(1.25),
        league_offense=0, league_defense=0, max_goals=10)
    assert 0 <= rating <= 100
    assert rating == pytest.approx(50, abs=2)


def test_spi_goal_expectancy_and_score_matrix() -> None:
    model = fitted_spi()
    result = model.predict(make_match(), make_snapshot())
    assert result.execution_status == ExecutionStatus.SUCCESS
    assert result.lambda_home > 0 and result.lambda_away > 0
    assert result.score_matrix is not None
    assert sum(map(sum, result.score_matrix)) == pytest.approx(1.0, abs=1e-8)
    assert result.p_home + result.p_draw + result.p_away == pytest.approx(1.0)


def test_spi_missing_xg_stays_goals_only() -> None:
    model = fitted_spi()
    result = model.predict(make_match(), make_snapshot())
    assert model.feature_mode == SPIFeatureMode.GOALS_ONLY
    assert result.metadata["feature_mode"] == "GOALS_ONLY"
    assert "shot_xg" not in result.metadata["features_used"]
    assert result.metadata["xg_sources"] == []


def test_spi_feature_mode_requires_complete_provenance_coverage() -> None:
    data = make_dataset(rows=24)
    cutoff = data.matches[-1].kickoff_time + timedelta(days=1)
    partial = TrainingXGObservation(match_id=data.matches[0].match_id, xg_home=1.2, xg_away=0.8,
        xg_source_id="fixture_xg", provider="PUBLIC_WEB", model_name="test-model", model_version="1",
        retrieved_at=data.matches[0].available_at, as_of_time=data.matches[0].available_at, data_hash="ab1234")
    contaminated = data.model_copy(update={"xg_observations": (partial,)})
    model = CoreSPILikeModel().fit(contaminated, cutoff,
        SPIConfig(**fit_config().model_dump(), spi_feature_mode="GOALS_XG"))
    assert model.feature_mode == SPIFeatureMode.GOALS_ONLY


def test_spi_fully_sourced_xg_mode() -> None:
    data = make_dataset(rows=24)
    observations = tuple(TrainingXGObservation(match_id=row.match_id, xg_home=1.1, xg_away=0.9,
        xg_source_id="fixture_xg", provider="PUBLIC_WEB", model_name="documented-model", model_version="v1",
        retrieved_at=row.available_at, as_of_time=row.available_at, data_hash=f"hash-{row.match_id}")
        for row in data.matches)
    with_xg = data.model_copy(update={"xg_observations": observations})
    model = CoreSPILikeModel().fit(with_xg, data.matches[-1].kickoff_time + timedelta(days=1),
        SPIConfig(**fit_config().model_dump(), spi_feature_mode="GOALS_XG"))
    assert model.feature_mode == SPIFeatureMode.GOALS_XG
    assert "shot_xg" in model.metadata["features_used"]
    assert model.metadata["xg_observation_count"] == 24


def test_spi_hierarchy_updates_are_pit_safe() -> None:
    data = make_dataset(rows=48)
    hierarchy = tuple(TeamHierarchyMembership(team_id=team,
        league_id="league_a" if index < 3 else "league_b", country_id="country_a",
        continent_id="continent_a", valid_from=START - timedelta(days=1), source="SYNTHETIC_TEST",
        retrieved_at=START - timedelta(hours=1), as_of_time=START - timedelta(hours=1),
        data_version="hierarchy-test-v1") for index, team in enumerate(sorted(data.known_team_ids)))
    with_hierarchy = data.model_copy(update={"team_hierarchy_timeline": hierarchy})
    model = CoreSPILikeModel().fit(with_hierarchy,
        data.matches[-1].kickoff_time + timedelta(days=1), fit_config())
    assert model.league_engine.method_id == "CORE_SPI_LEAGUE_STRENGTH_V1"
    assert all(item["as_of_time"] <= model.trained_until for item in model.league_engine.history)


def test_spi_season_transition_carries_state_and_mean_reverts() -> None:
    original = make_dataset(rows=24)
    matches = tuple(row.model_copy(update={"season": "2025"}) if index >= 12 else row
                    for index, row in enumerate(original.matches))
    data = original.model_copy(update={"matches": matches})
    model = CoreSPILikeModel().fit(data, matches[-1].kickoff_time + timedelta(days=1), fit_config())
    prior_post = next(state for state in model.state_store.journal
        if state.match_id == matches[11].match_id and state.team_id == matches[12].home_team_id
        and state.phase == "POST_MATCH")
    next_pre = next(state for state in model.state_store.journal
        if state.match_id == matches[12].match_id and state.team_id == matches[12].home_team_id
        and state.phase == "PRE_MATCH")
    assert next_pre.offensive_rating == pytest.approx(prior_post.offensive_rating * 0.75)
    assert next_pre.uncertainty > prior_post.uncertainty


def test_spi_score_matrix_tail_is_audited() -> None:
    model = fitted_spi()
    result = model.predict(make_match(), make_snapshot())
    assert result.metadata["score_matrix_retained_mass"] + result.metadata["score_matrix_tail_mass"] == pytest.approx(1)


def test_spi_draw_adjustment_is_configured_and_normalized() -> None:
    model = fitted_spi()
    prediction = model.predict(make_match(), make_snapshot())
    assert prediction.execution_status == ExecutionStatus.SUCCESS
    assert prediction.metadata["draw_adjustment"] == pytest.approx(1.0)
    assert sum(map(sum, prediction.score_matrix)) == pytest.approx(1.0)


def test_spi_draw_adjustment_uses_only_oos_validation() -> None:
    model = fitted_spi()
    predictions = []
    kickoffs = []
    for index in range(6):
        kickoff = START.replace(year=2026, month=9, day=10 + index)
        match = make_match(kickoff=kickoff).model_copy(update={
            "match_id": f"validation_match_{index}",
            "as_of_time": kickoff - timedelta(minutes=30),
            "retrieved_at": kickoff - timedelta(minutes=30),
        })
        snapshot = PredictionSnapshot(match_id=match.match_id,
            prediction_time=kickoff - timedelta(minutes=30), match_data_snapshot=match)
        predictions.append(model.predict(match, snapshot).model_copy(update={"is_oos": True}))
        kickoffs.append(kickoff)
    evidence = estimate_draw_adjustment(tuple(predictions), (0, 1, 2, 1, 0, 2), tuple(kickoffs),
        training_cutoff=model.trained_until, validation_start=START.replace(year=2026, month=9, day=1),
        final_test_start=START.replace(year=2026, month=10, day=1))
    assert evidence.sample_count == 6
    assert len(evidence.validation_data_hash) == 64
    assert 0.8 <= evidence.multiplier <= 1.2
    assert evidence.final_test_start > evidence.validation_start


def test_spi_predict_many_and_save_load(tmp_path) -> None:
    model = fitted_spi()
    match = make_match()
    predictions = model.predict_many([match, match], [make_snapshot(match), make_snapshot(match)])
    assert all(value.execution_status == ExecutionStatus.SUCCESS for value in predictions)
    model.save(tmp_path / "spi")
    loaded = CoreSPILikeModel.load(tmp_path / "spi")
    after = loaded.predict(match, make_snapshot(match))
    assert after.execution_status == ExecutionStatus.SUCCESS
    assert after.p_home == pytest.approx(predictions[0].p_home)


def test_spi_config_validation() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        SPIConfig(spi_season_carryover=0.8, spi_mean_reversion=0.1)
    with pytest.raises(ValueError, match="train/validation evidence hash"):
        SPIConfig(spi_draw_adjustment=1.05)
