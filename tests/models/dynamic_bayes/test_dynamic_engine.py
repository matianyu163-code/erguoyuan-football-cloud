"""Dynamic Bayesian Poisson tests use explicitly labelled synthetic fixtures."""

from __future__ import annotations

from datetime import timedelta

import pytest

from erguoyuan_football.contracts.common import ExecutionStatus
from erguoyuan_football.data.schemas import Fixture
from erguoyuan_football.data.snapshots import PredictionSnapshot
from erguoyuan_football.markets.from_score_matrix import derive_markets
from erguoyuan_football.models.config import ModelConfig
from erguoyuan_football.models.dynamic_bayes.model import (
    CoreDynamicBayesianPoissonModel,
)
from erguoyuan_football.models.dynamic_bayes.state_model import DynamicStateStore
from erguoyuan_football.models.registry import ModelRegistry
from erguoyuan_football.models.runner import ModelRunner
from erguoyuan_football.models.score_matrix import ScoreMatrix
from erguoyuan_football.models.training import (
    TrainingDataError,
    TrainingDataset,
    TrainingDatasetValidator,
)
from tests.unit.test_phase3_models import (
    START,
    TEAMS,
    available_report,
    fit_config,
    make_dataset,
    make_match,
    make_snapshot,
)

pytestmark = pytest.mark.model


def fitted_model(rows: int = 48) -> CoreDynamicBayesianPoissonModel:
    """Fit the real likelihood update on the marked test history."""
    data = make_dataset(rows=rows)
    cutoff = data.matches[-1].kickoff_time + timedelta(days=1)
    return CoreDynamicBayesianPoissonModel().fit(
        data, cutoff, fit_config(dynamic_posterior_draws=100),
    )


def test_dynamic_registry() -> None:
    model = ModelRegistry().get_model("DYNAMIC_BAYESIAN_POISSON_V1")
    assert isinstance(model, CoreDynamicBayesianPoissonModel)
    assert model.implementation_type == "REAL_IMPLEMENTATION"
    assert model.required_data == ("historical_goals",)
    assert model.supports_score_matrix and model.supports_expected_goals


def test_dynamic_training() -> None:
    model = fitted_model()
    assert model.fitted
    assert model.metadata["inference_method"] == "SEQUENTIAL_LAPLACE_POISSON"
    assert model.metadata["state_history_count"] == 96
    assert model.health_report is not None


def test_dynamic_prediction() -> None:
    model = fitted_model()
    prediction = model.predict(make_match(), make_snapshot())
    assert prediction.execution_status == ExecutionStatus.SUCCESS
    assert prediction.lambda_home > 0 and prediction.lambda_away > 0
    assert prediction.metadata["lambda_home_ci"][0] <= prediction.metadata["lambda_home_ci"][1]


def test_dynamic_probability_sum() -> None:
    prediction = fitted_model().predict(make_match(), make_snapshot())
    assert prediction.execution_status == ExecutionStatus.SUCCESS
    assert prediction.p_home + prediction.p_draw + prediction.p_away == pytest.approx(1, abs=1e-8)
    assert prediction.score_matrix is not None
    buckets = derive_markets(ScoreMatrix(values=prediction.score_matrix,
                                         max_goals=prediction.metadata["score_matrix_max_goals"],
                                         retained_mass=1, tail_mass=0))["TOTAL_GOALS_0_TO_7_PLUS"]
    assert set(buckets) == {"0", "1", "2", "3", "4", "5", "6", "7+"}
    assert sum(buckets.values()) == pytest.approx(1)


def test_dynamic_score_matrix_sum() -> None:
    model = fitted_model()
    prediction = model.predict(make_match(), make_snapshot())
    assert prediction.execution_status == ExecutionStatus.SUCCESS
    assert prediction.score_matrix is not None
    assert sum(map(sum, prediction.score_matrix)) == pytest.approx(1, abs=1e-8)
    assert prediction.metadata["score_matrix_max_goals"] == model.config.max_goals
    assert prediction.metadata["score_matrix_tail_mass"] >= 0


def test_dynamic_neutral_venue_disables_home_advantage() -> None:
    model = fitted_model()
    home_fixture = make_match().model_copy(update={"neutral_venue": False})
    neutral_fixture = home_fixture.model_copy(update={"neutral_venue": True})
    home = model.predict(home_fixture, make_snapshot(home_fixture))
    neutral = model.predict(neutral_fixture, make_snapshot(neutral_fixture))
    assert home.execution_status == neutral.execution_status == ExecutionStatus.SUCCESS
    assert home.lambda_home > neutral.lambda_home
    assert home.metadata["home_advantage"] == pytest.approx(model.inference.home_advantage)
    assert neutral.metadata["home_advantage"] == 0


def test_dynamic_posterior_predictive() -> None:
    model = fitted_model()
    checks = model.inference.diagnostics.posterior_predictive
    assert checks["observed_mean_goals"] >= 0
    assert checks["posterior_mean_goals"] >= 0
    assert 0 <= checks["observed_zero_goal_rate"] <= 1
    assert 0 <= checks["posterior_high_score_rate"] <= 1


def test_dynamic_pre_match_history_does_not_contain_result_state() -> None:
    model = fitted_model(rows=24)
    rows = make_dataset(rows=24).matches
    assert len(model.state_history) == len(rows) * 2
    for row in rows:
        pre_states = [state for state in model.state_history
                      if state.team_id in {row.home_team_id, row.away_team_id}
                      and state.as_of_time == row.kickoff_time - timedelta(microseconds=1)]
        assert len(pre_states) == 2
        assert all(state.as_of_time < row.kickoff_time for state in pre_states)


def test_dynamic_state_no_future_data() -> None:
    model = fitted_model(rows=24)
    row = make_dataset(rows=24).matches[-1]
    cutoff = row.kickoff_time + timedelta(hours=1)
    state = model.inference.state_store.latest_before(row.home_team_id, cutoff)
    assert state.as_of_time <= cutoff
    assert state.as_of_time < row.available_at


def test_dynamic_state_changes_over_time() -> None:
    model = fitted_model(rows=48)
    team = TEAMS[0]
    states = [state for state in model.state_history if state.team_id == team]
    assert len(states) >= 2
    assert any(abs(state.attack_mean - states[0].attack_mean) > 1e-6 for state in states[1:])


def test_dynamic_new_team_prior() -> None:
    model = fitted_model()
    match = make_match().model_copy(update={"home_team_id": "new_team_home", "away_team_id": "new_team_away"})
    prediction = model.predict(match, make_snapshot(match))
    assert prediction.execution_status == ExecutionStatus.SUCCESS
    assert prediction.metadata["home_attack_mean"] == pytest.approx(0)
    assert prediction.metadata["away_attack_mean"] == pytest.approx(0)


def test_dynamic_season_transition_shrinks_toward_prior() -> None:
    model = fitted_model()
    row = make_dataset(rows=48).matches[-1]
    state = model.inference.state_store.latest_before(row.home_team_id, row.available_at + timedelta(days=1))
    prior_distance = abs(state.attack_mean - model.inference.prior.attack_mean)
    transitioned = model.inference.state_store.apply_season_transition(
        state, row.available_at + timedelta(days=30), "2025", model.config,
    )
    assert abs(transitioned.attack_mean - model.inference.prior.attack_mean) <= prior_distance + 1e-12
    assert transitioned.season == "2025"


def test_dynamic_inactive_team_uncertainty_grows() -> None:
    model = fitted_model()
    state = model.inference.state_store.prior("idle", START)
    later = model.inference.state_store.transition(state, START + timedelta(days=70), model.config)
    assert later.attack_sd > state.attack_sd
    assert later.defence_sd > state.defence_sd


def test_dynamic_time_index_changes_process_units() -> None:
    store = DynamicStateStore(model_id="DYNAMIC_BAYESIAN_POISSON_V1", model_version="1.0.0",
                              league_attack_mean=0, league_defence_mean=0,
                              league_attack_sd=0.35, league_defence_sd=0.35)
    state = store.prior("team", START)
    weekly = store.transition(state, START + timedelta(days=7),
                              ModelConfig(dynamic_time_index="WEEKLY", dynamic_sigma_attack=0.1))
    event = store.transition(state, START + timedelta(days=7),
                             ModelConfig(dynamic_time_index="MATCH_EVENT_TIME", dynamic_sigma_attack=0.1))
    assert weekly.attack_sd < event.attack_sd


def test_dynamic_future_training_data_rejected() -> None:
    data = make_dataset(rows=24, include_future=True)
    with pytest.raises(TrainingDataError, match="future or unavailable"):
        TrainingDatasetValidator().validate(data, data.matches[-2].kickoff_time)


def test_dynamic_outlier_robustness() -> None:
    data = make_dataset(rows=24)
    unusual = data.matches[-1].model_copy(update={"home_goals": 7, "away_goals": 0})
    changed = TrainingDataset(matches=data.matches[:-1] + (unusual,), known_team_ids=data.known_team_ids,
                              dataset_kind=data.dataset_kind)
    cutoff = unusual.kickoff_time + timedelta(days=1)
    model = CoreDynamicBayesianPoissonModel().fit(changed, cutoff, fit_config(max_score=10))
    assert max(abs(state.attack_mean) for state in model.state_history) < 3
    assert max(abs(state.defence_mean) for state in model.state_history) < 3


def test_dynamic_save_load_prediction_is_identical(tmp_path) -> None:
    model = fitted_model()
    model.save(tmp_path / "dynamic")
    loaded = CoreDynamicBayesianPoissonModel.load(tmp_path / "dynamic")
    before = model.predict(make_match(), make_snapshot())
    after = loaded.predict(make_match(), make_snapshot())
    assert after.execution_status == ExecutionStatus.SUCCESS
    assert after.score_matrix == before.score_matrix
    assert after.p_home == before.p_home


def test_dynamic_artifact_future_rejected() -> None:
    model = fitted_model()
    match = make_match(kickoff=START + timedelta(days=50)).model_copy(update={
        "as_of_time": START + timedelta(days=46),
        "retrieved_at": START + timedelta(days=46),
    })
    snapshot = make_snapshot(match)
    prediction = model.predict(match, snapshot)
    assert prediction.execution_status == ExecutionStatus.FAILED
    assert prediction.failure_code == "TRAINING_CUTOFF_AFTER_PREDICTION"
    assert "POINT_IN_TIME_GUARD_V1" in prediction.reason
    assert prediction.training_end_time == model.trained_until
    assert prediction.p_home is prediction.p_draw is prediction.p_away is None


def test_dynamic_runner() -> None:
    data = make_dataset(rows=48)
    match = make_match()
    snapshot = make_snapshot(match, report=available_report(match.match_id))
    bundle = ModelRunner().run(
        match, snapshot, data, trained_until=data.matches[-1].kickoff_time + timedelta(days=1),
        config=fit_config(dynamic_posterior_draws=100), model_ids=("DYNAMIC_BAYESIAN_POISSON_V1",),
    )
    assert bundle.predictions[0].execution_status == ExecutionStatus.SUCCESS


def test_dynamic_runner_unavailable_without_required_data() -> None:
    data = make_dataset(rows=48)
    match = make_match()
    bundle = ModelRunner().run(
        match, make_snapshot(match), data,
        trained_until=data.matches[-1].kickoff_time + timedelta(days=1),
        config=fit_config(dynamic_posterior_draws=100), model_ids=("DYNAMIC_BAYESIAN_POISSON_V1",),
    )
    assert bundle.predictions[0].execution_status == ExecutionStatus.UNAVAILABLE
    assert bundle.predictions[0].p_home is None


def test_dynamic_partial_failure_does_not_stop_sibling_model(monkeypatch) -> None:
    from erguoyuan_football.models.registry import ModelRegistry

    registry = ModelRegistry()
    original_factory = registry.get_model

    def create(model_id: str):
        model = original_factory(model_id)
        if model_id == "DYNAMIC_BAYESIAN_POISSON_V1":
            def fail_fit(data, trained_until):
                raise RuntimeError("TEST_DYNAMIC_FAILURE")

            monkeypatch.setattr(model, "_fit", fail_fit)
        return model

    monkeypatch.setattr(registry, "get_model", create)
    data = make_dataset(rows=48)
    match = make_match()
    result = ModelRunner(registry).run(
        match, make_snapshot(match, report=available_report(match.match_id)), data,
        trained_until=data.matches[-1].kickoff_time + timedelta(days=1),
        config=fit_config(dynamic_posterior_draws=80),
        model_ids=("DYNAMIC_BAYESIAN_POISSON_V1", "DIXON_COLES_V1"),
    )
    assert result.by_model()["DYNAMIC_BAYESIAN_POISSON_V1"].execution_status == ExecutionStatus.FAILED
    assert result.by_model()["DIXON_COLES_V1"].execution_status == ExecutionStatus.SUCCESS


def test_dynamic_walk_forward_uses_oos_only_predictions() -> None:
    from erguoyuan_football.backtesting.base_model_backtest import walk_forward_split

    data = make_dataset(rows=16)
    windows = walk_forward_split(data, initial_matches=10, horizon=2, step=2)
    assert windows
    for window in windows:
        train_data = data.model_copy(update={
            "matches": tuple(row for row in data.matches if row.available_at <= window.training_end),
        })
        assert all(row.kickoff_time < window.training_end for row in train_data.matches)
        assert all(row.kickoff_time > window.prediction_time for row in window.test_rows)


def test_dynamic_oos_prediction_is_accepted_by_backtest() -> None:
    from erguoyuan_football.backtesting.base_model_backtest import evaluate_oos

    data = make_dataset(rows=44)
    target = data.matches[-1]
    training = TrainingDataset(matches=data.matches[:-1], known_team_ids=data.known_team_ids,
                               dataset_kind=data.dataset_kind)
    trained_until = training.matches[-1].available_at
    prediction_time = target.kickoff_time - timedelta(minutes=30)
    fixture = Fixture(
        match_id=target.match_id, competition_id=target.competition_id,
        home_team_id=target.home_team_id, away_team_id=target.away_team_id,
        kickoff_time=target.kickoff_time, source="SYNTHETIC_TEST",
        retrieved_at=prediction_time, as_of_time=prediction_time,
        data_version=target.data_version, season=target.season, neutral_venue=target.neutral_venue,
    )
    snapshot = PredictionSnapshot(match_id=fixture.match_id, prediction_time=prediction_time,
                                  match_data_snapshot=fixture)
    model = CoreDynamicBayesianPoissonModel().fit(
        training, trained_until, fit_config(dynamic_posterior_draws=80),
    )
    prediction = model.predict(fixture, snapshot).model_copy(update={"is_oos": True})
    metrics = evaluate_oos((prediction,), (target,), competition=target.competition_id,
                           model_version=model.model_version)
    assert metrics.sample_size == 1
    assert metrics.log_loss >= 0 and metrics.brier >= 0 and metrics.rps >= 0


def test_dynamic_benchmark_report() -> None:
    from erguoyuan_football.models.comparison import comparison_report

    data = make_dataset(rows=48)
    match = make_match()
    result = ModelRunner().run(
        match, make_snapshot(match, report=available_report(match.match_id)), data,
        trained_until=data.matches[-1].kickoff_time + timedelta(days=1),
        config=fit_config(dynamic_posterior_draws=80),
        model_ids=("DIXON_COLES_V1", "BAYESIAN_HIERARCHICAL_V1", "DYNAMIC_BAYESIAN_POISSON_V1"),
    )
    report = comparison_report(match, result)
    rows = {item["model_id"]: item for item in report["models"]}
    assert set(rows) == {"DIXON_COLES_V1", "BAYESIAN_HIERARCHICAL_V1", "DYNAMIC_BAYESIAN_POISSON_V1"}
    assert rows["DYNAMIC_BAYESIAN_POISSON_V1"]["execution_status"] == "SUCCESS"


def test_dynamic_diagnostics_are_attached_to_prediction() -> None:
    model = fitted_model()
    prediction = model.predict(make_match(), make_snapshot())
    assert prediction.execution_status == ExecutionStatus.SUCCESS
    diagnostics = prediction.metadata["diagnostics"]
    assert diagnostics["posterior_draws"] == 100
    assert diagnostics["divergences"] is None
    assert diagnostics["r_hat"] is None
    assert diagnostics["ess"] is None
    assert diagnostics["health"] in {"GOOD", "WARNING"}


@pytest.mark.parametrize("team_id", [TEAMS[0], TEAMS[1]])
def test_dynamic_state_store_prior_for_unknown_team(team_id: str) -> None:
    store = DynamicStateStore(model_id="DYNAMIC_BAYESIAN_POISSON_V1", model_version="1.0.0",
                              league_attack_mean=0, league_defence_mean=0,
                              league_attack_sd=0.35, league_defence_sd=0.35)
    prior = store.latest_before(team_id, START)
    assert prior.attack_sd == pytest.approx(0.35)
    assert prior.defence_sd == pytest.approx(0.35)
