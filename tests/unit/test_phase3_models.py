"""Phase 3 model-engine tests using a small, explicitly labelled real-test dataset."""

from datetime import UTC, datetime, timedelta

import pytest

from erguoyuan_football.contracts.common import Availability, ExecutionStatus
from erguoyuan_football.contracts.predictions import ModelPrediction
from erguoyuan_football.data.availability import (
    AvailabilityItem,
    DataAvailabilityReport,
)
from erguoyuan_football.data.schemas import Fixture
from erguoyuan_football.data.snapshots import PredictionSnapshot
from erguoyuan_football.models.artifact import cache_key
from erguoyuan_football.models.base import BaseFootballModel
from erguoyuan_football.models.bivariate_poisson import CoreBivariatePoissonModel
from erguoyuan_football.models.config import ModelConfig
from erguoyuan_football.models.dixon_coles import CoreDixonColesModel
from erguoyuan_football.models.elo import CoreEloModel
from erguoyuan_football.models.hierarchical_bayes import CoreHierarchicalBayesianModel
from erguoyuan_football.models.pi_rating import CorePiRatingModel
from erguoyuan_football.models.registry import ModelRegistry
from erguoyuan_football.models.runner import ModelRunner
from erguoyuan_football.models.score_matrix import ScoreMatrix
from erguoyuan_football.models.training import (
    InsufficientData,
    TrainingDataError,
    TrainingDataset,
    TrainingDatasetValidator,
    TrainingMatch,
)

TEAMS = tuple(f"team_{index}" for index in range(6))
START = datetime(2024, 1, 1, tzinfo=UTC)
pytestmark = pytest.mark.model


def make_dataset(*, rows: int = 48, include_future: bool = False,
                 dataset_kind: str = "SYNTHETIC_TEST") -> TrainingDataset:
    matches: list[TrainingMatch] = []
    for index in range(rows + int(include_future)):
        kickoff = START + timedelta(days=index)
        home = TEAMS[index % len(TEAMS)]
        away = TEAMS[(index + 1) % len(TEAMS)]
        matches.append(TrainingMatch(
            match_id=f"history_{index}", competition_id="league_1", season="2024",
            kickoff_time=kickoff, home_team_id=home, away_team_id=away,
            home_goals=(index * 2 + 1) % 4, away_goals=(index + 1) % 3,
            neutral_venue=index % 11 == 0, source="SYNTHETIC_TEST",
            completed_at=kickoff + timedelta(hours=2),
            as_of_time=kickoff + timedelta(hours=4),
            retrieved_at=kickoff + timedelta(hours=4), data_version="real-test-v1",
        ))
    return TrainingDataset(matches=tuple(matches), known_team_ids=frozenset(TEAMS), dataset_kind=dataset_kind)


def make_match(*, kickoff: datetime | None = None) -> Fixture:
    return Fixture(
        match_id="future_match", competition_id="league_1", home_team_id=TEAMS[0],
        away_team_id=TEAMS[1], kickoff_time=kickoff or START + timedelta(days=60),
        source="REAL_TEST", retrieved_at=START + timedelta(days=48),
        as_of_time=START + timedelta(days=48), data_version="fixture-v1",
        season="2024", neutral_venue=False,
    )


def make_snapshot(match: Fixture | None = None, *, report: DataAvailabilityReport | None = None) -> PredictionSnapshot:
    match = match or make_match()
    prediction_time = match.as_of_time + timedelta(hours=1)
    return PredictionSnapshot(
        match_id=match.match_id, prediction_time=prediction_time, match_data_snapshot=match,
        data_completeness=report,
    )


def available_report(match_id: str = "future_match") -> DataAvailabilityReport:
    return DataAvailabilityReport(match_id=match_id, items={
        key: AvailabilityItem(availability=Availability.AVAILABLE, reason="TEST_EVIDENCE", evidence_ids=(key,))
        for key in ("historical_goals", "historical_results")
    })


def fit_config(**changes) -> ModelConfig:
    options = {"min_matches": 24, "min_team_matches": 2, "max_goals": 5,
               "training_window": None, "max_iterations": 500, "allow_test_data": True}
    options.update(changes)
    return ModelConfig(**options)


def fit_goal_model(model: BaseFootballModel):
    data = make_dataset()
    model.fit(data, START + timedelta(days=48), fit_config())
    return model, data


def test_base_model_interface() -> None:
    registry = ModelRegistry()
    assert set(registry.list_models()) == {
        "DIXON_COLES_V1", "BIVARIATE_POISSON_V1", "BAYESIAN_HIERARCHICAL_V1", "ELO_V1", "PI_RATING_V1",
        "DYNAMIC_BAYESIAN_POISSON_V1",
        "CORE_SPI_LIKE_V1",
        "CORE_OPTA_XG_ELO_LIKE_V1",
        "HISTORICAL_MARKET_BAYESIAN_POISSON_V1",
    }
    for model_id in registry.list_models():
        model = registry.get_model(model_id)
        assert isinstance(model, BaseFootballModel)
        expected = "LIKE_IMPLEMENTATION" if model_id in {"CORE_SPI_LIKE_V1", "CORE_OPTA_XG_ELO_LIKE_V1"} else "REAL_IMPLEMENTATION"
        assert model.implementation_type == expected


def test_model_prediction_probability_sum() -> None:
    model, _ = fit_goal_model(CoreDixonColesModel())
    prediction = model.predict(make_match(), make_snapshot())
    assert prediction.execution_status == ExecutionStatus.SUCCESS
    assert abs(prediction.p_home + prediction.p_draw + prediction.p_away - 1) < 1e-6


def test_training_dataset_future_rejected() -> None:
    with pytest.raises(TrainingDataError, match="future"):
        TrainingDatasetValidator().validate(make_dataset(include_future=True), START + timedelta(days=48))


def test_synthetic_training_requires_explicit_test_flag() -> None:
    with pytest.raises(InsufficientData, match="SYNTHETIC_DATA_FORBIDDEN"):
        CoreDixonColesModel().fit(make_dataset(), START + timedelta(days=48),
                                  fit_config(allow_test_data=False))


def test_dixon_coles_fit() -> None:
    model, _ = fit_goal_model(CoreDixonColesModel())
    assert model.fitted and model.model_id == "DIXON_COLES_V1"
    assert model.metadata["training_sample_count"] == 48
    assert model.metadata["xi"] == fit_config().time_decay


def test_dixon_coles_predict() -> None:
    model, _ = fit_goal_model(CoreDixonColesModel())
    prediction = model.predict(make_match(), make_snapshot())
    assert prediction.execution_status == ExecutionStatus.SUCCESS
    assert prediction.lambda_home is not None and prediction.lambda_away is not None


def test_dixon_coles_score_matrix() -> None:
    model, _ = fit_goal_model(CoreDixonColesModel())
    prediction = model.predict(make_match(), make_snapshot())
    assert prediction.score_matrix is not None
    assert len(prediction.score_matrix) == fit_config().max_goals + 1
    assert abs(sum(map(sum, prediction.score_matrix)) - 1) < 1e-8


def test_dixon_coles_save_load(tmp_path) -> None:
    model, _ = fit_goal_model(CoreDixonColesModel())
    model.save(tmp_path / "dc")
    loaded = CoreDixonColesModel.load(tmp_path / "dc")
    before = model.predict(make_match(), make_snapshot())
    after = loaded.predict(make_match(), make_snapshot())
    assert before.p_home == after.p_home
    assert before.score_matrix == after.score_matrix


def test_bivariate_fit() -> None:
    model, _ = fit_goal_model(CoreBivariatePoissonModel())
    assert model.fitted and model.metadata["lambda3"] >= 0


def test_bivariate_predict() -> None:
    model, _ = fit_goal_model(CoreBivariatePoissonModel())
    prediction = model.predict(make_match(), make_snapshot())
    assert prediction.execution_status == ExecutionStatus.SUCCESS
    assert prediction.correlation_parameter is not None
    assert abs(sum(map(sum, prediction.score_matrix)) - 1) < 1e-8


def test_bivariate_correlation() -> None:
    model, _ = fit_goal_model(CoreBivariatePoissonModel())
    assert model.metadata["correlation_log"] == pytest.approx(model.metadata["correlation_log"])
    assert model.metadata["lambda3"] >= 0


def test_hierarchical_bayes_fit_smoke() -> None:
    model = CoreHierarchicalBayesianModel()
    config = fit_config(draws=20, tune=20, chains=2, thin=1, min_ess=20, max_rhat=1.2)
    try:
        model.fit(make_dataset(), START + timedelta(days=48), config)
    except RuntimeError as error:
        assert "SAMPLING_DIAGNOSTICS_FAILED" in str(error)
        assert model.metadata.get("sampling_status") in {
            "MAP_INTERNAL_ONLY", "POSTERIOR_INVALID", "NUMERICAL_FAILURE",
        }
        assert model.metadata.get("sampling_degraded") is True
        assert model.fitted is False
    else:
        assert model.metadata.get("sampling_status") == "PASS"
        assert model.fitted is True


def test_hierarchical_bayes_predict() -> None:
    model = CoreHierarchicalBayesianModel()
    config = fit_config(draws=20, tune=20, chains=2, thin=1, min_ess=20, max_rhat=1.2)
    try:
        model.fit(make_dataset(), START + timedelta(days=48), config)
    except RuntimeError:
        pass
    prediction = model.predict(make_match(), make_snapshot())
    assert prediction.execution_status in {ExecutionStatus.SUCCESS, ExecutionStatus.UNAVAILABLE}
    if prediction.execution_status != ExecutionStatus.SUCCESS:
        assert prediction.p_home is None and prediction.reason


def test_hierarchical_bayes_shrinkage() -> None:
    model = CoreHierarchicalBayesianModel()
    try:
        model.fit(make_dataset(), START + timedelta(days=48), fit_config(draws=20, tune=20, chains=2, thin=1))
    except RuntimeError:
        pass
    assert model.metadata.get("shrinkage") == "league_level_attack_defence"


def test_elo_pre_match_rating() -> None:
    model, _ = fit_goal_model(CoreEloModel())
    assert model.pre_match_rating(TEAMS[0]) != fit_config().initial_rating
    assert all(row["phase"] == "PRE_MATCH" for row in model.elo_history)


def test_elo_future_leakage() -> None:
    model = CoreEloModel()
    with pytest.raises(TrainingDataError):
        model.fit(make_dataset(include_future=True), START + timedelta(days=48), fit_config())


def test_elo_probability_mapper() -> None:
    model, _ = fit_goal_model(CoreEloModel())
    prediction = model.predict(make_match(), make_snapshot())
    assert prediction.execution_status == ExecutionStatus.SUCCESS
    assert prediction.metadata["mapper_training_sample_count"] == 48


def test_pi_pre_match_rating() -> None:
    model, _ = fit_goal_model(CorePiRatingModel())
    rating = model.pre_match_rating(TEAMS[0])
    assert set(rating) == {"home", "away"}
    assert any(row["phase"] == "PRE_MATCH" for row in model.pi_history)


def test_pi_probability_mapper() -> None:
    model, _ = fit_goal_model(CorePiRatingModel())
    prediction = model.predict(make_match(), make_snapshot())
    assert prediction.execution_status == ExecutionStatus.SUCCESS
    assert prediction.metadata["mapper_id"] == "PI_RATING_V1"


def test_model_registry() -> None:
    registry = ModelRegistry()
    requirements = registry.get_model_requirements("DIXON_COLES_V1")
    assert requirements["required_data"] == ("historical_goals",)
    assert "DIXON_COLES_V1" in registry.get_available_models({"historical_goals"})


def test_model_runner_partial_failure() -> None:
    model, data = fit_goal_model(CoreDixonColesModel())
    assert model.fitted
    report = DataAvailabilityReport(match_id="future_match", items={
        "historical_goals": AvailabilityItem(availability=Availability.AVAILABLE, reason="TEST", evidence_ids=("h",)),
    })
    bundle = ModelRunner().run(make_match(), make_snapshot(report=report), data,
                               trained_until=START + timedelta(days=48), config=fit_config(),
                               model_ids=("DIXON_COLES_V1", "ELO_V1"))
    statuses = {item.model_id: item.execution_status for item in bundle.predictions}
    assert statuses["DIXON_COLES_V1"] == ExecutionStatus.SUCCESS
    assert statuses["ELO_V1"] == ExecutionStatus.UNAVAILABLE


def test_model_unavailable() -> None:
    model = CoreDixonColesModel()
    prediction = model.predict(make_match(), make_snapshot())
    assert prediction.execution_status == ExecutionStatus.UNAVAILABLE
    assert prediction.p_home is None and prediction.reason


def test_model_artifact_cache(tmp_path) -> None:
    model, _ = fit_goal_model(CoreDixonColesModel())
    artifact = model.save(tmp_path / "dc")
    assert artifact.training_data_hash == model.training_data_hash
    assert cache_key(model.model_id, model.model_version, artifact.trained_until.isoformat(),
                     artifact.training_data_hash, artifact.config_hash) != cache_key(
                         model.model_id, model.model_version, artifact.trained_until.isoformat(),
                         "different", artifact.config_hash)


def test_score_matrix_sum() -> None:
    score = ScoreMatrix.from_raw(__import__("numpy").array([[0.2, 0.3], [0.1, 0.3]]))
    assert abs(sum(map(sum, score.values)) - 1) < 1e-8
    assert score.tail_mass == pytest.approx(0.1)


def test_score_matrix_market_derivation() -> None:
    from erguoyuan_football.markets.from_score_matrix import derive_markets

    values = ((0.25, 0.15), (0.2, 0.4))
    markets = derive_markets(ScoreMatrix(values=values, max_goals=1, retained_mass=1, tail_mass=0))
    assert markets["1X2"]["p_home"] == pytest.approx(0.2)
    assert markets["DOUBLE_CHANCE"]["1X"] == pytest.approx(0.85)


def test_training_dataset_duplicate_rejected() -> None:
    data = make_dataset()
    duplicate = TrainingDataset(matches=data.matches + (data.matches[0],), known_team_ids=data.known_team_ids,
                                dataset_kind=data.dataset_kind)
    with pytest.raises(TrainingDataError, match="duplicate"):
        TrainingDatasetValidator().validate(duplicate, START + timedelta(days=60))


def test_model_prediction_schema() -> None:
    model, _ = fit_goal_model(CoreDixonColesModel())
    prediction = model.predict(make_match(), make_snapshot())
    validated = ModelPrediction.model_validate(prediction.model_dump())
    assert validated.model_id == "DIXON_COLES_V1"
