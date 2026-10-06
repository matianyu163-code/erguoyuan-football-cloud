"""Execution isolation, immutable evidence ledger, and 39-match model rules."""

from __future__ import annotations

from datetime import timedelta

from erguoyuan_football.contracts.predictions import ProbabilityVector
from erguoyuan_football.prediction.execution_records import ModelExecutionStore
from erguoyuan_football.prediction.model_execution_engine import (
    ModelExecutionEngine,
    ModelExecutionResult,
)
from erguoyuan_football.prediction.model_execution_planner import ModelExecutionPlanner
from erguoyuan_football.prediction.model_input_adapter import EloInputAdapter
from erguoyuan_football.research.live_data.model_readiness import (
    ModelReadinessEvaluator,
)
from erguoyuan_football.research.live_data.model_requirements import (
    ModelDataRequirement,
    ModelSamplePolicy,
    RequirementLevel,
)
from erguoyuan_football.research.live_data.readiness import (
    DataAvailability,
    DataReadinessItem,
)
from erguoyuan_football.research.samples.sample_builder import SampleBuilder
from erguoyuan_football.research.samples.sample_repository import SampleRepository
from tests.phase14.conftest import (
    AWAY,
    COMPETITION,
    HOME,
    ROOT,
    START,
    make_readiness,
    make_samples,
)


def test_blocked_models_are_never_called_and_are_audited(sample_context, tmp_path) -> None:
    repository, hierarchy, cutoff = sample_context
    planner = ModelExecutionPlanner(ROOT / "config/phase14_model_requirements.yaml",
                                   ROOT / "config/model_registry.yaml")
    prediction_time = cutoff + timedelta(days=2)
    report = make_readiness(ready_models=("ELO_V1",))
    plan = planner.plan(report, prediction_time=prediction_time,
                        training_cutoff=cutoff)
    bundle = EloInputAdapter().build_input(repository, hierarchy, cutoff)
    calls: list[str] = []

    def executor(entry, model_input):
        calls.append(entry.model_id)
        return ModelExecutionResult(entry.model_id, "EXECUTED",
            ProbabilityVector(p_home=0.4, p_draw=0.3, p_away=0.3),
            {"source": "SYNTHETIC_TEST"}, "MEDIUM", "PENDING")

    store = ModelExecutionStore(tmp_path / "execution.sqlite")
    try:
        results = ModelExecutionEngine(store).execute(plan,
            {"ELO_V1": bundle}, {"ELO_V1": executor}, execute_ready_models=True)
        assert calls == ["ELO_V1"]
        assert {row.model_name: row.status for row in results}["DIXON_COLES_V1"] == "BLOCKED"
        saved = store.records(plan_id=plan.plan_id)
        assert len(saved) == 11
        assert sum(row["status"] == "BLOCKED" for row in saved) == 10
        elo_record = next(row for row in saved if row["model_name"] == "ELO_V1")
        assert elo_record["output_valid"]
        assert elo_record["output_probabilities"] == {
            "home": 0.4, "draw": 0.3, "away": 0.3,
        }
        assert elo_record["calibrated"] is False
    finally:
        store.close()


def test_39_matches_can_be_ready_for_one_model_and_blocked_for_another() -> None:
    rows = make_samples(count=39)
    cutoff = START + timedelta(days=90)
    repository = SampleRepository(rows)
    hierarchy = SampleBuilder(repository).build(HOME, AWAY, COMPETITION,
        cutoff=cutoff, competition_type="LEAGUE")
    items = {
        "HISTORICAL_RESULTS": DataReadinessItem("HISTORICAL_RESULTS",
            DataAvailability.AVAILABLE, evidence_ids=("SYNTHETIC_TEST_HISTORY",)),
        "MATCH_TIMESTAMPS": DataReadinessItem("MATCH_TIMESTAMPS",
            DataAvailability.AVAILABLE, evidence_ids=("SYNTHETIC_TEST_TIME",)),
    }
    dc = ModelDataRequirement("DIXON_COLES_V1",
        {"HISTORICAL_RESULTS": RequirementLevel.REQUIRED},
        ModelSamplePolicy(40, 3, 0, False, min_training_team_matches=3))
    dynamic = ModelDataRequirement("DYNAMIC_BAYESIAN_POISSON_V1",
        {"HISTORICAL_RESULTS": RequirementLevel.REQUIRED,
         "MATCH_TIMESTAMPS": RequirementLevel.REQUIRED},
        ModelSamplePolicy(20, 0, 20, False, min_training_team_matches=3))
    evaluator = ModelReadinessEvaluator()
    dc_result = evaluator.evaluate(dc, items, hierarchy, fixture_verified=True)
    dynamic_result = evaluator.evaluate(dynamic, items, hierarchy, fixture_verified=True)
    assert dc_result.ready is False
    assert dynamic_result.ready is True


def test_extreme_small_sample_remains_blocked() -> None:
    rows = make_samples(count=4)
    cutoff = START + timedelta(days=90)
    hierarchy = SampleBuilder(SampleRepository(rows)).build(
        HOME, AWAY, COMPETITION, cutoff=cutoff, competition_type="LEAGUE")
    items = {"HISTORICAL_RESULTS": DataReadinessItem("HISTORICAL_RESULTS",
        DataAvailability.AVAILABLE, evidence_ids=("SYNTHETIC_TEST_HISTORY",))}
    requirement = ModelDataRequirement("DIXON_COLES_V1",
        {"HISTORICAL_RESULTS": RequirementLevel.REQUIRED},
        ModelSamplePolicy(40, 3, 20, False, min_training_team_matches=3))
    result = ModelReadinessEvaluator().evaluate(requirement, items, hierarchy,
                                                 fixture_verified=True)
    assert result.ready is False
    assert result.uncertainty_level == "VERY_HIGH"
