"""Integration: real existing model code consumes adapter-produced inputs."""

from __future__ import annotations

from datetime import timedelta

import pytest

from erguoyuan_football.contracts.common import Availability
from erguoyuan_football.data.availability import (
    AvailabilityItem,
    DataAvailabilityReport,
)
from erguoyuan_football.data.schemas import Fixture
from erguoyuan_football.data.snapshots import PredictionSnapshot
from erguoyuan_football.models.config import ModelConfig
from erguoyuan_football.prediction.execution_records import ModelExecutionStore
from erguoyuan_football.prediction.existing_model_executor import ExistingModelExecutor
from erguoyuan_football.prediction.model_execution_engine import ModelExecutionEngine
from erguoyuan_football.prediction.model_execution_planner import (
    ModelExecutionPlanner,
)
from erguoyuan_football.prediction.model_input_adapter import (
    default_model_input_adapters,
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

pytestmark = pytest.mark.integration


def test_hierarchical_elo_and_goal_models_run_from_sample_bundles(tmp_path) -> None:
    rows = make_samples()
    training_cutoff = START + timedelta(days=90)
    repository = SampleRepository(rows)
    hierarchy = SampleBuilder(repository).build(
        HOME, AWAY, COMPETITION, cutoff=training_cutoff,
        competition_type="LEAGUE")
    prediction_time = training_cutoff + timedelta(hours=1)
    kickoff = training_cutoff + timedelta(days=12)
    fixture_match = Fixture(
        match_id="SYN_TARGET", competition_id="SYN_COMP", home_team_id="SYN_T0",
        away_team_id="SYN_T1", kickoff_time=kickoff, source="SYNTHETIC_TEST",
        retrieved_at=prediction_time - timedelta(minutes=30),
        as_of_time=prediction_time - timedelta(minutes=30), data_version="SYNTHETIC_TEST_V1",
        season="2025", neutral_venue=False,
    )
    evidence = ("SYNTHETIC_TEST_HISTORY",)
    availability = DataAvailabilityReport(match_id="SYN_TARGET", items={
        "historical_goals": AvailabilityItem(availability=Availability.AVAILABLE,
            reason="SYNTHETIC_TEST", evidence_ids=evidence),
        "historical_results": AvailabilityItem(availability=Availability.AVAILABLE,
            reason="SYNTHETIC_TEST", evidence_ids=evidence),
    })
    snapshot = PredictionSnapshot(match_id="SYN_TARGET", prediction_time=prediction_time,
        match_data_snapshot=fixture_match, data_completeness=availability)
    planner = ModelExecutionPlanner(ROOT / "config/phase14_model_requirements.yaml",
                                   ROOT / "config/model_registry.yaml")
    model_ids = ("BAYESIAN_HIERARCHICAL_V1", "ELO_V1", "DIXON_COLES_V1")
    ready_report = make_readiness(ready_models=model_ids)
    plan = planner.plan(ready_report, prediction_time=prediction_time,
                        training_cutoff=training_cutoff)
    adapters = default_model_input_adapters()
    bundles = {model_id: adapters[model_id].build_input(repository, hierarchy,
        training_cutoff) for model_id in model_ids}
    config = ModelConfig(min_matches=24, min_team_matches=2, max_goals=5,
        training_window=None, max_iterations=500, allow_test_data=True,
        draws=1000, tune=1000, chains=4, min_ess=100, max_rhat=1.2)
    executor = ExistingModelExecutor(fixture_match, snapshot, training_cutoff, config)
    store = ModelExecutionStore(tmp_path / "real_execution.sqlite")
    try:
        outputs = ModelExecutionEngine(store).execute(plan, bundles,
            {model_id: executor for model_id in model_ids}, execute_ready_models=True)
        selected = {row.model_name: row for row in outputs if row.model_name in model_ids}
        assert set(selected) == set(model_ids)
        assert all(row.status == "EXECUTED" and row.probabilities is not None
                   for row in selected.values())
        assert all(abs(row.probabilities.p_home + row.probabilities.p_draw
                       + row.probabilities.p_away - 1) < 1e-8
                   for row in selected.values())
        records = {row["model_name"]: row for row in store.records(plan_id=plan.plan_id)}
        assert all(records[model_id]["output_valid"] for model_id in model_ids)
        assert records["BAYESIAN_HIERARCHICAL_V1"]["sample_counts"]["training"] == len(
            bundles["BAYESIAN_HIERARCHICAL_V1"].training_dataset.matches)
        assert records["BAYESIAN_HIERARCHICAL_V1"]["prior_strength"] is not None
        bayesian = selected["BAYESIAN_HIERARCHICAL_V1"].raw_output["metadata"]
        assert bayesian["posterior_rate_pooling"]["prior_sample_count"] > 0
        assert bayesian["posterior_rate_pooling"]["direct_sample_count"] == len(
            bundles["BAYESIAN_HIERARCHICAL_V1"].direct_samples)
        assert records["BAYESIAN_HIERARCHICAL_V1"]["input_evidence_ids"]
    finally:
        store.close()
