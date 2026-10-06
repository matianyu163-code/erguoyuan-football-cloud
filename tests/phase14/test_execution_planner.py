"""Dry-run planning covers per-model readiness and fit/inference boundaries."""

from __future__ import annotations

from datetime import timedelta

import pytest

from erguoyuan_football.prediction.model_execution_planner import ModelExecutionPlanner
from tests.phase14.conftest import ROOT, START, make_readiness


def test_plan_runs_ready_models_and_blocks_unready_ones() -> None:
    planner = ModelExecutionPlanner(ROOT / "config/phase14_model_requirements.yaml",
                                    ROOT / "config/model_registry.yaml")
    prediction = START + timedelta(days=60)
    plan = planner.plan(make_readiness(), prediction_time=prediction,
                        training_cutoff=prediction - timedelta(days=1))
    entries = {row.model_id: row for row in plan.entries}
    assert entries["ELO_V1"].action == "RUN"
    assert entries["ELO_V1"].mode == "FIT_AND_PREDICT"
    assert entries["DIXON_COLES_V1"].action == "RUN"
    assert entries["CORE_XGBOOST_V1"].action == "BLOCK"
    assert "VALID_ARTIFACT_REQUIRED" in entries["CORE_XGBOOST_V1"].reasons
    assert plan.dry_run is True


def test_valid_artifact_bypasses_fit_gate_but_keeps_inference_gate() -> None:
    planner = ModelExecutionPlanner(ROOT / "config/phase14_model_requirements.yaml",
                                    ROOT / "config/model_registry.yaml")
    prediction = START + timedelta(days=60)
    report = make_readiness(ready_models=())
    plan = planner.plan(report, prediction_time=prediction,
                        training_cutoff=prediction - timedelta(days=1),
                        valid_artifacts=frozenset({"CORE_XGBOOST_V1"}))
    entry = {row.model_id: row for row in plan.entries}["CORE_XGBOOST_V1"]
    assert entry.mode == "PREDICT_ARTIFACT"
    assert entry.action == "RUN"
    missing = make_readiness(ready_models=(), data_available=False)
    plan_missing = planner.plan(missing, prediction_time=prediction,
        training_cutoff=prediction - timedelta(days=1),
        valid_artifacts=frozenset({"CORE_XGBOOST_V1"}))
    entry_missing = {row.model_id: row for row in plan_missing.entries}["CORE_XGBOOST_V1"]
    assert entry_missing.action == "BLOCK"
    assert any(reason.startswith("INFERENCE_INPUT_UNAVAILABLE")
               for reason in entry_missing.reasons)


def test_training_cutoff_after_prediction_blocks_every_model() -> None:
    planner = ModelExecutionPlanner(ROOT / "config/phase14_model_requirements.yaml",
                                    ROOT / "config/model_registry.yaml")
    prediction = START + timedelta(days=60)
    plan = planner.plan(make_readiness(), prediction_time=prediction,
                        training_cutoff=prediction + timedelta(seconds=1))
    assert all(row.action == "BLOCK" for row in plan.entries)
    assert all("TRAINING_CUTOFF_AFTER_PREDICTION" in row.reasons for row in plan.entries)


def test_unknown_neutral_venue_blocks_inference() -> None:
    planner = ModelExecutionPlanner(ROOT / "config/phase14_model_requirements.yaml",
                                    ROOT / "config/model_registry.yaml")
    prediction = START + timedelta(days=60)
    plan = planner.plan(make_readiness(), prediction_time=prediction,
        training_cutoff=prediction - timedelta(days=1), neutral_venue_known=False)
    assert all(row.action == "BLOCK" for row in plan.entries)
    assert all("INFERENCE_CONTEXT_MISSING:NEUTRAL_VENUE" in row.reasons
               for row in plan.entries)


def test_planner_rejects_naive_timestamps() -> None:
    planner = ModelExecutionPlanner(ROOT / "config/phase14_model_requirements.yaml",
                                    ROOT / "config/model_registry.yaml")
    with pytest.raises(ValueError, match="UTC_CUTOFF_REQUIRED"):
        planner.plan(make_readiness(), prediction_time=START.replace(tzinfo=None),
                     training_cutoff=START)
