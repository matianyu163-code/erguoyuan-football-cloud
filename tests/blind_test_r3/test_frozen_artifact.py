"""Opt-in real artifact load and fit-free inference with a synthetic future fixture."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from erguoyuan_football.blind_test_r3.derivation import standardize
from erguoyuan_football.data.schemas import Fixture
from erguoyuan_football.data.snapshots import PredictionSnapshot
from erguoyuan_football.models.dixon_coles import CoreDixonColesModel


def test_frozen_artifact_loads_and_runtime_never_fits(monkeypatch):
    path = os.environ.get("R3_TEST_ARTIFACT_DIR")
    if not path:
        pytest.skip("Set R3_TEST_ARTIFACT_DIR for frozen artifact integration")

    def forbidden_fit(*_args, **_kwargs):
        raise AssertionError("RUNTIME_FIT_FORBIDDEN")

    monkeypatch.setattr(CoreDixonColesModel, "fit", forbidden_fit)
    model = CoreDixonColesModel.load(Path(path))
    assert model.fitted and model.model_id == "DIXON_COLES_V1"
    fixture = Fixture(match_id="SYNTHETIC_TEST_R3_FUTURE_MATCH",
        competition_id="SENIOR_MENS_INTERNATIONAL",
        home_team_id="NATIONAL_CYP_M_SENIOR",
        away_team_id="NATIONAL_LVA_M_SENIOR",
        kickoff_time=datetime(2027, 1, 1, tzinfo=UTC),
        source="SYNTHETIC_TEST", retrieved_at=datetime(2026, 10, 5, tzinfo=UTC),
        as_of_time=datetime(2026, 10, 5, tzinfo=UTC),
        data_version="SYNTHETIC_TEST", neutral_venue=True)
    snapshot = PredictionSnapshot(match_id=fixture.match_id,
        prediction_time=datetime(2026, 10, 5, 7, tzinfo=UTC),
        match_data_snapshot=fixture)
    result = model.predict(fixture, snapshot)
    assert result.execution_status.value == "SUCCESS"
    assert result.score_matrix is not None
    standardized = standardize(result.model_dump(mode="json"), -1)
    assert standardized["one_x_two"]["top2"]
    assert standardized["exact_score"]["top2"]
