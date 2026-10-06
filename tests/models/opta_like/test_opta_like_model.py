"""Opta-like rating tests explicitly identify all estimates as CORE implementations."""

from datetime import timedelta

import numpy as np
import pytest

from erguoyuan_football.contracts.common import ExecutionStatus
from erguoyuan_football.models.dynamic_bayes.likelihood import score_matrix_from_lambdas
from erguoyuan_football.models.opta_like.model import (
    CoreOptaXGEloLikeModel,
    CorePowerTransformer,
    HierarchyDeltaAllocation,
    OptaLikeConfig,
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


def fit_like(*, xg: bool = False, hierarchy: bool = False) -> CoreOptaXGEloLikeModel:
    data = make_dataset(rows=48)
    update = {}
    if xg:
        update["xg_observations"] = tuple(TrainingXGObservation(
            match_id=row.match_id, xg_home=1.4, xg_away=0.7, xg_source_id="fixture-xg",
            provider="PUBLIC_WEB_FIXTURE", model_name="fixed-test-xg", model_version="v1",
            retrieved_at=row.available_at, as_of_time=row.available_at, data_hash=f"hash-{row.match_id}")
            for row in data.matches)
    if hierarchy:
        teams = sorted(data.known_team_ids)
        update["team_hierarchy_timeline"] = tuple(TeamHierarchyMembership(
            team_id=team, league_id=f"league_{index % 4}", country_id=f"country_{index % 2}",
            continent_id="continent_a" if index < 4 else "continent_b",
            valid_from=START - timedelta(days=10), source="SYNTHETIC_TEST",
            retrieved_at=START - timedelta(days=9), as_of_time=START - timedelta(days=9),
            data_version="hierarchy-test-v1") for index, team in enumerate(teams))
    data = data.model_copy(update=update)
    cutoff = data.matches[-1].kickoff_time + timedelta(days=1)
    return CoreOptaXGEloLikeModel().fit(data, cutoff, fit_config())


def test_opta_like_interface_and_identity() -> None:
    model = fit_like()
    result = model.predict(make_match(), make_snapshot())
    assert result.execution_status == ExecutionStatus.SUCCESS
    assert result.implementation_type == "LIKE_IMPLEMENTATION"
    assert result.metadata["official_model"] == "NOT_OFFICIAL_OPTA_MODEL"
    assert result.metadata["xg_mode"] == "WITHOUT_XG_RESULT_ONLY"


def test_team_layer_updates_every_match() -> None:
    model = fit_like()
    assert model.metadata["hierarchy_update_counts"]["TEAM"] == 48
    assert len(model.team_pre_match_history) == 96
    assert all(item["phase"] == "PRE_MATCH" and item["as_of_time"] < START + timedelta(days=48)
               for item in model.team_pre_match_history if item["match_id"] == "history_47")


def test_highest_affected_hierarchy_level_and_no_double_update() -> None:
    assert HierarchyDeltaAllocation.affected_level(None, None) is None
    model = fit_like(hierarchy=True)
    assert all(row["highest_affected_level"] in {"LEAGUE", "COUNTRY", "CONTINENT"}
               for row in model.hierarchy_history)
    assert all(len([key for key in ("LEAGUE", "COUNTRY", "CONTINENT") if key == row["highest_affected_level"]]) == 1
               for row in model.hierarchy_history)


def test_hierarchy_level_priority() -> None:
    teams = [TeamHierarchyMembership(team_id=f"t{i}", league_id=league, country_id=country,
        continent_id=continent, valid_from=START, source="TEST", retrieved_at=START,
        as_of_time=START, data_version="v1") for i, (league, country, continent) in enumerate((
            ("l1", "c1", "x"), ("l2", "c1", "x"), ("l3", "c2", "x"), ("l4", "c3", "y")))]
    assert HierarchyDeltaAllocation.affected_level(teams[0], teams[0]) is None
    assert HierarchyDeltaAllocation.affected_level(teams[0], teams[1]) == "LEAGUE"
    assert HierarchyDeltaAllocation.affected_level(teams[0], teams[2]) == "COUNTRY"
    assert HierarchyDeltaAllocation.affected_level(teams[0], teams[3]) == "CONTINENT"


def test_result_elo_is_deterministic_and_positive() -> None:
    model = fit_like()
    first = model.predict(make_match(), make_snapshot())
    second = model.predict(make_match(), make_snapshot())
    assert (first.p_home, first.p_draw, first.p_away) == pytest.approx(
        (second.p_home, second.p_draw, second.p_away))
    assert all(np.isfinite(value) for value in model.team_rating.values())


def test_xg_score_distribution_tail_is_accounted() -> None:
    matrix = score_matrix_from_lambdas(np.asarray([1.7]), np.asarray([0.9]), 8)
    assert sum(map(sum, matrix.values)) == pytest.approx(1)
    assert matrix.retained_mass + matrix.tail_mass == pytest.approx(1)
    assert matrix.tail_mass >= 0


def test_xg_elo_delta_uses_score_distribution() -> None:
    model = fit_like()
    delta = model._xg_delta(1.7, 0.7, difference=0, neutral=False)
    assert np.isfinite(delta)
    assert delta > 0


def test_80_20_reference_weights_and_xg_provenance() -> None:
    model = fit_like(xg=True)
    assert model.metadata["xg_weight_reference"] == [0.8, 0.2]
    assert model.metadata["xg_rows_used"] == 48
    assert model.metadata["xg_sources"] == ["PUBLIC_WEB_FIXTURE"]
    assert model.metadata["xg_mode"] == "WITH_XG"


def test_missing_xg_uses_result_only() -> None:
    model = fit_like()
    result = model.predict(make_match(), make_snapshot())
    assert model.metadata["xg_mode"] == "WITHOUT_XG_RESULT_ONLY"
    assert result.metadata["xg_mode"] == "WITHOUT_XG_RESULT_ONLY"


def test_xg_provenance_is_required() -> None:
    with pytest.raises(ValueError):
        TrainingXGObservation(match_id="m", xg_home=1, xg_away=1, xg_source_id="UNKNOWN",
            provider="UNKNOWN", model_name="UNKNOWN", model_version="UNKNOWN", retrieved_at=START,
            as_of_time=START, data_hash="UNKNOWN")


def test_core_power_transform_is_not_probability() -> None:
    values = [CorePowerTransformer.transform(100, [50, 100, 150]),
              CorePowerTransformer.transform(150, [50, 100, 150])]
    assert values == pytest.approx([50, 100])
    assert CorePowerTransformer.method == "CORE_ESTIMATED"


def test_mapper_trained_cutoff_and_predict_many() -> None:
    model = fit_like()
    assert model.mapper.trained_until == model.trained_until
    match = make_match()
    predictions = model.predict_many([match, match], [make_snapshot(match), make_snapshot(match)])
    assert len(predictions) == 2
    assert all(item.execution_status == ExecutionStatus.SUCCESS for item in predictions)


def test_opta_like_save_load(tmp_path) -> None:
    model = fit_like()
    match = make_match()
    before = model.predict(match, make_snapshot(match))
    model.save(tmp_path / "opta-like")
    loaded = CoreOptaXGEloLikeModel.load(tmp_path / "opta-like")
    after = loaded.predict(match, make_snapshot(match))
    assert after.p_home == pytest.approx(before.p_home)


def test_opta_like_config_weight_validation() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        OptaLikeConfig(result_weight=0.8, xg_weight=0.1)
