"""Deterministic SYNTHETIC_TEST history; no mocked ML probabilities."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from erguoyuan_football.data.schemas import Fixture, MatchResult
from erguoyuan_football.data.snapshots import HistoricalResult, PredictionSnapshot
from erguoyuan_football.ml.config import load_ml_config
from erguoyuan_football.ml.dataset_builder import MLDatasetBuilder
from erguoyuan_football.ml.feature_builder import MLFeatureBuilder
from erguoyuan_football.ml.feature_contract import build_feature_schema
from erguoyuan_football.ml.schemas import FeatureMode, ModelFamily
from erguoyuan_football.ml.splits import MLWalkForwardSplit


@pytest.fixture
def synthetic_history():
    start = datetime(2025, 1, 1, 18, tzinfo=UTC)
    fixtures, results, snapshots = [], {}, []
    for index in range(75):
        kickoff = start + timedelta(days=index)
        home = f"synthetic_team_{index % 4}"
        away = f"synthetic_team_{(index + 1 + index // 4 % 2) % 4}"
        if home == away:
            away = f"synthetic_team_{(index + 2) % 4}"
        fixture = Fixture(match_id=f"synthetic_ml_{index:03d}", competition_id="synthetic_league",
            home_team_id=home, away_team_id=away, kickoff_time=kickoff,
            source="SYNTHETIC_TEST", retrieved_at=kickoff - timedelta(days=7),
            as_of_time=kickoff - timedelta(days=7), data_version="synthetic-fixture-v1",
            season="2025", neutral_venue=False)
        home_goals, away_goals = ((2, 0), (1, 1), (0, 2))[index % 3]
        result = MatchResult(match_id=fixture.match_id, home_goals=home_goals,
            away_goals=away_goals, completed_at=kickoff + timedelta(hours=2),
            as_of_time=kickoff + timedelta(hours=3), retrieved_at=kickoff + timedelta(hours=3),
            source="SYNTHETIC_TEST", data_version="synthetic-result-v1")
        prediction_time = kickoff - timedelta(hours=1)
        history = tuple(HistoricalResult(fixture=past, result=results[past.match_id])
                        for past in fixtures if results[past.match_id].retrieved_at <= prediction_time)
        snapshots.append(PredictionSnapshot(match_id=fixture.match_id,
            prediction_time=prediction_time, match_data_snapshot=fixture,
            historical_results=history))
        fixtures.append(fixture)
        results[fixture.match_id] = result
    return tuple(snapshots), results


@pytest.fixture
def ml_dataset_factory(synthetic_history, project_root):
    snapshots, results = synthetic_history

    def build(family: ModelFamily = ModelFamily.XGBOOST,
              mode: FeatureMode = FeatureMode.NO_MARKET):
        builder = MLFeatureBuilder()
        vectors = builder.build_many(snapshots, {}, mode=mode, family=family, for_training=True)
        schema = build_feature_schema(mode, family)
        dataset = MLDatasetBuilder().build(vectors, results, schema=schema,
            competition_ids={snapshot.match_id: snapshot.match_data_snapshot.competition_id
                             for snapshot in snapshots},
            seasons={snapshot.match_id: "2025" for snapshot in snapshots},
            dataset_cutoff=max(result.retrieved_at for result in results.values()),
            dataset_kind="SYNTHETIC_TEST")
        split = MLWalkForwardSplit.rolling(dataset, initial_train=45,
                                          validation_size=10, test_size=10, step=10)[0]
        name = "xgboost" if family == ModelFamily.XGBOOST else "catboost"
        config = load_ml_config(project_root / f"config/{name}.yaml", family=family,
                                profile="development", allow_test_data=True)
        return dataset, split, config

    return build
