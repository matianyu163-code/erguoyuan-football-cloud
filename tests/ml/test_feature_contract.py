"""Fixed schema, immutable hashes, temporal lineage and missing-value tests."""

import math

import pytest

from erguoyuan_football.ml.dataset_builder import MLDatasetBuilder, make_dataset
from erguoyuan_football.ml.feature_builder import MLFeatureBuilder
from erguoyuan_football.ml.feature_contract import build_feature_schema
from erguoyuan_football.ml.feature_store import MLFeatureStore
from erguoyuan_football.ml.missing import feature_frame
from erguoyuan_football.ml.schemas import FeatureMode, ModelFamily

pytestmark = pytest.mark.fast


def test_feature_schema_and_order(ml_dataset_factory) -> None:
    dataset, _, _ = ml_dataset_factory()
    assert dataset.feature_schema.feature_names[0] == "horizon_seconds"
    assert tuple(dataset.rows[0].vector.features) == dataset.feature_schema.feature_names
    assert dataset.feature_schema.schema_hash == dataset.manifest.feature_schema_hash
    assert "market_available" not in dataset.feature_schema.feature_names
    assert "home_team_id" not in dataset.feature_schema.feature_names


def test_feature_hash_and_point_in_time(synthetic_history) -> None:
    snapshots, _ = synthetic_history
    vector = MLFeatureBuilder().build(snapshots[-1], (), mode=FeatureMode.NO_MARKET,
                                      family=ModelFamily.XGBOOST, for_training=True)
    assert vector.prediction_time < vector.kickoff_time
    assert all(max(item.as_of_time, item.retrieved_at) <= vector.prediction_time
               for item in vector.feature_lineage)
    with pytest.raises(ValueError, match="FEATURE_DATA_HASH_MISMATCH"):
        vector.model_copy(update={"feature_data_hash": "wrong"}).model_validate(
            vector.model_copy(update={"feature_data_hash": "wrong"}).model_dump())


def test_missing_not_zero_and_indicator(ml_dataset_factory) -> None:
    dataset, _, _ = ml_dataset_factory()
    first = dataset.rows[0].vector
    assert first.features["dc_p_home"] is None
    assert first.features["dc_available"] == 0.0
    frame = feature_frame((first,), dataset.feature_schema)
    assert math.isnan(frame.loc[0, "dc_p_home"])
    assert frame.loc[0, "dc_available"] == 0.0


def test_build_many_and_feature_store_immutable(tmp_path, synthetic_history) -> None:
    snapshots, _ = synthetic_history
    vectors = MLFeatureBuilder().build_many(snapshots[:3], {}, mode=FeatureMode.NO_MARKET,
        family=ModelFamily.XGBOOST, for_training=True)
    with MLFeatureStore(tmp_path / "features.duckdb") as store:
        store.append_many(vectors)
        store.append_many(vectors)
        assert store.connection.execute("SELECT COUNT(*) FROM ml_feature_vectors").fetchone()[0] == 3
        assert store.load_vector(vectors[0].feature_data_hash) == vectors[0]


def test_dataset_manifest_and_label_availability(ml_dataset_factory) -> None:
    dataset, _, _ = ml_dataset_factory()
    assert dataset.manifest.row_count == 75
    assert len({row.vector.match_id for row in dataset.rows}) == 75
    assert all(row.label_available_at > row.vector.kickoff_time for row in dataset.rows)


def test_dataset_store_content_idempotence(tmp_path, ml_dataset_factory) -> None:
    dataset, _, _ = ml_dataset_factory()
    with MLFeatureStore(tmp_path / "datasets.duckdb") as store:
        store.save_dataset(dataset)
        store.save_dataset(make_dataset(dataset.rows, dataset.feature_schema,
                                        dataset_kind=dataset.dataset_kind))
        assert store.load_dataset(dataset.manifest.data_hash) == dataset
        assert store.connection.execute("SELECT COUNT(*) FROM ml_datasets").fetchone()[0] == 1


def test_future_label_rejected(synthetic_history) -> None:
    snapshots, results = synthetic_history
    vector = MLFeatureBuilder().build(snapshots[0], (), mode=FeatureMode.NO_MARKET,
        family=ModelFamily.XGBOOST, for_training=True)
    with pytest.raises(ValueError, match="FUTURE_LABEL_REJECTED"):
        MLDatasetBuilder().build((vector,), results,
            schema=build_feature_schema(FeatureMode.NO_MARKET, ModelFamily.XGBOOST),
            competition_ids={vector.match_id: "synthetic_league"}, seasons={},
            dataset_cutoff=vector.prediction_time, dataset_kind="SYNTHETIC_TEST")
