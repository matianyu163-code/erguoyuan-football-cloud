"""Indexed artifact metadata and exact-key cache lookup tests."""

from datetime import timedelta

import pytest

from erguoyuan_football.contracts.common import now
from erguoyuan_football.models.artifact import ModelArtifact
from erguoyuan_football.models.artifact_index import ArtifactIndex, FeatureArtifact

pytestmark = pytest.mark.fast


def test_feature_artifact_roundtrip_and_point_in_time_lookup(tmp_path) -> None:
    at = now()
    artifact = FeatureArtifact(feature_set_id="spi-state", feature_version="1",
        as_of_time=at, data_hash="data-hash", config_hash="config-hash",
        path=str(tmp_path / "features.parquet"), created_at=at)
    index = ArtifactIndex(tmp_path / "artifact-index.duckdb")
    index.register_feature(artifact)
    assert index.find_feature("spi-state", "1", at, "config-hash", "data-hash") == artifact
    assert index.find_feature("spi-state", "1", at + timedelta(seconds=1),
                              "config-hash", "data-hash") is None
    index.close()


def test_model_artifact_index_exact_hash_lookup(tmp_path) -> None:
    at = now()
    artifact = ModelArtifact(model_id="CORE_SPI_LIKE_V1", model_version="1.0.0",
        trained_until=at, training_data_hash="data-hash", config_hash="config-hash",
        artifact_path=str(tmp_path / "model"), payload_sha256="abc", dependencies={})
    assert artifact.code_version
    index = ArtifactIndex(tmp_path / "artifact-index.duckdb")
    index.register_model(artifact)
    assert index.find_model(artifact.model_id, artifact.model_version, at,
                            artifact.config_hash, artifact.training_data_hash) == artifact.artifact_path
    assert index.find_model(artifact.model_id, artifact.model_version, at,
                            artifact.config_hash, "other-data") is None
    index.close()
