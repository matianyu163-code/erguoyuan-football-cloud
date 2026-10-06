"""Focused tests for Phase 9.1 runtime safety and cache behavior."""

from __future__ import annotations

import json
from collections.abc import MutableMapping
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from erguoyuan_football.meta.artifact import load_artifact
from erguoyuan_football.meta.candidate import CandidateDatasetCache, VerifiedCandidate
from erguoyuan_football.meta.runtime import (
    ArtifactIndex,
    CoreRuntimeContext,
    LoadedArtifactCache,
    PerformanceTelemetry,
)


def test_candidate_cache_returns_isolated_dataframes(tmp_path: Path) -> None:
    """Mutating a caller's copy must not corrupt the cached candidate."""
    source = tmp_path / "candidate.parquet"
    source.write_bytes(b"immutable fixture")
    candidate = VerifiedCandidate(
        "dataset", "hash", {}, pd.DataFrame({"value": [1]}),
        pd.DataFrame({"target": [0]}), pd.DataFrame({"lineage": ["a"]}),
    )
    cache = CandidateDatasetCache()
    key = ("db", "dataset")
    cache.put(key, candidate, (source,))

    first = cache.get(key)
    assert first is not None
    first.features.loc[0, "value"] = 99

    second = cache.get(key)
    assert second is not None
    assert second.features.loc[0, "value"] == 1
    assert cache.hits == 2


def test_candidate_cache_invalidates_changed_source(tmp_path: Path) -> None:
    """A source-file signature change must invalidate a cached candidate."""
    source = tmp_path / "candidate.parquet"
    source.write_bytes(b"before")
    candidate = VerifiedCandidate(
        "dataset", "hash", {}, pd.DataFrame({"value": [1]}),
        pd.DataFrame({"target": [0]}), pd.DataFrame({"lineage": ["a"]}),
    )
    cache = CandidateDatasetCache()
    key = ("db", "dataset")
    cache.put(key, candidate, (source,))
    source.write_bytes(b"after with a different size")

    assert cache.get(key) is None
    assert cache.misses == 1


def test_artifact_index_rejects_tampered_path_identity(tmp_path: Path) -> None:
    """An edited index cannot redirect an artifact ID to a different artifact."""
    index_path = tmp_path / "index.json"
    artifact_path = tmp_path / "verified-id"
    artifact_path.mkdir()
    (artifact_path / "manifest.json").write_text(
        json.dumps({"artifact_id": "verified-id"}), encoding="utf-8")
    index = ArtifactIndex(index_path)
    index.register("verified-id", artifact_path)
    assert index.resolve("verified-id") == artifact_path.resolve()

    other_path = tmp_path / "other-id"
    other_path.mkdir()
    (other_path / "manifest.json").write_text(
        json.dumps({"artifact_id": "other-id"}), encoding="utf-8")
    index_path.write_text(json.dumps({
        "schema_version": "PHASE9_ARTIFACT_INDEX_V1",
        "artifacts": {"verified-id": str(other_path)},
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="ARTIFACT_INDEX_ID_MISMATCH"):
        ArtifactIndex(index_path).resolve("verified-id")


def test_runtime_context_freezes_top_level_config(tmp_path: Path) -> None:
    """The run context exposes a read-only configuration mapping."""
    context = CoreRuntimeContext.create(
        config={"alpha": 1, "nested": {"features": ["x"]}},
        config_hash="config-hash", database_path=tmp_path / "db.duckdb",
        artifact_index=ArtifactIndex(tmp_path / "index.json"),
        artifact_cache=LoadedArtifactCache(),
    )
    with pytest.raises(TypeError):
        cast(MutableMapping[str, object], context.config)["alpha"] = 2
    with pytest.raises(TypeError):
        context.config["nested"]["features"] = ()
    assert context.config["nested"]["features"] == ("x",)


def test_performance_telemetry_appends_structured_events(tmp_path: Path) -> None:
    """Telemetry is JSONL, bounded to metadata, and append-only."""
    path = tmp_path / "logs" / "runtime.jsonl"
    telemetry = PerformanceTelemetry(path)
    telemetry.record(stage="candidate", duration_ms=2.5, rows=3, db_queries=1,
                     cache_hit=False, artifact_cache_hit=False, memory_peak_bytes=1024,
                     run_id="run-1")
    telemetry.record(stage="inference", duration_ms=1.0, rows=3, db_queries=0,
                     cache_hit=True, artifact_cache_hit=True, memory_peak_bytes=1024,
                     run_id="run-1")

    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [event["stage"] for event in events] == ["candidate", "inference"]
    assert all(event["run_id"] == "run-1" for event in events)


def test_phase9_1_compatibility_is_limited_to_frozen_artifact() -> None:
    """Only the audited frozen Phase 9 artifact accepts the wrapper refactor."""
    project_root = Path(__file__).resolve().parents[2]
    candidate_hash = "10cec92f96327bfb9d2fe0d3f0782ac5aa884c14419642c127a479786ccec470"
    artifact_path = (project_root / "artifacts" / "phase9" / "meta_no_market" /
                     "13a49d400f358e9e8ea69962c07e64db")
    if not artifact_path.is_dir():
        pytest.skip("Frozen Phase 9 reference artifact is unavailable")

    with pytest.raises(ValueError, match="META_ARTIFACT_PROVENANCE_REJECTED"):
        load_artifact(artifact_path, expected_candidate_hash=candidate_hash)
    artifact = load_artifact(artifact_path, expected_candidate_hash=candidate_hash,
                             allow_phase9_1_frozen_compatibility=True)
    assert artifact.artifact_id == artifact_path.name
