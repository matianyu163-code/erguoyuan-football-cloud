"""Auditable cache manifests and checked local-only model persistence."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any

from pydantic import Field

from erguoyuan_football.contracts.common import Contract, UTCTime, now


def dependency_versions() -> dict[str, str]:
    """Include numerical/runtime identities in every artifact and cache key."""
    names = ("penaltyblog", "numpy", "pandas", "scipy", "scikit-learn", "joblib", "pydantic", "duckdb", "httpx")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "NOT_INSTALLED"
    return versions


class ModelArtifact(Contract):
    """Manifest; the checksum protects against accidental corruption, not malicious pickle."""

    model_id: str
    model_version: str
    code_version: str = Field(default_factory=lambda: os.environ.get("CORE_CODE_VERSION", "UNVERSIONED_SOURCE_TREE"))
    trained_until: UTCTime
    training_data_hash: str
    config_hash: str
    created_at: UTCTime = Field(default_factory=now)
    artifact_path: str
    payload_sha256: str
    dependencies: dict[str, str]
    state_process: str | None = None
    time_index_version: str | None = None
    posterior_method: str | None = None
    sampling_diagnostics: dict[str, Any] = Field(default_factory=dict)
    state_history_count: int | None = Field(default=None, ge=0)


def cache_key(model_id: str, version: str, cutoff: str, data_hash: str, config_hash: str) -> str:
    """Change cache identity for data, cutoff, config, implementation or dependency changes."""
    value = json.dumps([model_id, version, cutoff, data_hash, config_hash, dependency_versions()], sort_keys=True)
    return hashlib.sha256(value.encode()).hexdigest()


def file_hash(path: Path) -> str:
    """Hash an artifact payload before any deserialization."""
    return hashlib.sha256(path.read_bytes()).hexdigest()
