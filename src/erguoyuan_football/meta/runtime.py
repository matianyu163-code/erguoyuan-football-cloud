"""Small Phase 9 runtime context, explicit artifact index, and load cache."""

from __future__ import annotations

import ctypes
import hashlib
import json
import threading
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

from erguoyuan_football.meta.artifact import MetaArtifact, load_artifact


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _freeze(value: Any) -> Any:
    """Recursively protect a run's configuration snapshot from mutation."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


def peak_working_set_bytes() -> int:
    """Return process peak working set through the Windows API when available."""
    if not hasattr(ctypes, "windll"):
        return 0

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("page_fault_count", ctypes.c_ulong),
                    ("peak_working_set", ctypes.c_size_t), ("working_set", ctypes.c_size_t),
                    ("quota_peak_paged_pool", ctypes.c_size_t), ("quota_paged_pool", ctypes.c_size_t),
                    ("quota_peak_nonpaged_pool", ctypes.c_size_t),
                    ("quota_nonpaged_pool", ctypes.c_size_t), ("pagefile_usage", ctypes.c_size_t),
                    ("peak_pagefile_usage", ctypes.c_size_t)]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ProcessMemoryCounters), ctypes.c_ulong]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    process = kernel32.GetCurrentProcess()
    ok = psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb)
    if not ok:
        raise OSError("GET_PROCESS_MEMORY_INFO_FAILED")
    return int(counters.peak_working_set)


class ArtifactIndex:
    """Resolve explicitly registered artifact IDs without walking directories."""

    def __init__(self, index_path: str | Path) -> None:
        self.path = Path(index_path)
        self._entries: dict[str, str] = {}
        if self.path.is_file():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != "PHASE9_ARTIFACT_INDEX_V1":
                raise ValueError("ARTIFACT_INDEX_SCHEMA_INVALID")
            self._entries = dict(payload.get("artifacts", {}))

    def register(self, artifact_id: str, artifact_path: str | Path) -> None:
        """Record a verified artifact path in an explicit index."""
        directory = Path(artifact_path).resolve()
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("artifact_id") != artifact_id or directory.name != artifact_id:
            raise ValueError("ARTIFACT_INDEX_ID_MISMATCH")
        self._entries[artifact_id] = str(directory)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps({"schema_version": "PHASE9_ARTIFACT_INDEX_V1",
            "artifacts": self._entries}, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self.path)

    def resolve(self, artifact_id: str) -> Path:
        """Resolve only an explicit index entry; never fall back to recursive search."""
        location = self._entries.get(artifact_id)
        if location is None:
            raise ValueError("ARTIFACT_NOT_INDEXED")
        path = Path(location).resolve()
        manifest_path = path / "manifest.json"
        if not path.is_dir() or not manifest_path.is_file():
            raise ValueError("INDEXED_ARTIFACT_MISSING")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if path.name != artifact_id or manifest.get("artifact_id") != artifact_id:
            raise ValueError("ARTIFACT_INDEX_ID_MISMATCH")
        return path


class LoadedArtifactCache:
    """Cache a verified META/calibrator pair by all compatibility identities."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._items: dict[tuple[str, str, str, str, bool], MetaArtifact] = {}
        self.loads = 0
        self.hits = 0

    def get(self, path: str | Path, *, expected_candidate_hash: str,
            expected_config_hash: str | None = None,
            allow_phase9_1_frozen_compatibility: bool = False
            ) -> tuple[MetaArtifact, bool]:
        """Validate disk identity on each access and return cached immutable weights."""
        directory = Path(path).resolve()
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if expected_config_hash is not None and manifest.get("config_hash") != expected_config_hash:
            raise ValueError("META_ARTIFACT_CONFIG_MISMATCH")
        key = self._key(directory, manifest, allow_phase9_1_frozen_compatibility)
        with self._lock:
            cached = self._items.get(key)
            if cached is not None:
                self.hits += 1
                return cached, True
        loaded = load_artifact(directory, expected_candidate_hash=expected_candidate_hash,
            allow_phase9_1_frozen_compatibility=allow_phase9_1_frozen_compatibility)
        with self._lock:
            self._items[key] = loaded
            self.loads += 1
        return loaded, False

    @staticmethod
    def _key(directory: Path, manifest: Mapping[str, Any], compatibility: bool = False
             ) -> tuple[str, str, str, str, bool]:
        manifest_path = directory / "manifest.json"
        artifact_hash = hashlib.sha256("".join((
            _file_hash(manifest_path), _file_hash(directory / "meta.joblib"),
            _file_hash(directory / "calibrator" / "calibrator.joblib"),
            _file_hash(directory / "calibrator" / "manifest.json"),
        )).encode("ascii")).hexdigest()
        return (str(manifest["artifact_id"]), artifact_hash,
                str(manifest["feature_schema_version"]), str(manifest["config_hash"]),
                compatibility)

    def remember_verified(self, artifact: MetaArtifact) -> None:
        """Cache an artifact that was just created and verified by atomic save."""
        directory = artifact.path.resolve()
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("artifact_id") != artifact.artifact_id:
            raise ValueError("ARTIFACT_CACHE_SEED_ID_MISMATCH")
        key = self._key(directory, manifest)
        with self._lock:
            self._items[key] = artifact
            self.loads += 1

    def clear(self) -> None:
        """Drop cached objects and reset telemetry counters."""
        with self._lock:
            self._items.clear()
            self.loads = 0
            self.hits = 0


@dataclass(frozen=True)
class CoreRuntimeContext:
    """Scoped Phase 9 runtime dependencies; configuration is read-only."""

    config: Mapping[str, Any]
    config_hash: str
    database_path: Path
    artifact_index: ArtifactIndex
    artifact_cache: LoadedArtifactCache
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    meta_artifact: MetaArtifact | None = None
    calibrator: Any | None = None
    cache: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, *, config: Mapping[str, Any], config_hash: str,
               database_path: str | Path, artifact_index: ArtifactIndex,
               artifact_cache: LoadedArtifactCache,
               meta_artifact: MetaArtifact | None = None) -> CoreRuntimeContext:
        """Freeze a single-run config view and bind explicit dependencies."""
        return cls(config=_freeze(config), config_hash=config_hash,
                   database_path=Path(database_path), artifact_index=artifact_index,
                   artifact_cache=artifact_cache, meta_artifact=meta_artifact,
                   calibrator=meta_artifact.calibrator if meta_artifact else None)


class PerformanceTelemetry:
    """Append small structured runtime events without logging model/data arrays."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def record(self, *, stage: str, duration_ms: float, rows: int,
               db_queries: int, cache_hit: bool, artifact_cache_hit: bool,
               memory_peak_bytes: int, dataset_hash: str | None = None,
               artifact_id: str | None = None, calibrator_id: str | None = None,
               run_id: str | None = None) -> dict[str, Any]:
        """Append one bounded event and return the serialized telemetry record."""
        event = {
            "run_id": run_id or uuid.uuid4().hex,
            "stage": stage,
            "recorded_at": datetime.now(UTC).isoformat(),
            "duration_ms": round(duration_ms, 3),
            "rows": rows,
            "db_queries": db_queries,
            "cache_hit": cache_hit,
            "artifact_cache_hit": artifact_cache_hit,
            "memory_peak_bytes": memory_peak_bytes,
            "dataset_hash": dataset_hash,
            "artifact_id": artifact_id,
            "calibrator_id": calibrator_id,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, separators=(",", ":")) + "\n")
        return event
