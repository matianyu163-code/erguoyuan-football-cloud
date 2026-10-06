"""DuckDB connection helper that keeps Phase 9 temporary spill on the workspace drive."""

from __future__ import annotations

import os
from pathlib import Path

import duckdb


def connect_project_duckdb(path: str | Path, *, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open DuckDB with conservative threads and project-local temporary spill by default."""
    settings: dict[str, str | bool | int | float | list[str]] = {
        "threads": int(os.environ.get("CORE_DUCKDB_THREADS", "2"))
    }
    temp_directory = os.environ.get("DUCKDB_TEMP_DIRECTORY")
    target = Path(temp_directory).resolve() if temp_directory else Path(__file__).resolve().parents[3] / ".workspace" / "duckdb"
    target.mkdir(parents=True, exist_ok=True)
    settings["temp_directory"] = str(target)
    return duckdb.connect(str(path), read_only=read_only, config=settings)
