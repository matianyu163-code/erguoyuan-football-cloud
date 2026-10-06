"""Immutable version registration for model, provider, config, and knowledge updates."""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

UpdateCategory = Literal["models", "providers", "configs", "knowledge", "rules"]
_VERSION = re.compile(
    r"^(?:MODEL_V1(?:\.1)?|PROVIDER_V1|CONFIG_V1|KNOWLEDGE_V1|RULE_V1)$"
)
_CATEGORIES = {"models", "providers", "configs", "knowledge", "rules"}


class UpdateManager:
    """Register version metadata without replacing code or existing versions."""

    def __init__(self, history_path: Path) -> None:
        self.history_path = history_path
        self.version_root = history_path.parent / "versions"

    def register_update(self, *, category: UpdateCategory, version: str,
                        change: str, operator: str,
                        changed_at: datetime | None = None) -> Path:
        """Create one immutable version manifest and append its audit entry."""
        if category not in _CATEGORIES:
            raise ValueError("UPDATE_CATEGORY_INVALID")
        if not _VERSION.fullmatch(version):
            raise ValueError("UPDATE_VERSION_INVALID")
        if not change.strip() or not operator.strip():
            raise ValueError("UPDATE_CHANGE_AND_OPERATOR_REQUIRED")
        timestamp = changed_at or datetime.now(UTC)
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("UPDATE_TIMEZONE_REQUIRED")
        timestamp = timestamp.astimezone(UTC)
        manifest_path = self.version_root / category / f"{version}.json"
        if manifest_path.exists():
            raise FileExistsError("UPDATE_VERSION_ALREADY_EXISTS")
        history = self._read_history()
        if any(item.get("category") == category and item.get("version") == version
               for item in history):
            raise FileExistsError("UPDATE_HISTORY_VERSION_ALREADY_EXISTS")
        manifest = {
            "category": category, "version": version,
            "time": timestamp.isoformat(), "change": change.strip(),
            "operator": operator.strip(),
        }
        _write_exclusive_json(manifest_path, manifest)
        history.append(manifest)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.history_path.with_suffix(self.history_path.suffix + ".tmp")
        temporary.write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
        os.replace(temporary, self.history_path)
        return manifest_path

    def _read_history(self) -> list[dict[str, Any]]:
        if not self.history_path.exists():
            return []
        payload: Any = json.loads(self.history_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
            raise ValueError("UPDATE_HISTORY_SCHEMA_INVALID")
        return payload


def _write_exclusive_json(path: Path, payload: dict[str, str]) -> None:
    """Atomically create a version manifest and never overwrite an existing one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".tmp",
                                     prefix=f".{path.name}.", dir=path.parent,
                                     delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
