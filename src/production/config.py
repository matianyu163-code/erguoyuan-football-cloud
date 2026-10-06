"""Validated configuration for the production trial launcher."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml


@dataclass(frozen=True)
class ProductionConfig:
    """Trial-mode feature switches and project-local persistent paths."""

    mode: str
    enable_live_data: bool
    enable_prediction: bool
    enable_record: bool
    enable_update: bool
    auto_bet: bool
    auto_publish: bool
    record_database: Path
    update_history: Path
    enable_official_fixture_research: bool = False

    @classmethod
    def load(cls, path: Path) -> ProductionConfig:
        """Load YAML and reject unsafe trial switches or escaped state paths."""
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError("PRODUCTION_CONFIG_INVALID")
        features = raw.get("features")
        safety = raw.get("safety")
        storage = raw.get("storage")
        if not isinstance(features, dict):
            raise TypeError("PRODUCTION_CONFIG_FEATURES_INVALID")
        if not isinstance(safety, dict):
            raise TypeError("PRODUCTION_CONFIG_SAFETY_INVALID")
        if not isinstance(storage, dict):
            raise TypeError("PRODUCTION_CONFIG_STORAGE_INVALID")
        if raw.get("mode") != "TRIAL":
            raise ValueError("PRODUCTION_MODE_MUST_BE_TRIAL")
        if safety.get("auto_bet") is not False or safety.get("auto_publish") is not False:
            raise ValueError("TRIAL_AUTO_ACTIONS_MUST_BE_DISABLED")
        project_root = path.resolve().parents[1]

        storage_values: dict[str, Any] = cast(dict[str, Any], storage)

        def local_path(key: str) -> Path:
            value = storage_values.get(key)
            if not isinstance(value, str) or not value:
                raise ValueError(f"PRODUCTION_STORAGE_PATH_MISSING:{key}")
            resolved = (project_root / value).resolve()
            if not resolved.is_relative_to(project_root):
                raise ValueError(f"PRODUCTION_STORAGE_PATH_ESCAPES_PROJECT:{key}")
            return resolved

        for key in ("enable_live_data", "enable_prediction", "enable_record",
                    "enable_update"):
            if type(features.get(key)) is not bool:
                raise ValueError(f"PRODUCTION_FEATURE_FLAG_INVALID:{key}")
        feature_values: dict[str, Any] = cast(dict[str, Any], features)
        official_research = feature_values.get("enable_official_fixture_research", False)
        if type(official_research) is not bool:
            raise ValueError("PRODUCTION_FEATURE_FLAG_INVALID:enable_official_fixture_research")
        return cls(
            mode="TRIAL",
            enable_live_data=feature_values["enable_live_data"],
            enable_prediction=feature_values["enable_prediction"],
            enable_record=feature_values["enable_record"],
            enable_update=feature_values["enable_update"],
            auto_bet=False,
            auto_publish=False,
            record_database=local_path("trial_record_database"),
            update_history=local_path("update_history"),
            enable_official_fixture_research=official_research,
        )


def default_config_path() -> Path:
    """Return the checked-in trial config for this source checkout."""
    return Path(__file__).resolve().parents[2] / "config" / "production.yaml"
