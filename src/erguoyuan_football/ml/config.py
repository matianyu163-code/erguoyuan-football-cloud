"""Versioned, bounded ML training configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import Field

from erguoyuan_football.contracts.common import Contract
from erguoyuan_football.ml.schemas import ModelFamily, stable_hash


class MLConfig(Contract):
    family: ModelFamily
    profile: Literal["development", "production"] = "production"
    params: dict[str, Any]
    early_stopping_rounds: int = Field(ge=1)
    max_trials: int = Field(default=6, ge=1, le=24)
    min_train_rows: int = Field(default=30, ge=3)
    allow_test_data: bool = False

    @property
    def config_hash(self) -> str:
        return stable_hash(self.model_dump(mode="json"))


def load_ml_config(path: str | Path, *, family: ModelFamily, profile: str = "production",
                   allow_test_data: bool = False) -> MLConfig:
    """Select a named profile; no tuning against final-test outcomes."""
    if profile not in {"development", "production"}:
        raise ValueError("UNKNOWN_ML_PROFILE")
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get(profile), dict):
        raise TypeError("INVALID_ML_CONFIG")
    selected = payload[profile]
    if not isinstance(selected.get("params"), dict):
        raise TypeError("INVALID_ML_MODEL_PARAMS")
    return MLConfig(family=family, profile=cast(Literal["development", "production"], profile),
        params=selected["params"],
        early_stopping_rounds=selected["early_stopping_rounds"],
        max_trials=selected.get("max_trials", 6),
        min_train_rows=selected.get("min_train_rows", 30), allow_test_data=allow_test_data)
