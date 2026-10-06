"""Dry-run execution plans that separate fit-time and inference-time needs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import yaml

from erguoyuan_football.research.live_data.readiness import (
    DataAvailability,
    LiveDataReadinessReport,
)


@dataclass(frozen=True)
class ModelRequirementPolicy:
    """Distinct data dependencies for fitting and inference."""

    model_id: str
    training_inputs: tuple[str, ...]
    inference_inputs: tuple[str, ...]
    prior_support: str
    sample_levels: tuple[str, ...]
    model_version: str = "UNKNOWN"
    requires_artifact: bool = False
    training_min_matches: int | None = None
    training_min_team_matches: int | None = None


@dataclass(frozen=True)
class ExecutionPlanEntry:
    """One model action with a human-readable reason and policy provenance."""

    model_id: str
    action: Literal["RUN", "RUN_DEGRADED", "BLOCK"]
    mode: Literal["FIT_AND_PREDICT", "PREDICT_ARTIFACT", "NONE"]
    requirements: ModelRequirementPolicy
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ExecutionPlan:
    """Cutoff-bound immutable plan; construction never imports or calls models."""

    plan_id: str
    created_at: datetime
    prediction_time: datetime
    training_cutoff: datetime
    readiness_status: str
    entries: tuple[ExecutionPlanEntry, ...]
    dry_run: bool = True
    execution_mode: Literal["REAL_DRY_RUN", "REAL_EXECUTE", "REPLAY_EXECUTE"] = "REAL_DRY_RUN"


class ModelExecutionPlanner:
    """Translate readiness and artifact evidence into per-model run decisions."""

    def __init__(self, requirements_path: Path | str,
                 model_registry_path: Path | str) -> None:
        raw = yaml.safe_load(Path(requirements_path).read_text(encoding="utf-8"))
        registry = yaml.safe_load(Path(model_registry_path).read_text(encoding="utf-8"))
        models = raw.get("models") if isinstance(raw, dict) else None
        registry_rows = {row.get("model_id"): row for row in registry.get("models", [])
                         if isinstance(row, dict) and row.get("enabled") is True}
        enabled = list(registry_rows)
        if not isinstance(models, dict) or set(models) != set(enabled):
            raise ValueError("PHASE14_REQUIREMENT_REGISTRY_DRIFT")
        self.input_schema_version = str(raw["input_schema_version"])
        self.sample_hierarchy_version = str(raw["sample_hierarchy_version"])
        readiness_path = Path(model_registry_path).parent / "phase13_6_model_readiness.yaml"
        readiness_raw = yaml.safe_load(readiness_path.read_text(encoding="utf-8"))
        sample_policies = readiness_raw.get("models", {})
        self.policies = {
            model_id: ModelRequirementPolicy(
                model_id, tuple(value.get("training_inputs", ())),
                tuple(value.get("inference_inputs", ())),
                str(value["prior_support"]), tuple(value.get("sample_levels", ())),
                str(registry_rows[model_id].get("version", "UNKNOWN")),
                value.get("requires_artifact") is True,
                sample_policies.get(model_id, {}).get("training_min_matches"),
                sample_policies.get(model_id, {}).get("min_training_team_matches"),
            ) for model_id, value in models.items()
        }

    @staticmethod
    def _is_utc(value: datetime) -> bool:
        return value.tzinfo is not None and value.utcoffset() == UTC.utcoffset(value)

    def plan(self, readiness: LiveDataReadinessReport, *,
             prediction_time: datetime, training_cutoff: datetime,
             valid_artifacts: frozenset[str] = frozenset(),
             neutral_venue_known: bool = True) -> ExecutionPlan:
        """Build a plan. Artifacts bypass training gates but never inference gates."""
        if not self._is_utc(prediction_time) or not self._is_utc(training_cutoff):
            raise ValueError("UTC_CUTOFF_REQUIRED")
        readiness_rows = {row.model_name: row for row in readiness.model_readiness}
        data_items = {row.data_type: row for row in readiness.data_items}
        entries: list[ExecutionPlanEntry] = []
        for model_id, policy in self.policies.items():
            reasons: list[str] = []
            if training_cutoff > prediction_time:
                reasons.append("TRAINING_CUTOFF_AFTER_PREDICTION")
            artifact_mode = model_id in valid_artifacts
            if not neutral_venue_known:
                reasons.append("INFERENCE_CONTEXT_MISSING:NEUTRAL_VENUE")
            readiness_row = readiness_rows.get(model_id)
            if artifact_mode:
                for input_name in policy.inference_inputs:
                    item = data_items.get(input_name)
                    if item is None or item.status != DataAvailability.AVAILABLE:
                        reasons.append(f"INFERENCE_INPUT_UNAVAILABLE:{input_name}")
                    elif input_name == "ML_ARTIFACT" and model_id not in item.model_ids:
                        reasons.append("MODEL_ARTIFACT_ID_MISMATCH")
                if policy.requires_artifact and not artifact_mode:
                    reasons.append("VALID_ARTIFACT_REQUIRED")
            else:
                if policy.requires_artifact:
                    reasons.append("VALID_ARTIFACT_REQUIRED")
                if readiness_row is None:
                    reasons.append("MODEL_READINESS_MISSING")
                elif not readiness_row.ready:
                    reasons.extend(readiness_row.reasons or ("MODEL_READINESS_BLOCKED",))
                for input_name in policy.training_inputs:
                    item = data_items.get(input_name)
                    if item is None or item.status != DataAvailability.AVAILABLE:
                        reasons.append(f"TRAINING_INPUT_UNAVAILABLE:{input_name}")
            reasons = list(dict.fromkeys(reasons))
            if reasons:
                action: Literal["RUN", "RUN_DEGRADED", "BLOCK"] = "BLOCK"
                mode: Literal["FIT_AND_PREDICT", "PREDICT_ARTIFACT", "NONE"] = "NONE"
            elif not artifact_mode and readiness_row is not None and readiness_row.degraded:
                action, mode = "RUN_DEGRADED", "FIT_AND_PREDICT"
            elif artifact_mode:
                action, mode = "RUN", "PREDICT_ARTIFACT"
            else:
                action, mode = "RUN", "FIT_AND_PREDICT"
            entries.append(ExecutionPlanEntry(model_id, action, mode, policy, tuple(reasons)))
        plan_key = "|".join((readiness.overall_status, prediction_time.isoformat(),
                             training_cutoff.isoformat(),
                             *(f"{row.model_id}:{row.action}:{','.join(row.reasons)}"
                               for row in entries)))
        return ExecutionPlan(hashlib.sha256(plan_key.encode()).hexdigest()[:24],
            datetime.now(UTC), prediction_time, training_cutoff,
            readiness.overall_status, tuple(entries), True)
