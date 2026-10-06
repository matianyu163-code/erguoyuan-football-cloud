"""Readiness-gated, isolated model execution with persisted evidence records."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from erguoyuan_football.contracts.predictions import ProbabilityVector
from erguoyuan_football.prediction.execution_records import (
    ModelExecutionRecord,
    ModelExecutionStore,
)
from erguoyuan_football.prediction.model_execution_planner import (
    ExecutionPlan,
    ExecutionPlanEntry,
)
from erguoyuan_football.prediction.model_input_adapter import ModelInputBundle

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelExecutionResult:
    """Primitive result of one called model; absent outputs remain null."""

    model_name: str
    status: str
    probabilities: ProbabilityVector | None
    raw_output: Mapping[str, Any] | None
    uncertainty: str
    execution_record_id: str
    error: str | None = None


class ModelExecutionEngine:
    """Call only RUN entries after explicit execute authorization."""

    def __init__(self, store: ModelExecutionStore) -> None:
        self.store = store

    def execute(self, plan: ExecutionPlan,
                inputs: Mapping[str, ModelInputBundle],
                executors: Mapping[str, Callable[[ExecutionPlanEntry, ModelInputBundle],
                                                 ModelExecutionResult]],
                *, execute_ready_models: bool) -> tuple[ModelExecutionResult, ...]:
        """Persist blocked decisions without calling them; isolate model failures."""
        if not execute_ready_models:
            return ()
        results: list[ModelExecutionResult] = []
        for entry in plan.entries:
            bundle = inputs.get(entry.model_id)
            started = datetime.now(UTC)
            if entry.action == "BLOCK":
                record = self._record(plan, entry, bundle, "BLOCKED", started,
                    datetime.now(UTC), block_reason=";".join(entry.reasons))
                self.store.save(record)
                results.append(ModelExecutionResult(entry.model_id, "BLOCKED", None,
                    None, "VERY_HIGH", record.execution_record_id,
                    ";".join(entry.reasons)))
                continue
            executor = executors.get(entry.model_id)
            if bundle is None or executor is None:
                reason = "MODEL_INPUT_BUNDLE_MISSING" if bundle is None else "EXECUTOR_NOT_CONFIGURED"
                record = self._record(plan, entry, bundle, "BLOCKED", started,
                    datetime.now(UTC), block_reason=reason)
                self.store.save(record)
                results.append(ModelExecutionResult(entry.model_id, "BLOCKED", None,
                    None, "VERY_HIGH", record.execution_record_id, reason))
                continue
            try:
                output = executor(entry, bundle)
                if output.model_name != entry.model_id or output.probabilities is None:
                    raise ValueError("MODEL_OUTPUT_ID_OR_PROBABILITY_MISSING")
                if output.status not in {"EXECUTED", "DEGRADED_EXECUTED"}:
                    raise ValueError(f"INVALID_EXECUTOR_STATUS:{output.status}")
                status = "DEGRADED_EXECUTED" if entry.action == "RUN_DEGRADED" else "EXECUTED"
                metadata = output.raw_output.get("metadata") if output.raw_output else None
                bayesian_diagnostic = (metadata.get("bayesian_diagnostic")
                    if entry.model_id == "BAYESIAN_HIERARCHICAL_V1"
                    and isinstance(metadata, Mapping) else None)
                record = self._record(plan, entry, bundle, status, started,
                    datetime.now(UTC), output_valid=True,
                    output_probabilities={
                        "home": output.probabilities.p_home,
                        "draw": output.probabilities.p_draw,
                        "away": output.probabilities.p_away,
                    }, bayesian_diagnostic=bayesian_diagnostic)
                self.store.save(record)
                results.append(ModelExecutionResult(entry.model_id, status,
                    output.probabilities, output.raw_output, output.uncertainty,
                    record.execution_record_id))
            except Exception as error:  # Model failures are isolated and persisted.
                LOGGER.exception("model execution failed for %s", entry.model_id)
                reason = type(error).__name__
                record = self._record(plan, entry, bundle, "FAILED", started,
                    datetime.now(UTC), block_reason=str(error), error_code=reason,
                    bayesian_diagnostic=getattr(error, "diagnostic", None))
                self.store.save(record)
                results.append(ModelExecutionResult(entry.model_id, "FAILED", None,
                    None, "VERY_HIGH", record.execution_record_id,
                    f"{reason}:{error}"))
        return tuple(results)

    @staticmethod
    def _record(plan: ExecutionPlan, entry: ExecutionPlanEntry,
                bundle: ModelInputBundle | None, status: str,
                started: datetime, finished: datetime, *,
                block_reason: str | None = None, error_code: str | None = None,
                output_valid: bool = False,
                output_probabilities: dict[str, float] | None = None,
                bayesian_diagnostic: dict[str, Any] | None = None,
                ) -> ModelExecutionRecord:
        """Create audit evidence including exact category counts and priors."""
        counts = bundle.sample_counts if bundle else {}
        prior_types = (tuple({bundle.prior_type}) if bundle and bundle.prior_type else ())
        training_failed = any(reason.startswith("TRAINING_") for reason in entry.reasons)
        inference_failed = any(reason.startswith(("INFERENCE_", "DIRECT_SAMPLE_LOW",
                                                   "COMPETITION_SAMPLE_LOW"))
                               for reason in entry.reasons)
        return ModelExecutionRecord.create(
            plan_id=plan.plan_id, model_name=entry.model_id,
            model_version=entry.requirements.model_version, status=status, started_at=started,
            finished_at=finished,
            input_evidence_ids=bundle.evidence_ids if bundle else (),
            sample_counts=counts, prior_types=prior_types,
            cutoff=plan.prediction_time, training_cutoff=plan.training_cutoff,
            degraded=entry.action == "RUN_DEGRADED", output_valid=output_valid,
            block_reason=block_reason, error_code=error_code,
            prior_parameter_source=bundle.prior_parameter_source if bundle else None,
            prior_strength=bundle.prior_strength if bundle else None,
            prior_derived_from=bundle.prior_derived_from if bundle else (),
            feature_provenance=(dict(bundle.lineage) if bundle else None),
            data_origin=(bundle.training_dataset.dataset_kind if bundle else "UNKNOWN"),
            sample_window=(";".join(f"{key}={value}" for key, value in sorted(counts.items()))
                           if counts else "UNKNOWN"),
            training_status=("BLOCKED" if training_failed else "READY" if bundle else "UNKNOWN"),
            inference_status=("BLOCKED" if inference_failed else "READY" if bundle else "UNKNOWN"),
            execution_mode=plan.execution_mode,
            output_probabilities=output_probabilities, calibrated=False,
            bayesian_diagnostic=bayesian_diagnostic,
        )
