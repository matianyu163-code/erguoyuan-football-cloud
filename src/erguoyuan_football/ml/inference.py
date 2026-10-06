"""Run independent fitted ML models without cascading one failure to another."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path

from erguoyuan_football.contracts.common import ExecutionStatus
from erguoyuan_football.contracts.predictions import ModelPrediction
from erguoyuan_football.ml.base import CoreMLModel, MLDependencyUnavailable
from erguoyuan_football.ml.registry import MLModelRegistry
from erguoyuan_football.ml.schemas import MLFeatureVector, ModelFamily
from erguoyuan_football.models.runner import BaseModelPredictionBundle

LOGGER = logging.getLogger(__name__)


class MLModelRunner:
    """Load each trusted artifact at most once per run; preserve per-model status."""

    def __init__(self, registry: MLModelRegistry | None = None) -> None:
        self.registry = registry or MLModelRegistry()

    def run(self, vectors: Mapping[ModelFamily, MLFeatureVector], *,
            fitted_models: Mapping[str, CoreMLModel] | None = None,
            artifact_paths: Mapping[str, str | Path] | None = None,
            production: bool = False, network_gate_status: str | None = None,
            market_gate_status: str | None = None, oos: bool = False) -> BaseModelPredictionBundle:
        outputs: list[ModelPrediction] = []
        for model_id in self.registry.list_models():
            model = fitted_models.get(model_id) if fitted_models else None
            if model is None:
                model = self.registry.get_model(model_id)
            vector = vectors.get(model.family)
            if vector is None:
                reference = next(iter(vectors.values()), None)
                if reference is not None:
                    outputs.append(model._record(reference, ExecutionStatus.UNAVAILABLE,
                                                 reason="ML_FEATURE_VECTOR_UNAVAILABLE"))
                continue
            try:
                if not model.fitted and artifact_paths and model_id in artifact_paths:
                    model = type(model).load(artifact_paths[model_id])
                outputs.append(model.predict(vector, production=production,
                    network_gate_status=network_gate_status,
                    market_gate_status=market_gate_status, oos=oos))
            except MLDependencyUnavailable as error:
                outputs.append(model._record(vector, ExecutionStatus.UNAVAILABLE, reason=str(error)))
            except Exception as error:
                LOGGER.exception("ML model %s failed independently", model_id)
                outputs.append(model._record(vector, ExecutionStatus.FAILED,
                                             reason=f"{type(error).__name__}:{error}"))
        return BaseModelPredictionBundle(tuple(outputs))
