"""Isolated model capability accounting for frozen blind-test publication."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any, cast


def _distribution(value: Mapping[str, Any]) -> dict[str, float]:
    keys = ("HOME", "DRAW", "AWAY")
    probabilities = {key: value.get(key) for key in keys}
    if any(isinstance(p, bool) or not isinstance(p, (int, float)) or
           not math.isfinite(p) or p < 0 or p > 1 for p in probabilities.values()):
        raise ValueError("INVALID_MODEL_1X2_DISTRIBUTION")
    result = {key: float(cast(float, probabilities[key])) for key in keys}
    if not math.isclose(sum(result.values()), 1.0, abs_tol=1e-9):
        raise ValueError("MODEL_1X2_NOT_NORMALIZED")
    return result


def execute_available(
    runners: Mapping[str, Callable[[], Mapping[str, Any]]],
) -> dict[str, Any]:
    """Run each registered frozen model independently; preserve failures."""
    if not runners:
        raise ValueError("NO_REGISTERED_FROZEN_MODELS")
    available: dict[str, dict[str, float]] = {}
    unavailable: dict[str, str] = {}
    for name, runner in sorted(runners.items()):
        try:
            available[name] = _distribution(runner())
        except (OSError, ValueError, TypeError, RuntimeError, KeyError) as error:
            unavailable[name] = f"{type(error).__name__}:{error}"
    ensemble = None
    if available:
        count = len(available)
        ensemble = {key: sum(row[key] for row in available.values()) / count
                    for key in ("HOME", "DRAW", "AWAY")}
    return {"available_models": sorted(available), "unavailable_models": unavailable,
            "available_model_count": len(available), "registered_model_count": len(runners),
            "model_distributions": available,
            "available_model_ensemble": ensemble,
            "ensemble_method": "EQUAL_WEIGHT_AVAILABLE_MODELS_V1" if ensemble else "UNAVAILABLE"}
