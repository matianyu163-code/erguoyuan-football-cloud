"""Isolated model capability accounting for frozen blind-test publication."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

NATIONAL_TEAM = "NATIONAL_TEAM"
CLUB = "CLUB"
UNKNOWN = "UNKNOWN"
UNIVERSAL = "UNIVERSAL"


@dataclass(frozen=True)
class ModelRegistration:
    """One model loaded from a hash-verified frozen release."""

    name: str
    domain: str
    run: Callable[[], Mapping[str, Any]]


def _distribution(value: Mapping[str, Any]) -> dict[str, float]:
    keys = ("HOME", "DRAW", "AWAY")
    direct = all(key in value for key in keys)
    if direct:
        probabilities = {key: value.get(key) for key in keys}
    else:
        probabilities = {"HOME": value.get("p_home"),
                         "DRAW": value.get("p_draw"),
                         "AWAY": value.get("p_away")}
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
    available: dict[str, dict[str, float]] = {}
    outputs: dict[str, Mapping[str, Any]] = {}
    unavailable: dict[str, str] = {}
    for name, runner in sorted(runners.items()):
        try:
            output = runner()
            available[name] = _distribution(output)
            outputs[name] = output
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
            "model_outputs": outputs,
            "available_model_ensemble": ensemble,
            "ensemble_method": "EQUAL_WEIGHT_AVAILABLE_MODELS_V1" if ensemble else "UNAVAILABLE"}


class ModelCapabilityRouter:
    """Route one fixture to compatible frozen model domains and ensemble successes."""

    def __init__(self, registrations: list[ModelRegistration],
                 minimum_available_models: int = 1) -> None:
        if minimum_available_models < 1:
            raise ValueError("MINIMUM_AVAILABLE_MODELS_MUST_BE_POSITIVE")
        self.registrations = tuple(registrations)
        self.minimum_available_models = minimum_available_models

    def execute(self, domain: str) -> dict[str, Any]:
        """Attempt every compatible frozen model without letting failures cascade."""
        eligible = [item for item in self.registrations
                    if item.domain == domain or item.domain == UNIVERSAL]
        incompatible = {item.name: "DOMAIN_MISMATCH" for item in self.registrations
                        if item not in eligible}
        executed = execute_available({item.name: item.run for item in eligible})
        unavailable = {**incompatible, **executed["unavailable_models"]}
        errors = dict(executed["unavailable_models"])
        count = int(executed["available_model_count"])
        ratio = count / len(eligible) if eligible else 0.0
        if count < self.minimum_available_models or not eligible:
            coverage = "UNAVAILABLE"
        elif ratio < 0.4:
            coverage = "MINIMAL"
        elif ratio < 0.8:
            coverage = "PARTIAL"
        else:
            coverage = "FULL"
        if not eligible:
            reason = f"NO_MODEL_REGISTERED_FOR_DOMAIN:{domain}"
            unavailable[f"NO_{domain}_MODEL_REGISTERED"] = reason
        else:
            reason = None
        return {
            "match_domain": domain,
            "available_models": executed["available_models"],
            "unavailable_models": unavailable,
            "model_errors": errors,
            "available_model_count": count,
            "registered_model_count": len(self.registrations),
            "eligible_model_count": len(eligible),
            "minimum_available_models": self.minimum_available_models,
            "model_coverage": coverage,
            "model_distributions": executed["model_distributions"],
            "model_outputs": executed["model_outputs"],
            "available_model_ensemble": (executed["available_model_ensemble"]
                                          if count >= self.minimum_available_models
                                          else None),
            "ensemble_method": executed["ensemble_method"],
            "reason": reason,
        }
