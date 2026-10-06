"""Health and posterior-predictive diagnostics for Dynamic Bayesian Poisson."""

from __future__ import annotations

from dataclasses import asdict
from typing import Literal

from pydantic import Field

from erguoyuan_football.contracts.common import Contract
from erguoyuan_football.models.dynamic_bayes.inference import DynamicDiagnostics


class ModelHealthReport(Contract):
    """Serializable diagnostics that gate whether a fit may be used."""

    status: Literal["GOOD", "WARNING", "FAILED"]
    r_hat: float | None = Field(default=None, ge=1)
    ess: float | None = Field(default=None, ge=0)
    divergences: int | None = Field(default=None, ge=0)
    sampling_time_seconds: float = Field(default=0, ge=0)
    posterior_draws: int = Field(default=0, ge=0)
    posterior_predictive: dict[str, float] = Field(default_factory=dict)
    parameter_extremes: dict[str, float] = Field(default_factory=dict)
    probability_valid: bool = False
    reason: str | None = None

    @classmethod
    def from_diagnostics(cls, diagnostics: DynamicDiagnostics) -> ModelHealthReport:
        """Convert internal diagnostics into an immutable artifact field."""
        values = asdict(diagnostics)
        values["status"] = values.pop("health")
        values["probability_valid"] = values["status"] in {"GOOD", "WARNING"}
        return cls.model_validate(values)
