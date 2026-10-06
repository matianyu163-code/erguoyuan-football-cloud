"""Typed diagnostics for the existing penaltyblog hierarchical sampler."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from typing import Any

import numpy as np


class BayesianDiagnosticStatus(StrEnum):
    """Failure classes are kept distinct for operational diagnosis."""

    PASS = "PASS"
    R_HAT_FAILED = "R_HAT_FAILED"
    ESS_FAILED = "ESS_FAILED"
    DIVERGENCE_HIGH = "DIVERGENCE_HIGH"
    TREE_DEPTH_LIMIT = "TREE_DEPTH_LIMIT"
    NUMERICAL_FAILURE = "NUMERICAL_FAILURE"
    POSTERIOR_INVALID = "POSTERIOR_INVALID"
    MAP_INTERNAL_ONLY = "MAP_INTERNAL_ONLY"


@dataclass(frozen=True)
class BayesianDiagnosticResult:
    """Sampler settings and convergence/parameter validity for one attempt."""

    success: bool
    status: BayesianDiagnosticStatus
    r_hat_ok: bool
    ess_ok: bool
    divergence_count: int
    tree_depth_hits: int
    numerical_error: str | None
    posterior_valid: bool
    recommendation: str
    sampler: str
    execution_mode: str
    iterations: int
    warmup: int
    chains: int
    max_r_hat: float | None
    min_ess: float | None
    divergence_metric_applicable: bool = False
    tree_depth_metric_applicable: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-ready audit fields with enum values serialized as strings."""
        result = asdict(self)
        result["status"] = self.status.value
        return result

    @classmethod
    def numerical_failure(cls, *, execution_mode: str, iterations: int,
                          warmup: int, chains: int, error: BaseException) -> BayesianDiagnosticResult:
        """Record a sampler/runtime exception without disguising it as convergence."""
        return cls(False, BayesianDiagnosticStatus.NUMERICAL_FAILURE, False, False,
                   0, 0, f"{type(error).__name__}:{error}", False,
                   "RETRY_OR_BLOCK", "PENALTYBLOG_DIFFERENTIAL_EVOLUTION",
                   execution_mode, iterations, warmup, chains, None, None)


def diagnose_sampler(*, diagnostics: Any, posterior: Any, max_r_hat: float,
                     min_ess: float, execution_mode: str, iterations: int,
                     warmup: int, chains: int) -> BayesianDiagnosticResult:
    """Validate all reported convergence metrics and posterior parameters."""
    try:
        r_values = np.asarray(diagnostics["r_hat"], dtype=np.float64)
        ess_values = np.asarray(diagnostics["ess"], dtype=np.float64)
        posterior_values = np.asarray(posterior, dtype=np.float64)
        if not r_values.size or not ess_values.size:
            raise ValueError("EMPTY_DIAGNOSTIC_SERIES")
        if not np.all(np.isfinite(r_values)) or not np.all(np.isfinite(ess_values)):
            raise ValueError("NONFINITE_DIAGNOSTIC")
        posterior_valid = bool(posterior_values.size and np.all(np.isfinite(posterior_values)))
        observed_rhat = float(np.max(r_values))
        observed_ess = float(np.min(ess_values))
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        return BayesianDiagnosticResult.numerical_failure(
            execution_mode=execution_mode, iterations=iterations, warmup=warmup,
            chains=chains, error=error)

    r_hat_ok = observed_rhat <= max_r_hat
    ess_ok = observed_ess >= min_ess
    status = (BayesianDiagnosticStatus.POSTERIOR_INVALID if not posterior_valid else
              BayesianDiagnosticStatus.R_HAT_FAILED if not r_hat_ok else
              BayesianDiagnosticStatus.ESS_FAILED if not ess_ok else
              BayesianDiagnosticStatus.PASS)
    success = status == BayesianDiagnosticStatus.PASS
    return BayesianDiagnosticResult(
        success, status, r_hat_ok, ess_ok, 0, 0, None, posterior_valid,
        "ACCEPT" if success else "RETRY_OR_BLOCK",
        "PENALTYBLOG_DIFFERENTIAL_EVOLUTION", execution_mode, iterations,
        warmup, chains, observed_rhat, observed_ess,
    )


def for_execution_mode(result: BayesianDiagnosticResult, mode: str,
                       recommendation: str) -> BayesianDiagnosticResult:
    """Copy a diagnostic with the operational attempt mode and disposition."""
    return replace(result, execution_mode=mode, recommendation=recommendation)
