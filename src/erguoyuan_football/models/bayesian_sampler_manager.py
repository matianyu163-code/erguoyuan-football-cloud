"""Explicit retry policy for MCMC and validation-only MAP estimation."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from erguoyuan_football.models.bayesian_diagnostics import (
    BayesianDiagnosticResult,
    BayesianDiagnosticStatus,
    for_execution_mode,
)


@dataclass(frozen=True)
class BayesianSamplerOutcome:
    """Production-eligible MCMC result or an explicitly non-production fallback."""

    diagnostic: BayesianDiagnosticResult
    production_eligible: bool
    degraded: bool
    map_parameters: tuple[float, ...] | None = None
    attempts: tuple[BayesianDiagnosticResult, ...] = ()


class BayesianSamplerManager:
    """Try configured MCMC, then an operational retry, then validation-only MAP."""

    def run(self, *, full_mcmc: Callable[[], BayesianDiagnosticResult],
            simplified_mcmc: Callable[[], BayesianDiagnosticResult],
            map_estimation: Callable[[], tuple[tuple[float, ...] | None, str | None]]
            ) -> BayesianSamplerOutcome:
        """Never promote MAP to a production posterior prediction."""
        first = full_mcmc()
        first = for_execution_mode(first, "FULL_MCMC",
                                   "ACCEPT" if first.success else "RETRY_SIMPLIFIED_MCMC")
        if first.success:
            return BayesianSamplerOutcome(first, True, False, attempts=(first,))

        second = simplified_mcmc()
        second = for_execution_mode(second, "SIMPLIFIED_MCMC",
                                    "ACCEPT" if second.success else "RETRY_MAP_INTERNAL_ONLY")
        if second.success:
            return BayesianSamplerOutcome(second, True, False,
                                          attempts=(first, second))

        try:
            parameters, error = map_estimation()
        except (ArithmeticError, RuntimeError, ValueError, TypeError) as caught:
            parameters = None
            error = f"{type(caught).__name__}:{caught}"
        map_valid = parameters is not None and bool(parameters) and all(
            math.isfinite(value) for value in parameters)
        map_diag = BayesianDiagnosticResult(
            False,
            BayesianDiagnosticStatus.MAP_INTERNAL_ONLY if map_valid else
            BayesianDiagnosticStatus.POSTERIOR_INVALID,
            False, False, 0, 0, error, False,
            "INTERNAL_VALIDATION_ONLY" if map_valid else "BLOCKED",
            "PENALTYBLOG_HIERARCHICAL_MAP", "MAP_FALLBACK", 0, 0, 1,
            None, None,
        )
        return BayesianSamplerOutcome(map_diag, False, map_valid,
                                      tuple(parameters or ()) if map_valid else None,
                                      (first, second, map_diag))
