"""No Bayesian fallback can bypass convergence gates."""

from __future__ import annotations

from erguoyuan_football.models.bayesian_diagnostics import (
    BayesianDiagnosticStatus,
    diagnose_sampler,
)
from erguoyuan_football.models.bayesian_sampler_manager import BayesianSamplerManager


def _failed_mcmc():
    return diagnose_sampler(
        diagnostics={"r_hat": [1.2], "ess": [50]}, posterior=[[0.2]],
        max_r_hat=1.05, min_ess=200, execution_mode="FULL_MCMC",
        iterations=500, warmup=250, chains=4,
    )


def test_retry_is_full_then_simplified_then_internal_map_only() -> None:
    sequence: list[str] = []
    result = BayesianSamplerManager().run(
        full_mcmc=lambda: (sequence.append("FULL_MCMC") or _failed_mcmc()),
        simplified_mcmc=lambda: (sequence.append("SIMPLIFIED_MCMC") or _failed_mcmc()),
        map_estimation=lambda: (sequence.append("MAP_ESTIMATION") or ((0.3, 0.4), None)),
    )
    assert sequence == ["FULL_MCMC", "SIMPLIFIED_MCMC", "MAP_ESTIMATION"]
    assert result.diagnostic.status == BayesianDiagnosticStatus.MAP_INTERNAL_ONLY
    assert result.production_eligible is False
    assert result.degraded is True


def test_successful_retry_requires_same_diagnostic_thresholds() -> None:
    passed = diagnose_sampler(
        diagnostics={"r_hat": [1.01], "ess": [350]}, posterior=[[0.2]],
        max_r_hat=1.05, min_ess=200, execution_mode="SIMPLIFIED_MCMC",
        iterations=1000, warmup=500, chains=4,
    )
    result = BayesianSamplerManager().run(
        full_mcmc=_failed_mcmc, simplified_mcmc=lambda: passed,
        map_estimation=lambda: (None, "MUST_NOT_RUN"),
    )
    assert result.production_eligible is True
    assert result.diagnostic.execution_mode == "SIMPLIFIED_MCMC"
