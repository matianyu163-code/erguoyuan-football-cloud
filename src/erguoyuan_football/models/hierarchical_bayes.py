"""Hierarchical Bayesian goal model V1 through penaltyblog MCMC adapter."""

from __future__ import annotations

import math
import warnings

import numpy as np
from penaltyblog.models import HierarchicalBayesianGoalModel

from erguoyuan_football.models.base import BaseFootballModel
from erguoyuan_football.models.bayesian_diagnostics import (
    BayesianDiagnosticResult,
    diagnose_sampler,
)
from erguoyuan_football.models.bayesian_sampler_manager import BayesianSamplerManager
from erguoyuan_football.models.penaltyblog_adapter import PenaltyblogAdapter
from erguoyuan_football.models.score_matrix import ScoreMatrix
from erguoyuan_football.models.training import TrainingDataset
from erguoyuan_football.research.samples.bayesian_prior import BayesianPrior
from erguoyuan_football.research.samples.prior_influence_report import (
    build_prior_influence_report,
)


class BayesianSamplingError(RuntimeError):
    """A Bayesian run failed explicit sampling gates; carries immutable audit detail."""

    def __init__(self, diagnostic: dict[str, object]) -> None:
        self.diagnostic = diagnostic
        final = diagnostic.get("final", diagnostic)
        final_diagnostic = final if isinstance(final, dict) else {}
        status = final_diagnostic.get("status", "UNKNOWN")
        mode = final_diagnostic.get("execution_mode", "FAILED")
        super().__init__(f"SAMPLING_DIAGNOSTICS_FAILED:{mode}:{status}")


class CoreHierarchicalBayesianModel(BaseFootballModel):
    """League-level attack/defence shrinkage with convergence diagnostics."""

    model_id = "BAYESIAN_HIERARCHICAL_V1"
    model_name = "Bayesian Hierarchical"
    model_version = "1.0.0"
    required_data = ("historical_goals",)

    def __init__(self) -> None:
        super().__init__()
        self.adapter = PenaltyblogAdapter(HierarchicalBayesianGoalModel)
        self.diagnostics: dict[str, float] = {}
        self.bayesian_diagnostic: dict[str, object] = {}
        self.prediction_prior: BayesianPrior | None = None
        self.prior_direct_count = 0

    def set_prediction_prior(self, prior: BayesianPrior | None, *, direct_count: int) -> None:
        """Attach a disjoint, cutoff-validated empirical prior for prediction."""
        if direct_count < 0:
            raise ValueError("DIRECT_COUNT_MUST_BE_NONNEGATIVE")
        if prior is not None and (prior.sample_count < 1 or prior.effective_strength <= 0):
            raise ValueError("INVALID_BAYESIAN_PRIOR")
        self.prediction_prior = prior
        self.prior_direct_count = direct_count

    def _fit(self, data: TrainingDataset) -> None:
        warning_counts: dict[str, int] = {"FULL_MCMC": 0, "SIMPLIFIED_MCMC": 0}

        def run_mcmc(mode: str, *, chains: int, draws: int, tune: int,
                     seed: int) -> BayesianDiagnosticResult:
            attempt_config = self.config.model_copy(update={
                "chains": chains, "draws": draws, "tune": tune, "seed": seed,
            })
            np.random.seed(seed)
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    self.adapter.fit_bayesian(data, attempt_config, n_cores=1)
                warning_counts[mode] = len(caught)
                backend = self.adapter.backend
                if backend is None or backend.trace is None:
                    raise RuntimeError("BAYESIAN_BACKEND_OR_TRACE_MISSING")
                return diagnose_sampler(
                    diagnostics=backend.get_diagnostics(), posterior=backend.trace,
                    max_r_hat=self.config.max_rhat, min_ess=self.config.min_ess,
                    execution_mode=mode, iterations=draws, warmup=tune, chains=chains,
                )
            except (ArithmeticError, RuntimeError, ValueError, TypeError) as error:
                return BayesianDiagnosticResult.numerical_failure(
                    execution_mode=mode, iterations=draws, warmup=tune,
                    chains=chains, error=error)

        def full_mcmc() -> BayesianDiagnosticResult:
            return run_mcmc("FULL_MCMC", chains=self.config.chains,
                            draws=self.config.draws, tune=self.config.tune,
                            seed=self.config.seed)

        def simplified_mcmc() -> BayesianDiagnosticResult:
            # Keep the same likelihood, prior and convergence thresholds. The retry
            # adds independent chains and a distinct seed while remaining sequential.
            return run_mcmc("SIMPLIFIED_MCMC", chains=max(4, self.config.chains),
                            draws=self.config.draws, tune=self.config.tune,
                            seed=self.config.seed + 1)

        manager = BayesianSamplerManager()
        outcome = manager.run(full_mcmc=full_mcmc, simplified_mcmc=simplified_mcmc,
            map_estimation=lambda: self.adapter.fit_bayesian_map_internal(data, self.config))
        self.bayesian_diagnostic = {
            "final": outcome.diagnostic.to_dict(),
            "attempts": [attempt.to_dict() for attempt in outcome.attempts],
            "warning_counts": warning_counts,
        }
        self.diagnostics = {
            "max_rhat": float(outcome.diagnostic.max_r_hat or 0.0),
            "min_ess": float(outcome.diagnostic.min_ess or 0.0),
            "divergences": float(outcome.diagnostic.divergence_count),
            "sampling_warnings": float(sum(warning_counts.values())),
        }
        self.metadata.update({"sampling_status": outcome.diagnostic.status.value,
                              "sampling_mode": outcome.diagnostic.execution_mode,
                              "sampling_degraded": outcome.degraded,
                              "shrinkage": "league_level_attack_defence",
                              "diagnostics": self.diagnostics,
                              "bayesian_diagnostic": self.bayesian_diagnostic})
        if not outcome.production_eligible:
            raise BayesianSamplingError(self.bayesian_diagnostic)
        backend = self.adapter.backend
        if backend is None or backend.trace is None:
            raise BayesianSamplingError(self.bayesian_diagnostic)
        self.metadata["posterior_samples"] = int(backend.trace.shape[0])

    def _predict_values(self, match):
        values = self.adapter.predict(match.home_team_id, match.away_team_id,
                                       max_goals=self.config.max_goals, neutral_venue=bool(match.neutral_venue))
        values["metadata"]["sampling_status"] = self.metadata.get("sampling_status")
        intervals = self.adapter.posterior_lambda_intervals(
            match.home_team_id, match.away_team_id, neutral_venue=bool(match.neutral_venue))
        if intervals is not None:
            values["metadata"]["lambda_home_ci"] = intervals[0]
            values["metadata"]["lambda_away_ci"] = intervals[1]
        else:
            values["metadata"]["credible_interval_status"] = "UNAVAILABLE"
        if self.prediction_prior is not None:
            values = self._apply_empirical_prior(values, self.prediction_prior,
                                                 self.prior_direct_count)
        return values

    def _apply_empirical_prior(self, values: dict, prior: BayesianPrior,
                               direct_count: int) -> dict:
        """Pool direct posterior rates with a capped Gamma-rate prior summary."""
        prior_strength = prior.effective_strength
        denominator = direct_count + prior_strength
        if denominator <= 0:
            raise ValueError("BAYESIAN_PRIOR_WEIGHT_INVALID")
        before_home = float(values["lambda_home"])
        before_away = float(values["lambda_away"])
        home_rate = (prior.home_goal_mean * prior_strength + before_home * direct_count) / denominator
        away_rate = (prior.away_goal_mean * prior_strength + before_away * direct_count) / denominator
        matrix_info = values["metadata"].get("score_matrix")
        if not isinstance(matrix_info, dict):
            raise TypeError("BAYESIAN_PRIOR_SCORE_MATRIX_MISSING")
        original = np.asarray(values["score_matrix"], dtype=np.float64)
        old_home, old_away = before_home, before_away
        adjusted = np.zeros_like(original)
        for home_goals in range(original.shape[0]):
            for away_goals in range(original.shape[1]):
                ratio = _poisson_ratio(home_goals, home_rate, old_home)
                ratio *= _poisson_ratio(away_goals, away_rate, old_away)
                adjusted[home_goals, away_goals] = original[home_goals, away_goals] * ratio
        conditional = adjusted / float(adjusted.sum())
        max_goals = original.shape[0] - 1
        retained = (_poisson_support_mass(home_rate, max_goals)
                    * _poisson_support_mass(away_rate, max_goals))
        score = ScoreMatrix(values=tuple(tuple(float(value) for value in row)
                                         for row in conditional),
                            max_goals=max_goals, retained_mass=retained,
                            tail_mass=1 - retained)
        outcome = score.outcome()
        prior_influence = build_prior_influence_report(
            direct_sample_count=direct_count, prior_sample_count=prior.sample_count,
            prior_strength=prior_strength,
            direct_posterior_parameters=(before_home, before_away),
            posterior_parameters=(home_rate, away_rate))
        values.update({"lambda_home": home_rate, "lambda_away": away_rate,
                       "expected_home_goals": home_rate,
                       "expected_away_goals": away_rate,
                       "score_matrix": score.values,
                       "p_home": outcome.p_home, "p_draw": outcome.p_draw,
                       "p_away": outcome.p_away})
        values["metadata"].update({
            "score_matrix": score.model_dump(mode="json"),
            "bayesian_prior": {
                "prior_type": prior.prior_type,
                "sample_count": prior.sample_count,
                "effective_strength": prior.effective_strength,
                "attack_mean": prior.attack_mean,
                "attack_variance": prior.attack_variance,
                "defense_mean": prior.defense_mean,
                "defense_variance": prior.defense_variance,
                "home_advantage_mean": prior.home_advantage_mean,
                "home_advantage_variance": prior.home_advantage_variance,
                "home_goal_mean": prior.home_goal_mean,
                "away_goal_mean": prior.away_goal_mean,
                "source_tier": prior.source_tier,
                "recency_factor": prior.recency_factor,
                "similarity_factor": prior.similarity_factor,
                "evidence_ids": list(prior.evidence_ids),
            },
            "posterior_rate_pooling": {
                "direct_sample_count": direct_count,
                "prior_sample_count": prior.sample_count,
                "posterior_lambda_home": home_rate,
                "posterior_lambda_away": away_rate,
                "pre_prior_lambda_home": before_home,
                "pre_prior_lambda_away": before_away,
                "prior_weight": prior_strength / denominator,
                "method": "GAMMA_RATE_MEAN_POOLING_V1",
                "tail_estimation_method": "INDEPENDENT_POISSON_MARGINAL_TAIL",
                "pre_prior_retained_mass": matrix_info["retained_mass"],
            },
            "prior_influence_report": prior_influence.to_dict(),
        })
        return values


def _poisson_ratio(goals: int, new_rate: float, old_rate: float) -> float:
    """Stable ratio of Poisson cell masses used to retain fitted score dependence."""
    if min(new_rate, old_rate) <= 0 or not all(
            math.isfinite(rate) for rate in (new_rate, old_rate)):
        raise ValueError("INVALID_POISSON_RATE")
    return math.exp(old_rate - new_rate + goals * math.log(new_rate / old_rate))


def _poisson_support_mass(rate: float, max_goals: int) -> float:
    """Compute the finite marginal mass represented by an inclusive score support."""
    return sum(math.exp(-rate + goals * math.log(rate) - math.lgamma(goals + 1))
               for goals in range(max_goals + 1))
