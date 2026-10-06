"""Market-anchored Bayesian Poisson with fitted history/market dependence."""

from __future__ import annotations

import math
import time
from datetime import datetime
from typing import Any, ClassVar, Literal, Self

import numpy as np
import yaml
from pydantic import Field
from scipy.optimize import minimize
from scipy.special import gammaln

from erguoyuan_football.contracts.common import (
    Availability,
    ExecutionStatus,
    ImplementationType,
    utc,
)
from erguoyuan_football.contracts.predictions import ModelPrediction
from erguoyuan_football.data.schemas import Fixture
from erguoyuan_football.data.snapshots import PredictionSnapshot
from erguoyuan_football.markets.schemas import (
    DependencyTag,
    MarketGoalFeatures,
    PredictionHorizon,
)
from erguoyuan_football.models.base import BaseFootballModel
from erguoyuan_football.models.config import ModelConfig
from erguoyuan_football.models.dynamic_bayes.likelihood import score_matrix_from_lambdas
from erguoyuan_football.models.training import InsufficientData, TrainingDataset

MARKET_DEPENDENCIES = tuple(item.value for item in (
    DependencyTag.MARKET_RAW, DependencyTag.MARKET_DEVIG, DependencyTag.MARKET_CONSENSUS,
    DependencyTag.MARKET_IMPLIED_GOALS, DependencyTag.MARKET_BAYES,
))


class HistoricalMarketBayesConfig(ModelConfig):
    """Priors and reliability controls; market influence is fitted, not fixed 50/50."""

    market_mode: Literal["FUSION", "HISTORICAL_ONLY"] = "FUSION"
    market_min_training_matches: int = Field(default=30, ge=12)
    market_prediction_horizon: PredictionHorizon = PredictionHorizon.CUSTOM
    market_attack_prior_sd: float = Field(default=0.45, gt=0)
    market_defence_prior_sd: float = Field(default=0.45, gt=0)
    market_global_log_rate_prior_sd: float = Field(default=0.7, gt=0)
    market_home_advantage_prior_mean: float = Field(default=0.12)
    market_home_advantage_prior_sd: float = Field(default=0.35, gt=0)
    market_anchor_prior_mean: float = 0.0
    market_anchor_prior_sd: float = Field(default=0.5, gt=0)
    market_posterior_draws: int = Field(default=400, ge=40, le=20_000)
    market_seed: int = 314159
    market_optimizer_max_iterations: int = Field(default=3_000, ge=100)
    market_reliability_bookmaker_scale: float = Field(default=3.0, gt=0)
    market_reliability_dispersion_scale: float = Field(default=5.0, ge=0)
    market_reliability_freshness_scale_seconds: int = Field(default=21_600, ge=1)
    market_reliability_fit_error_scale: float = Field(default=0.12, gt=0)
    market_max_fit_error: float = Field(default=0.12, gt=0)


class HistoricalMarketBayesianPoissonModel(BaseFootballModel):
    """Joint MAP/Laplace inference over team scoring strengths and a fitted market anchor."""

    model_id = "HISTORICAL_MARKET_BAYESIAN_POISSON_V1"
    model_name = "Historical + Market Bayesian Poisson"
    model_version = "1.0.0"
    implementation_type = ImplementationType.REAL_IMPLEMENTATION.value
    required_data = ("historical_goals", "market_odds")
    supports_score_matrix = True
    supports_expected_goals = True
    supports_1x2 = True
    allows_multiple_competitions: ClassVar[bool] = False

    def __init__(self) -> None:
        super().__init__()
        self.market_config = HistoricalMarketBayesConfig()
        self.parameter_mean: np.ndarray | None = None
        self.parameter_covariance: np.ndarray | None = None
        self.team_ids: tuple[str, ...] = ()
        self.team_index: dict[str, int] = {}
        self.market_center_home = 0.0
        self.market_center_away = 0.0
        self.market_training_features: dict[str, dict[str, float]] = {}
        self.posterior_draws: np.ndarray | None = None

    def prepare_config(self, config: ModelConfig) -> HistoricalMarketBayesConfig:
        """Convert shared controls to this model's independently hashed typed controls."""
        if isinstance(config, HistoricalMarketBayesConfig):
            return config
        return HistoricalMarketBayesConfig(**config.model_dump())

    def required_data_for(self, config: ModelConfig) -> tuple[str, ...]:
        """Expose market odds as required only for the actual fusion model path."""
        prepared = self.prepare_config(config)
        return ("historical_goals",) if prepared.market_mode == "HISTORICAL_ONLY" else self.required_data

    def prepare_training_data(self, data: TrainingDataset,
                              config: ModelConfig) -> TrainingDataset:
        """Remove all market rows from a historical-only ablation and its artifact hash."""
        prepared = self.prepare_config(config)
        return data.model_copy(update={"market_goal_features": ()}) if prepared.market_mode == "HISTORICAL_ONLY" else data

    def prediction_record(self, *args, **kwargs) -> ModelPrediction:
        """Keep market dependency tags even when the result is UNAVAILABLE or FAILED."""
        kwargs.setdefault("dependency_tags", MARKET_DEPENDENCIES
                          if self.market_config.market_mode == "FUSION" else ())
        return super().prediction_record(*args, **kwargs)

    def fit(self, training_data: TrainingDataset, trained_until: datetime,
            config: ModelConfig) -> Self:
        market_config = self.prepare_config(config)
        training_data = self.prepare_training_data(training_data, market_config)
        self.market_config = market_config
        return super().fit(training_data, trained_until, config)

    def _fit(self, data: TrainingDataset) -> None:
        use_market = self.market_config.market_mode == "FUSION"
        feature_by_match = {item.match_id: item for item in data.market_goal_features
                            if item.availability == Availability.AVAILABLE
                            and item.prediction_horizon == self.market_config.market_prediction_horizon}
        matches = tuple(sorted((row for row in data.matches if not use_market or row.match_id in feature_by_match),
                               key=lambda row: (row.kickoff_time, row.match_id)))
        minimum = self.market_config.market_min_training_matches if use_market else self.market_config.min_matches
        if len(matches) < minimum:
            raise InsufficientData(
                f"INSUFFICIENT_MARKET_TRAINING_MATCHES:{len(matches)}<"
                f"{minimum}"
            )
        self.team_ids = tuple(sorted({team for row in matches for team in (row.home_team_id, row.away_team_id)}))
        self.team_index = {team: index for index, team in enumerate(self.team_ids)}
        self.market_training_features = {
            row.match_id: (_reliability_features(feature_by_match[row.match_id], self.market_config)
                           if use_market else {"reliability": 0.0}) for row in matches
        }
        self.market_center_home = (float(np.mean([
            math.log(_required_market_lambda(feature_by_match[row.match_id], "home")) for row in matches
        ])) if use_market else 0.0)
        self.market_center_away = (float(np.mean([
            math.log(_required_market_lambda(feature_by_match[row.match_id], "away")) for row in matches
        ])) if use_market else 0.0)
        mean_goals = (sum(row.home_goals + row.away_goals for row in matches)
                      / (2.0 * len(matches)))
        mu_prior = math.log(max(0.1, mean_goals))
        initial = np.zeros(2 * len(self.team_ids) + 1, dtype=float)
        initial[0] = mu_prior
        initial[1] = self.market_config.market_home_advantage_prior_mean
        initial[-1] = self.market_config.market_anchor_prior_mean

        def negative_log_posterior(parameters: np.ndarray) -> float:
            global_rate, home_advantage, attack, defence, anchor = self._decode(parameters)
            log_likelihood = 0.0
            for row in matches:
                feature = feature_by_match.get(row.match_id)
                reliability = self.market_training_features[row.match_id]["reliability"]
                home = self.team_index[row.home_team_id]
                away = self.team_index[row.away_team_id]
                market_home = (math.log(_required_market_lambda(feature, "home")) - self.market_center_home
                               if feature is not None else 0.0)
                market_away = (math.log(_required_market_lambda(feature, "away")) - self.market_center_away
                               if feature is not None else 0.0)
                eta_home = (global_rate + (home_advantage if not row.neutral_venue else 0.0)
                            + attack[home] - defence[away] + anchor * reliability * market_home)
                eta_away = global_rate + attack[away] - defence[home] + anchor * reliability * market_away
                if not -8 < eta_home < 5 or not -8 < eta_away < 5:
                    return 1e100
                rate_home, rate_away = math.exp(eta_home), math.exp(eta_away)
                log_likelihood += (row.home_goals * eta_home - rate_home - gammaln(row.home_goals + 1)
                                   + row.away_goals * eta_away - rate_away - gammaln(row.away_goals + 1))
            penalty = 0.5 * ((global_rate - mu_prior) / self.market_config.market_global_log_rate_prior_sd) ** 2
            penalty += 0.5 * ((home_advantage - self.market_config.market_home_advantage_prior_mean)
                              / self.market_config.market_home_advantage_prior_sd) ** 2
            penalty += 0.5 * float(np.sum((attack / self.market_config.market_attack_prior_sd) ** 2))
            penalty += 0.5 * float(np.sum((defence / self.market_config.market_defence_prior_sd) ** 2))
            penalty += 0.5 * ((anchor - self.market_config.market_anchor_prior_mean)
                              / self.market_config.market_anchor_prior_sd) ** 2
            return -(log_likelihood - penalty)

        fitted = minimize(negative_log_posterior, initial, method="L-BFGS-B",
                          options={"maxiter": self.market_config.market_optimizer_max_iterations,
                                   "ftol": 1e-11, "gtol": 1e-7, "maxls": 40})
        if not fitted.success or not np.isfinite(fitted.x).all() or not math.isfinite(float(fitted.fun)):
            raise RuntimeError(f"MARKET_BAYESIAN_OPTIMIZER_FAILED:{fitted.message}")
        covariance = _laplace_covariance(negative_log_posterior, fitted.x)
        if not np.isfinite(covariance).all():
            raise RuntimeError("MARKET_BAYESIAN_POSTERIOR_COVARIANCE_INVALID")
        eigenvalues = np.linalg.eigvalsh(covariance)
        if np.min(eigenvalues) <= 0:
            raise RuntimeError("MARKET_BAYESIAN_POSTERIOR_NOT_POSITIVE_DEFINITE")
        self.parameter_mean = np.asarray(fitted.x, dtype=float)
        self.parameter_covariance = covariance
        rng = np.random.default_rng(self.market_config.market_seed)
        self.posterior_draws = rng.multivariate_normal(
            self.parameter_mean, self.parameter_covariance, size=self.market_config.market_posterior_draws,
            check_valid="raise",
        )
        anchor = float(self._decode(self.parameter_mean)[-1])
        devig_versions = {feature.devig_policy_version for feature in feature_by_match.values()
                          if feature.devig_policy_version is not None}
        self.metadata.update({
            "method_family": "MARKET_ANCHORED_BAYESIAN_GOAL_MODEL",
            "posterior_method": "JOINT_POISSON_MAP_WITH_LAPLACE_APPROXIMATION",
            "market_anchor_structure": "FITTED_LOG_LAMBDA_COVARIATE_WITH_MARKET_UNCERTAINTY_RELIABILITY",
            "market_anchor_strength_posterior_mean": anchor,
            "market_anchor_strength_posterior_sd": float(math.sqrt(covariance[-1, -1])),
            "market_center_home": self.market_center_home,
            "market_center_away": self.market_center_away,
            "market_training_sample_count": len(matches),
            "market_mode": self.market_config.market_mode,
            "market_prediction_horizon": self.market_config.market_prediction_horizon.value,
            "market_snapshot_ids": sorted({feature_by_match[row.match_id].market_snapshot_id
                                            for row in matches if row.match_id in feature_by_match}),
            "devig_policy_versions": sorted(devig_versions),
            "market_dependency_ids": sorted({identifier for feature in feature_by_match.values()
                                              for identifier in feature.dependency_ids}),
            "uses_market": use_market,
            "uses_historical_goals": True,
            "dependency_tags": MARKET_DEPENDENCIES if use_market else (),
            "training_market_reliability_mean": float(np.mean([
                self.market_training_features[row.match_id]["reliability"] for row in matches
            ])),
            "laplace_covariance_min_eigenvalue": float(np.min(eigenvalues)),
            "posterior_draw_count": len(self.posterior_draws),
            "optimizer_iterations": int(fitted.nit),
            "optimizer_objective": float(fitted.fun),
            "market_config": self.market_config.model_dump(mode="json"),
        })

    def _decode(self, parameters: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray, float]:
        """Expand identifiable sum-to-zero attack and defence coordinates."""
        count = len(self.team_ids)
        global_rate, home_advantage = float(parameters[0]), float(parameters[1])
        attack_free = parameters[2:2 + count - 1]
        defence_free = parameters[2 + count - 1:2 + 2 * (count - 1)]
        attack = np.append(attack_free, -float(np.sum(attack_free)))
        defence = np.append(defence_free, -float(np.sum(defence_free)))
        return global_rate, home_advantage, attack, defence, float(parameters[-1])

    def _predict_values(self, match: Fixture) -> dict[str, Any]:
        raise RuntimeError("MARKET_MODEL_REQUIRES_FROZEN_MARKET_FEATURES")

    def predict(self, match: Fixture, prediction_snapshot: PredictionSnapshot) -> ModelPrediction:
        """Predict only from a matching, successful market-goal fit frozen at this cutoff."""
        started = time.perf_counter()
        try:
            self.validate_prediction_input(match, prediction_snapshot)
            feature = None
            if self.market_config.market_mode == "FUSION":
                feature = prediction_snapshot.market_goal_features
                if feature is None or feature.availability != Availability.AVAILABLE:
                    raise InsufficientData("MISSING_REQUIRED_MARKET_GOAL_FEATURES")
                if (feature.match_id != match.match_id or feature.prediction_time != prediction_snapshot.prediction_time
                        or feature.prediction_horizon != self.market_config.market_prediction_horizon
                        or feature.as_of_time is None or feature.retrieved_at is None
                        or max(feature.as_of_time, feature.retrieved_at) > prediction_snapshot.prediction_time):
                    raise InsufficientData("POINT_IN_TIME_GUARD_V1_MARKET_FEATURE_MISMATCH")
                if feature.fit_error is None or feature.fit_error > self.market_config.market_max_fit_error:
                    raise InsufficientData("MARKET_GOAL_FIT_POOR")
            if match.home_team_id not in self.team_index or match.away_team_id not in self.team_index:
                raise InsufficientData("UNSEEN_TEAM")
            values = self._predict_with_market(match, feature)
            return self.prediction_record(match, prediction_snapshot, ExecutionStatus.SUCCESS,
                elapsed_ms=(time.perf_counter() - started) * 1000, values=values,
                dependency_tags=MARKET_DEPENDENCIES if feature is not None else ())
        except InsufficientData as error:
            return self.prediction_record(match, prediction_snapshot, ExecutionStatus.UNAVAILABLE,
                                          reason=str(error), dependency_tags=MARKET_DEPENDENCIES
                                          if self.market_config.market_mode == "FUSION" else ())
        except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            return self.prediction_record(match, prediction_snapshot, ExecutionStatus.FAILED,
                reason=f"{type(error).__name__}:{error}", dependency_tags=MARKET_DEPENDENCIES
                if self.market_config.market_mode == "FUSION" else ())

    def _predict_with_market(self, match: Fixture, feature: MarketGoalFeatures | None) -> dict[str, Any]:
        if self.posterior_draws is None or self.parameter_mean is None:
            raise InsufficientData("MODEL_NOT_FITTED")
        feature_reliability = (_reliability_features(feature, self.market_config) if feature is not None
                               else {"reliability": 0.0})
        home_index, away_index = self.team_index[match.home_team_id], self.team_index[match.away_team_id]
        lambdas_home, lambdas_away = [], []
        historical_home, historical_away = [], []
        for draw in self.posterior_draws:
            global_rate, home_advantage, attack, defence, anchor = self._decode(draw)
            history_h = global_rate + (home_advantage if not match.neutral_venue else 0.0) + attack[home_index] - defence[away_index]
            history_a = global_rate + attack[away_index] - defence[home_index]
            history_h_rate, history_a_rate = math.exp(history_h), math.exp(history_a)
            market_h = (math.log(_required_market_lambda(feature, "home")) - self.market_center_home
                        if feature is not None else 0.0)
            market_a = (math.log(_required_market_lambda(feature, "away")) - self.market_center_away
                        if feature is not None else 0.0)
            lambdas_home.append(math.exp(history_h + anchor * feature_reliability["reliability"] * market_h))
            lambdas_away.append(math.exp(history_a + anchor * feature_reliability["reliability"] * market_a))
            historical_home.append(history_h_rate)
            historical_away.append(history_a_rate)
        matrix = score_matrix_from_lambdas(np.asarray(lambdas_home), np.asarray(lambdas_away),
                                           self.market_config.max_goals)
        outcome = matrix.outcome()
        home_ci = np.quantile(lambdas_home, [0.025, 0.975])
        away_ci = np.quantile(lambdas_away, [0.025, 0.975])
        return {
            "lambda_home": float(np.mean(lambdas_home)), "lambda_away": float(np.mean(lambdas_away)),
            "expected_home_goals": float(np.mean(lambdas_home)),
            "expected_away_goals": float(np.mean(lambdas_away)), "score_matrix": matrix.values,
            "p_home": outcome.p_home, "p_draw": outcome.p_draw, "p_away": outcome.p_away,
            "metadata": {
                "historical_lambda_home": float(np.mean(historical_home)),
                "historical_lambda_away": float(np.mean(historical_away)),
                "market_lambda_home": feature.market_implied_lambda_home if feature is not None else None,
                "market_lambda_away": feature.market_implied_lambda_away if feature is not None else None,
                "market_anchor_strength": (float(self._decode(self.parameter_mean)[-1])
                                            if feature is not None else 0.0),
                "market_anchor_reliability": feature_reliability["reliability"],
                "market_reliability_features": feature_reliability,
                "market_consensus_source_count": feature.source_count if feature is not None else 0,
                "market_consensus_bookmaker_count": feature.bookmaker_count if feature is not None else 0,
                "market_dispersion": feature.dispersion if feature is not None else {},
                "market_fit_error": feature.fit_error if feature is not None else None,
                "market_snapshot_id": feature.market_snapshot_id if feature is not None else None,
                "devig_policy_version": feature.devig_policy_version if feature is not None else None,
                "market_dependency_ids": feature.dependency_ids if feature is not None else (),
                "market_source_ids": feature.source_ids if feature is not None else (),
                "inference_mode": feature.inference_mode if feature is not None else "HISTORICAL_ONLY_ABLATION",
                "lambda_home_ci": tuple(float(value) for value in home_ci),
                "lambda_away_ci": tuple(float(value) for value in away_ci),
                "posterior_method": "JOINT_POISSON_MAP_WITH_LAPLACE_APPROXIMATION",
                "score_matrix_retained_mass": matrix.retained_mass,
                "score_matrix_tail_mass": matrix.tail_mass,
                "market_dependency_tags": MARKET_DEPENDENCIES if feature is not None else (),
                "uses_market": feature is not None,
                "uses_historical_goals": True,
            },
        }


def _reliability_features(feature: MarketGoalFeatures,
                          config: HistoricalMarketBayesConfig) -> dict[str, float]:
    """Configurable market-uncertainty reliability; beta remains jointly fit on goals."""
    if feature.as_of_time is None:
        raise ValueError("MARKET_FEATURE_SOURCE_TIME_MISSING")
    age = max(0.0, (utc(feature.prediction_time) - utc(feature.as_of_time)).total_seconds())
    mad_values = [value for key, value in feature.dispersion.items() if key.endswith(".mad")]
    dispersion_values = mad_values or list(feature.dispersion.values())
    dispersion = float(np.mean(dispersion_values)) if dispersion_values else 0.0
    count_factor = feature.bookmaker_count / (feature.bookmaker_count + config.market_reliability_bookmaker_scale)
    dispersion_factor = math.exp(-config.market_reliability_dispersion_scale * max(0.0, dispersion))
    freshness_factor = math.exp(-age / config.market_reliability_freshness_scale_seconds)
    fit_error = feature.fit_error or 0.0
    fit_factor = math.exp(-fit_error / config.market_reliability_fit_error_scale)
    reliability = min(1.0, max(0.0, count_factor * dispersion_factor * freshness_factor * fit_factor))
    return {"age_seconds": age, "mean_dispersion": dispersion, "bookmaker_count_factor": count_factor,
            "dispersion_factor": dispersion_factor, "freshness_factor": freshness_factor,
            "fit_error_factor": fit_factor, "reliability": reliability}


def _required_market_lambda(feature: MarketGoalFeatures, side: Literal["home", "away"]) -> float:
    """Return a validated positive fitted market lambda or fail without fallback."""
    value = feature.market_implied_lambda_home if side == "home" else feature.market_implied_lambda_away
    if value is None or not math.isfinite(value) or value <= 0:
        raise InsufficientData(f"MISSING_OR_INVALID_MARKET_LAMBDA:{side.upper()}")
    return value


def _laplace_covariance(objective, optimum: np.ndarray) -> np.ndarray:
    """Numerically form a symmetric Hessian for a local Laplace posterior."""
    count = len(optimum)
    step = 1e-4
    hessian = np.zeros((count, count), dtype=float)
    center = float(objective(optimum))
    for left in range(count):
        left_shift = np.zeros(count)
        left_shift[left] = step
        hessian[left, left] = (objective(optimum + left_shift) - 2.0 * center
                               + objective(optimum - left_shift)) / (step * step)
        for right in range(left):
            right_shift = np.zeros(count)
            right_shift[right] = step
            value = (objective(optimum + left_shift + right_shift)
                     - objective(optimum + left_shift - right_shift)
                     - objective(optimum - left_shift + right_shift)
                     + objective(optimum - left_shift - right_shift)) / (4.0 * step * step)
            hessian[left, right] = value
            hessian[right, left] = value
    hessian = (hessian + hessian.T) / 2
    if not np.isfinite(hessian).all():
        raise RuntimeError("MARKET_BAYESIAN_HESSIAN_NON_FINITE")
    eigenvalues = np.linalg.eigvalsh(hessian)
    if np.min(eigenvalues) <= 0:
        raise RuntimeError("MARKET_BAYESIAN_HESSIAN_NOT_POSITIVE_DEFINITE")
    return np.linalg.inv(hessian)


def load_historical_market_bayes_config(path: str) -> HistoricalMarketBayesConfig:
    """Load an explicitly versioned model config; unknown keys are rejected."""
    with open(path, encoding="utf-8") as stream:
        values = yaml.safe_load(stream)
    if not isinstance(values, dict):
        raise TypeError("MARKET_BAYES_CONFIG_MUST_BE_A_MAPPING")
    expected_id = HistoricalMarketBayesianPoissonModel.model_id
    expected_version = HistoricalMarketBayesianPoissonModel.model_version
    if values.pop("model_id", None) != expected_id or values.pop("model_version", None) != expected_version:
        raise ValueError("MARKET_BAYES_CONFIG_MODEL_ID_OR_VERSION_MISMATCH")
    return HistoricalMarketBayesConfig.model_validate(values)
