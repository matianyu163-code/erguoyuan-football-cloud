"""Hierarchical Elo with optional provenance-checked xG score-distribution updates."""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any, ClassVar, Self

import numpy as np
from pydantic import Field, model_validator
from sklearn.linear_model import LogisticRegression

from erguoyuan_football.contracts.common import ImplementationType, utc
from erguoyuan_football.contracts.predictions import ProbabilityVector
from erguoyuan_football.models.base import BaseFootballModel
from erguoyuan_football.models.config import ModelConfig
from erguoyuan_football.models.dynamic_bayes.likelihood import score_matrix_from_lambdas
from erguoyuan_football.models.training import (
    TeamHierarchyMembership,
    TrainingDataset,
    TrainingXGObservation,
)


class OptaLikeConfig(ModelConfig):
    """CORE-estimated model controls; none claim to reproduce private Opta parameters."""

    result_weight: float = Field(default=0.8, ge=0, le=1)
    xg_weight: float = Field(default=0.2, ge=0, le=1)
    hierarchy_k_factor: float = Field(default=20.0, gt=0, le=200)
    hierarchy_home_advantage: float = Field(default=60.0, ge=-300, le=300)
    xg_max_goals: int = Field(default=10, ge=2, le=30)
    power_min_rating: float = 0.0
    power_max_rating: float = 400.0

    @model_validator(mode="after")
    def weights_sum(self) -> OptaLikeConfig:
        if not math.isclose(self.result_weight + self.xg_weight, 1.0, abs_tol=1e-9):
            raise ValueError("result_weight and xg_weight must sum to one")
        if self.power_max_rating <= self.power_min_rating:
            raise ValueError("power rating range must increase")
        return self


class HierarchyDeltaAllocation:
    """Allocate a cross-competition result to only its highest affected shared level."""

    @staticmethod
    def affected_level(home: TeamHierarchyMembership | None,
                       away: TeamHierarchyMembership | None) -> str | None:
        if home is None or away is None:
            return None
        if home.league_id == away.league_id:
            return None
        if home.country_id == away.country_id:
            return "LEAGUE"
        if home.continent_id == away.continent_id:
            return "COUNTRY"
        return "CONTINENT"


class HierarchicalProbabilityMapper:
    """PIT-fitted multinomial logistic mapping from raw hierarchy strength to 1X2."""

    version = "1.0.0"

    def __init__(self) -> None:
        self.model: LogisticRegression | None = None
        self.trained_until: datetime | None = None
        self.training_sample_count = 0

    def fit(self, features: list[tuple[float, float, float]], outcomes: list[int], trained_until) -> None:
        if len(features) < 12 or len(set(outcomes)) != 3:
            raise ValueError("HIERARCHICAL_MAPPER_REQUIRES_12_ROWS_AND_3_OUTCOMES")
        self.model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        self.model.fit(np.asarray(features, dtype=float), np.asarray(outcomes, dtype=int))
        self.trained_until = utc(trained_until)
        self.training_sample_count = len(features)

    def predict(self, rating_difference: float, home_advantage: float, cross_league: bool) -> ProbabilityVector:
        if self.model is None:
            raise ValueError("HIERARCHICAL_MAPPER_UNAVAILABLE")
        raw = self.model.predict_proba([[rating_difference, home_advantage, float(cross_league)]])[0]
        values = dict(zip(self.model.classes_, raw, strict=True))
        return ProbabilityVector(p_home=float(values.get(0, 0)), p_draw=float(values.get(1, 0)),
                                 p_away=float(values.get(2, 0)))


class CorePowerTransformer:
    """CORE_ESTIMATED monotonic raw-rating min-max transform; not a win probability."""

    method = "CORE_ESTIMATED"

    @staticmethod
    def transform(raw: float, observed: list[float]) -> float:
        if not observed:
            raise ValueError("POWER_TRANSFORM_REQUIRES_OBSERVED_RATINGS")
        low, high = min(observed), max(observed)
        if math.isclose(low, high):
            return 50.0
        return 100.0 * (raw - low) / (high - low)


class CoreOptaXGEloLikeModel(BaseFootballModel):
    """CORE xG-Elo-inspired model with team updates and optional versioned hierarchy."""

    model_id = "CORE_OPTA_XG_ELO_LIKE_V1"
    model_name = "CORE Opta xG-Elo-like"
    model_version = "1.0.0"
    implementation_type = ImplementationType.LIKE_IMPLEMENTATION.value
    required_data = ("historical_results",)
    optional_data = ("xg", "league_hierarchy")
    supports_score_matrix = False
    supports_expected_goals = False
    allows_multiple_competitions: ClassVar[bool] = True

    def __init__(self) -> None:
        super().__init__()
        self.opta_config = OptaLikeConfig()
        self.team_rating: dict[str, float] = {}
        self.league_rating: dict[str, float] = {}
        self.country_rating: dict[str, float] = {}
        self.continent_rating: dict[str, float] = {}
        self.hierarchy_history: list[dict[str, Any]] = []
        self.team_pre_match_history: list[dict[str, Any]] = []
        self.memberships: tuple[TeamHierarchyMembership, ...] = ()
        self.mapper = HierarchicalProbabilityMapper()
        self.xg_history: tuple[TrainingXGObservation, ...] = ()

    def fit(self, training_data: TrainingDataset, trained_until, config: ModelConfig) -> Self:
        """Fit the result mapper and update hierarchy in strict kickoff order."""
        self.opta_config = OptaLikeConfig(**config.model_dump())
        self.team_rating = {}
        self.league_rating = {}
        self.country_rating = {}
        self.continent_rating = {}
        self.hierarchy_history = []
        self.team_pre_match_history = []
        self.mapper = HierarchicalProbabilityMapper()
        self.memberships = training_data.team_hierarchy_timeline
        self.xg_history = training_data.xg_observations
        return super().fit(training_data, trained_until, self.opta_config)

    def prepare_config(self, config: ModelConfig) -> ModelConfig:
        """Expose the typed effective configuration to artifact-cache lookups."""
        return OptaLikeConfig(**config.model_dump())

    def _membership(self, team_id: str, at) -> TeamHierarchyMembership | None:
        cutoff = utc(at)
        valid = [item for item in self.memberships
                 if item.team_id == team_id and item.as_of_time <= cutoff
                 and item.valid_from <= cutoff and (item.valid_to is None or cutoff < item.valid_to)]
        valid.sort(key=lambda item: (item.valid_from, item.as_of_time))
        return valid[-1] if valid else None

    def _raw_rating(self, team_id: str, membership: TeamHierarchyMembership | None) -> float:
        return (self.team_rating.get(team_id, self.config.initial_rating)
                + (self.league_rating.get(membership.league_id, 0) if membership else 0)
                + (self.country_rating.get(membership.country_id, 0) if membership else 0)
                + (self.continent_rating.get(membership.continent_id, 0) if membership else 0))

    def _expected_home(self, difference: float, neutral: bool) -> float:
        advantage = 0 if neutral else self.opta_config.hierarchy_home_advantage
        return 1 / (1 + 10 ** (-(difference + advantage) / self.config.elo_scale))

    def _xg_delta(self, xg_home: float, xg_away: float, difference: float, neutral: bool) -> float:
        matrix = score_matrix_from_lambdas(np.asarray([xg_home]), np.asarray([xg_away]),
                                           self.opta_config.xg_max_goals)
        expected = self._expected_home(difference, neutral)
        delta = 0.0
        for home_goals, row in enumerate(matrix.values):
            for away_goals, probability in enumerate(row):
                result = 1.0 if home_goals > away_goals else 0.5 if home_goals == away_goals else 0.0
                margin = math.log1p(abs(home_goals - away_goals)) if self.config.goal_difference_adjustment else 1.0
                delta += probability * self.opta_config.hierarchy_k_factor * max(1.0, margin) * (result - expected)
        return delta

    def _fit(self, data: TrainingDataset) -> None:
        self.team_rating = {team: self.config.initial_rating for team in self.teams}
        xg_by_match = {item.match_id: item for item in data.xg_observations}
        mapper_features: list[tuple[float, float, float]] = []
        outcomes: list[int] = []
        xg_count = 0
        allocation_counts = {"TEAM": 0, "LEAGUE": 0, "COUNTRY": 0, "CONTINENT": 0}
        for row in sorted(data.matches, key=lambda item: (item.kickoff_time, item.match_id)):
            pre_time = utc(row.kickoff_time) - timedelta(microseconds=1)
            home_membership = self._membership(row.home_team_id, pre_time)
            away_membership = self._membership(row.away_team_id, pre_time)
            home_raw = self._raw_rating(row.home_team_id, home_membership)
            away_raw = self._raw_rating(row.away_team_id, away_membership)
            neutral = row.neutral_venue
            advantage = 0.0 if neutral else self.opta_config.hierarchy_home_advantage
            cross_league = bool(home_membership and away_membership and
                                home_membership.league_id != away_membership.league_id)
            mapper_features.append((home_raw - away_raw, advantage, float(cross_league)))
            outcome = 0 if row.home_goals > row.away_goals else 1 if row.home_goals == row.away_goals else 2
            outcomes.append(outcome)
            self.team_pre_match_history.extend((
                {"team_id": row.home_team_id, "match_id": row.match_id, "as_of_time": pre_time,
                 "rating": home_raw, "phase": "PRE_MATCH"},
                {"team_id": row.away_team_id, "match_id": row.match_id, "as_of_time": pre_time,
                 "rating": away_raw, "phase": "PRE_MATCH"},
            ))
            expected = self._expected_home(home_raw - away_raw, neutral)
            actual = 1.0 if outcome == 0 else 0.5 if outcome == 1 else 0.0
            margin = math.log1p(abs(row.home_goals - row.away_goals)) if self.config.goal_difference_adjustment else 1.0
            result_delta = self.opta_config.hierarchy_k_factor * max(1.0, margin) * (actual - expected)
            xg = xg_by_match.get(row.match_id)
            xg_delta = 0.0
            if xg:
                xg_delta = self._xg_delta(xg.xg_home, xg.xg_away, home_raw - away_raw, neutral)
                xg_count += 1
            delta = ((self.opta_config.result_weight * result_delta + self.opta_config.xg_weight * xg_delta)
                     if xg else result_delta)
            # Team components are updated for every match, regardless of hierarchy coverage.
            self.team_rating[row.home_team_id] += delta
            self.team_rating[row.away_team_id] -= delta
            allocation_counts["TEAM"] += 1
            level = HierarchyDeltaAllocation.affected_level(home_membership, away_membership)
            if level:
                container = {"LEAGUE": self.league_rating, "COUNTRY": self.country_rating,
                             "CONTINENT": self.continent_rating}[level]
                home_key = getattr(home_membership, f"{level.lower()}_id")
                away_key = getattr(away_membership, f"{level.lower()}_id")
                container[home_key] = container.get(home_key, 0.0) + delta * 0.5
                container[away_key] = container.get(away_key, 0.0) - delta * 0.5
                allocation_counts[level] += 1
                self.hierarchy_history.append({"match_id": row.match_id, "as_of_time": row.available_at,
                    "highest_affected_level": level, "delta": delta, "phase": "POST_MATCH"})
        self.mapper.fit(mapper_features, outcomes, self.trained_until)
        if self.mapper.trained_until is None:
            raise ValueError("HIERARCHICAL_MAPPER_TRAINING_CUTOFF_MISSING")
        self.metadata.update({
            "method_family": "OPTA_POWER_RANKINGS_PUBLIC_METHOD_INSPIRED",
            "official_model": "NOT_OFFICIAL_OPTA_MODEL",
            "rating_names": ["CORE_HIERARCHICAL_ELO", "CORE_POWER_RATING"],
            "hierarchy_delta_allocation": "CORE_ESTIMATED_HIGHEST_AFFECTED_LEVEL",
            "hierarchy_update_counts": allocation_counts,
            "xg_distribution_method": "CORE_XG_SCORE_DISTRIBUTION_V1:INDEPENDENT_POISSON",
            "xg_weight_reference": [self.opta_config.result_weight, self.opta_config.xg_weight],
            "xg_rows_used": xg_count,
            "xg_mode": "WITH_XG" if xg_count else "WITHOUT_XG_RESULT_ONLY",
            "xg_sources": sorted({item.provider for item in data.xg_observations}) if xg_count else [],
            "probability_mapper_version": self.mapper.version,
            "probability_mapper_trained_until": self.mapper.trained_until.isoformat(),
            "probability_mapper_training_sample_count": self.mapper.training_sample_count,
            "power_transform": CorePowerTransformer.method,
            "team_count": len(self.team_rating),
        })

    def _predict_values(self, match) -> dict[str, Any]:
        if self.mapper.trained_until is None:
            raise ValueError("HIERARCHICAL_MAPPER_UNAVAILABLE")
        home_membership = self._membership(match.home_team_id, self._active_prediction_time)
        away_membership = self._membership(match.away_team_id, self._active_prediction_time)
        home = self._raw_rating(match.home_team_id, home_membership)
        away = self._raw_rating(match.away_team_id, away_membership)
        advantage = 0.0 if match.neutral_venue else self.opta_config.hierarchy_home_advantage
        cross_league = bool(home_membership and away_membership and
                            home_membership.league_id != away_membership.league_id)
        vector = self.mapper.predict(home - away, advantage, cross_league)
        ratings = list(self.team_rating.values())
        return {"p_home": vector.p_home, "p_draw": vector.p_draw, "p_away": vector.p_away,
            "metadata": {"raw_hierarchical_elo_home": home, "raw_hierarchical_elo_away": away,
                "core_power_rating_home": CorePowerTransformer.transform(home, ratings),
                "core_power_rating_away": CorePowerTransformer.transform(away, ratings),
                "rating_difference": home-away, "home_advantage": advantage,
                "cross_league_flag": cross_league,
                "method_family": "OPTA_POWER_RANKINGS_PUBLIC_METHOD_INSPIRED",
                "official_model": "NOT_OFFICIAL_OPTA_MODEL",
                "xg_mode": "WITHOUT_XG_RESULT_ONLY",
                "mapper_version": self.mapper.version,
                "mapper_trained_until": self.mapper.trained_until.isoformat(),
                "mapper_training_sample_count": self.mapper.training_sample_count,
                "prediction_valid_from": self._active_prediction_time.isoformat()}}

    def predict(self, match, prediction_snapshot):
        self._active_prediction_time = prediction_snapshot.prediction_time
        return super().predict(match, prediction_snapshot)

    def update_once(self, training_data: TrainingDataset, trained_until, config: ModelConfig) -> Self:
        """Build hierarchy and mapper once before a batch of independent predictions."""
        return self.fit(training_data, trained_until, config)
