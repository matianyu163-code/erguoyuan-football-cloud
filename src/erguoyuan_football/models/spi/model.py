"""CORE SPI-inspired strength model; explicitly not the official FiveThirtyEight SPI."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import timedelta
from typing import Any, ClassVar, Self

import numpy as np
from pydantic import Field, model_validator

from erguoyuan_football.contracts.common import (
    Contract,
    ImplementationType,
    UTCTime,
    utc,
)
from erguoyuan_football.contracts.predictions import ModelPrediction
from erguoyuan_football.models.base import BaseFootballModel
from erguoyuan_football.models.config import ModelConfig
from erguoyuan_football.models.dynamic_bayes.likelihood import score_matrix_from_lambdas
from erguoyuan_football.models.score_matrix import ScoreMatrix
from erguoyuan_football.models.spi.state import (
    SPIFeatureMode,
    SPIMatchPerformanceEstimator,
    SPIState,
    SPIStateStore,
)
from erguoyuan_football.models.training import (
    TeamHierarchyMembership,
    TrainingDataset,
    TrainingMatch,
)


class SPIConfig(ModelConfig):
    """Parameters whose values affect SPI training and predictions."""

    spi_feature_mode: SPIFeatureMode = SPIFeatureMode.GOALS_ONLY
    spi_baseline_goals: float = Field(default=1.25, gt=0)
    spi_smoothing: float = Field(default=0.5, gt=0)
    spi_learning_rate: float = Field(default=0.12, gt=0, le=1)
    spi_season_carryover: float = Field(default=0.75, ge=0, le=1)
    spi_mean_reversion: float = Field(default=0.25, ge=0, le=1)
    spi_home_advantage: float = Field(default=0.15, ge=-1, le=1)
    spi_draw_adjustment: float = Field(default=1.0, ge=0.8, le=1.2)
    spi_draw_adjustment_validation_hash: str | None = None
    spi_league_k: float = Field(default=8.0, gt=0, le=100)
    spi_xg_weight: float = Field(default=0.2, ge=0, le=1)

    @model_validator(mode="after")
    def validate_spi_weights(self) -> SPIConfig:
        if not math.isclose(self.spi_season_carryover + self.spi_mean_reversion, 1, abs_tol=1e-9):
            raise ValueError("SPI season carryover and mean reversion must sum to one")
        if self.spi_draw_adjustment != 1.0 and not self.spi_draw_adjustment_validation_hash:
            raise ValueError("non-identity draw adjustment requires a train/validation evidence hash")
        return self


class OverallRatingMapper:
    """Map attack/defence into neutral expected league points share (0-100)."""

    @staticmethod
    def map_rating(offense: float, defense: float, *, league_log_goal: float,
                   league_offense: float, league_defense: float, max_goals: int) -> float:
        home = math.exp(league_log_goal + offense + league_defense)
        away = math.exp(league_log_goal + league_offense + defense)
        matrix = score_matrix_from_lambdas(np.asarray([home]), np.asarray([away]), max_goals)
        vector = matrix.outcome()
        home_points = 3 * vector.p_home + vector.p_draw
        away_points = 3 * vector.p_away + vector.p_draw
        return 100 * home_points / (home_points + away_points)


class SPIValidationDrawAdjustment(Contract):
    """Train/validation-only diagonal multiplier estimate with frozen test boundary."""

    multiplier: float = Field(ge=0.8, le=1.2)
    training_cutoff: UTCTime
    validation_start: UTCTime
    final_test_start: UTCTime
    sample_count: int = Field(ge=1)
    validation_data_hash: str
    validation_log_loss: float = Field(ge=0)


def estimate_draw_adjustment(validation_predictions: tuple[ModelPrediction, ...],
                             outcomes: tuple[int, ...], kickoff_times: tuple[Any, ...], *,
                             training_cutoff, validation_start, final_test_start,
                             candidates: tuple[float, ...] = (0.8, 0.9, 1.0, 1.1, 1.2)) -> SPIValidationDrawAdjustment:
    """Select a diagonal multiplier from chronological OOS validation only."""
    cutoff, validation_at, test_at = utc(training_cutoff), utc(validation_start), utc(final_test_start)
    if not cutoff < validation_at < test_at:
        raise ValueError("draw tuning requires training < validation < final test time")
    if not validation_predictions or len(validation_predictions) != len(outcomes) or len(outcomes) != len(kickoff_times):
        raise ValueError("matching nonempty validation predictions, outcomes and kickoff times required")
    if any(candidate < 0.8 or candidate > 1.2 for candidate in candidates) or 1.0 not in candidates:
        raise ValueError("draw adjustment candidates must include identity within [0.8, 1.2]")
    for prediction, label, kickoff in zip(validation_predictions, outcomes, kickoff_times, strict=True):
        if label not in {0, 1, 2} or prediction.model_id != "CORE_SPI_LIKE_V1" or not prediction.is_oos:
            raise ValueError("draw tuning accepts only SPI OOS predictions and H/D/A labels")
        if prediction.execution_status.value != "SUCCESS" or prediction.score_matrix is None:
            raise ValueError("draw tuning requires successful SPI score matrices")
        if prediction.training_end_time is None or not prediction.training_end_time <= prediction.prediction_time < utc(kickoff):
            raise ValueError("POINT_IN_TIME_GUARD_V1: invalid validation prediction lineage")
        if prediction.prediction_time < validation_at or utc(kickoff) >= test_at:
            raise ValueError("validation row falls outside the frozen validation interval")
    scored: list[tuple[float, float]] = []
    for candidate in candidates:
        losses = []
        for prediction, label in zip(validation_predictions, outcomes, strict=True):
            matrix = np.asarray(prediction.score_matrix, dtype=float).copy()
            np.fill_diagonal(matrix, matrix.diagonal() * candidate)
            matrix /= matrix.sum()
            vector = (float(np.tril(matrix, -1).sum()), float(np.trace(matrix)),
                      float(np.triu(matrix, 1).sum()))
            losses.append(-math.log(max(1e-15, vector[label])))
        scored.append((float(np.mean(losses)), candidate))
    loss, selected = min(scored, key=lambda item: (item[0], abs(item[1] - 1.0)))
    evidence = hashlib.sha256(json.dumps({
        "prediction_ids": [item.prediction_id for item in validation_predictions],
        "outcomes": outcomes, "kickoffs": [utc(value).isoformat() for value in kickoff_times],
        "candidates": candidates, "selected": selected, "training_cutoff": cutoff.isoformat(),
        "validation_start": validation_at.isoformat(), "final_test_start": test_at.isoformat(),
    }, sort_keys=True).encode()).hexdigest()
    return SPIValidationDrawAdjustment(multiplier=selected, training_cutoff=cutoff,
        validation_start=validation_at, final_test_start=test_at, sample_count=len(outcomes),
        validation_data_hash=evidence, validation_log_loss=loss)


class SPILeagueStrengthEngine:
    """PIT Elo-style relative league ratings from observed inter-league games."""

    method_id = "CORE_SPI_LEAGUE_STRENGTH_V1"

    def __init__(self, k_factor: float) -> None:
        self.k_factor = k_factor
        self.ratings: dict[str, float] = {}
        self.history: list[dict[str, Any]] = []

    def update_after_match(self, row: TrainingMatch, home_league: str | None,
                           away_league: str | None, prediction_time) -> None:
        if not home_league or not away_league or home_league == away_league:
            return
        home = self.ratings.get(home_league, 0.0)
        away = self.ratings.get(away_league, 0.0)
        expected_home = 1 / (1 + 10 ** (-(home - away) / 400))
        actual_home = 1.0 if row.home_goals > row.away_goals else 0.5 if row.home_goals == row.away_goals else 0.0
        margin = math.log1p(abs(row.home_goals - row.away_goals))
        delta = self.k_factor * max(1.0, margin) * (actual_home - expected_home)
        self.history.append({"as_of_time": utc(prediction_time), "home_league": home_league,
                             "away_league": away_league, "delta": delta, "phase": "POST_MATCH"})
        self.ratings[home_league] = home + delta
        self.ratings[away_league] = away - delta


class CoreSPILikeModel(BaseFootballModel):
    """Train one chronological attack/defence state per team and derive Poisson outcomes."""

    model_id = "CORE_SPI_LIKE_V1"
    model_name = "CORE SPI-like"
    model_version = "1.0.0"
    implementation_type = ImplementationType.LIKE_IMPLEMENTATION.value
    required_data = ("historical_results",)
    optional_data = ("xg", "league_hierarchy", "rest_days")
    allows_multiple_competitions: ClassVar[bool] = True

    def __init__(self) -> None:
        super().__init__()
        self.spi_config = SPIConfig()
        self.state_store = SPIStateStore()
        self.league_engine = SPILeagueStrengthEngine(self.spi_config.spi_league_k)
        self.estimator = SPIMatchPerformanceEstimator(baseline_goals=self.spi_config.spi_baseline_goals,
            smoothing=self.spi_config.spi_smoothing, learning_rate=self.spi_config.spi_learning_rate)
        self.team_matches: dict[str, int] = {}
        self.team_leagues: dict[str, str] = {}
        self._last_update_once: tuple[str, str] | None = None
        self._actual_feature_mode = SPIFeatureMode.GOALS_ONLY

    @property
    def feature_mode(self) -> SPIFeatureMode:
        return self._actual_feature_mode

    def prepare_config(self, config: ModelConfig) -> ModelConfig:
        """Expose the typed effective configuration to artifact-cache lookups."""
        return SPIConfig(**config.model_dump())

    def fit(self, training_data: TrainingDataset, trained_until, config: ModelConfig) -> Self:
        """Fit once on all validated historical rows; data without xG stays goals-only."""
        self.spi_config = SPIConfig(**config.model_dump())
        self.estimator = SPIMatchPerformanceEstimator(baseline_goals=self.spi_config.spi_baseline_goals,
            smoothing=self.spi_config.spi_smoothing, learning_rate=self.spi_config.spi_learning_rate)
        self.league_engine = SPILeagueStrengthEngine(self.spi_config.spi_league_k)
        self.state_store = SPIStateStore()
        self.team_matches = {}
        self.team_leagues = {}
        self._actual_feature_mode = SPIFeatureMode.GOALS_ONLY
        self._last_update_once = (training_data.data_hash, utc(trained_until).isoformat())
        return super().fit(training_data, trained_until, self.spi_config)

    def _fit(self, data: TrainingDataset) -> None:
        current: dict[str, dict[str, Any]] = {
            team: {"offense": 0.0, "defense": 0.0, "uncertainty": 1.0, "season": ""}
            for team in self.teams
        }
        ordered = sorted(data.matches, key=lambda row: (row.kickoff_time, row.match_id))
        xg_by_match = {item.match_id: item for item in data.xg_observations}
        use_xg = (self.spi_config.spi_feature_mode in {SPIFeatureMode.GOALS_XG, SPIFeatureMode.FULL_AVAILABLE}
                  and len(xg_by_match) == len(ordered))
        self._actual_feature_mode = SPIFeatureMode.GOALS_XG if use_xg else SPIFeatureMode.GOALS_ONLY
        self.team_leagues = {team: (membership.league_id if membership else "")
            for team in self.teams
            for membership in [self._membership(data, team, self.trained_until)]}
        for row in ordered:
            pre_time = utc(row.kickoff_time) - timedelta(microseconds=1)
            home, away = current[row.home_team_id], current[row.away_team_id]
            season_shift = home["season"] not in {"", row.season} or away["season"] not in {"", row.season}
            if season_shift:
                for state in (home, away):
                    state["offense"] *= self.spi_config.spi_season_carryover
                    state["defense"] *= self.spi_config.spi_season_carryover
                    state["uncertainty"] += self.spi_config.spi_mean_reversion
            pre_home = self._state(row.home_team_id, row, pre_time, home, data)
            pre_away = self._state(row.away_team_id, row, pre_time, away, data)
            attack_h, defense_h = self.estimator.estimate(row.home_goals, row.away_goals,
                                                           away["offense"], away["defense"])
            attack_a, defense_a = self.estimator.estimate(row.away_goals, row.home_goals,
                                                           home["offense"], home["defense"])
            if use_xg:
                xg = xg_by_match[row.match_id]
                xg_attack_h, xg_defense_h = self.estimator.estimate(xg.xg_home, xg.xg_away,
                    away["offense"], away["defense"])
                xg_attack_a, xg_defense_a = self.estimator.estimate(xg.xg_away, xg.xg_home,
                    home["offense"], home["defense"])
                weight = self.spi_config.spi_xg_weight
                attack_h, defense_h = ((1-weight)*attack_h + weight*xg_attack_h,
                                       (1-weight)*defense_h + weight*xg_defense_h)
                attack_a, defense_a = ((1-weight)*attack_a + weight*xg_attack_a,
                                       (1-weight)*defense_a + weight*xg_defense_a)
            alpha = self.spi_config.spi_learning_rate
            post_home = {**home, "offense": (1 - alpha) * home["offense"] + alpha * attack_h,
                         "defense": (1 - alpha) * home["defense"] + alpha * defense_h,
                         "uncertainty": max(0.05, home["uncertainty"] * (1 - alpha)), "season": row.season}
            post_away = {**away, "offense": (1 - alpha) * away["offense"] + alpha * attack_a,
                         "defense": (1 - alpha) * away["defense"] + alpha * defense_a,
                         "uncertainty": max(0.05, away["uncertainty"] * (1 - alpha)), "season": row.season}
            self.state_store.append(pre_home, self._state(row.home_team_id, row, row.available_at, post_home, data, phase="POST_MATCH"))
            self.state_store.append(pre_away, self._state(row.away_team_id, row, row.available_at, post_away, data, phase="POST_MATCH"))
            current[row.home_team_id], current[row.away_team_id] = post_home, post_away
            self.team_matches[row.home_team_id] = self.team_matches.get(row.home_team_id, 0) + 1
            self.team_matches[row.away_team_id] = self.team_matches.get(row.away_team_id, 0) + 1
            home_membership = self._membership(data, row.home_team_id, pre_time)
            away_membership = self._membership(data, row.away_team_id, pre_time)
            self.league_engine.update_after_match(row,
                home_membership.league_id if home_membership else None,
                away_membership.league_id if away_membership else None, row.available_at)
        self._states = current
        self.metadata.update({
            "method_family": "FIVETHIRTYEIGHT_SPI_INSPIRED",
            "official_model": "NOT_OFFICIAL_FIVETHIRTYEIGHT_SPI",
            "league_strength_method": self.league_engine.method_id,
            "league_strength_extension": "CORE_EXTENSION",
            "feature_mode": self.feature_mode.value,
            "features_used": ["historical_goals", "opponent_strength", "home_away"] + (["shot_xg"] if use_xg else []),
            "xg_observation_count": len(xg_by_match),
            "xg_sources": sorted({item.provider for item in data.xg_observations}) if use_xg else [],
            "draw_adjustment": self.spi_config.spi_draw_adjustment,
            "draw_adjustment_status": "UNADJUSTED_UNTIL_VALIDATION_ESTIMATION",
            "training_rows": len(ordered),
        })

    @staticmethod
    def _membership(data: TrainingDataset, team_id: str, at) -> TeamHierarchyMembership | None:
        cutoff = utc(at)
        candidates = [item for item in data.team_hierarchy_timeline if item.team_id == team_id
            and item.as_of_time <= cutoff and item.valid_from <= cutoff
            and (item.valid_to is None or cutoff < item.valid_to)]
        candidates.sort(key=lambda item: (item.valid_from, item.as_of_time))
        return candidates[-1] if candidates else None

    def _state(self, team_id: str, row: TrainingMatch, as_of_time, values: dict[str, Any],
               data: TrainingDataset, phase: str = "PRE_MATCH") -> SPIState:
        membership = self._membership(data, team_id, as_of_time)
        league_id = membership.league_id if membership else ""
        league_strength = self.league_engine.ratings.get(league_id, 0.0)
        overall = OverallRatingMapper.map_rating(values["offense"], values["defense"],
            league_log_goal=math.log(self.spi_config.spi_baseline_goals), league_offense=0,
            league_defense=0, max_goals=self.spi_config.max_goals)
        return SPIState(team_id=team_id, as_of_time=as_of_time, competition_id=row.competition_id,
            offensive_rating=values["offense"], defensive_rating=values["defense"],
            overall_rating=overall, league_strength=league_strength, uncertainty=values["uncertainty"],
            model_version=self.model_version, feature_mode=self.feature_mode, match_id=row.match_id,
            phase=phase, season=row.season)

    def _predict_values(self, match) -> dict[str, Any]:
        home = self._states[match.home_team_id]
        away = self._states[match.away_team_id]
        home_league = self.team_leagues.get(match.home_team_id, match.competition_id)
        away_league = self.team_leagues.get(match.away_team_id, match.competition_id)
        league_home = self.league_engine.ratings.get(home_league, 0.0)
        league_away = self.league_engine.ratings.get(away_league, 0.0)
        baseline = self.spi_config.spi_baseline_goals
        lambda_home = baseline * math.exp(home["offense"] + away["defense"] +
            (0 if match.neutral_venue else self.spi_config.spi_home_advantage) + league_home / 400)
        lambda_away = baseline * math.exp(away["offense"] + home["defense"] + league_away / 400)
        matrix = score_matrix_from_lambdas(np.asarray([lambda_home]), np.asarray([lambda_away]), self.spi_config.max_goals)
        if self.spi_config.spi_draw_adjustment != 1.0:
            values = np.asarray(matrix.values).copy()
            np.fill_diagonal(values, values.diagonal() * self.spi_config.spi_draw_adjustment)
            values /= values.sum()
            matrix = ScoreMatrix(values=tuple(tuple(float(value) for value in row) for row in values),
                max_goals=matrix.max_goals, retained_mass=matrix.retained_mass,
                tail_mass=matrix.tail_mass, tail_policy=matrix.tail_policy)
        vector = matrix.outcome()
        return {"lambda_home": lambda_home, "lambda_away": lambda_away,
            "expected_home_goals": lambda_home, "expected_away_goals": lambda_away,
            "score_matrix": matrix.values, "p_home": vector.p_home, "p_draw": vector.p_draw,
            "p_away": vector.p_away,
            "metadata": {"home_offense": home["offense"], "home_defense": home["defense"],
                "away_offense": away["offense"], "away_defense": away["defense"],
                "home_overall": OverallRatingMapper.map_rating(home["offense"], home["defense"],
                    league_log_goal=math.log(baseline), league_offense=0, league_defense=0,
                    max_goals=self.spi_config.max_goals),
                "away_overall": OverallRatingMapper.map_rating(away["offense"], away["defense"],
                    league_log_goal=math.log(baseline), league_offense=0, league_defense=0,
                    max_goals=self.spi_config.max_goals),
                "league_strength_home": league_home, "league_strength_away": league_away,
                "feature_mode": self.feature_mode.value,
                "features_used": self.metadata["features_used"],
                "draw_adjustment": self.spi_config.spi_draw_adjustment,
                "draw_adjustment_validation_hash": self.spi_config.spi_draw_adjustment_validation_hash,
                "score_matrix_retained_mass": matrix.retained_mass,
                "score_matrix_tail_mass": matrix.tail_mass,
                "uncertainty": home["uncertainty"] + away["uncertainty"],
                "method_family": "FIVETHIRTYEIGHT_SPI_INSPIRED",
                "official_model": "NOT_OFFICIAL_FIVETHIRTYEIGHT_SPI"}}

    def update_once(self, training_data: TrainingDataset, trained_until, config: ModelConfig) -> Self:
        """Fit the shared team/league state once; callers then predict many fixtures."""
        return self.fit(training_data, trained_until, config)
