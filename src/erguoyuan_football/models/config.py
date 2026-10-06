"""Versioned model configuration, with independent development/production sampling."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal, cast

import yaml
from pydantic import Field

from erguoyuan_football.contracts.common import Contract


class ModelConfig(Contract):
    """Explicit knobs; defaults are engineering starting points, not OOS-tuned values."""

    max_goals: int = Field(default=10, ge=2, le=50)
    time_decay: float = Field(default=0.001, ge=0, le=1)
    min_matches: int = Field(default=40, ge=3)
    # Kept as the serialized V1 key for compatibility; this is the minimum
    # distinct, resolved team coverage of a training corpus, not appearances
    # required from every sparse opponent.
    min_team_matches: int = Field(default=3, ge=1)
    training_window: int | None = Field(default=1095, ge=1)
    max_score: int = Field(default=30, ge=1)
    allow_test_data: bool = False
    max_iterations: int = Field(default=2000, ge=10)
    initial_rating: float = 1500.0
    k_factor: float = Field(default=20.0, gt=0)
    home_advantage: float = 60.0
    elo_scale: float = Field(default=400.0, gt=0)
    goal_difference_adjustment: bool = True
    pi_learning_rate: float = Field(default=0.035, gt=0, lt=1)
    pi_coupling: float = Field(default=0.7, ge=0, le=1)
    mapper_c: float = Field(default=1.0, gt=0)
    profile: Literal["development", "production"] = "production"
    chains: int = Field(default=4, ge=2)
    draws: int = Field(default=4000, ge=20)
    tune: int = Field(default=4000, ge=0)
    thin: int = Field(default=2, ge=1)
    seed: int = 314159
    max_rhat: float = Field(default=1.05, ge=1, le=1.2)
    min_ess: int = Field(default=200, ge=20)
    # Dynamic Bayesian Poisson V1. These are deliberately separate from the
    # Phase 3 static Bayesian sampler settings.
    dynamic_state_process: Literal["RANDOM_WALK", "AR1"] = "RANDOM_WALK"
    dynamic_time_index: Literal["MATCH_EVENT_TIME", "WEEKLY"] = "WEEKLY"
    dynamic_sigma_attack: float = Field(default=0.08, gt=0, le=2)
    dynamic_sigma_defence: float = Field(default=0.08, gt=0, le=2)
    dynamic_phi_attack: float = Field(default=0.98, gt=0, le=1)
    dynamic_phi_defence: float = Field(default=0.98, gt=0, le=1)
    dynamic_league_attack_mean: float = 0.0
    dynamic_league_defence_mean: float = 0.0
    dynamic_league_attack_sd: float = Field(default=0.35, gt=0, le=5)
    dynamic_league_defence_sd: float = Field(default=0.35, gt=0, le=5)
    dynamic_home_advantage: float = Field(default=0.15, ge=-2, le=2)
    dynamic_season_transition_weight: float = Field(default=0.75, ge=0, le=1)
    dynamic_uncertainty_growth_per_day: float = Field(default=0.01, ge=0, le=1)
    dynamic_posterior_draws: int = Field(default=400, ge=20, le=5000)
    dynamic_learning_rate: float = Field(default=0.35, gt=0, le=1)

    @property
    def config_hash(self) -> str:
        """Hash every option so test and production caches are disjoint."""
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()


def load_config(path: str | Path, profile: str = "production", *, allow_test_data: bool = False) -> ModelConfig:
    """Read a named YAML sampling profile without changing production defaults."""
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    selected_profile = cast(Literal["development", "production"], profile)
    dynamic_values = value.get("dynamic_bayes", {}).get(profile, {})
    return ModelConfig(**value["defaults"], **value["hierarchical_bayes"][profile], **dynamic_values,
                       profile=selected_profile, allow_test_data=allow_test_data)
