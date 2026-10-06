"""League-level priors used for new teams and season transitions."""

from __future__ import annotations

from dataclasses import dataclass

from erguoyuan_football.models.config import ModelConfig


@dataclass(frozen=True)
class LeaguePrior:
    """Exchangeable prior means and standard deviations for one competition."""

    attack_mean: float
    attack_sd: float
    defence_mean: float
    defence_sd: float

    @classmethod
    def from_config(cls, config: ModelConfig) -> LeaguePrior:
        """Build the explicit prior configured for the fit."""
        return cls(config.dynamic_league_attack_mean, config.dynamic_league_attack_sd,
                   config.dynamic_league_defence_mean, config.dynamic_league_defence_sd)

