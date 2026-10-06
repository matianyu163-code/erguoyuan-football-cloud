"""Empirical, lineage-bearing league-level prior for hierarchical goal rates."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BayesianPrior:
    """A bounded prior distribution summary, never counted as direct training rows."""

    prior_type: str
    attack_mean: float
    attack_variance: float
    defense_mean: float
    defense_variance: float
    home_advantage_mean: float
    home_advantage_variance: float
    home_goal_mean: float
    away_goal_mean: float
    effective_strength: float
    sample_count: int
    evidence_ids: tuple[str, ...]
    source_tier: int
    recency_factor: float
    similarity_factor: float

