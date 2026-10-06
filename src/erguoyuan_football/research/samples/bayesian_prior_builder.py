"""Build a bounded empirical prior from PIT-compatible non-direct samples."""

from __future__ import annotations

import math
import statistics
from datetime import datetime

from erguoyuan_football.research.samples.bayesian_prior import BayesianPrior
from erguoyuan_football.research.samples.match_deduplicator import HistoricalMatchSample
from erguoyuan_football.research.samples.sample_hierarchy import SampleHierarchy
from erguoyuan_football.research.samples.sample_repository import SampleRepository


class BayesianPriorBuilder:
    """Prefer same-competition context and cap effective prior information at 20."""

    MAX_EFFECTIVE_STRENGTH = 20.0
    MIN_PRIOR_SAMPLE_COUNT = 8

    def build(self, repository: SampleRepository, hierarchy: SampleHierarchy,
              cutoff: datetime) -> BayesianPrior | None:
        """Return the strongest available compatible group without row overlap."""
        direct_ids = {row.match_id for row in (*hierarchy.home_direct, *hierarchy.away_direct)}
        candidates = (
            ("SAME_COMPETITION", tuple(hierarchy.competition), 1.0),
            ("COMPARABLE_COMPETITION", tuple(hierarchy.comparable), 0.75),
            ("FEDERATION", tuple(hierarchy.federation_prior), 0.60),
            ("AGE_GROUP", tuple(hierarchy.age_group_prior), 0.45),
            ("GENDER", tuple(hierarchy.gender_prior), 0.30),
        )
        by_id = {row.match_id: row for row in repository.samples}
        choices: list[tuple[str, tuple[HistoricalMatchSample, ...], float]] = []
        for label, lineages, similarity in candidates:
            rows = tuple(by_id[item.match_id] for item in lineages
                         if item.match_id in by_id and item.match_id not in direct_ids
                         and by_id[item.match_id].kickoff < cutoff
                         and by_id[item.match_id].fetched_at <= cutoff)
            if rows:
                choices.append((label, rows, similarity))
        if not choices:
            return None
        selected = next((item for item in choices
                         if len(item[1]) >= self.MIN_PRIOR_SAMPLE_COUNT), choices[0])
        prior_type, rows, similarity = selected
        home_mean = sum(row.home_goals for row in rows) / len(rows)
        away_mean = sum(row.away_goals for row in rows) / len(rows)
        overall = (sum(row.home_goals + row.away_goals for row in rows)
                   / (2 * len(rows)))
        if min(home_mean, away_mean, overall) <= 0:
            return None
        per_team_scored: dict[str, list[int]] = {}
        per_team_conceded: dict[str, list[int]] = {}
        for row in rows:
            per_team_scored.setdefault(row.home_team_id, []).append(row.home_goals)
            per_team_conceded.setdefault(row.home_team_id, []).append(row.away_goals)
            per_team_scored.setdefault(row.away_team_id, []).append(row.away_goals)
            per_team_conceded.setdefault(row.away_team_id, []).append(row.home_goals)
        attack_effects = [math.log(max(0.1, statistics.mean(values)) / overall)
                          for values in per_team_scored.values()]
        defense_effects = [math.log(max(0.1, statistics.mean(values)) / overall)
                           for values in per_team_conceded.values()]
        ages = [(cutoff - row.kickoff).total_seconds() / 86400 for row in rows]
        median_age = statistics.median(ages)
        recency = math.exp(-median_age / 1825.0)
        source_quality = min(1.0, statistics.mean(row.source_tier for row in rows) / 5.0)
        raw_strength = min(float(len(rows)), self.MAX_EFFECTIVE_STRENGTH)
        strength = min(self.MAX_EFFECTIVE_STRENGTH,
                       raw_strength * source_quality * recency * similarity)
        if strength <= 0:
            return None
        return BayesianPrior(
            prior_type=prior_type,
            attack_mean=0.0,
            attack_variance=max(0.05, statistics.pvariance(attack_effects)),
            defense_mean=0.0,
            defense_variance=max(0.05, statistics.pvariance(defense_effects)),
            home_advantage_mean=math.log(home_mean / away_mean),
            home_advantage_variance=max(0.02, (math.log(home_mean / away_mean) ** 2) + 0.05),
            home_goal_mean=home_mean,
            away_goal_mean=away_mean,
            effective_strength=strength,
            sample_count=len(rows),
            evidence_ids=tuple(sorted({evidence for row in rows
                                       for evidence in row.evidence_ids})),
            source_tier=min(row.source_tier for row in rows),
            recency_factor=recency,
            similarity_factor=similarity,
        )
