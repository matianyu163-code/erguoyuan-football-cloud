"""Versioned coverage score for research completeness, never a model confidence."""

from __future__ import annotations

from typing import ClassVar

from erguoyuan_football.research.match_package import MatchResearchPackage

QUALITY_POLICY_VERSION = "RESEARCH_COVERAGE_V1"


class DataQualityEvaluator:
    """Score source-backed coverage with explicit weights totaling one."""

    weights: ClassVar[dict[str, float]] = {
        "team_identity": 0.20,
        "competition": 0.10,
        "fixture": 0.10,
        "injury": 0.05,
        "lineup": 0.05,
        "xg": 0.20,
        "odds": 0.20,
        "news": 0.10,
    }

    def evaluate(self, package: MatchResearchPackage) -> float:
        """Return 0–1 coverage; no evidence value is interpreted as a probability."""
        coverage = {
            "team_identity": (package.home_team is not None
                              and package.away_team is not None
                              and package.home_team.team_id != package.away_team.team_id),
            "competition": package.competition is not None,
            **{key: bool(package.available_data.get(key))
               for key in ("fixture", "injury", "lineup", "xg", "odds", "news")},
        }
        return round(sum(weight for key, weight in self.weights.items()
                         if coverage[key]), 10)
