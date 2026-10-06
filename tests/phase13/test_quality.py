"""SYNTHETIC_TEST quality score is coverage, never prediction confidence."""

import math

from erguoyuan_football.knowledge.competitions.competition_resolver import (
    CompetitionResolver,
)
from erguoyuan_football.knowledge.teams.team_resolver import TeamResolver
from erguoyuan_football.research.data_quality import DataQualityEvaluator
from erguoyuan_football.research.match_package import MatchResearchPackage


def test_quality_weights_and_range() -> None:
    """Weights sum to one and all-empty coverage scores zero."""
    evaluator = DataQualityEvaluator()
    assert math.isclose(sum(evaluator.weights.values()), 1.0)
    empty = MatchResearchPackage(None, None, None, None)
    assert evaluator.evaluate(empty) == 0.0
    assert 0 <= evaluator.evaluate(empty) <= 1


def test_identity_and_competition_weights() -> None:
    """Known identities score coverage without treating missing data as found."""
    resolver = TeamResolver()
    home = resolver.resolve("Arsenal")
    away = resolver.resolve("Liverpool")
    competition = CompetitionResolver().resolve("UEFA Champions League")
    assert home is not None and away is not None and competition is not None
    package = MatchResearchPackage(None, home, away, competition)
    assert DataQualityEvaluator().evaluate(package) == 0.3
