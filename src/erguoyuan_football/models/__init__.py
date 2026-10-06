"""Phase 3–5 base-model engine and auditable adapters."""

from erguoyuan_football.models.artifact import ModelArtifact
from erguoyuan_football.models.base import BaseFootballModel
from erguoyuan_football.models.bivariate_poisson import CoreBivariatePoissonModel
from erguoyuan_football.models.dixon_coles import CoreDixonColesModel
from erguoyuan_football.models.dynamic_bayes.model import (
    CoreDynamicBayesianPoissonModel,
)
from erguoyuan_football.models.elo import CoreEloModel
from erguoyuan_football.models.hierarchical_bayes import CoreHierarchicalBayesianModel
from erguoyuan_football.models.opta_like import CoreOptaXGEloLikeModel
from erguoyuan_football.models.pi_rating import CorePiRatingModel
from erguoyuan_football.models.registry import ModelRegistry
from erguoyuan_football.models.spi import CoreSPILikeModel

__all__ = [
    "BaseFootballModel",
    "CoreBivariatePoissonModel",
    "CoreDixonColesModel",
    "CoreDynamicBayesianPoissonModel",
    "CoreEloModel",
    "CoreHierarchicalBayesianModel",
    "CoreOptaXGEloLikeModel",
    "CorePiRatingModel",
    "CoreSPILikeModel",
    "ModelArtifact",
    "ModelRegistry",
]
