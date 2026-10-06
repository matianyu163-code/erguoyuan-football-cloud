"""Explicit model registry; lookup never fits or predicts."""

from __future__ import annotations

from collections.abc import Callable

from erguoyuan_football.models.base import BaseFootballModel
from erguoyuan_football.models.bivariate_poisson import CoreBivariatePoissonModel
from erguoyuan_football.models.dixon_coles import CoreDixonColesModel
from erguoyuan_football.models.dynamic_bayes.model import (
    CoreDynamicBayesianPoissonModel,
)
from erguoyuan_football.models.elo import CoreEloModel
from erguoyuan_football.models.hierarchical_bayes import CoreHierarchicalBayesianModel
from erguoyuan_football.models.historical_market_bayes import (
    HistoricalMarketBayesianPoissonModel,
)
from erguoyuan_football.models.opta_like import CoreOptaXGEloLikeModel
from erguoyuan_football.models.pi_rating import CorePiRatingModel
from erguoyuan_football.models.spi import CoreSPILikeModel

FACTORIES: dict[str, Callable[[], BaseFootballModel]] = {
    "DIXON_COLES_V1": CoreDixonColesModel,
    "BIVARIATE_POISSON_V1": CoreBivariatePoissonModel,
    "BAYESIAN_HIERARCHICAL_V1": CoreHierarchicalBayesianModel,
    "ELO_V1": CoreEloModel,
    "PI_RATING_V1": CorePiRatingModel,
    "DYNAMIC_BAYESIAN_POISSON_V1": CoreDynamicBayesianPoissonModel,
    "CORE_SPI_LIKE_V1": CoreSPILikeModel,
    "CORE_OPTA_XG_ELO_LIKE_V1": CoreOptaXGEloLikeModel,
    "HISTORICAL_MARKET_BAYESIAN_POISSON_V1": HistoricalMarketBayesianPoissonModel,
}


class ModelRegistry:
    """Register legacy models and expose Phase 7 ML models without changing the old runner."""

    def list_models(self) -> tuple[str, ...]:
        """Return stable registered IDs without executing models."""
        return tuple(FACTORIES)

    def list_all_models(self) -> tuple[str, ...]:
        """Return all eleven base model IDs; legacy runner still consumes its nine-model list."""
        from erguoyuan_football.ml.registry import MLModelRegistry

        return self.list_models() + MLModelRegistry().list_models()

    def get_ml_model(self, model_id: str):
        """Return an unfitted XGBoost/CatBoost wrapper via the dedicated ML lifecycle."""
        from erguoyuan_football.ml.registry import MLModelRegistry

        return MLModelRegistry().get_model(model_id)

    def get_model(self, model_id: str) -> BaseFootballModel:
        """Construct an unfitted model or report an unknown ID."""
        try:
            return FACTORIES[model_id]()
        except KeyError as error:
            raise KeyError(f"UNKNOWN_MODEL:{model_id}") from error

    def get_model_requirements(self, model_id: str) -> dict[str, object]:
        """Read class capabilities without fitting."""
        model = self.get_model(model_id)
        return {"model_id": model.model_id, "required_data": model.required_data,
                "optional_data": model.optional_data, "supports_score_matrix": model.supports_score_matrix,
                "supports_expected_goals": model.supports_expected_goals, "supports_1x2": model.supports_1x2}

    def get_available_models(self, available_data: set[str]) -> tuple[str, ...]:
        """List models whose declared required inputs are available; never run them."""
        return tuple(model_id for model_id in self.list_models()
                     if set(self.get_model(model_id).required_data) <= available_data)
