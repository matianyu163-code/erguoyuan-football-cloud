"""ML-only registration; looking up a model never fits or predicts it."""

from __future__ import annotations

from collections.abc import Callable

from erguoyuan_football.ml.base import CoreMLModel
from erguoyuan_football.ml.catboost_model import CoreCatBoostModel
from erguoyuan_football.ml.xgboost_model import CoreXGBoostModel

ML_FACTORIES: dict[str, Callable[[], CoreMLModel]] = {
    CoreXGBoostModel.model_id: CoreXGBoostModel,
    CoreCatBoostModel.model_id: CoreCatBoostModel,
}


class MLModelRegistry:
    def list_models(self) -> tuple[str, ...]:
        return tuple(ML_FACTORIES)

    def get_model(self, model_id: str) -> CoreMLModel:
        try:
            return ML_FACTORIES[model_id]()
        except KeyError as error:
            raise KeyError(f"UNKNOWN_ML_MODEL:{model_id}") from error
