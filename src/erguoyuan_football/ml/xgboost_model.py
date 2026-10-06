"""CORE wrapper around the official XGBoost multiclass classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from erguoyuan_football.ml.base import CoreMLModel, MLDependencyUnavailable
from erguoyuan_football.ml.config import MLConfig
from erguoyuan_football.ml.schemas import ModelFamily


class CoreXGBoostModel(CoreMLModel):
    model_id = "CORE_XGBOOST_V1"
    model_name = "CORE XGBoost multiclass 1X2"
    family = ModelFamily.XGBOOST
    library_name = "xgboost"
    model_extension = ".json"

    def _new_estimator(self, config: MLConfig) -> Any:
        if config.params.get("objective") != "multi:softprob":
            raise ValueError("XGBOOST_REQUIRES_MULTICLASS_PROBABILITY_OBJECTIVE")
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise MLDependencyUnavailable("XGBOOST_NOT_INSTALLED") from error
        return XGBClassifier(**config.params, early_stopping_rounds=config.early_stopping_rounds)

    def _fit_estimator(self, train_x: Any, train_y: np.ndarray, validation_x: Any,
                       validation_y: np.ndarray, config: MLConfig) -> Any:
        estimator = self._new_estimator(config)
        estimator.fit(train_x, train_y, eval_set=[(validation_x, validation_y)], verbose=False)
        return estimator

    def _best_metadata(self) -> dict[str, Any]:
        return {"best_iteration": int(self.estimator.best_iteration),
                "best_score": float(self.estimator.best_score)}

    def _save_native(self, path: Path) -> None:
        self.estimator.save_model(str(path))

    def _load_native(self, path: Path) -> Any:
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise MLDependencyUnavailable("XGBOOST_NOT_INSTALLED") from error
        estimator = XGBClassifier()
        estimator.load_model(str(path))
        return estimator
