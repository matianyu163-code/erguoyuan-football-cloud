"""CORE wrapper around the official CatBoost multiclass classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from erguoyuan_football.ml.base import CoreMLModel, MLDependencyUnavailable
from erguoyuan_football.ml.config import MLConfig
from erguoyuan_football.ml.schemas import ModelFamily


class CoreCatBoostModel(CoreMLModel):
    model_id = "CORE_CATBOOST_V1"
    model_name = "CORE CatBoost multiclass 1X2"
    family = ModelFamily.CATBOOST
    library_name = "catboost"
    model_extension = ".cbm"

    def _new_estimator(self, config: MLConfig) -> Any:
        if config.params.get("loss_function") != "MultiClass":
            raise ValueError("CATBOOST_REQUIRES_MULTICLASS_OBJECTIVE")
        try:
            from catboost import CatBoostClassifier
        except ImportError as error:
            raise MLDependencyUnavailable("CATBOOST_NOT_INSTALLED") from error
        return CatBoostClassifier(**config.params)

    def _fit_estimator(self, train_x: Any, train_y: np.ndarray, validation_x: Any,
                       validation_y: np.ndarray, config: MLConfig) -> Any:
        try:
            from catboost import Pool
        except ImportError as error:
            raise MLDependencyUnavailable("CATBOOST_NOT_INSTALLED") from error
        if self.schema is None:
            raise ValueError("FEATURE_SCHEMA_MISSING")
        cats = list(self.schema.categorical_features)
        feature_names = list(self.schema.feature_names)
        train_pool = Pool(train_x, label=train_y, cat_features=cats, feature_names=feature_names)
        validation_pool = Pool(validation_x, label=validation_y,
                               cat_features=cats, feature_names=feature_names)
        estimator = self._new_estimator(config)
        estimator.fit(train_pool, eval_set=validation_pool, use_best_model=True,
                      early_stopping_rounds=config.early_stopping_rounds, verbose=False)
        return estimator

    def _best_metadata(self) -> dict[str, Any]:
        scores = self.estimator.get_best_score()
        validation = scores.get("validation", {})
        return {"best_iteration": int(self.estimator.get_best_iteration()),
                "best_score": float(validation.get("MultiClass")) if "MultiClass" in validation else None}

    def _save_native(self, path: Path) -> None:
        self.estimator.save_model(str(path), format="cbm")

    def _load_native(self, path: Path) -> Any:
        try:
            from catboost import CatBoostClassifier
        except ImportError as error:
            raise MLDependencyUnavailable("CATBOOST_NOT_INSTALLED") from error
        estimator = CatBoostClassifier()
        estimator.load_model(str(path))
        return estimator
