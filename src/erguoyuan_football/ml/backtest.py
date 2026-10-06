"""Strictly chronological, raw-probability ML evaluation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

from erguoyuan_football.backtesting.base_model_backtest import (
    accuracy,
    brier_score,
    log_loss,
    ranked_probability_score,
)
from erguoyuan_football.contracts.common import ExecutionStatus
from erguoyuan_football.contracts.predictions import ModelPrediction
from erguoyuan_football.ml.base import CoreMLModel
from erguoyuan_football.ml.config import MLConfig
from erguoyuan_football.ml.schemas import MLDataset, MLTrainingRow
from erguoyuan_football.ml.splits import MLTimeSplit, MLWalkForwardSplit


@dataclass(frozen=True)
class MLMetricSet:
    sample_size: int
    log_loss: float
    brier: float
    rps: float
    ece: float
    accuracy: float
    draw_log_loss: float | None
    draw_recall: float | None


@dataclass(frozen=True)
class MLOOSObservation:
    row: MLTrainingRow
    prediction: ModelPrediction
    window_index: int


@dataclass(frozen=True)
class MLBacktestReport:
    model_id: str
    model_version: str
    dataset_kind: str
    feature_mode: str
    observations: tuple[MLOOSObservation, ...]
    overall: MLMetricSet
    strata: dict[str, dict[str, MLMetricSet]]
    frozen_test_hashes: tuple[str, ...]
    performance_claim: str


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray,
                               *, bins: int = 10) -> float:
    """Top-class ECE; diagnostic only, never a calibration operation."""
    if bins < 1 or probabilities.ndim != 2 or probabilities.shape[1] != 3 or len(labels) != len(probabilities):
        raise ValueError("INVALID_ECE_INPUT")
    confidence = probabilities.max(axis=1)
    correct = (probabilities.argmax(axis=1) == labels).astype(float)
    error = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        membership = (confidence >= lower) & (confidence <= upper if index == bins - 1 else confidence < upper)
        if membership.any():
            error += float(membership.mean() * abs(correct[membership].mean() - confidence[membership].mean()))
    return error


def metrics_for(observations: tuple[MLOOSObservation, ...]) -> MLMetricSet:
    if not observations:
        raise ValueError("EMPTY_OOS_METRIC_SET")
    probabilities = np.asarray([(item.prediction.p_home, item.prediction.p_draw, item.prediction.p_away)
                                for item in observations], dtype=float)
    labels = np.asarray([int(item.row.label) for item in observations], dtype=int)
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6):
        raise ValueError("INVALID_OOS_PROBABILITIES")
    draw = labels == 1
    return MLMetricSet(sample_size=len(observations), log_loss=log_loss(probabilities, labels),
        brier=brier_score(probabilities, labels), rps=ranked_probability_score(probabilities, labels),
        ece=expected_calibration_error(probabilities, labels), accuracy=accuracy(probabilities, labels),
        draw_log_loss=float(-np.log(np.maximum(probabilities[draw, 1], 1e-15)).mean()) if draw.any() else None,
        draw_recall=float((probabilities[draw].argmax(axis=1) == 1).mean()) if draw.any() else None)


def _strata(observations: tuple[MLOOSObservation, ...]) -> dict[str, dict[str, MLMetricSet]]:
    groups: dict[str, dict[str, list[MLOOSObservation]]] = defaultdict(lambda: defaultdict(list))
    for item in observations:
        vector = item.row.vector
        horizon = (vector.kickoff_time - vector.prediction_time).total_seconds() / 3600
        values = {
            "competition": item.row.competition_id,
            "season": item.row.season or "UNKNOWN",
            "prediction_horizon": "<24h" if horizon < 24 else "24-72h" if horizon <= 72 else ">72h",
            "market_availability": "AVAILABLE" if vector.features.get("market_available") == 1 else "UNAVAILABLE",
            "feature_availability": "BASE_AVAILABLE" if any(
                name.endswith("_available") and name not in {"form_available", "xg_available", "market_available"}
                and value == 1 for name, value in vector.features.items()) else "BASE_UNAVAILABLE",
            "favorite_bucket": "EVEN" if max(item.prediction.p_home or 0, item.prediction.p_draw or 0,
                item.prediction.p_away or 0) < 0.45 else "FAVORITE" if max(
                item.prediction.p_home or 0, item.prediction.p_draw or 0,
                item.prediction.p_away or 0) >= 0.60 else "MIDDLE",
        }
        for name, value in values.items():
            groups[name][value].append(item)
    return {name: {value: metrics_for(tuple(rows)) for value, rows in segments.items()}
            for name, segments in groups.items()}


ModelT = TypeVar("ModelT", bound=CoreMLModel)


class MLWalkForwardBacktester:
    """Refit for each window; test labels never reach the estimator or early stopping."""

    def run(self, dataset: MLDataset, model_class: type[ModelT], config: MLConfig, *,
            initial_train: int, validation_size: int, test_size: int,
            step: int) -> MLBacktestReport:
        windows = MLWalkForwardSplit.rolling(dataset, initial_train=initial_train,
            validation_size=validation_size, test_size=test_size, step=step)
        if not windows:
            raise ValueError("NO_VALID_ML_WALK_FORWARD_WINDOW")
        return self.run_windows(windows, model_class, config)

    def run_windows(self, windows: tuple[MLTimeSplit, ...], model_class: type[ModelT],
                    config: MLConfig) -> MLBacktestReport:
        if not windows:
            raise ValueError("NO_VALID_ML_WALK_FORWARD_WINDOW")
        observations: list[MLOOSObservation] = []
        seen: set[str] = set()
        for index, window in enumerate(windows):
            window.assert_test_unchanged()
            model = model_class().fit(window.train, window.validation, config)
            predictions = model.predict_many(tuple(row.vector for row in window.test.rows), oos=True)
            for row, prediction in zip(window.test.rows, predictions, strict=True):
                if row.vector.match_id in seen:
                    raise ValueError("DUPLICATED_WALK_FORWARD_TEST_MATCH")
                seen.add(row.vector.match_id)
                if (prediction.execution_status != ExecutionStatus.SUCCESS or not prediction.is_oos
                        or prediction.training_end_time is None
                        or prediction.training_end_time > prediction.prediction_time
                        or prediction.prediction_time >= row.vector.kickoff_time):
                    raise ValueError("NON_OOS_ML_BACKTEST_PREDICTION")
                observations.append(MLOOSObservation(row=row, prediction=prediction, window_index=index))
        items = tuple(observations)
        return MLBacktestReport(model_id=model_class.model_id, model_version=model_class.model_version,
            dataset_kind=windows[0].train.dataset_kind,
            feature_mode=windows[0].train.feature_schema.mode.value,
            observations=items, overall=metrics_for(items), strata=_strata(items),
            frozen_test_hashes=tuple(window.frozen_test_hash for window in windows),
            performance_claim="REAL_OOS" if windows[0].train.dataset_kind == "REAL"
            else "SYNTHETIC_TEST_ONLY")
