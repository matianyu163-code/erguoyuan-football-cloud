"""Chronologically fitted multiclass calibration methods for Phase 9 development."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression

CalibrationMethod = Literal["NONE", "TEMPERATURE_SCALING", "MULTICLASS_LOGIT_CALIBRATION"]
EPS = 1e-12


def multiclass_log_loss(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Mean natural log loss with explicit H/D/A class order."""
    _validate(probabilities)
    if len(probabilities) != len(labels) or not np.isin(labels, [0, 1, 2]).all() or len(labels) == 0:
        raise ValueError("INVALID_CALIBRATION_LABELS")
    return float(-np.log(np.clip(probabilities[np.arange(len(labels)), labels], EPS, 1)).mean())


def _validate(probabilities: np.ndarray) -> None:
    if probabilities.ndim != 2 or probabilities.shape[1] != 3 or not np.isfinite(probabilities).all():
        raise ValueError("INVALID_CALIBRATION_PROBABILITIES")
    if (np.any(probabilities < 0) or np.any(probabilities > 1) or
            np.any(np.abs(probabilities.sum(axis=1) - 1) >= 1e-6)):
        raise ValueError("INVALID_CALIBRATION_PROBABILITIES")


@dataclass
class MulticlassCalibrator:
    """NONE, scalar temperature or regularized logistic mapping of log probabilities."""

    method: CalibrationMethod
    temperature: float | None = None
    mapping: LogisticRegression | None = None
    fit_sample_count: int = 0

    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> MulticlassCalibrator:
        """Fit on the calibration-fit interval only, never on final holdout."""
        _validate(probabilities)
        if len(probabilities) < 50 or set(labels) != {0, 1, 2}:
            raise ValueError("INSUFFICIENT_CALIBRATION_FIT_DATA")
        self.fit_sample_count = len(labels)
        if self.method == "NONE":
            return self
        logits = np.log(np.clip(probabilities, EPS, 1))
        if self.method == "TEMPERATURE_SCALING":
            result = minimize_scalar(lambda log_t: multiclass_log_loss(
                softmax(logits / math.exp(log_t), axis=1), labels),
                bounds=(-2.0, 2.0), method="bounded")
            if not result.success or not math.isfinite(float(result.fun)):
                raise ValueError("TEMPERATURE_CALIBRATION_FAILED")
            self.temperature = math.exp(float(result.x))
        elif self.method == "MULTICLASS_LOGIT_CALIBRATION":
            mapping = LogisticRegression(C=1.0, max_iter=2000)
            mapping.fit(logits, labels)
            if tuple(mapping.classes_) != (0, 1, 2):
                raise ValueError("CALIBRATION_CLASS_ORDER_INVALID")
            self.mapping = mapping
        else:
            raise ValueError("UNKNOWN_CALIBRATION_METHOD")
        return self

    def predict_proba(self, probabilities: np.ndarray) -> np.ndarray:
        """Compatibility method routed through batch calibration."""
        return self.calibrate_many(probabilities)

    def calibrate_many(self, probabilities: np.ndarray) -> np.ndarray:
        """Calibrate finite batched 1X2 probabilities without creating a market signal."""
        _validate(probabilities)
        if self.fit_sample_count == 0:
            raise ValueError("CALIBRATOR_NOT_FITTED")
        if self.method == "NONE":
            output = probabilities.copy()
        elif self.method == "TEMPERATURE_SCALING" and self.temperature is not None:
            output = softmax(np.log(np.clip(probabilities, EPS, 1)) / self.temperature, axis=1)
        elif self.method == "MULTICLASS_LOGIT_CALIBRATION" and self.mapping is not None:
            output = self.mapping.predict_proba(np.log(np.clip(probabilities, EPS, 1)))
        else:
            raise ValueError("CALIBRATOR_STATE_INVALID")
        _validate(output)
        return output
