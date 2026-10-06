"""Reorder third-party multiclass probability columns into CORE HOME/DRAW/AWAY."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from erguoyuan_football.ml.labels import CLASS_ORDER


class ProbabilityClassOrderGuard:
    """Never assume a library's predict_proba column order."""

    @staticmethod
    def reorder(probabilities: np.ndarray, classes: Iterable[int]) -> np.ndarray:
        labels = tuple(int(value) for value in classes)
        if len(labels) != 3 or set(labels) != set(CLASS_ORDER):
            raise ValueError("ML_CLASS_MAPPING_MISMATCH")
        values = np.asarray(probabilities, dtype=float)
        if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
            raise ValueError("ML_PROBABILITY_SHAPE_OR_FINITE_ERROR")
        ordered = values[:, [labels.index(label) for label in CLASS_ORDER]]
        if (ordered < 0).any() or (ordered > 1).any() or not np.allclose(
                ordered.sum(axis=1), 1.0, rtol=0, atol=1e-6):
            raise ValueError("ML_PROBABILITY_NOT_NORMALIZED")
        # Tree libraries commonly return float32 probabilities. Remove only their
        # rounding drift after checking the unadjusted mass; this is not calibration.
        ordered = ordered / ordered.sum(axis=1, keepdims=True)
        ordered[:, 2] = 1.0 - ordered[:, 0] - ordered[:, 1]
        return ordered
