"""Regularized multinomial logistic META on verified no-market OOS inputs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from erguoyuan_football.meta.candidate import MODEL_PREFIXES, VerifiedCandidate


@dataclass(frozen=True)
class MetaRows:
    """Chronological feature/label view; labels never enter the encoder."""

    frame: pd.DataFrame
    labels: np.ndarray
    indices: np.ndarray

    @property
    def count(self) -> int:
        return len(self.indices)


def select_rows(candidate: VerifiedCandidate, start: date, end: date, *,
                min_available_models: int) -> MetaRows:
    """Select a fixed date interval without reading final-holdout rows."""
    if end < start or end >= date(2026, 8, 1) or min_available_models not in range(1, 9):
        raise ValueError("INVALID_META_WINDOW_OR_MIN_MODELS")
    dates = candidate.features["prediction_date"].astype(str)
    count = candidate.features[[f"{name}_available" for name in MODEL_PREFIXES]].sum(axis=1)
    mask = (dates >= start.isoformat()) & (dates <= end.isoformat()) & (count >= min_available_models)
    indices = np.flatnonzero(mask.to_numpy())
    frame = candidate.features.iloc[indices].copy()
    labels = candidate.labels.iloc[indices]["target"].to_numpy(dtype=int)
    if len(frame) and (not frame["prediction_date"].is_monotonic_increasing or
                       len(set(frame["match_id"])) != len(frame)):
        raise ValueError("META_ROWS_NOT_CHRONOLOGICAL_OR_UNIQUE")
    return MetaRows(frame, labels, indices)


def encode_features(frame: pd.DataFrame, *, epsilon: float = 1e-6) -> np.ndarray:
    """Encode shared-epsilon log ratios and mask; source missing probabilities remain null."""
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("INVALID_SHARED_LOG_RATIO_EPSILON")
    blocks: list[np.ndarray] = []
    for prefix in MODEL_PREFIXES:
        availability = frame[f"{prefix}_available"].to_numpy(dtype=float)
        raw = frame[[f"{prefix}_home", f"{prefix}_draw", f"{prefix}_away"]].to_numpy(dtype=float)
        if not np.isin(availability, [0.0, 1.0]).all():
            raise ValueError("INVALID_MODEL_AVAILABILITY_MASK")
        present = availability == 1.0
        if (np.isnan(raw[present]).any() or np.isfinite(raw[~present]).any() or
                not np.isfinite(raw[present]).all()):
            raise ValueError("PROBABILITY_MASK_MISMATCH")
        if present.any() and (np.any(raw[present] < 0) or np.any(raw[present] > 1) or
                              np.any(np.abs(raw[present].sum(axis=1) - 1) >= 1e-6)):
            raise ValueError("INVALID_BASE_PROBABILITY")
        home_ratio = np.zeros((len(frame), 1))
        away_ratio = np.zeros((len(frame), 1))
        if present.any():
            home_ratio[present, 0] = np.log((raw[present, 0] + epsilon) /
                                             (raw[present, 1] + epsilon))
            away_ratio[present, 0] = np.log((raw[present, 2] + epsilon) /
                                             (raw[present, 1] + epsilon))
        blocks.extend((home_ratio, away_ratio, availability[:, None]))
    encoded = np.hstack(blocks)
    if not np.isfinite(encoded).all():
        raise ValueError("NONFINITE_META_FEATURE")
    return encoded


class NoMarketMetaModel:
    """Independent META_NO_MARKET_V1; no market or base-model fit data access."""

    model_id = "META_NO_MARKET_V1"
    version = "1.1.0"

    def __init__(self, *, regularization_c: float = 1.0, epsilon: float = 1e-6) -> None:
        if not math.isfinite(regularization_c) or regularization_c <= 0:
            raise ValueError("INVALID_REGULARIZATION_C")
        self.regularization_c = regularization_c
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("INVALID_SHARED_LOG_RATIO_EPSILON")
        self.epsilon = epsilon
        self.pipeline: Pipeline | None = None
        self.trained_until: date | None = None
        self.training_sample_count = 0

    def fit(self, rows: MetaRows) -> NoMarketMetaModel:
        """Fit only chronologically preselected OOS samples."""
        if rows.count < 100 or set(rows.labels) != {0, 1, 2}:
            raise ValueError("INSUFFICIENT_META_TRAINING_DATA")
        array = encode_features(rows.frame, epsilon=self.epsilon)
        pipe = Pipeline([("scale", StandardScaler()),
                         ("logit", LogisticRegression(C=self.regularization_c,
                                                      max_iter=2000))])
        pipe.fit(array, rows.labels)
        if tuple(pipe.named_steps["logit"].classes_) != (0, 1, 2):
            raise ValueError("META_CLASS_ORDER_INVALID")
        self.pipeline = pipe
        self.trained_until = date.fromisoformat(str(rows.frame.iloc[-1]["prediction_date"]))
        self.training_sample_count = rows.count
        return self

    def predict_proba(self, rows: MetaRows) -> np.ndarray:
        """Compatibility method routed through the batch prediction implementation."""
        return self.predict_many(rows)

    def predict_many(self, rows: MetaRows) -> np.ndarray:
        """Predict a batch in one vectorized feature/model call."""
        if self.pipeline is None or self.trained_until is None:
            raise ValueError("META_NOT_FITTED")
        if rows.count and date.fromisoformat(str(rows.frame.iloc[0]["prediction_date"])) <= self.trained_until:
            raise ValueError("META_IN_SAMPLE_PREDICTION_REJECTED")
        probabilities = self.pipeline.predict_proba(encode_features(rows.frame, epsilon=self.epsilon))
        if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(
                np.abs(probabilities.sum(axis=1) - 1) >= 1e-6):
            raise ValueError("META_INVALID_PROBABILITIES")
        return probabilities
