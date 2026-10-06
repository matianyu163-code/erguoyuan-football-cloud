"""Descriptive disagreement measures for independently sourced base forecasts."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from erguoyuan_football.meta.contracts import ModelDisagreementReport


def measure_disagreement(match_id: str, probabilities: list[tuple[float, float, float]]) -> ModelDisagreementReport:
    """Describe spread without creating a selection, stake, or replacement probability."""
    values = np.asarray(probabilities, dtype=float)
    if (values.ndim != 2 or values.shape[1] != 3 or len(values) == 0 or
            not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1) or
            np.any(np.abs(values.sum(axis=1) - 1) >= 1e-6)):
        raise ValueError("INVALID_DISAGREEMENT_INPUT")
    names = ("home", "draw", "away")
    pair_distances = [float(np.abs(values[left] - values[right]).sum() / 2)
                      for left, right in combinations(range(len(values)), 2)]
    mean = values.mean(axis=0)
    entropy = float(-(mean * np.log(np.clip(mean, 1e-12, 1))).sum())
    return ModelDisagreementReport(
        match_id=match_id, available_model_count=len(values),
        probability_std={name: float(values[:, index].std()) for index, name in enumerate(names)},
        probability_range={name: float(np.ptp(values[:, index])) for index, name in enumerate(names)},
        mean_pairwise_distance=float(np.mean(pair_distances)) if pair_distances else 0.0,
        entropy=entropy,
    )
