"""Robust dispersion summaries over independently de-vigged bookmaker probabilities."""

from __future__ import annotations

import numpy as np


def market_dispersion(bookmaker_probabilities: tuple[dict[str, float], ...]) -> dict[str, dict[str, float]]:
    """Compute standard deviation, IQR, range and MAD by aligned selection."""
    if not bookmaker_probabilities:
        raise ValueError("market dispersion requires bookmaker probabilities")
    selections = set(bookmaker_probabilities[0])
    if any(set(row) != selections for row in bookmaker_probabilities):
        raise ValueError("bookmaker probability selections must align")
    result = {}
    for selection in sorted(selections):
        values = np.asarray([row[selection] for row in bookmaker_probabilities], dtype=float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError("dispersion requires finite probabilities in [0, 1]")
        median = float(np.median(values))
        q25, q75 = np.quantile(values, [0.25, 0.75])
        result[selection] = {
            "std": float(np.std(values, ddof=0)),
            "iqr": float(q75 - q25),
            "range": float(np.max(values) - np.min(values)),
            "mad": float(np.median(np.abs(values - median))),
        }
    return result


def mean_dispersion(summary: dict[str, dict[str, float]]) -> float | None:
    """Return mean absolute deviation spread across selections for quality filters."""
    if not summary:
        return None
    return float(np.mean([values["mad"] for values in summary.values()]))
