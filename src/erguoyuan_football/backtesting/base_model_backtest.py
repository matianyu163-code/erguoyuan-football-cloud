"""Small, strictly chronological base-model evaluation harness."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from erguoyuan_football.contracts.predictions import ModelPrediction, ProbabilityVector
from erguoyuan_football.models.training import (
    TrainingDataError,
    TrainingDataset,
    TrainingMatch,
)


@dataclass(frozen=True)
class WalkForwardWindow:
    """One expanding-window train/predict boundary."""

    training_end: datetime
    prediction_time: datetime
    test_rows: tuple[TrainingMatch, ...]


def walk_forward_split(data: TrainingDataset, *, initial_matches: int, horizon: int,
                       step: int = 1) -> tuple[WalkForwardWindow, ...]:
    """Create expanding windows from sorted matches, never random-split rows."""
    rows = tuple(sorted(data.matches, key=lambda row: (row.kickoff_time, row.match_id)))
    if initial_matches < 1 or horizon < 1 or step < 1:
        raise ValueError("initial_matches, horizon and step must be positive")
    windows: list[WalkForwardWindow] = []
    start = initial_matches
    while start < len(rows):
        test = rows[start:min(start + horizon, len(rows))]
        if not test:
            break
        training_end = rows[start - 1].available_at
        prediction_time = test[0].kickoff_time - timedelta(microseconds=1)
        if training_end > prediction_time:
            # A late result cannot be included in a forecast immediately before
            # the next kickoff; omit this window rather than leaking it forward.
            start += step
            continue
        windows.append(WalkForwardWindow(training_end=training_end, prediction_time=prediction_time,
                                         test_rows=test))
        start += step
    return tuple(windows)


def _vector(prediction: ModelPrediction | ProbabilityVector) -> tuple[float, float, float]:
    values: tuple[float | None, float | None, float | None]
    if isinstance(prediction, ProbabilityVector):
        values = (prediction.p_home, prediction.p_draw, prediction.p_away)
    else:
        if prediction.execution_status.value != "SUCCESS":
            raise ValueError("metrics require successful OOS predictions")
        values = (prediction.p_home, prediction.p_draw, prediction.p_away)
    if any(value is None for value in values):
        raise ValueError("metrics require probabilities")
    home, draw, away = values
    assert home is not None and draw is not None and away is not None
    return (float(home), float(draw), float(away))


def log_loss(probabilities: Iterable[tuple[float, float, float]], outcomes: Iterable[int]) -> float:
    """Multiclass logarithmic loss with finite probabilities only."""
    values = list(probabilities)
    labels = list(outcomes)
    if len(values) != len(labels) or not values:
        raise ValueError("non-empty matching metric inputs required")
    return float(-np.mean([math.log(max(1e-15, min(1.0, row[label]))) for row, label in zip(values, labels, strict=True)]))


def brier_score(probabilities: Iterable[tuple[float, float, float]], outcomes: Iterable[int]) -> float:
    """Multiclass Brier score."""
    values = list(probabilities)
    labels = list(outcomes)
    if len(values) != len(labels) or not values:
        raise ValueError("non-empty matching metric inputs required")
    return float(np.mean([sum((probability - (index == label)) ** 2 for index, probability in enumerate(row))
                          for row, label in zip(values, labels, strict=True)]))


def ranked_probability_score(probabilities: Iterable[tuple[float, float, float]], outcomes: Iterable[int]) -> float:
    """Three-category ranked probability score in home/draw/away order."""
    values = list(probabilities)
    labels = list(outcomes)
    if len(values) != len(labels) or not values:
        raise ValueError("non-empty matching metric inputs required")
    return float(np.mean([sum((sum(row[: index + 1]) - (label <= index)) ** 2 for index in range(2))
                          for row, label in zip(values, labels, strict=True)]))


def accuracy(probabilities: Iterable[tuple[float, float, float]], outcomes: Iterable[int]) -> float:
    """Auxiliary argmax accuracy."""
    values = list(probabilities)
    labels = list(outcomes)
    if len(values) != len(labels) or not values:
        raise ValueError("non-empty matching metric inputs required")
    return float(np.mean([int(int(np.argmax(row)) == label) for row, label in zip(values, labels, strict=True)]))


@dataclass(frozen=True)
class BacktestResult:
    """Metrics plus audit fields for an OOS run."""

    model_version: str
    sample_size: int
    competition: str
    date_range: tuple[datetime, datetime]
    log_loss: float
    brier: float
    rps: float
    accuracy: float
    predictions: tuple[ModelPrediction, ...]


def evaluate_oos(predictions: Iterable[ModelPrediction], rows: Iterable[TrainingMatch], *, competition: str,
                 model_version: str) -> BacktestResult:
    """Evaluate only successful OOS predictions and verify their time provenance."""
    pairs = list(zip(predictions, rows, strict=True))
    if not pairs:
        raise ValueError("empty OOS evaluation")
    for prediction, row in pairs:
        if not prediction.is_oos or prediction.training_end_time is None:
            raise TrainingDataError("IN_SAMPLE_OR_MISSING_OOS_FLAG")
        if not prediction.training_end_time <= prediction.prediction_time < row.kickoff_time:
            raise TrainingDataError("OOS_TIME_ORDER_VIOLATION")
    selected = [(prediction, row) for prediction, row in pairs if prediction.execution_status.value == "SUCCESS"]
    if not selected:
        raise ValueError("no successful OOS predictions")
    probs = [_vector(prediction) for prediction, _ in selected]
    labels = [0 if row.home_goals > row.away_goals else 1 if row.home_goals == row.away_goals else 2
              for _, row in selected]
    dates = [row.kickoff_time for _, row in selected]
    return BacktestResult(model_version=model_version, sample_size=len(selected), competition=competition,
                          date_range=(min(dates), max(dates)), log_loss=log_loss(probs, labels),
                          brier=brier_score(probs, labels), rps=ranked_probability_score(probs, labels),
                          accuracy=accuracy(probs, labels), predictions=tuple(prediction for prediction, _ in selected))


def naive_league_frequency(rows: Iterable[TrainingMatch]) -> ProbabilityVector:
    """Historical H/D/A frequency benchmark from an explicitly bounded training set."""
    matches = list(rows)
    if not matches:
        raise ValueError("empty benchmark data")
    counts = [sum(row.home_goals > row.away_goals for row in matches),
              sum(row.home_goals == row.away_goals for row in matches),
              sum(row.home_goals < row.away_goals for row in matches)]
    total = len(matches)
    return ProbabilityVector(p_home=counts[0] / total, p_draw=counts[1] / total, p_away=counts[2] / total)


def independent_poisson_benchmark(rows: Iterable[TrainingMatch], max_goals: int = 12) -> tuple[ProbabilityVector, tuple[tuple[float, ...], ...]]:
    """Simple bounded independent-Poisson benchmark, with the same score-matrix contract."""
    matches = list(rows)
    if not matches:
        raise ValueError("empty benchmark data")
    home_rate = sum(row.home_goals for row in matches) / len(matches)
    away_rate = sum(row.away_goals for row in matches) / len(matches)
    from scipy.stats import poisson
    matrix = np.outer(poisson.pmf(np.arange(max_goals + 1), home_rate),
                      poisson.pmf(np.arange(max_goals + 1), away_rate))
    matrix /= matrix.sum()
    values = tuple(tuple(float(v) for v in row) for row in matrix)
    vector = ProbabilityVector(p_home=float(np.tril(matrix, -1).sum()), p_draw=float(np.trace(matrix)),
                               p_away=float(np.triu(matrix, 1).sum()))
    return vector, values
