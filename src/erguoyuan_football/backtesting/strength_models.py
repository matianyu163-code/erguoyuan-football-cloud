"""Rolling OOS evaluation for Phase 5 strength models without random splits."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import numpy as np

from erguoyuan_football.backtesting.base_model_backtest import (
    BacktestResult,
    evaluate_oos,
    walk_forward_split,
)
from erguoyuan_football.data.schemas import Fixture
from erguoyuan_football.data.snapshots import PredictionSnapshot
from erguoyuan_football.models.base import BaseFootballModel
from erguoyuan_football.models.config import ModelConfig
from erguoyuan_football.models.opta_like.model import CoreOptaXGEloLikeModel
from erguoyuan_football.models.spi.model import CoreSPILikeModel, SPIConfig
from erguoyuan_football.models.training import TrainingDataset, TrainingMatch


def _training_slice(data: TrainingDataset, cutoff) -> TrainingDataset:
    rows = tuple(row for row in data.matches if row.kickoff_time < cutoff and row.available_at <= cutoff)
    ids = {row.match_id for row in rows}
    return TrainingDataset(matches=rows, known_team_ids=data.known_team_ids,
        competition_hierarchy=data.competition_hierarchy, team_hierarchy=data.team_hierarchy,
        team_hierarchy_timeline=tuple(item for item in data.team_hierarchy_timeline
            if item.as_of_time <= cutoff),
        xg_observations=tuple(item for item in data.xg_observations
            if item.match_id in ids and item.as_of_time <= cutoff and item.retrieved_at <= cutoff),
        dataset_kind=data.dataset_kind)


def _forecast_fixture(row: TrainingMatch) -> tuple[Fixture, PredictionSnapshot]:
    prediction_time = row.kickoff_time - timedelta(minutes=30)
    fixture = Fixture(match_id=row.match_id, competition_id=row.competition_id,
        home_team_id=row.home_team_id, away_team_id=row.away_team_id,
        kickoff_time=row.kickoff_time, source=row.source, retrieved_at=prediction_time,
        as_of_time=prediction_time, data_version=row.data_version, season=row.season,
        neutral_venue=row.neutral_venue)
    return fixture, PredictionSnapshot(match_id=row.match_id, prediction_time=prediction_time,
                                       match_data_snapshot=fixture)


def walk_forward_strength_model(model_id: str, data: TrainingDataset, *,
                                config: ModelConfig | None = None,
                                initial_matches: int = 24, horizon: int = 4,
                                step: int = 4) -> BacktestResult:
    """Fit once per fold and evaluate only later matches; metadata keeps synthetic tests labelled."""
    if model_id not in {"CORE_SPI_LIKE_V1", "CORE_OPTA_XG_ELO_LIKE_V1"}:
        raise ValueError(f"unsupported Phase 5 model: {model_id}")
    active_config = config or ModelConfig(min_matches=max(12, initial_matches),
                                           min_team_matches=1, training_window=None,
                                           allow_test_data=data.dataset_kind == "SYNTHETIC_TEST")
    predictions = []
    targets = []
    model: BaseFootballModel
    windows = walk_forward_split(data, initial_matches=initial_matches, horizon=horizon, step=step)
    for window in windows:
        training = _training_slice(data, window.training_end)
        if model_id == "CORE_SPI_LIKE_V1":
            model = CoreSPILikeModel().fit(training, window.training_end,
                                           SPIConfig(**active_config.model_dump()))
        else:
            model = CoreOptaXGEloLikeModel().fit(training, window.training_end, active_config)
        for row in window.test_rows:
            fixture, snapshot = _forecast_fixture(row)
            prediction = model.predict(fixture, snapshot)
            predictions.append(prediction.model_copy(update={"is_oos": True}))
            targets.append(row)
    if not predictions:
        raise ValueError("NO_VALID_WALK_FORWARD_WINDOWS")
    competition = ",".join(sorted({row.competition_id for row in targets}))
    return evaluate_oos(predictions, targets, competition=competition,
                        model_version=f"{model_id}:1.0.0:{data.dataset_kind}")


def expected_calibration_error(probabilities: list[tuple[float, float, float]],
                               outcomes: list[int], bins: int = 10) -> float:
    """Top-class multiclass ECE, reported as auxiliary OOS calibration evidence."""
    if not probabilities or len(probabilities) != len(outcomes) or bins < 1:
        raise ValueError("matching nonempty probabilities and outcomes required")
    rows = [(int(np.argmax(p)), max(p), int(int(np.argmax(p)) == outcome))
            for p, outcome in zip(probabilities, outcomes, strict=True)]
    error = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        members = [row for row in rows if lower <= row[1] < upper or (index == bins - 1 and row[1] == 1)]
        if members:
            error += len(members) / len(rows) * abs(float(np.mean([r[1] for r in members]))
                                                      - float(np.mean([r[2] for r in members])))
    return error


def correlation_report(predictions: dict[str, tuple[tuple[float, float, float], ...]]) -> dict[str, Any]:
    """Return per-outcome prediction correlations for aligned forecasts only."""
    if len(predictions) < 2:
        return {"status": "INSUFFICIENT_MODELS", "models": sorted(predictions)}
    lengths = {len(values) for values in predictions.values()}
    if len(lengths) != 1 or not next(iter(lengths)):
        raise ValueError("prediction vectors must be nonempty and aligned")
    names = sorted(predictions)
    result: dict[str, Any] = {"models": names, "outcomes": {}}
    for column, name in enumerate(("home", "draw", "away")):
        rows = [[row[column] for row in predictions[model]] for model in names]
        matrix: list[list[float | None]] = []
        for left in rows:
            correlations: list[float | None] = []
            for right in rows:
                if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
                    correlations.append(None)
                else:
                    correlations.append(float(np.corrcoef(left, right)[0, 1]))
            matrix.append(correlations)
        result["outcomes"][name] = matrix
    return result
