"""DATE_SAFE_BATCH is date-granular and never admits same-day results or quotes."""

from datetime import UTC, date, datetime, time, timedelta

import pytest

from erguoyuan_football.backtesting.date_safe_oos import (
    _predict_record,
    _prior_history_rows,
)
from erguoyuan_football.contracts.common import Availability
from erguoyuan_football.data.temporal import PredictionTemporalMode
from erguoyuan_football.markets.schemas import MarketGoalFeatures
from erguoyuan_football.models.training import (
    TrainingDataError,
    TrainingDataset,
    TrainingDatasetValidator,
    TrainingMatch,
)


def row(match_id: str, day: date) -> TrainingMatch:
    start = datetime.combine(day, time.min, UTC)
    end = start + timedelta(days=1) - timedelta(microseconds=1)
    return TrainingMatch(match_id=match_id, competition_id="EPL", season="2024-25",
        kickoff_time=start, completed_at=end, as_of_time=end,
        home_team_id="home", away_team_id="away", home_goals=1, away_goals=0,
        neutral_venue=False, source="SYNTHETIC_TEST", retrieved_at=end + timedelta(days=100),
        data_version=f"hash-{match_id}")


def test_same_day_result_not_training() -> None:
    rows = (row("prior", date(2025, 1, 1)), row("same-day-a", date(2025, 1, 2)),
            row("same-day-b", date(2025, 1, 2)))
    selected = _prior_history_rows(rows, date(2025, 1, 2))
    assert [match.match_id for match in selected] == ["prior"]


def test_next_day_can_use_previous_day() -> None:
    selected = _prior_history_rows((row("d1", date(2025, 1, 1)),
                                    row("d2", date(2025, 1, 2))), date(2025, 1, 3))
    assert {match.match_id for match in selected} == {"d1", "d2"}


def test_future_match_excluded_from_history() -> None:
    selected = _prior_history_rows((row("prior", date(2025, 1, 1)),
                                    row("future", date(2025, 1, 3))), date(2025, 1, 2))
    assert [match.match_id for match in selected] == ["prior"]


def test_date_safe_not_valid_for_hourly_market() -> None:
    match = row("history", date(2025, 1, 1))
    prediction_time = datetime(2025, 1, 2, tzinfo=UTC)
    quote = MarketGoalFeatures(availability=Availability.UNAVAILABLE, match_id=match.match_id,
        market_snapshot_id="snapshot", prediction_time=prediction_time,
        reason="TEST_FIXTURE", as_of_time=prediction_time - timedelta(hours=1),
        retrieved_at=prediction_time - timedelta(minutes=30))
    data = TrainingDataset(matches=(match,), known_team_ids=frozenset({"home", "away"}),
        dataset_kind="SYNTHETIC_TEST", temporal_mode="DATE_SAFE_BATCH", market_goal_features=(quote,))
    with pytest.raises(TrainingDataError, match="FORBIDS_SNAPSHOT_AND_MARKET"):
        TrainingDatasetValidator().validate(data, datetime(2025, 1, 2, tzinfo=UTC))


def test_training_dataset_marks_temporal_mode() -> None:
    data = TrainingDataset(matches=(row("history", date(2025, 1, 1)),),
        known_team_ids=frozenset({"home", "away"}), dataset_kind="SYNTHETIC_TEST",
        temporal_mode=PredictionTemporalMode.DATE_SAFE_BATCH.value,
        assumptions=("same-date rows excluded",))
    assert data.temporal_mode == "DATE_SAFE_BATCH"
    assert "same-date rows excluded" in data.assumptions


def test_date_safe_metadata() -> None:
    history = row("history", date(2025, 1, 1))
    data = TrainingDataset(matches=(history,), known_team_ids=frozenset({"home", "away"}),
        dataset_kind="SYNTHETIC_TEST", temporal_mode="DATE_SAFE_BATCH",
        assumptions=("same-date rows excluded",))
    boundary = datetime(2025, 1, 2, tzinfo=UTC)
    prediction = _predict_record("TEST_MODEL", "1.0", match_id="target", competition_id="EPL",
        season_id="2024-25", target_date=date(2025, 1, 2), training_data=data,
        cutoff=boundary, snapshot_id="snapshot", config_hash="config", sources=("TEST_SOURCE",),
        values={"p_home": 0.5, "p_draw": 0.25, "p_away": 0.25})
    assert prediction.metadata["prediction_temporal_mode"] == "DATE_SAFE_BATCH"
    assert prediction.metadata["training_match_count"] == 1
    assert prediction.is_oos and prediction.prediction_time == boundary
