"""Auditable 2026 FIFA World Cup dataset, PIT prediction, and metric contracts."""

from __future__ import annotations

import math
from collections import Counter
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from erguoyuan_football.backtesting.base_model_backtest import (
    accuracy,
    brier_score,
    log_loss,
)
from erguoyuan_football.contracts.common import Contract, UTCTime


class WorldCupStage(StrEnum):
    """Stages in the 48-team 2026 tournament, including the bronze match."""

    GROUP = "GROUP"
    ROUND32 = "ROUND32"
    ROUND16 = "ROUND16"
    QUARTER = "QUARTER"
    SEMI = "SEMI"
    THIRD_PLACE = "THIRD_PLACE"
    FINAL = "FINAL"


class WorldCupSourceEvidence(Contract):
    """Source lineage for a fixture, result, or pre-match input fact."""

    evidence_id: str
    source: str
    source_url: str
    retrieved_at: UTCTime
    as_of_time: UTCTime
    content_sha256: str | None = None

    @model_validator(mode="after")
    def validate_source(self) -> WorldCupSourceEvidence:
        if not self.source_url.startswith(("https://", "http://")):
            raise ValueError("SOURCE_URL_MUST_BE_HTTP")
        return self


class WorldCupResult(Contract):
    """Post-match labels separated into regulation, extra-time, and penalties."""

    regulation_home_goals: int = Field(ge=0)
    regulation_away_goals: int = Field(ge=0)
    extra_time_home_goals: int = Field(default=0, ge=0)
    extra_time_away_goals: int = Field(default=0, ge=0)
    penalties_home: int | None = Field(default=None, ge=0)
    penalties_away: int | None = Field(default=None, ge=0)
    winner_team_id: str | None = None

    @model_validator(mode="after")
    def validate_penalties(self) -> WorldCupResult:
        if (self.penalties_home is None) != (self.penalties_away is None):
            raise ValueError("PENALTY_SCORES_MUST_BE_PAIRED")
        if self.penalties_home is not None and self.penalties_home == self.penalties_away:
            raise ValueError("PENALTY_WINNER_MUST_BE_UNIQUE")
        return self


class WorldCupMatch(Contract):
    """One official fixture; labels remain distinct from pre-match features."""

    match_id: str
    competition: Literal["FIFA_WORLD_CUP_2026"] = "FIFA_WORLD_CUP_2026"
    match_number: int = Field(ge=1, le=104)
    stage: WorldCupStage
    group: str | None = None
    home_team_id: str
    home_team: str
    away_team_id: str
    away_team: str
    kickoff_time: UTCTime
    venue: str
    fixture_evidence: tuple[WorldCupSourceEvidence, ...] = ()
    result: WorldCupResult | None = None
    result_evidence: tuple[WorldCupSourceEvidence, ...] = ()

    @model_validator(mode="after")
    def validate_fixture(self) -> WorldCupMatch:
        if self.home_team_id == self.away_team_id:
            raise ValueError("WORLD_CUP_TEAMS_MUST_DIFFER")
        if self.stage == WorldCupStage.GROUP and not self.group:
            raise ValueError("GROUP_STAGE_FIXTURE_REQUIRES_GROUP")
        if self.stage != WorldCupStage.GROUP and self.group is not None:
            raise ValueError("KNOCKOUT_FIXTURE_MUST_NOT_HAVE_GROUP")
        if self.result is not None and not self.result_evidence:
            raise ValueError("RESULT_REQUIRES_SOURCE_EVIDENCE")
        if any(evidence.as_of_time > self.kickoff_time for evidence in self.fixture_evidence):
            raise ValueError("FIXTURE_EVIDENCE_AS_OF_AFTER_KICKOFF")
        if self.result is not None and any(
            evidence.as_of_time < self.kickoff_time for evidence in self.result_evidence
        ):
            raise ValueError("RESULT_EVIDENCE_AS_OF_BEFORE_KICKOFF")
        return self


class TournamentContext(Contract):
    """Strictly pre-match tournament state; values are supplied with evidence."""

    match_id: str
    stage: WorldCupStage
    group_position: dict[str, int] = Field(default_factory=dict)
    qualification_pressure: float | None = Field(default=None, ge=0, le=1)
    elimination_risk: float | None = Field(default=None, ge=0, le=1)
    rotation_probability: float | None = Field(default=None, ge=0, le=1)
    as_of_time: UTCTime
    source_evidence: tuple[WorldCupSourceEvidence, ...]


class WorldCupFavorite(Contract):
    """Pre-match reference favorite used only to define an upset label."""

    team_id: str
    evidence: WorldCupSourceEvidence


class WorldCupPrediction(Contract):
    """Frozen pre-match 1X2 forecast with training and input lineage."""

    match_id: str
    model_id: str
    model_version: str
    prediction_time: UTCTime
    kickoff_time: UTCTime
    training_end_time: UTCTime | None
    input_data_version: str
    input_evidence: tuple[WorldCupSourceEvidence, ...]
    status: Literal["SUCCESS", "UNAVAILABLE", "FAILED"]
    p_home: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    p_draw: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    p_away: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    reason: str | None = None

    @model_validator(mode="after")
    def validate_prediction(self) -> WorldCupPrediction:
        if not self.prediction_time < self.kickoff_time:
            raise ValueError("PREDICTION_MUST_PRECEDE_KICKOFF")
        if self.status == "SUCCESS":
            if self.training_end_time is None or self.training_end_time > self.prediction_time:
                raise ValueError("TRAINING_CUTOFF_MUST_PRECEDE_PREDICTION")
            if not self.input_evidence:
                raise ValueError("SUCCESS_REQUIRES_INPUT_EVIDENCE")
            if any(
                item.as_of_time > self.prediction_time
                or item.retrieved_at > self.prediction_time
                for item in self.input_evidence
            ):
                raise ValueError("FUTURE_INPUT_EVIDENCE")
            values = (self.p_home, self.p_draw, self.p_away)
            if any(value is None for value in values):
                raise ValueError("SUCCESS_REQUIRES_ALL_PROBABILITIES")
            assert self.p_home is not None and self.p_draw is not None and self.p_away is not None
            if not math.isclose(self.p_home + self.p_draw + self.p_away, 1.0, abs_tol=1e-6):
                raise ValueError("PROBABILITIES_MUST_SUM_TO_ONE")
        elif any(value is not None for value in (self.p_home, self.p_draw, self.p_away)):
            raise ValueError("NON_SUCCESS_PROBABILITIES_MUST_BE_NULL")
        elif not self.reason:
            raise ValueError("NON_SUCCESS_REQUIRES_REASON")
        return self


class WorldCupDataset(Contract):
    """A versioned special-event fixture set; completion is explicitly checked."""

    dataset_version: Literal["GOLDEN_WC2026_V1"] = "GOLDEN_WC2026_V1"
    dataset_type: Literal["SPECIAL_EVENT_DATASET"] = "SPECIAL_EVENT_DATASET"
    source_evidence: tuple[WorldCupSourceEvidence, ...]
    matches: tuple[WorldCupMatch, ...]

    @model_validator(mode="after")
    def validate_ids(self) -> WorldCupDataset:
        ids = [match.match_id for match in self.matches]
        numbers = [match.match_number for match in self.matches]
        if len(ids) != len(set(ids)):
            raise ValueError("DUPLICATE_WORLD_CUP_MATCH_ID")
        if len(numbers) != len(set(numbers)):
            raise ValueError("DUPLICATE_WORLD_CUP_MATCH_NUMBER")
        return self

    def validate_complete(self) -> None:
        """Require all 104 sourced fixtures and result labels before scoring."""
        expected = {
            WorldCupStage.GROUP: 72,
            WorldCupStage.ROUND32: 16,
            WorldCupStage.ROUND16: 8,
            WorldCupStage.QUARTER: 4,
            WorldCupStage.SEMI: 2,
            WorldCupStage.THIRD_PLACE: 1,
            WorldCupStage.FINAL: 1,
        }
        actual = Counter(match.stage for match in self.matches)
        if len(self.matches) != 104 or actual != expected:
            raise ValueError("WORLD_CUP_DATASET_MUST_HAVE_104_MATCHES_WITH_EXPECTED_STAGES")
        if any(not match.fixture_evidence for match in self.matches):
            raise ValueError("ALL_FIXTURES_REQUIRE_SOURCE_EVIDENCE")
        if any(match.result is None for match in self.matches):
            raise ValueError("ALL_RESULTS_REQUIRED_FOR_METRICS")


def load_worldcup_dataset(path: str | Path) -> WorldCupDataset:
    """Load a validated JSON dataset without fetching or inventing source data."""
    return WorldCupDataset.model_validate_json(Path(path).read_text(encoding="utf-8"))


class WorldCupMetricReport(Contract):
    """Metrics are post-freeze diagnostics and never feed model training."""

    sample_size: int = Field(ge=1)
    group_stage_accuracy: float = Field(ge=0, le=1)
    knockout_regulation_accuracy: float = Field(ge=0, le=1)
    upset_match_accuracy: float | None = Field(default=None, ge=0, le=1)
    upset_sample_size: int = Field(default=0, ge=0)
    multiclass_brier: float = Field(ge=0)
    log_loss: float = Field(ge=0)
    top_label_ece: float = Field(ge=0, le=1)
    calibration_bins: int = Field(ge=1)


def _outcome(match: WorldCupMatch) -> int:
    result = match.result
    if result is None:
        raise ValueError("MATCH_RESULT_UNAVAILABLE")
    if result.regulation_home_goals > result.regulation_away_goals:
        return 0
    if result.regulation_home_goals == result.regulation_away_goals:
        return 1
    return 2


def evaluate_worldcup_predictions(
    matches: tuple[WorldCupMatch, ...],
    predictions: tuple[WorldCupPrediction, ...],
    *,
    pre_match_favorites: dict[str, WorldCupFavorite],
    calibration_bins: int = 10,
) -> WorldCupMetricReport:
    """Evaluate frozen 1X2 predictions; upset reference must be pre-match sourced."""
    if calibration_bins < 1:
        raise ValueError("CALIBRATION_BINS_MUST_BE_POSITIVE")
    by_match = {match.match_id: match for match in matches}
    successful = [row for row in predictions if row.status == "SUCCESS"]
    if not successful:
        raise ValueError("NO_SUCCESSFUL_WORLD_CUP_PREDICTIONS")
    if len({row.match_id for row in successful}) != len(successful):
        raise ValueError("DUPLICATE_WORLD_CUP_PREDICTIONS")
    paired: list[tuple[WorldCupMatch, WorldCupPrediction, tuple[float, float, float], int]] = []
    for prediction in successful:
        match = by_match.get(prediction.match_id)
        if match is None or match.result is None:
            raise ValueError("PREDICTION_MATCH_OR_RESULT_MISSING")
        if prediction.kickoff_time != match.kickoff_time:
            raise ValueError("PREDICTION_KICKOFF_MISMATCH")
        outcome = _outcome(match)
        assert prediction.p_home is not None
        assert prediction.p_draw is not None
        assert prediction.p_away is not None
        probs = (float(prediction.p_home), float(prediction.p_draw), float(prediction.p_away))
        paired.append((match, prediction, probs, outcome))

    group_rows = [row for row in paired if row[0].stage == WorldCupStage.GROUP]
    knockout_rows = [row for row in paired if row[0].stage != WorldCupStage.GROUP]
    if not group_rows or not knockout_rows:
        raise ValueError("GROUP_AND_KNOCKOUT_PREDICTIONS_REQUIRED")
    group_accuracy = accuracy([row[2] for row in group_rows], [row[3] for row in group_rows])
    knockout_accuracy = accuracy([row[2] for row in knockout_rows], [row[3] for row in knockout_rows])
    outcomes = [row[3] for row in paired]
    probabilities = [row[2] for row in paired]

    upset_rows: list[tuple[tuple[float, float, float], int]] = []
    for match, prediction, probs, outcome in paired:
        favorite = pre_match_favorites.get(match.match_id)
        if favorite is None:
            continue
        if favorite.team_id not in {match.home_team_id, match.away_team_id}:
            raise ValueError("PRE_MATCH_FAVORITE_NOT_IN_MATCH")
        if (favorite.evidence.as_of_time > prediction.prediction_time
                or favorite.evidence.retrieved_at > prediction.prediction_time):
            raise ValueError("UPSET_REFERENCE_EVIDENCE_NOT_PRE_MATCH")
        actual_winner = (match.home_team_id if outcome == 0 else
                         match.away_team_id if outcome == 2 else None)
        if actual_winner is not None and actual_winner != favorite.team_id:
            upset_rows.append((probs, outcome))
    upset_accuracy = (
        accuracy([row[0] for row in upset_rows], [row[1] for row in upset_rows])
        if upset_rows else None
    )

    confidence = [max(row) for row in probabilities]
    predicted = [max(range(3), key=row.__getitem__) for row in probabilities]
    bin_indices = [min(calibration_bins - 1, int(value * calibration_bins)) for value in confidence]
    ece = 0.0
    for index in range(calibration_bins):
        members = [i for i, bin_index in enumerate(bin_indices) if bin_index == index]
        if members:
            mean_confidence = sum(confidence[i] for i in members) / len(members)
            mean_accuracy = sum(predicted[i] == outcomes[i] for i in members) / len(members)
            ece += len(members) / len(paired) * abs(mean_confidence - mean_accuracy)
    return WorldCupMetricReport(
        sample_size=len(paired),
        group_stage_accuracy=group_accuracy,
        knockout_regulation_accuracy=knockout_accuracy,
        upset_match_accuracy=upset_accuracy,
        upset_sample_size=len(upset_rows),
        multiclass_brier=brier_score(probabilities, outcomes),
        log_loss=log_loss(probabilities, outcomes),
        top_label_ece=ece,
        calibration_bins=calibration_bins,
    )


def validate_worldcup_prediction_pit(
    *,
    prediction_time: datetime,
    kickoff_time: datetime,
    training_end_time: datetime,
    input_evidence: tuple[WorldCupSourceEvidence, ...],
) -> None:
    """Reject post-kickoff, post-prediction, or timezone-naive forecast inputs."""
    times = (prediction_time, kickoff_time, training_end_time)
    if any(value.tzinfo is None or value.utcoffset() is None for value in times):
        raise ValueError("PIT_TIMESTAMPS_MUST_BE_TIMEZONE_AWARE")
    boundary = prediction_time.astimezone(UTC)
    if training_end_time.astimezone(UTC) > boundary:
        raise ValueError("TRAINING_END_AFTER_PREDICTION_TIME")
    if boundary >= kickoff_time.astimezone(UTC):
        raise ValueError("PREDICTION_NOT_BEFORE_KICKOFF")
    if any(evidence.as_of_time > boundary or evidence.retrieved_at > boundary
           for evidence in input_evidence):
        raise ValueError("FUTURE_DATA_DETECTED")
