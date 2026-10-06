"""Synthetic provider and point-in-time cases; no match enters a real ledger."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime

import pytest

from erguoyuan_football.blind_test_r3.national_history import (
    PINNED_COMMIT,
    NationalTeamHistoryProvider,
)
from erguoyuan_football.models.training import TrainingDatasetValidator


def _provider(tmp_path, source_time: str) -> NationalTeamHistoryProvider:
    csv_path = tmp_path / "results.csv"
    content = ("date,home_team,away_team,home_score,away_score,tournament,neutral\n"
        "2025-03-01,Turkey,France,1,0,FIFA World Cup qualification,FALSE\n"
        "2025-06-01,Bosnia and Herzegovina,Turkey,0,0,UEFA Nations League,TRUE\n"
        "2025-07-01,France,Turkey,2,1,FIFA World Cup,TRUE\n"
        "2026-10-05,Turkey,France,7,7,UEFA Nations League,FALSE\n")
    csv_path.write_text(content, encoding="utf-8")
    metadata_path = tmp_path / "results.json"
    metadata_path.write_text(json.dumps({"commit": PINNED_COMMIT,
        "sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "source_timestamp": source_time}), encoding="utf-8")
    return NationalTeamHistoryProvider(csv_path, metadata_path)


def test_provider_maps_canonical_teams_and_excludes_cup_score_from_training(tmp_path):
    provider = _provider(tmp_path, "2026-10-05T06:00:00+00:00")
    cutoff = datetime(2026, 10, 5, 7, tzinfo=UTC)
    rows = provider.rows_as_of(cutoff, train_start=date(2024, 1, 1))
    assert len(rows) == 3
    assert rows[0].home_team_id == "NATIONAL_TUR_M_SENIOR"
    assert rows[1].home_team_id == "NATIONAL_BIH_M_SENIOR"
    assert rows[1].neutral_venue is True and rows[1].kickoff_time is None
    dataset = provider.training_dataset(cutoff, train_start=date(2024, 1, 1))
    assert len(dataset.matches) == 2
    assert dataset.temporal_mode == "DATE_SAFE_BATCH"
    TrainingDatasetValidator().validate(dataset, cutoff)


def test_provider_rejects_snapshot_retrieved_after_cutoff(tmp_path):
    provider = _provider(tmp_path, "2026-10-05T08:00:00+00:00")
    with pytest.raises(ValueError, match="RETRIEVED_AFTER_CUTOFF"):
        provider.rows_as_of(datetime(2026, 10, 5, 7, tzinfo=UTC),
                            train_start=date(2024, 1, 1))
