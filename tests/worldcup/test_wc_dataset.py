"""Dataset contract tests use synthetic fixtures only, never real match labels."""

from datetime import UTC, datetime, timedelta

import pytest

from erguoyuan_football.backtesting.worldcup_dataset import (
    WorldCupDataset,
    WorldCupMatch,
    WorldCupResult,
    WorldCupSourceEvidence,
    WorldCupStage,
)


def _evidence(evidence_id: str, at: datetime) -> WorldCupSourceEvidence:
    return WorldCupSourceEvidence(
        evidence_id=evidence_id,
        source="SYNTHETIC_TEST",
        source_url="https://example.invalid/synthetic",
        retrieved_at=at,
        as_of_time=at,
    )


def _all_stages() -> list[WorldCupStage]:
    return ([WorldCupStage.GROUP] * 72 + [WorldCupStage.ROUND32] * 16
            + [WorldCupStage.ROUND16] * 8 + [WorldCupStage.QUARTER] * 4
            + [WorldCupStage.SEMI] * 2 + [WorldCupStage.THIRD_PLACE]
            + [WorldCupStage.FINAL])


def test_wc_dataset_validates_all_104_fixture_stages() -> None:
    start = datetime(2026, 6, 11, tzinfo=UTC)
    matches = []
    for number, stage in enumerate(_all_stages(), 1):
        kickoff = start + timedelta(hours=number)
        matches.append(WorldCupMatch(
            match_id=f"SYNTHETIC_TEST_WC_{number}",
            match_number=number,
            stage=stage,
            group="A" if stage == WorldCupStage.GROUP else None,
            home_team_id=f"H{number}",
            home_team=f"Home {number}",
            away_team_id=f"A{number}",
            away_team=f"Away {number}",
            kickoff_time=kickoff,
            venue="Synthetic Venue",
            fixture_evidence=(_evidence(f"fixture-{number}", kickoff - timedelta(days=1)),),
            result=WorldCupResult(regulation_home_goals=1, regulation_away_goals=0,
                                  winner_team_id=f"H{number}"),
            result_evidence=(_evidence(f"result-{number}", kickoff + timedelta(hours=2)),),
        ))
    dataset = WorldCupDataset(
        source_evidence=(_evidence("dataset-source", start - timedelta(days=30)),),
        matches=tuple(matches),
    )
    dataset.validate_complete()
    assert len(dataset.matches) == 104
    assert sum(match.stage == WorldCupStage.THIRD_PLACE for match in dataset.matches) == 1


def test_wc_dataset_rejects_duplicate_match_ids() -> None:
    kickoff = datetime(2026, 6, 11, tzinfo=UTC)
    values = {
        "match_id": "duplicate",
        "match_number": 1,
        "stage": WorldCupStage.GROUP,
        "group": "A",
        "home_team_id": "H",
        "home_team": "Home",
        "away_team_id": "A",
        "away_team": "Away",
        "kickoff_time": kickoff,
        "venue": "Synthetic Venue",
        "fixture_evidence": (_evidence("fixture", kickoff - timedelta(days=1)),),
    }
    with pytest.raises(ValueError, match="DUPLICATE_WORLD_CUP_MATCH_ID"):
        WorldCupDataset(
            source_evidence=(_evidence("source", kickoff - timedelta(days=2)),),
            matches=(WorldCupMatch(**values), WorldCupMatch(**{**values, "match_number": 2})),
        )


def test_wc_result_cannot_be_added_without_result_evidence() -> None:
    kickoff = datetime(2026, 6, 11, tzinfo=UTC)
    with pytest.raises(ValueError, match="RESULT_REQUIRES_SOURCE_EVIDENCE"):
        WorldCupMatch(
            match_id="SYNTHETIC_TEST_WC_1",
            match_number=1,
            stage=WorldCupStage.GROUP,
            group="A",
            home_team_id="H",
            home_team="Home",
            away_team_id="A",
            away_team="Away",
            kickoff_time=kickoff,
            venue="Synthetic Venue",
            result=WorldCupResult(regulation_home_goals=1, regulation_away_goals=0),
        )
