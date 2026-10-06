"""Explicitly labelled synthetic entity and result records for adapter tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from erguoyuan_football.knowledge.competitions.competition_schema import (
    CompetitionIdentity,
)
from erguoyuan_football.knowledge.teams.team_schema import TeamIdentity
from erguoyuan_football.research.samples.match_deduplicator import HistoricalMatchSample
from erguoyuan_football.research.samples.sample_builder import SampleBuilder
from erguoyuan_football.research.samples.sample_repository import SampleRepository

START = datetime(2025, 1, 1, tzinfo=UTC)
ROOT = Path(__file__).resolve().parents[2]
TEAMS = ("SYN_T0", "SYN_T1", "SYN_T2", "SYN_T3")
COMPETITION = CompetitionIdentity("SYN_COMP", "Synthetic League", "UEFA", "GER", "LEAGUE")
HOME = TeamIdentity("SYN_T0", "Synthetic Team 0", "GER", "UEFA", ["SYN_T0"],
                    entity_type="CLUB", gender="MEN", age_group="SENIOR",
                    squad_level="FIRST_TEAM")
AWAY = TeamIdentity("SYN_T1", "Synthetic Team 1", "GER", "UEFA", ["SYN_T1"],
                    entity_type="CLUB", gender="MEN", age_group="SENIOR",
                    squad_level="FIRST_TEAM")


def make_samples(*, count: int = 48, age_group: str = "SENIOR",
                 gender: str = "MEN", competition_id: str = "SYN_COMP",
                 start_index: int = 0) -> tuple[HistoricalMatchSample, ...]:
    """Create labeled test evidence with valid event and retrieval timestamps."""
    result = []
    for offset in range(count):
        index = start_index + offset
        kickoff = START + timedelta(days=index)
        home = TEAMS[index % len(TEAMS)]
        away = TEAMS[(index + 1) % len(TEAMS)]
        result.append(HistoricalMatchSample(
            f"SYN_MATCH_{age_group}_{index}", home, away, competition_id, kickoff,
            index % 4, (index + 1) % 3, (f"SYNTHETIC_TEST_E_{index}",),
            ("SYNTHETIC_TEST",), 2, kickoff + timedelta(hours=3), "UEFA",
            gender, age_group, "LEAGUE", index % 13 == 0,
        ))
    return tuple(result)


def make_readiness(*, ready_models: tuple[str, ...] = ("ELO_V1", "DIXON_COLES_V1"),
                   data_available: bool = True):
    """Build deterministic readiness evidence for planner and execution tests."""
    from erguoyuan_football.prediction.model_execution_planner import (
        ModelExecutionPlanner,
    )
    from erguoyuan_football.research.live_data.readiness import (
        DataAvailability,
        DataReadinessItem,
        LiveDataReadinessReport,
        ModelReadiness,
    )

    planner = ModelExecutionPlanner(ROOT / "config/phase14_model_requirements.yaml",
                                    ROOT / "config/model_registry.yaml")
    names = {name for policy in planner.policies.values()
             for name in (*policy.training_inputs, *policy.inference_inputs)}
    items = tuple(DataReadinessItem(
        name, DataAvailability.AVAILABLE if data_available else DataAvailability.MISSING,
        evidence_ids=("SYNTHETIC_TEST",) if data_available else (),
        model_ids=("CORE_XGBOOST_V1", "CORE_CATBOOST_V1")
        if name == "ML_ARTIFACT" and data_available else (),
    ) for name in sorted(names))
    rows = tuple(ModelReadiness(
        model_id, model_id in ready_models, False,
        () if model_id in ready_models else ("MODEL_READINESS_BLOCKED",), (), (), (),
        () if model_id in ready_models else ("MODEL_READINESS_BLOCKED",),
        "READY" if model_id in ready_models else "BLOCKED", "MEDIUM",
    ) for model_id in planner.policies)
    return LiveDataReadinessReport(True, "READY", items, rows, ready_models,
        tuple(row.model_name for row in rows if not row.ready), (), (),
        START + timedelta(days=50), sample_hierarchy=None)


@pytest.fixture
def sample_context():
    """Return synthetic sample repository, target identities, and built hierarchy."""
    rows = make_samples()
    cutoff = START + timedelta(days=90)
    repository = SampleRepository(rows)
    hierarchy = SampleBuilder(repository).build(HOME, AWAY, COMPETITION,
        cutoff=cutoff, competition_type="LEAGUE")
    return repository, hierarchy, cutoff
