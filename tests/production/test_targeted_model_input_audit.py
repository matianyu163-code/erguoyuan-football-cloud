"""Regression coverage for bridge history identity and training-team coverage."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from datetime import UTC, timedelta
from pathlib import Path

import pytest

from erguoyuan_football.knowledge.match_identity import GlobalKnowledgeResolver
from erguoyuan_football.models.config import ModelConfig
from erguoyuan_football.models.elo import CoreEloModel
from erguoyuan_football.models.training import (
    InsufficientData,
    TrainingDatasetValidator,
)
from erguoyuan_football.prediction.model_input_adapter import (
    DixonColesInputAdapter,
    ModelInputError,
    default_model_input_adapters,
)
from erguoyuan_football.research.live_data.model_requirements import (
    load_model_requirements,
)
from erguoyuan_football.research.samples.sample_builder import SampleBuilder
from erguoyuan_football.research.samples.sample_repository import SampleRepository
from production.bridge.contracts import BridgeHistory, CoreDataPacketV1
from production.bridge.importer import bridge_competition, import_history
from production.bridge.runner import _readiness
from production.bridge.validator import BridgeRejected

ROOT = Path(__file__).resolve().parents[2]
PACKET_PATH = ROOT / "data/bridge/ready/00e173de-4f86-4be5-b81f-114cd200f954.packet.json"


def _packet() -> CoreDataPacketV1:
    return CoreDataPacketV1.model_validate_json(PACKET_PATH.read_text(encoding="utf-8"))


def _imported(packet: CoreDataPacketV1):
    resolver = GlobalKnowledgeResolver()
    imported = import_history(packet, resolver)
    target = resolver.resolve_names(packet.match.home, packet.match.away,
                                    packet.match.competition, allow_discovery=False)
    assert target.home_team is not None and target.away_team is not None
    competition = bridge_competition(packet.match.competition,
                                     target.home_team.federation,
                                     target.home_team.entity_type)
    hierarchy = SampleBuilder(imported.repository).build(
        target.home_team, target.away_team, competition,
        cutoff=packet.research_as_of.astimezone(UTC),
        competition_type="INTERNATIONAL",
    )
    return resolver, imported, target, competition, hierarchy


def test_bridge_history_preserves_opponent_entities() -> None:
    _resolver, imported, _target, _competition, _hierarchy = _imported(_packet())
    team_ids = {team for row in imported.repository.samples
                for team in (row.home_team_id, row.away_team_id)}

    assert imported.imported_count == 63
    assert imported.deduplicated_count == 56
    assert len(team_ids) == 31
    assert "NATIONAL_FRA_M_SENIOR" in team_ids
    assert "NATIONAL_BEL_M_SENIOR" in team_ids
    assert all(team.startswith("NATIONAL_") and team.endswith("_M_SENIOR")
               for team in team_ids)


def test_national_team_history_has_unique_team_coverage() -> None:
    _resolver, imported, _target, _competition, _hierarchy = _imported(_packet())
    teams = {team for row in imported.repository.samples
             for team in (row.home_team_id, row.away_team_id)}
    assert len(teams) == 31
    assert len(teams) >= 3


def test_training_team_coverage_uses_all_training_entities() -> None:
    packet = _packet()
    _resolver, imported, _target, _competition, hierarchy = _imported(packet)
    outcomes = {0 if row.home_goals > row.away_goals else
                1 if row.home_goals == row.away_goals else 2
                for row in imported.repository.samples}
    report = _readiness(hierarchy, imported.deduplicated_count, outcomes,
                        packet.research_as_of.astimezone(UTC),
                        ROOT / "config/model_registry.yaml")
    rows = {row.model_name: row for row in report.model_readiness}

    for model_id in ("DIXON_COLES_V1", "BIVARIATE_POISSON_V1", "ELO_V1",
                     "PI_RATING_V1", "CORE_SPI_LIKE_V1"):
        assert rows[model_id].training_team_coverage == 31
        assert "TRAINING_TEAM_COVERAGE_LOW:1<3" not in rows[model_id].reasons
    assert rows["DIXON_COLES_V1"].training_sample_count == 56


def test_bridge_history_is_training_eligible_and_pit_bounded() -> None:
    packet = _packet()
    _resolver, imported, _target, _competition, hierarchy = _imported(packet)
    cutoff = packet.research_as_of.astimezone(UTC)
    bundle = DixonColesInputAdapter().build_input(
        imported.repository, hierarchy, cutoff)

    TrainingDatasetValidator().validate(bundle.training_dataset, cutoff)
    assert len(bundle.training_dataset.matches) == 56
    assert len(bundle.training_dataset.known_team_ids) == 31
    assert all(row.kickoff_time < cutoff and row.completed_at <= cutoff
               for row in bundle.training_dataset.matches)
    assert bundle.training_dataset.dataset_kind == "REAL"


def test_base_models_are_independent_of_bayesian_prior_readiness() -> None:
    packet = _packet()
    _resolver, imported, _target, _competition, hierarchy = _imported(packet)
    outcomes = {0 if row.home_goals > row.away_goals else
                1 if row.home_goals == row.away_goals else 2
                for row in imported.repository.samples}
    report = _readiness(hierarchy, imported.deduplicated_count, outcomes,
                        packet.research_as_of.astimezone(UTC),
                        ROOT / "config/model_registry.yaml")
    rows = {row.model_name: row for row in report.model_readiness}

    assert rows["DIXON_COLES_V1"].ready
    assert rows["BIVARIATE_POISSON_V1"].ready
    assert rows["ELO_V1"].ready
    assert rows["PI_RATING_V1"].ready
    assert rows["CORE_SPI_LIKE_V1"].ready
    assert rows["BAYESIAN_HIERARCHICAL_V1"].ready is False
    assert "INFERENCE_COMPETITION_SAMPLE_LOW:4<20" in rows[
        "BAYESIAN_HIERARCHICAL_V1"].reasons


def test_france_belgium_snapshot_team_count() -> None:
    packet = _packet()
    _resolver, imported, target, _competition, _hierarchy = _imported(packet)
    home_away = {row.home_team_id for row in imported.repository.samples}
    away_teams = {row.away_team_id for row in imported.repository.samples}
    team_ids = home_away | away_teams

    assert target.home_team.team_id == "NATIONAL_FRA_M_SENIOR"
    assert target.away_team.team_id == "NATIONAL_BEL_M_SENIOR"
    assert len(imported.repository.samples) == 56
    assert len(home_away) > 1 and len(away_teams) > 1
    assert len(team_ids) == 31


def test_unknown_opponent_is_rejected_not_collapsed_to_generic_team() -> None:
    packet = _packet()
    first = packet.history.home_matches[0]
    unknown = first.model_copy(update={
        "home": "Unlisted National Side 9c42f6",
        "home_entity": None,
        "match_id": "UNKNOWN_OPPONENT_TEST",
    })
    history = BridgeHistory.model_validate({
        **packet.history.model_dump(),
        "home_matches": (unknown,),
        "away_matches": (), "direct_matches": (), "competition_matches": (),
    })
    changed = packet.model_copy(update={"history": history})

    with pytest.raises(BridgeRejected, match="ENTITY_RESOLUTION_FAILED"):
        import_history(changed, GlobalKnowledgeResolver())


def test_duplicate_entities_and_youth_senior_separation() -> None:
    _resolver, imported, _target, _competition, _hierarchy = _imported(_packet())
    fingerprint_counts = Counter(row.fingerprint
                                 for row in imported.repository.samples)
    team_ids = {team for row in imported.repository.samples
                for team in (row.home_team_id, row.away_team_id)}
    assert len(imported.repository.samples) == 56
    assert all(count == 1 for count in fingerprint_counts.values())
    assert all(team.endswith("_M_SENIOR") for team in team_ids)


def test_future_history_is_still_rejected_by_model_input_adapter() -> None:
    packet = _packet()
    _resolver, imported, _target, _competition, hierarchy = _imported(packet)
    cutoff = packet.research_as_of.astimezone(UTC)
    row = imported.repository.samples[0]
    future = replace(row, fetched_at=cutoff + timedelta(microseconds=1))
    repository = SampleRepository(tuple(
        future if sample.match_id == row.match_id else sample
        for sample in imported.repository.samples))

    with pytest.raises(ModelInputError, match="FUTURE_EVIDENCE_REJECTED"):
        DixonColesInputAdapter().build_input(repository, hierarchy, cutoff)


def test_multi_competition_relaxation_does_not_apply_to_club_training() -> None:
    packet = _packet()
    _resolver, imported, _target, _competition, hierarchy = _imported(packet)
    cutoff = packet.research_as_of.astimezone(UTC)
    dataset = default_model_input_adapters()["ELO_V1"].build_input(
        imported.repository, hierarchy, cutoff).training_dataset
    club_rows = tuple(row.model_copy(update={
        "home_team_id": f"CLUB_{row.home_team_id}",
        "away_team_id": f"CLUB_{row.away_team_id}",
        "competition_id": f"CLUB_COMP_{row.competition_id}",
    }) for row in dataset.matches)
    club_ids = frozenset(team for row in club_rows
                         for team in (row.home_team_id, row.away_team_id))
    club_dataset = dataset.model_copy(update={
        "matches": club_rows,
        "known_team_ids": club_ids,
    })

    with pytest.raises(InsufficientData, match="V1_REQUIRES_ONE_COMPETITION_PER_FIT"):
        CoreEloModel().fit(club_dataset, cutoff + timedelta(seconds=1),
                           ModelConfig(min_matches=12, min_team_matches=3,
                                       training_window=None))


def test_model_readiness_reports_model_specific_sample_and_team_counts() -> None:
    packet = _packet()
    _resolver, imported, _target, _competition, hierarchy = _imported(packet)
    outcomes = {0 if row.home_goals > row.away_goals else
                1 if row.home_goals == row.away_goals else 2
                for row in imported.repository.samples}
    report = _readiness(hierarchy, imported.deduplicated_count, outcomes,
                        packet.research_as_of.astimezone(UTC),
                        ROOT / "config/model_registry.yaml")
    requirements = {row.model_id: row for row in load_model_requirements(
        ROOT / "config/model_registry.yaml")}
    readiness = {row.model_name: row for row in report.model_readiness}

    for model_id in ("DIXON_COLES_V1", "BIVARIATE_POISSON_V1", "ELO_V1",
                     "PI_RATING_V1", "CORE_SPI_LIKE_V1"):
        policy = requirements[model_id].sample_policy
        assert policy is not None
        assert readiness[model_id].training_sample_count == 56
        assert readiness[model_id].training_team_coverage == 31
        assert policy.min_training_team_matches == 3


def test_default_adapter_registry_still_contains_all_model_inputs() -> None:
    assert len(default_model_input_adapters()) == 11
