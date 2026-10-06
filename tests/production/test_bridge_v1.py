"""SYNTHETIC_TEST packets exercise only bridge gates, never real predictions."""

from __future__ import annotations

import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from production.bridge.contracts import CoreDataPacketV1, HistoryMatch
from production.bridge.importer import import_history
from production.bridge.runner import BridgeRunner
from production.bridge.validator import BridgeRejected, validate_packet
from production.config import ProductionConfig
from production.trial_record import TrialRecordStore


def _packet(*, history: list[dict[str, object]] | None = None,
            source_url: str = "https://www.uefa.com/uefanationsleague/fixtures-results/") -> CoreDataPacketV1:
    now = datetime.now(UTC).replace(microsecond=0)
    kickoff = now + timedelta(days=2)
    return CoreDataPacketV1.model_validate({
        "schema_version": "CORE_DATA_PACKET_V1",
        "request_id": "SYNTHETIC_TEST_BRIDGE_001",
        "created_at": (now - timedelta(minutes=2)).isoformat(),
        "research_as_of": now.isoformat(),
        "match": {"competition": "UEFA Nations League", "home": "Kosovo",
                  "away": "Austria", "kickoff_original": kickoff.isoformat(),
                  "kickoff_timezone": "UTC", "kickoff_utc": kickoff.isoformat(),
                  "source_type": "RESEARCH_TEST", "neutral_venue": False},
        "entities": {"home_entity": "NATIONAL_XKK_M_SENIOR",
                     "away_entity": "NATIONAL_AUT_M_SENIOR"},
        "fixture_evidence": {"sources": [{"source": "UEFA_SYNTHETIC_TEST",
            "source_url": source_url, "source_tier": "A",
            "fetched_at": (now - timedelta(minutes=1)).isoformat()}]},
        "history": {"home_matches": history or []},
    })


def _history(*, goals: int = 1, fetched_offset: int = -1) -> dict[str, object]:
    now = datetime.now(UTC).replace(microsecond=0)
    old = now - timedelta(days=10)
    return {"date": old.date().isoformat(), "kickoff": old.isoformat(),
        "competition": "UEFA Nations League", "home": "Kosovo",
        "away": "Slovenia", "home_goals": goals, "away_goals": 0,
        "neutral_venue": False, "source": "UEFA_SYNTHETIC_TEST",
        "source_url": "https://www.uefa.com/uefanationsleague/fixtures-results/",
        "source_tier": "A",
        "fetched_at": (now + timedelta(days=fetched_offset)).isoformat()}


def test_valid_packet() -> None:
    validate_packet(_packet())


def test_invalid_packet_schema() -> None:
    with pytest.raises(ValidationError):
        CoreDataPacketV1.model_validate({"schema_version": "CORE_DATA_PACKET_V2"})


def test_future_leakage() -> None:
    with pytest.raises(BridgeRejected, match="PIT_REJECTED"):
        validate_packet(_packet(history=[_history(fetched_offset=1)]))


def test_source_rejection() -> None:
    with pytest.raises(BridgeRejected, match="BRIDGE_SOURCE_REJECTED"):
        validate_packet(_packet(source_url="https://prediction-blog.example/match"))


def test_duplicate_history_and_conflict() -> None:
    from erguoyuan_football.knowledge.match_identity import GlobalKnowledgeResolver

    first = _history()
    packet = _packet(history=[first, first])
    validate_packet(packet)
    imported = import_history(packet, GlobalKnowledgeResolver())
    assert imported.imported_count == 2
    assert imported.deduplicated_count == 1
    conflict = _packet(history=[first, first | {"home_goals": 2}])
    validate_packet(conflict)
    with pytest.raises(BridgeRejected, match="HISTORY_RESULT_CONFLICT"):
        import_history(conflict, GlobalKnowledgeResolver())


def test_snapshot_result_export_and_database_record(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    (tmp_path / "config").mkdir()
    for name in ("model_registry.yaml", "phase13_6_model_readiness.yaml",
                 "phase14_model_requirements.yaml"):
        shutil.copy2(repo / "config" / name, tmp_path / "config" / name)
    config = ProductionConfig("TRIAL", False, False, True, False, False, False,
        tmp_path / "trial_prediction_records.sqlite", tmp_path / "updates.json")
    runner = BridgeRunner(config, project_root=tmp_path, synthetic_test=True)
    try:
        result = runner.run(_packet())
    finally:
        runner.close()
    assert result.primary_blocker == "HISTORY_INSUFFICIENT"
    assert result.snapshot_id is not None
    assert result.prediction_id is not None
    assert result.core_probability is None
    assert (tmp_path / "data/bridge/results" / f"{result.prediction_id}.json").is_file()
    store = TrialRecordStore(config.record_database)
    try:
        row = store.list_records()[0]
    finally:
        store.close()
    assert row["input_snapshot"]["synthetic_data"] is True
    assert row["source_type"] == "CHATGPT_WORK_BRIDGE"
    assert row["prediction_output"]["primary_blocker"] == "HISTORY_INSUFFICIENT"


def test_synthetic_packet_rejected_in_production(tmp_path: Path) -> None:
    config = ProductionConfig("TRIAL", False, True, True, False, False, False,
        tmp_path / "trial_prediction_records.sqlite", tmp_path / "updates.json")
    runner = BridgeRunner(config, project_root=tmp_path)
    try:
        result = runner.run(_packet())
    finally:
        runner.close()
    assert result.primary_blocker == "BRIDGE_INPUT_INVALID"
    assert not result.models_executed


def test_exact_history_kickoff_required() -> None:
    row = HistoryMatch.model_validate(_history() | {"kickoff": None})
    packet = _packet(history=[row.model_dump(mode="json")])
    with pytest.raises(BridgeRejected, match="EXACT_HISTORY_KICKOFF_REQUIRED"):
        validate_packet(packet)


def test_synthetic_history_runs_real_rating_code_without_mock(tmp_path: Path) -> None:
    """SYNTHETIC_TEST verifies the model execution path, never real performance."""
    repo = Path(__file__).resolve().parents[2]
    (tmp_path / "config").mkdir()
    for name in ("model_registry.yaml", "phase13_6_model_readiness.yaml",
                 "phase14_model_requirements.yaml"):
        shutil.copy2(repo / "config" / name, tmp_path / "config" / name)
    now = datetime.now(UTC).replace(microsecond=0)
    pairings = [("Kosovo", "Slovenia"), ("Austria", "Albania"),
                ("Kosovo", "Albania"), ("Austria", "Slovenia")]
    history = []
    for index in range(16):
        kickoff = now - timedelta(days=50 - index * 2)
        home, away = pairings[index % len(pairings)]
        score = ((1, 0), (0, 0), (0, 1))[index % 3]
        history.append({
            "date": kickoff.date().isoformat(), "kickoff": kickoff.isoformat(),
            "competition": "UEFA Nations League", "home": home, "away": away,
            "home_goals": score[0], "away_goals": score[1],
            "neutral_venue": False, "source": "UEFA_SYNTHETIC_TEST",
            "source_url": "https://www.uefa.com/uefanationsleague/fixtures-results/",
            "source_tier": "A",
            "fetched_at": (now - timedelta(days=1)).isoformat(),
        })
    config = ProductionConfig("TRIAL", False, True, True, False, False, False,
        tmp_path / "trial_prediction_records.sqlite", tmp_path / "updates.json")
    runner = BridgeRunner(config, project_root=tmp_path, synthetic_test=True,
                          execute_synthetic_models=True)
    try:
        result = runner.run(_packet(history=history))
    finally:
        runner.close()
    assert result.snapshot_id is not None
    assert result.prediction_id is not None
    assert {"ELO_V1", "PI_RATING_V1"} <= set(result.models_planned)
    assert {"ELO_V1", "PI_RATING_V1"} <= set(result.models_executed)
    assert result.core_probability is None
    assert all(abs(sum(values.values()) - 1) < 1e-6
               for values in result.raw_probabilities.values())
