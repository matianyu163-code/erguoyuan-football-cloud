"""Opt-in real-data integration for the configured major-league match path."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from erguoyuan_football.prediction.model_execution_planner import ModelExecutionPlanner
from erguoyuan_football.research.live_data.model_requirements import (
    load_model_requirements,
)
from erguoyuan_football.research.live_data.openligadb import (
    OpenLigaDBConfig,
    OpenLigaDBProvider,
)
from erguoyuan_football.research.live_data.readiness import (
    LiveDataReadinessGate,
    ReadinessConfig,
)
from erguoyuan_football.research.pipeline_execution_record import PipelineExecutionStore
from erguoyuan_football.research.production_match_pipeline import (
    ProductionMatchPipeline,
    ProductionMatchRequest,
    ProductionPipelineError,
)
from erguoyuan_football.web_research.evidence.evidence_store import EvidenceStore

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.live_network
def test_live_arsenal_liverpool_bundle_stops_before_prediction(
        request: pytest.FixtureRequest) -> None:
    """Verify fixture/history/hierarchy from the real registered source."""
    if "live_network" not in request.config.option.markexpr:
        pytest.skip("explicit -m live_network required")
    provider = OpenLigaDBProvider(OpenLigaDBConfig.from_yaml(
        ROOT / "config/phase13_5_provider.yaml"))
    evidence = EvidenceStore(":memory:", provider.sources)
    audit = PipelineExecutionStore(":memory:")
    try:
        gate = LiveDataReadinessGate(
            ReadinessConfig.from_yaml(ROOT / "config/phase13_5_readiness.yaml"),
            load_model_requirements(ROOT / "config/model_registry.yaml"))
        pipeline = ProductionMatchPipeline(provider, evidence, gate,
            planner=ModelExecutionPlanner(ROOT / "config/phase14_model_requirements.yaml",
                                          ROOT / "config/model_registry.yaml"),
            audit_store=audit)
        cutoff = datetime.now(UTC) + timedelta(minutes=10)
        try:
            bundle = pipeline.build(ProductionMatchRequest("Arsenal vs Liverpool",
                "Premier League"), prediction_cutoff=cutoff)
        except ProductionPipelineError as error:
            assert "HISTORY_PROVIDER_UNAVAILABLE" in str(error)
            record = audit._db.execute(
                "SELECT record_json FROM pipeline_stage_records "
                "WHERE stage='HISTORY_FETCH' ORDER BY rowid DESC LIMIT 1").fetchone()
            assert record is not None
            stage = json.loads(record[0])["stage"]
            assert stage["status"] == "FAILED"
            assert stage["error_code"] == "HISTORY_PROVIDER_UNAVAILABLE"
            assert audit.stage_count() == 5
            return
        assert bundle.canonical_match.verified
        assert bundle.historical_matches
        assert bundle.sample_hierarchy.quality.source_conflict_count == 0
        assert bundle.production_readiness.status == "NOT_READY"
        assert bundle.prediction_executed is False
        assert not bundle.model_input_bundles
        assert any(stage.stage == "PIT_FILTER" and stage.status == "PASS"
                   for stage in bundle.stages)
        assert len(audit_records := audit._db.execute(
            "SELECT stage FROM pipeline_stage_records").fetchall()) >= 10
        assert audit_records
    finally:
        audit.close()
        evidence.close()
        provider.close()

