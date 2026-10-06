"""Phase 14.1 data-only pipeline contracts and fail-closed behavior."""

from __future__ import annotations

import pytest

from erguoyuan_football.prediction.model_input_adapter import DixonColesInputAdapter
from erguoyuan_football.research.canonical_match import CanonicalMatchIdentity
from erguoyuan_football.research.model_input_validator import ModelInputValidator
from erguoyuan_football.research.production_match_data import (
    ProductionMatchDataBundle,
)
from erguoyuan_football.research.production_match_pipeline import (
    ProductionMatchPipeline,
    ProductionMatchRequest,
    ProductionPipelineError,
)
from tests.phase14.conftest import START


def test_production_input_rejects_synthetic_test_samples(sample_context) -> None:
    repository, hierarchy, cutoff = sample_context
    bundle = DixonColesInputAdapter().build_input(repository, hierarchy, cutoff)
    result = ModelInputValidator().validate(bundle, production=True)
    assert result.status == "INVALID"
    assert result.reason == "PRODUCTION_INPUT_REJECTED"


def test_canonical_match_requires_real_provider_evidence() -> None:
    identity = CanonicalMatchIdentity("canonical1", "HOME", "AWAY", "COMP",
        START, (("PROVIDER", "42"),), "HOME_AWAY", True, ("E_FIXTURE",))
    assert identity.match_id == "canonical1"
    with pytest.raises(ValueError, match="VERIFIED_MATCH_REQUIRES_PROVIDER_EVIDENCE"):
        CanonicalMatchIdentity("canonical2", "HOME", "AWAY", "COMP", START,
                               (), "UNKNOWN", True, ())


def test_replay_requires_persisted_as_of_evidence_before_network_access() -> None:
    pipeline = ProductionMatchPipeline.__new__(ProductionMatchPipeline)
    with pytest.raises(ProductionPipelineError,
                       match="REPLAY_EVIDENCE_UNAVAILABLE"):
        pipeline.build(ProductionMatchRequest("Arsenal vs Liverpool"),
            prediction_cutoff=START, mode="REPLAY")


def test_bundle_refuses_prediction_execution() -> None:
    from datetime import timedelta

    from erguoyuan_football.prediction.model_execution_planner import (
        ExecutionPlan,
    )
    from erguoyuan_football.research.match_package import MatchResearchPackage
    from erguoyuan_football.research.production_match_data import (
        PipelineStageResult,
        ProductionDataReadiness,
        SampleQualityReport,
    )
    from erguoyuan_football.research.research_status import ResearchStatus
    from erguoyuan_football.research.samples.sample_hierarchy import (
        SampleHierarchy,
        SampleQuality,
    )

    identity = CanonicalMatchIdentity("canonical1", "HOME", "AWAY", "COMP",
        START + timedelta(days=2), (("PROVIDER", "42"),), "HOME_AWAY", True,
        ("E_FIXTURE",))
    stage = PipelineStageResult("FIXTURE_VERIFICATION", "PASS", START, START, 1, 1)
    quality = SampleQuality(None, None, 0, None, 0.0, 0, 0, 0, None, 0, 0,
                            False, 0)
    hierarchy = SampleHierarchy("COMP", (), (), (), (), (), (), (), (), START, quality)
    sample_quality = SampleQualityReport(0, 0, 0, 0, 0, 0.0, 0.0, 0.0,
        0, 0, 0, "VERY_HIGH", None, None, None, 0, "HIGH")
    plan = ExecutionPlan("p", START, START, START, "NOT_READY", ())
    package = MatchResearchPackage("42", None, None, None, status=ResearchStatus.PARTIAL)
    with pytest.raises(ValueError, match="PREDICTION_EXECUTION_FORBIDDEN"):
        ProductionMatchDataBundle(
            research_session_id="session", canonical_match=identity,
            prediction_cutoff=START, mode="LIVE", historical_matches=(),
            team_direct_samples=(), competition_samples=(), comparable_samples=(),
            prior_samples=(), research_package=package, sample_hierarchy=hierarchy,
            model_input_bundles={}, execution_plan=plan, model_input_readiness=(),
            data_quality_report=sample_quality, readiness_report=None,
            production_readiness=ProductionDataReadiness("NOT_READY", ("NO_DATA",)),
            evidence_ids=(), provider_ids=(), acquisition_metrics={}, stages=(stage,),
            generated_at=START, prediction_executed=True)
