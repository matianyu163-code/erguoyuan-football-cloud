"""Auditable data-only bundle produced before any model is invoked."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from erguoyuan_football.prediction.model_execution_planner import ExecutionPlan
from erguoyuan_football.prediction.model_input_adapter import ModelInputBundle
from erguoyuan_football.research.canonical_match import CanonicalMatchIdentity
from erguoyuan_football.research.historical_match_repository import HistoricalMatch
from erguoyuan_football.research.match_package import MatchResearchPackage
from erguoyuan_football.research.samples.sample_hierarchy import SampleHierarchy
from erguoyuan_football.research.samples.sample_windows import MatchWindowSet


@dataclass(frozen=True)
class PipelineStageResult:
    """One auditable pipeline stage and its input/output cardinality."""

    stage: str
    status: str
    started_at: datetime
    finished_at: datetime
    input_count: int
    output_count: int
    error_code: str | None = None
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SampleQualityReport:
    """Separate sample counts and descriptive quality values."""

    direct_home_count: int
    direct_away_count: int
    competition_count: int
    comparable_count: int
    prior_count: int
    opponent_diversity: float
    recency_score: float
    source_quality_score: float
    conflict_count: int
    deduplicated_count: int
    stale_count: int
    uncertainty_level: str
    earliest_match: datetime | None
    latest_match: datetime | None
    median_age_days: float | None
    total_span_days: int
    roster_continuity_risk: str


@dataclass(frozen=True)
class ModelInputReadiness:
    """Input validity and execution permission are different decisions."""

    model_id: str
    input_status: str
    execution_status: str
    reason: str | None = None


@dataclass(frozen=True)
class ProductionDataReadiness:
    """Data readiness only; this is not final probability readiness."""

    status: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ProductionMatchDataBundle:
    """Full evidence lineage for a frozen, pre-model production input."""

    research_session_id: str
    canonical_match: CanonicalMatchIdentity
    prediction_cutoff: datetime
    mode: str
    historical_matches: tuple[HistoricalMatch, ...]
    team_direct_samples: tuple[HistoricalMatch, ...]
    competition_samples: tuple[HistoricalMatch, ...]
    comparable_samples: tuple[HistoricalMatch, ...]
    prior_samples: tuple[HistoricalMatch, ...]
    research_package: MatchResearchPackage
    sample_hierarchy: SampleHierarchy
    model_input_bundles: dict[str, ModelInputBundle]
    execution_plan: ExecutionPlan
    model_input_readiness: tuple[ModelInputReadiness, ...]
    data_quality_report: SampleQualityReport
    readiness_report: Any
    production_readiness: ProductionDataReadiness
    evidence_ids: tuple[str, ...]
    provider_ids: tuple[str, ...]
    acquisition_metrics: dict[str, int | str]
    stages: tuple[PipelineStageResult, ...]
    generated_at: datetime
    prediction_executed: bool = False
    sample_windows: MatchWindowSet | None = None
    match_universe: str = "GLOBAL_RESEARCH"
    jc_verification: Any | None = None
    resource_plan: Any | None = None
    match_source_audit: Any | None = None

    def __post_init__(self) -> None:
        if self.prediction_executed:
            raise ValueError("PHASE14_1_PREDICTION_EXECUTION_FORBIDDEN")
        if self.mode not in {"LIVE", "REPLAY"}:
            raise ValueError("INVALID_PIPELINE_MODE")
        if self.match_universe not in {"JC_PRODUCTION", "GLOBAL_RESEARCH"}:
            raise ValueError("INVALID_MATCH_UNIVERSE")
        if self.mode == "LIVE" and self.prediction_cutoff >= self.canonical_match.kickoff_at:
            raise ValueError("LIVE_CUTOFF_MUST_PRECEDE_KICKOFF")
