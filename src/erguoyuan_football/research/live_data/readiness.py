"""Evidence-backed, prediction-free live data and model readiness decisions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path

import yaml

from erguoyuan_football.research.live_data.model_requirements import (
    ModelDataRequirement,
)
from erguoyuan_football.research.live_data.schemas import VerifiedFixture
from erguoyuan_football.research.match_package import MatchResearchPackage
from erguoyuan_football.research.samples.sample_hierarchy import (
    SampleHierarchy,
    SampleQuality,
)
from erguoyuan_football.web_research.evidence.evidence_schema import EvidenceRecord
from erguoyuan_football.web_research.policies.freshness_policy import (
    is_valid_for_cutoff,
)
from erguoyuan_football.web_research.time_utils import parse_utc


class DataAvailability(StrEnum):
    """Readiness vocabulary, distinct from provider network status."""

    AVAILABLE = "AVAILABLE"
    PARTIAL = "PARTIAL"
    MISSING = "MISSING"
    STALE = "STALE"
    CONFLICT = "CONFLICT"
    INVALID = "INVALID"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass(frozen=True)
class DataReadinessItem:
    """One data category with full source/evidence ancestry when available."""

    data_type: str
    status: DataAvailability
    source_tier: int | None = None
    provider_id: str | None = None
    freshness: str | None = None
    confidence: str | None = None
    reasons: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    source_urls: tuple[str, ...] = ()
    fetched_at: tuple[str, ...] = ()
    data_origin: str | None = None
    derived_from: tuple[str, ...] = ()
    sample_count: int | None = None
    sample_sufficiency: str | None = None
    lineup_state: str | None = None
    odds_state: str | None = None
    model_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelReadiness:
    """Eligibility only; READY never asserts successful model training."""

    model_name: str
    ready: bool
    degraded: bool
    missing_required: tuple[str, ...]
    missing_optional: tuple[str, ...]
    stale_inputs: tuple[str, ...]
    conflicts: tuple[str, ...]
    reasons: tuple[str, ...]
    status: str = "BLOCKED"
    uncertainty_level: str = "VERY_HIGH"
    stages: dict[str, str] | None = None
    training_sample_count: int = 0
    training_team_coverage: int = 0
    unique_training_team_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class LiveDataReadinessReport:
    """One cutoff-specific verdict; prediction execution is always false."""

    fixture_verified: bool
    overall_status: str
    data_items: tuple[DataReadinessItem, ...]
    model_readiness: tuple[ModelReadiness, ...]
    usable_models: tuple[str, ...]
    blocked_models: tuple[str, ...]
    degraded_models: tuple[str, ...]
    critical_missing: tuple[str, ...]
    generated_at: datetime
    prediction_executed: bool = False
    entity_resolution_status: str = "UNVERIFIED"
    sample_hierarchy: SampleHierarchy | None = None
    sample_quality: SampleQuality | None = None
    uncertainty_level: str = "VERY_HIGH"
    blocked_reasons: dict[str, tuple[str, ...]] | None = None
    model_groups: dict[str, str] | None = None

    def item(self, data_type: str) -> DataReadinessItem:
        """Look up a category by stable name."""
        return next(item for item in self.data_items if item.data_type == data_type)


@dataclass(frozen=True)
class ReadinessConfig:
    """Readiness policy, separate from model training thresholds."""

    min_recent_form_matches: int
    core_strength_models: tuple[str, ...]
    required_for_ready: tuple[str, ...]
    degraded_when_optional_missing: bool
    core_model_groups: dict[str, tuple[str, ...]] | None = None
    required_core_groups: tuple[str, ...] = ()
    min_ready_core_groups: int = 1

    @classmethod
    def from_yaml(cls, path: Path | str) -> ReadinessConfig:
        """Load the explicit phase policy."""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError("INVALID_READINESS_CONFIG")
        recent = int(raw["min_recent_form_matches"])
        if recent < 1:
            raise ValueError("INVALID_READINESS_SAMPLE_THRESHOLD")
        policy_raw = yaml.safe_load((Path(path).parent / "phase13_6_model_readiness.yaml")
                                    .read_text(encoding="utf-8"))
        groups = {str(key): tuple(value) for key, value in
                  policy_raw["core_model_groups"].items()}
        return cls(recent, tuple(raw["core_strength_models"]),
                   tuple(raw["required_for_ready"]),
                   bool(raw["degraded_when_optional_missing"]), groups,
                   tuple(policy_raw["required_core_groups"]),
                   int(policy_raw["min_ready_core_groups"]))


_CATEGORIES = ("FIXTURE", "HISTORICAL_RESULTS", "MATCH_TIMESTAMPS",
               "HOME_AWAY_SPLIT", "THREE_WAY_MAPPER_TRAINING", "RECENT_FORM",
               "TEAM_STATS", "COMPETITION_CONTEXT",
               "ODDS", "XG", "XGA", "NEWS", "INJURY", "LINEUP",
               "ELO_EXTERNAL", "ELO_LOCAL_DERIVED", "ML_FEATURE_VECTOR", "ML_ARTIFACT")
_KEYS = {"FIXTURE": "fixture", "HISTORICAL_RESULTS": "historical_results",
         "RECENT_FORM": "recent_form", "TEAM_STATS": "team_stats",
         "ODDS": "odds", "XG": "xg", "XGA": "xga", "NEWS": "news",
         "INJURY": "injury", "LINEUP": "lineup",
         "ELO_EXTERNAL": "elo_external", "ELO_LOCAL_DERIVED": "elo_local_derived",
         "ML_FEATURE_VECTOR": "ml_feature_vector", "ML_ARTIFACT": "ml_artifact"}


def _item(name: str, evidence: list[EvidenceRecord], cutoff: datetime,
          *, minimum: int = 1) -> DataReadinessItem:
    valid = [record for record in evidence if is_valid_for_cutoff(record, cutoff)]
    if not valid:
        reason = "FUTURE_EVIDENCE_REJECTED" if evidence else f"{name}_MISSING"
        return DataReadinessItem(name, DataAvailability.INVALID if evidence else
                                 DataAvailability.MISSING, reasons=(reason,))
    status = DataAvailability.AVAILABLE if len(valid) >= minimum else DataAvailability.PARTIAL
    reasons = () if status == DataAvailability.AVAILABLE else ("LOW_SAMPLE",)
    if name in {"ODDS", "LINEUP"}:
        ttl = timedelta(seconds=900)
        if all(cutoff - parse_utc(record.as_of_time) > ttl for record in valid):
            status, reasons = DataAvailability.STALE, (f"{name}_STALE",)
    if name == "ODDS" and len({str(record.value) for record in valid}) > 1:
        status, reasons = DataAvailability.CONFLICT, ("ODDS_CONFLICT",)
    if name == "ML_ARTIFACT":
        def valid_artifact(record: EvidenceRecord) -> bool:
            value = record.value
            if not isinstance(value, dict) or not isinstance(value.get("model_id"), str):
                return False
            trained_until = value.get("trained_until")
            if not isinstance(trained_until, str):
                return False
            try:
                training_time = parse_utc(trained_until)
            except ValueError:
                return False
            return (value.get("artifact_exists") is True
                    and value.get("schema_compatible") is True
                    and training_time <= cutoff)
        valid = [record for record in valid if valid_artifact(record)]
        if not valid:
            status, reasons = DataAvailability.INVALID, ("ML_ARTIFACT_INVALID",)
    if name == "ML_FEATURE_VECTOR" and not all(
        isinstance(record.value, dict) and
        record.value.get("coverage_complete") is True and
        record.value.get("schema_compatible") is True for record in valid
    ):
        status, reasons = DataAvailability.INVALID, ("ML_FEATURE_SCHEMA_OR_COVERAGE_INVALID",)
    if not valid:
        return DataReadinessItem(name, status, reasons=reasons)
    first = valid[0]
    origin = ("LOCAL_DERIVED" if all(isinstance(record.value, dict) and
              record.value.get("data_origin") == "LOCAL_DERIVED" for record in valid)
              else "EXTERNAL")
    ancestry = tuple(sorted({parent for record in valid if isinstance(record.value, dict)
                             for parent in record.value.get("derived_from", [])}))
    if origin == "LOCAL_DERIVED" and not ancestry:
        status, reasons = DataAvailability.INVALID, ("DERIVATION_LINEAGE_MISSING",)
    lineup_state = None
    if name == "LINEUP":
        lineup_state = "OFFICIAL_LINEUP" if any(isinstance(record.value, dict)
            and record.value.get("lineup_type") == "OFFICIAL" for record in valid) else "EXPECTED_LINEUP"
    return DataReadinessItem(
        name, status, first.source_tier, first.provider_id or first.source_id,
        "CURRENT" if status == DataAvailability.AVAILABLE else status.value,
        first.confidence, reasons, tuple(record.evidence_id for record in valid),
        tuple(sorted({record.source_url for record in valid})),
        tuple(record.fetched_time for record in valid), origin, ancestry, len(valid),
        "SUFFICIENT" if len(valid) >= minimum else "LOW_SAMPLE", lineup_state,
        status.value if name == "ODDS" else None,
        tuple(sorted({str(record.value["model_id"]) for record in valid
                      if name == "ML_ARTIFACT" and isinstance(record.value, dict)})),
    )


class LiveDataReadinessGate:
    """Evaluate validated evidence and declared requirements without importing models."""

    def __init__(self, config: ReadinessConfig,
                 requirements: tuple[ModelDataRequirement, ...]) -> None:
        self.config = config
        self.requirements = requirements

    def evaluate(self, package: MatchResearchPackage, *,
                 fixture: VerifiedFixture | None, cutoff: datetime,
                 team_provider_ids: tuple[int, int] | None = None,
                 sample_hierarchy: SampleHierarchy | None = None,
                 entity_resolution_status: str = "UNVERIFIED") -> LiveDataReadinessReport:
        """Fail closed on fixture/PIT/sample gaps; never execute predictions."""
        if cutoff.tzinfo is None or cutoff.utcoffset() is None:
            raise ValueError("UTC_TIMESTAMP_REQUIRED")
        verified = (fixture is not None and package.match_id == fixture.provider_match_id
                    and fixture.verified_at <= cutoff < fixture.kickoff_at
                    and fixture.fixture_confidence in {"EXACT", "HIGH"})
        items: dict[str, DataReadinessItem] = {}
        for name in _CATEGORIES:
            records = package.available_data.get(_KEYS.get(name, ""), [])
            minimum = self.config.min_recent_form_matches if name == "RECENT_FORM" else 1
            items[name] = _item(name, records, cutoff, minimum=minimum)
        if verified:
            items["FIXTURE"] = _item("FIXTURE", package.available_data.get("fixture", []), cutoff)
        else:
            items["FIXTURE"] = DataReadinessItem("FIXTURE", DataAvailability.INVALID,
                                                  reasons=("FIXTURE_NOT_VERIFIED",))
        history = [record for record in package.available_data.get("historical_results", [])
                   if is_valid_for_cutoff(record, cutoff)]
        if history:
            items["MATCH_TIMESTAMPS"] = _item("MATCH_TIMESTAMPS", history, cutoff)
            items["HOME_AWAY_SPLIT"] = _item("HOME_AWAY_SPLIT", history, cutoff)
        outcomes = {0 if record.value["home_goals"] > record.value["away_goals"] else
                    1 if record.value["home_goals"] == record.value["away_goals"] else 2
                    for record in history if isinstance(record.value, dict)
                    and isinstance(record.value.get("home_goals"), int)
                    and isinstance(record.value.get("away_goals"), int)}
        if len(history) >= 12 and len(outcomes) == 3:
            items["THREE_WAY_MAPPER_TRAINING"] = _item(
                "THREE_WAY_MAPPER_TRAINING", history, cutoff)
        elif history:
            items["THREE_WAY_MAPPER_TRAINING"] = DataReadinessItem(
                "THREE_WAY_MAPPER_TRAINING", DataAvailability.PARTIAL,
                reasons=("RATING_MAPPER_REQUIRES_12_ROWS_AND_3_OUTCOMES",),
                sample_count=len(history), sample_sufficiency="LOW_SAMPLE")
        if package.competition is not None and verified:
            fixture_item = items["FIXTURE"]
            items["COMPETITION_CONTEXT"] = DataReadinessItem(
                **{**fixture_item.__dict__, "data_type": "COMPETITION_CONTEXT"})
        if package.conflicts:
            for name in ("ODDS", "FIXTURE"):
                if items[name].status == DataAvailability.AVAILABLE:
                    old = items[name]
                    items[name] = DataReadinessItem(
                        **{**old.__dict__, "status": DataAvailability.CONFLICT,
                           "reasons": ("SOURCE_CONFLICT",)})
        from erguoyuan_football.research.live_data.model_readiness import (
            ModelReadinessEvaluator,
        )
        evaluator = ModelReadinessEvaluator()
        model_rows = [evaluator.evaluate(requirement, items, sample_hierarchy,
                                         fixture_verified=verified)
                      for requirement in self.requirements]
        critical = tuple(name for name in self.config.required_for_ready
                         if items[name].status != DataAvailability.AVAILABLE)
        usable = tuple(row.model_name for row in model_rows if row.ready)
        groups = self.config.core_model_groups or {
            "BASE_STRENGTH": self.config.core_strength_models}
        group_states = {name: "READY" if any(model_id in usable for model_id in model_ids)
                        else "BLOCKED" for name, model_ids in groups.items()}
        core_names = self.config.required_core_groups or tuple(groups)
        ready_core = sum(group_states.get(name) == "READY" for name in core_names)
        if not verified or items["FIXTURE"].status == DataAvailability.CONFLICT or (
            entity_resolution_status in {"TEAM_IDENTITY_CONFLICT", "TEAM_DISCOVERY_AMBIGUOUS"}):
            overall = "FAILED"
        elif critical or ready_core < self.config.min_ready_core_groups:
            overall = "NOT_READY"
        elif self.config.degraded_when_optional_missing and (
            any(row.degraded or not row.ready for row in model_rows)
            or any(item.status != DataAvailability.AVAILABLE for item in items.values())):
            overall = "DEGRADED"
        else:
            overall = "READY"
        uncertainty = ("VERY_HIGH" if not usable else "HIGH" if overall == "DEGRADED"
                       else "LOW")
        return LiveDataReadinessReport(
            verified, overall, tuple(items.values()), tuple(model_rows), usable,
            tuple(row.model_name for row in model_rows if not row.ready),
            tuple(row.model_name for row in model_rows if row.degraded), critical,
            cutoff, False, entity_resolution_status, sample_hierarchy,
            sample_hierarchy.quality if sample_hierarchy else None, uncertainty,
            {row.model_name: row.reasons for row in model_rows if not row.ready},
            group_states,
        )
