"""Build a pre-model production data bundle from verified live match research."""

from __future__ import annotations

import hashlib
import statistics
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime, time
from uuid import uuid4

from erguoyuan_football.app.input.match_input import (
    MatchInputParserV2,
    MatchRequest,
)
from erguoyuan_football.jc_verification.jc_match_verifier import JCMatchVerifier
from erguoyuan_football.jc_verification.jc_resource_scheduler import (
    CompetitionProfile,
    JCResourceScheduler,
)
from erguoyuan_football.jc_verification.jc_schema import (
    JCMatchQuery,
    JCStatus,
    JCVerificationResult,
)
from erguoyuan_football.knowledge.competitions.competition_schema import (
    CompetitionIdentity,
)
from erguoyuan_football.knowledge.teams.team_resolver import TeamResolver
from erguoyuan_football.match_source.match_source import MatchSourceType
from erguoyuan_football.match_source.match_source_audit import (
    MatchSourceAudit,
    MatchSourceAuditStore,
)
from erguoyuan_football.match_source.match_source_classifier import (
    MatchSourceClassifier,
)
from erguoyuan_football.prediction.model_execution_planner import (
    ModelExecutionPlanner,
)
from erguoyuan_football.prediction.model_input_adapter import (
    ModelInputAdapter,
    ModelInputBundle,
    default_model_input_adapters,
)
from erguoyuan_football.research.canonical_match import CanonicalMatchIdentity
from erguoyuan_football.research.historical_data_acquirer import (
    HistoricalDataAcquirer,
    HistoricalProviderUnavailable,
)
from erguoyuan_football.research.historical_match_repository import (
    HistoricalMatch,
)
from erguoyuan_football.research.historical_provider_router import (
    HistoricalProviderOption,
)
from erguoyuan_football.research.live_data.openligadb import (
    OpenLigaDBProvider,
)
from erguoyuan_football.research.live_data.readiness import (
    LiveDataReadinessGate,
)
from erguoyuan_football.research.live_data.service import (
    LiveResearchService,
)
from erguoyuan_football.research.match_universe import MatchUniverse
from erguoyuan_football.research.model_input_validator import ModelInputValidator
from erguoyuan_football.research.pipeline_execution_record import (
    PipelineExecutionRecord,
    PipelineExecutionStore,
)
from erguoyuan_football.research.production_match_data import (
    ModelInputReadiness,
    PipelineStageResult,
    ProductionDataReadiness,
    ProductionMatchDataBundle,
    SampleQualityReport,
)
from erguoyuan_football.research.samples.match_deduplicator import (
    HistoricalMatchSample,
    deduplicate_matches,
)
from erguoyuan_football.research.samples.sample_builder import SampleBuilder
from erguoyuan_football.research.samples.sample_repository import SampleRepository
from erguoyuan_football.research.samples.sample_windows import build_match_windows
from erguoyuan_football.research.venue_context import resolve_venue_context
from erguoyuan_football.web_research.evidence.evidence_store import EvidenceStore


class ProductionPipelineError(ValueError):
    """Stage-coded fail-closed input pipeline error."""


@dataclass(frozen=True)
class ProductionMatchRequest:
    """Raw match request kept separate from resolved canonical identities."""

    raw_text: str
    competition: str | None = None
    date_hint: datetime | None = None
    match_universe: MatchUniverse = MatchUniverse.GLOBAL_RESEARCH


class ProductionMatchPipeline:
    """Resolve, verify, collect and package data; never invoke prediction models."""

    def __init__(self, provider: OpenLigaDBProvider, evidence_store: EvidenceStore,
                 readiness_gate: LiveDataReadinessGate,
                 *, planner: ModelExecutionPlanner,
                 adapters: dict[str, ModelInputAdapter] | None = None,
                 audit_store: PipelineExecutionStore | None = None,
                 team_resolver: TeamResolver | None = None,
                 historical_fallbacks: tuple[HistoricalProviderOption, ...] = (),
                 jc_verifier: JCMatchVerifier | None = None,
                 jc_scheduler: JCResourceScheduler | None = None,
                 competition_profiles: Mapping[str, CompetitionProfile] | None = None,
                 source_audit_store: MatchSourceAuditStore | None = None) -> None:
        self.provider = provider
        self.evidence_store = evidence_store
        self.readiness_gate = readiness_gate
        self.planner = planner
        self.adapters = adapters or default_model_input_adapters()
        self.audit_store = audit_store
        self.team_resolver = team_resolver or TeamResolver()
        self.historical_fallbacks = historical_fallbacks
        self.jc_verifier = jc_verifier or JCMatchVerifier()
        self.jc_scheduler = jc_scheduler or JCResourceScheduler()
        self.competition_profiles = dict(competition_profiles or {})
        self.source_classifier = MatchSourceClassifier()
        self.source_audit_store = (source_audit_store or
            (MatchSourceAuditStore(audit_store.connection) if audit_store else None))
        self.parser = MatchInputParserV2()

    def build(self, request: ProductionMatchRequest | MatchRequest | str, *,
              prediction_cutoff: datetime, mode: str = "LIVE",
              fetch_live: bool = True) -> ProductionMatchDataBundle:
        """Return verified data and model inputs without calling ``predict``."""
        if mode not in {"LIVE", "REPLAY"}:
            raise ProductionPipelineError("INVALID_PIPELINE_MODE")
        self._validate_cutoff(prediction_cutoff)
        if mode == "REPLAY":
            raise ProductionPipelineError("REPLAY_EVIDENCE_UNAVAILABLE")
        if not fetch_live:
            raise ProductionPipelineError("LIVE_FETCH_DISABLED_NO_CACHED_FIXTURE_PIPELINE")
        pipeline_id = str(uuid4())
        stages: list[PipelineStageResult] = []

        started = datetime.now(UTC)
        raw_text = _request_raw_text(request)
        source_audit = self.source_classifier.classify(
            raw_text,
            jc_confirmed=bool(getattr(request, "jc_confirmed", False)),
            declared_type=getattr(request, "match_source_type", None),
        )
        if self.source_audit_store:
            self.source_audit_store.append(source_audit)
        parsed = self._parse_request(
            request, cleaned_text=self.source_classifier.clean_match_text(raw_text))
        parsed = replace(parsed, match_source_type=source_audit.source_type,
                         jc_confirmed=source_audit.source_type
                         == MatchSourceType.USER_JC_CONFIRMED)
        self._stage(pipeline_id, None, stages, "SOURCE_CLASSIFICATION",
                    source_audit.source_type.value, started, 1, 1,
                    (source_audit.audit_id,))
        self._stage(pipeline_id, None, stages, "INPUT_PARSE", "PASS", started,
                    1, 1, ())
        if parsed.validation_status != "VALID" or not parsed.home_team or not parsed.away_team:
            raise ProductionPipelineError(parsed.error_code or "INPUT_INVALID")

        started = datetime.now(UTC)
        home = self.team_resolver.resolve(parsed.home_team)
        away = self.team_resolver.resolve(parsed.away_team)
        if home is None or away is None or home.team_id == away.team_id:
            self._stage(pipeline_id, None, stages, "ENTITY_RESOLUTION", "BLOCKED",
                        started, 2, int(home is not None) + int(away is not None), ())
            raise ProductionPipelineError("TEAM_NOT_FOUND_OR_AMBIGUOUS")
        self._stage(pipeline_id, None, stages, "ENTITY_RESOLUTION", "PASS",
                    started, 2, 2, ())

        config = self.provider.config
        requested_competition = (request.competition if isinstance(request, ProductionMatchRequest)
                                 else parsed.competition)
        allowed_competitions = {config.competition_name.casefold(),
                                *(value.casefold() for value in config.competition_aliases)}
        if requested_competition and requested_competition.casefold() not in allowed_competitions:
            raise ProductionPipelineError("PROVIDER_COVERAGE_MISSING")
        if home.team_id not in config.team_bindings or away.team_id not in config.team_bindings:
            raise ProductionPipelineError("PROVIDER_TEAM_BINDING_MISSING")
        started = datetime.now(UTC)
        self._stage(pipeline_id, None, stages, "COMPETITION_RESOLUTION", "PASS",
                    started, 1, 1, ())

        try:
            acquirer = HistoricalDataAcquirer(
                fallback_providers=self.historical_fallbacks)
            try:
                acquisition = acquirer.acquire(self.provider, cutoff=prediction_cutoff)
            finally:
                acquirer.close()
            batch = acquisition.batch
        except Exception as error:
            error_code = ("HISTORY_PROVIDER_UNAVAILABLE" if
                          isinstance(error, HistoricalProviderUnavailable) else
                          "HISTORY_FETCH_FAILED")
            self._stage(pipeline_id, None, stages, "HISTORY_FETCH", "FAILED",
                        started, 0, 0, (), error_code)
            raise ProductionPipelineError(
                f"{error_code}:{type(error).__name__}:{error}") from error
        self._stage(pipeline_id, None, stages, "HISTORY_FETCH", "PASS", started,
                    0, len(batch.matches), ())
        if batch.retrieved_at > prediction_cutoff:
            self._stage(pipeline_id, None, stages, "FIXTURE_VERIFICATION", "BLOCKED",
                        started, 1, 0, (), "EVIDENCE_AFTER_PREDICTION_CUTOFF")
            raise ProductionPipelineError("EVIDENCE_AFTER_PREDICTION_CUTOFF")
        research = LiveResearchService(self.provider, self.evidence_store, self.readiness_gate,
                                       team_database=self.team_resolver.database)
        result = research.build(batch, home.team_id, away.team_id,
            cutoff=prediction_cutoff,
            date_hint=(request.date_hint if isinstance(request, ProductionMatchRequest)
                       else datetime.combine(request.date, time.min, UTC)
                       if isinstance(request, MatchRequest) and request.date is not None
                       else None))
        self._stage(pipeline_id, result.fixture_check.fixture.provider_match_id
                    if result.fixture_check.fixture else None, stages,
                    "FIXTURE_VERIFICATION", "PASS" if result.fixture_check.fixture
                    else "BLOCKED", started, len(batch.matches),
                    int(result.fixture_check.fixture is not None),
                    tuple(record.evidence_id for record in (
                        result.package.available_data.get("fixture", [])
                        if result.package else [])),
                    None if result.fixture_check.fixture else result.fixture_check.status)
        if result.package is None or result.readiness is None or result.fixture_check.fixture is None:
            raise ProductionPipelineError("FIXTURE_NOT_VERIFIED")
        package = result.package
        model_readiness = result.readiness
        fixture = result.fixture_check.fixture
        if prediction_cutoff >= fixture.kickoff_at:
            raise ProductionPipelineError("LIVE_CUTOFF_MUST_PRECEDE_KICKOFF")
        jc_query = JCMatchQuery(
            fixture.home_team_id, home.official_name,
            fixture.away_team_id, away.official_name,
            fixture.competition_id, config.competition_name, fixture.kickoff_at)
        jc_verification = _jc_result_for_source(source_audit, jc_query, self.jc_verifier)
        profile = self.competition_profiles.get(fixture.competition_id)
        resource_plan = self.jc_scheduler.plan(jc_verification.status, profile)
        self._stage(pipeline_id, fixture.provider_match_id, stages, "JC_VERIFICATION",
                    jc_verification.status.value, datetime.now(UTC), 1,
                    int(jc_verification.status.value != "UNKNOWN"),
                    (jc_verification.evidence_id,) if jc_verification.evidence_id else (),
                    jc_verification.reason if jc_verification.status.value == "UNKNOWN" else None)
        self._stage(pipeline_id, fixture.provider_match_id, stages, "RESOURCE_PLAN",
                    resource_plan.mode, datetime.now(UTC), 1, 1, ())
        venue = resolve_venue_context(neutral_venue=fixture.neutral_venue,
            fixture_verified=True, competition_type=config.competition_type,
            domestic_competition=config.team_country not in {"UNKNOWN", "INT"})

        # Prior-season results retain the exact retrieval time and endpoint that
        # supplied them. Duplicate evidence already persisted is left untouched.
        provider_options = {option.provider_id: option
                            for option in self.historical_fallbacks}
        for historic_batch, provider_id in zip(acquisition.batches,
                                               acquisition.batch_provider_ids,
                                               strict=True):
            if historic_batch.season == batch.season:
                continue
            if provider_id == self.provider.config.provider_id:
                records = self.provider.historical_results(
                    historic_batch, fixture, prediction_cutoff)
            else:
                option = provider_options.get(provider_id)
                if option is None:
                    raise ProductionPipelineError("HISTORY_PROVIDER_PROVENANCE_MISSING")
                records = option.historical_results(historic_batch, fixture,
                                                    prediction_cutoff)
            for record in records:
                if self.evidence_store.get(record.evidence_id) is None:
                    self.evidence_store.save(record)
                package.available_data.setdefault("historical_results", []).append(record)
        sample_batch = replace(batch, matches=tuple(
            row for source_batch in acquisition.batches for row in source_batch.matches))
        stage_started = datetime.now(UTC)
        samples = SampleRepository.from_openligadb(sample_batch, package,
            self.provider.config, prediction_cutoff,
            provider_team_ids_by_source={option.provider_id: option.canonical_team_ids
                for option in self.historical_fallbacks})
        received_count = len(samples.samples)
        dedup = deduplicate_matches(samples.samples)
        samples = SampleRepository(dedup.samples)
        self._stage(pipeline_id, fixture.provider_match_id, stages, "DEDUP", "PASS",
                    stage_started, received_count, len(dedup.samples),
                    tuple(evidence for row in dedup.samples for evidence in row.evidence_ids))
        self._stage(pipeline_id, fixture.provider_match_id, stages, "CONFLICT_FILTER",
                    "PARTIAL" if dedup.conflicts else "PASS", stage_started,
                    received_count, len(dedup.samples),
                    tuple(evidence for group in dedup.conflicts for evidence in group),
                    "RESULT_CONFLICT" if dedup.conflicts else None)
        self._stage(pipeline_id, fixture.provider_match_id, stages, "PIT_FILTER", "PASS",
                    stage_started, result.historical_count, len(samples.samples),
                    tuple(evidence for row in samples.samples for evidence in row.evidence_ids))
        self._stage(pipeline_id, fixture.provider_match_id, stages, "HISTORY_NORMALIZE",
                    "PASS" if samples.samples else "PARTIAL", stage_started,
                    result.historical_count, len(samples.samples),
                    tuple(evidence for row in samples.samples for evidence in row.evidence_ids),
                    None if samples.samples else "NO_NORMALIZED_FINAL_RESULTS")
        if package.home_team is None or package.away_team is None:
            raise ProductionPipelineError("TEAM_IDENTITY_NOT_IN_CATALOG")
        home_identity = replace(package.home_team,
            entity_type=config.competition_entity_type, gender=config.competition_gender,
            age_group=config.competition_age_group, federation=config.team_federation)
        away_identity = replace(package.away_team,
            entity_type=config.competition_entity_type, gender=config.competition_gender,
            age_group=config.competition_age_group, federation=config.team_federation)
        competition_identity = CompetitionIdentity(
            fixture.competition_id, config.competition_name, config.team_federation,
            config.team_country, "CLUB")
        hierarchy = SampleBuilder(samples).build(home_identity, away_identity,
            competition_identity, cutoff=prediction_cutoff, competition_type="LEAGUE")
        provider_ids = (config.team_bindings[home.team_id].provider_team_id,
                        config.team_bindings[away.team_id].provider_team_id)
        readiness_report = self.readiness_gate.evaluate(
            package, fixture=fixture, cutoff=prediction_cutoff,
            team_provider_ids=provider_ids, sample_hierarchy=hierarchy,
            entity_resolution_status="VERIFIED")
        package.readiness_report = readiness_report
        model_readiness = readiness_report
        all_samples = _to_historical_matches(samples.samples)
        sample_map = {source.match_id: normalized for source, normalized
                      in zip(samples.samples, all_samples, strict=True)}
        direct_ids = {row.match_id for row in (*hierarchy.home_direct, *hierarchy.away_direct)}
        competition_ids = {row.match_id for row in hierarchy.competition}
        comparable_ids = {row.match_id for row in hierarchy.comparable}
        prior_ids = {row.match_id for row in (
            *hierarchy.federation_prior, *hierarchy.age_group_prior,
            *hierarchy.gender_prior)}
        if direct_ids & prior_ids:
            raise ProductionPipelineError("DIRECT_PRIOR_OVERLAP")
        direct_samples = tuple(sample_map[item] for item in sorted(direct_ids) if item in sample_map)
        competition_samples = tuple(sample_map[item] for item in sorted(competition_ids)
                                    if item in sample_map)
        comparable_samples = tuple(sample_map[item] for item in sorted(comparable_ids)
                                   if item in sample_map)
        prior_samples = tuple(sample_map[item] for item in sorted(prior_ids) if item in sample_map)
        for stage_name, count in (("DIRECT_SAMPLE", len(direct_samples)),
                                  ("COMPETITION_SAMPLE", len(competition_samples)),
                                  ("COMPARABLE_SAMPLE", len(comparable_samples)),
                                  ("PRIOR_SAMPLE", len(prior_samples))):
            self._stage(pipeline_id, fixture.provider_match_id, stages, stage_name,
                        "PASS" if count else "PARTIAL", stage_started,
                        len(samples.samples), count,
                        tuple(evidence for row in samples.samples
                              if row.match_id in direct_ids | competition_ids |
                              comparable_ids | prior_ids for evidence in row.evidence_ids))

        adapter_bundles: dict[str, ModelInputBundle] = {}
        input_readiness: list[ModelInputReadiness] = []
        for model in model_readiness.model_readiness:
            if not model.ready:
                input_readiness.append(ModelInputReadiness(model.model_name,
                    "BLOCKED", "BLOCKED", ";".join(model.reasons)))
                continue
            adapter = self.adapters.get(model.model_name)
            if adapter is None:
                input_readiness.append(ModelInputReadiness(model.model_name,
                    "BLOCKED", "BLOCKED", "MODEL_INPUT_ADAPTER_MISSING"))
                continue
            bundle = adapter.build_input(samples, hierarchy, prediction_cutoff,
                                         degraded=model.degraded)
            validation = ModelInputValidator().validate(bundle, production=True)
            if validation.status not in {"VALID", "DEGRADED_VALID"}:
                input_readiness.append(ModelInputReadiness(model.model_name,
                    "INVALID", "BLOCKED", validation.reason))
                continue
            adapter_bundles[model.model_name] = bundle
            execution = ("READY" if model.model_name != "BAYESIAN_HIERARCHICAL_V1"
                         or bundle.bayesian_prior is not None else "BLOCKED")
            reason = ("BAYESIAN_PRIOR_UNAVAILABLE" if execution == "BLOCKED" else None)
            input_readiness.append(ModelInputReadiness(model.model_name,
                "DEGRADED_VALID" if model.degraded else "VALID", execution, reason))
        evidence_ids = tuple(sorted({record.evidence_id for records in
            package.available_data.values() for record in records} | {
            evidence for row in samples.samples for evidence in row.evidence_ids}))
        self._stage(pipeline_id, fixture.provider_match_id, stages, "MODEL_INPUT_BUILD",
                    "PASS" if adapter_bundles else "PARTIAL", stage_started,
                    len(model_readiness.usable_models), len(adapter_bundles),
                    tuple(evidence for bundle in adapter_bundles.values()
                          for evidence in bundle.evidence_ids),
                    None if adapter_bundles else "NO_VALID_MODEL_INPUT_BUNDLE")
        plan = self.planner.plan(model_readiness,
            prediction_time=prediction_cutoff, training_cutoff=prediction_cutoff,
            neutral_venue_known=venue.context.value != "UNKNOWN")
        quality = _quality(hierarchy, direct_samples, all_samples, prediction_cutoff,
                           deduplicated_count=max(0, received_count - len(dedup.samples)
                               - sum(len(group) for group in dedup.conflicts)))
        ready_inputs = [row for row in input_readiness if row.input_status in
                        {"VALID", "DEGRADED_VALID"}]
        status = "NOT_READY" if not ready_inputs else (
            "DEGRADED_READY" if any(row.input_status == "DEGRADED_VALID"
                                    for row in ready_inputs) else "PRODUCTION_READY")
        production_readiness = ProductionDataReadiness(status, () if ready_inputs else
            ("NO_VALID_MODEL_INPUT_BUNDLE",))
        self._stage(pipeline_id, fixture.provider_match_id, stages, "SAMPLE_QUALITY",
                    "PASS", stage_started, len(all_samples), len(all_samples),
                    tuple(evidence for row in all_samples for evidence in row.source_evidence_ids))
        self._stage(pipeline_id, fixture.provider_match_id, stages, "READINESS",
                    "PASS" if ready_inputs else "BLOCKED", stage_started,
                    len(input_readiness), len(ready_inputs), evidence_ids,
                    None if ready_inputs else "NO_VALID_MODEL_INPUT_BUNDLE")
        canonical_id = _canonical_match_id(fixture.provider_id,
            fixture.provider_match_id, fixture.home_team_id, fixture.away_team_id,
            fixture.kickoff_at, fixture.competition_id)
        canonical = CanonicalMatchIdentity(canonical_id, fixture.home_team_id,
            fixture.away_team_id, fixture.competition_id, fixture.kickoff_at,
            ((fixture.provider_id, fixture.provider_match_id),),
            venue.context.value,
            True, tuple(record.evidence_id for record in
                        package.available_data.get("fixture", [])))
        package.verified_fixture = fixture
        return ProductionMatchDataBundle(
            pipeline_id, canonical, prediction_cutoff,
            mode, all_samples, direct_samples, competition_samples, comparable_samples,
            prior_samples, package, hierarchy, adapter_bundles, plan,
            tuple(input_readiness), quality, model_readiness, production_readiness,
            evidence_ids,
            (self.provider.config.provider_id,), {
                "pages_fetched": acquisition.pages_fetched,
                "records_received": acquisition.records_received,
                "pagination_supported": str(acquisition.pagination_supported),
                "cache_status": acquisition.cache_status,
                "seasons_loaded": ",".join(str(item.season) for item in acquisition.batches),
            }, tuple(stages), datetime.now(UTC), False,
            build_match_windows(samples.samples, home.team_id, away.team_id,
                                fixture.competition_id, prediction_cutoff),
            resource_plan.universe.value,
            jc_verification, resource_plan, source_audit)

    def _parse_request(self, request: ProductionMatchRequest | MatchRequest | str,
                       *, cleaned_text: str | None = None
                       ) -> MatchRequest:
        if isinstance(request, MatchRequest):
            return request
        if isinstance(request, ProductionMatchRequest):
            parsed = self.parser.parse(cleaned_text if cleaned_text is not None
                                       else request.raw_text)
            return MatchRequest(parsed.match_id, parsed.home_team, parsed.away_team,
                request.competition, parsed.date, parsed.input_source,
                parsed.validation_status, parsed.reason, parsed.raw_text,
                parsed.match_source_type, parsed.jc_confirmed)
        return self.parser.parse(cleaned_text if cleaned_text is not None else request)

    @staticmethod
    def _validate_cutoff(cutoff: datetime) -> None:
        offset = cutoff.utcoffset() if cutoff.tzinfo is not None else None
        if offset is None or offset.total_seconds() != 0:
            raise ProductionPipelineError("UTC_PREDICTION_CUTOFF_REQUIRED")

    def _stage(self, pipeline_id: str, match_id: str | None,
               stages: list[PipelineStageResult], name: str, status: str,
               started: datetime, input_count: int, output_count: int,
               evidence_ids: tuple[str, ...], error_code: str | None = None) -> None:
        stage = PipelineStageResult(name, status, started, datetime.now(UTC),
                                    input_count, output_count, error_code, evidence_ids)
        stages.append(stage)
        if self.audit_store:
            self.audit_store.append(PipelineExecutionRecord(pipeline_id, match_id, stage))


def _to_historical_matches(rows: tuple[HistoricalMatchSample, ...]
                           ) -> tuple[HistoricalMatch, ...]:
    result = []
    for row in rows:
        provider_ids = tuple((provider, row.match_id) for provider in row.provider_ids)
        result.append(HistoricalMatch(_canonical_match_id(
            row.provider_ids[0], row.match_id, row.home_team_id, row.away_team_id,
            row.kickoff, row.competition_id), provider_ids,
            row.home_team_id, row.away_team_id, row.competition_id, row.kickoff,
            row.home_goals, row.away_goals, None, None,
            "NEUTRAL" if row.neutral_venue else "HOME_AWAY",
            "FINISHED", row.evidence_ids, row.fetched_at, row.gender,
            row.age_group, "CLUB", "FIRST_TEAM", str(row.kickoff.year)))
    return tuple(result)


def _request_raw_text(request: ProductionMatchRequest | MatchRequest | str) -> str:
    """Recover original input text for source classification and audit."""
    if isinstance(request, ProductionMatchRequest):
        return request.raw_text
    if isinstance(request, MatchRequest):
        return request.raw_text or "".join((request.home_team or "", " VS ",
                                             request.away_team or ""))
    return request


def _jc_result_for_source(source_audit: MatchSourceAudit, query: JCMatchQuery,
                          verifier: JCMatchVerifier) -> JCVerificationResult:
    """Honor explicit source provenance before consulting an external JC provider."""
    if source_audit.source_type == MatchSourceType.USER_JC_CONFIRMED:
        return JCVerificationResult(JCStatus.USER_CONFIRMED, 1.0,
            query.competition_id, query.competition_name, query.kickoff_time,
            "USER_INPUT", source_audit.audit_id, datetime.now(UTC),
            "USER_DECLARED_JC;OFFICIAL_PROVIDER_SKIPPED")
    if source_audit.source_type == MatchSourceType.RESEARCH_TEST:
        return JCVerificationResult.unknown(datetime.now(UTC), "RESEARCH_TEST_BYPASS")
    return verifier.verify(query)


def _canonical_match_id(provider: str, provider_id: str, home: str, away: str,
                        kickoff: datetime, competition: str) -> str:
    material = f"{provider}|{provider_id}|{home}|{away}|{kickoff.isoformat()}|{competition}"
    return hashlib.sha256(material.encode()).hexdigest()[:24]


def _quality(hierarchy, direct: tuple[HistoricalMatch, ...],
             rows: tuple[HistoricalMatch, ...], cutoff: datetime, *,
             deduplicated_count: int) -> SampleQualityReport:
    ages = sorted((cutoff - row.kickoff_at).total_seconds() / 86400 for row in direct)
    median_age = statistics.median(ages) if ages else None
    span = ((hierarchy.quality.latest_match - hierarchy.quality.earliest_match).days
            if hierarchy.quality.latest_match and hierarchy.quality.earliest_match else 0)
    recency = max(0.0, 1.0 - (median_age or 3650) / 3650)
    source_quality = min(1.0, (hierarchy.quality.min_source_tier or 0) / 5)
    return SampleQualityReport(
        hierarchy.home_team_direct_matches, hierarchy.away_team_direct_matches,
        hierarchy.competition_matches, hierarchy.comparable_competition_matches,
        hierarchy.federation_prior_matches + hierarchy.age_group_prior_matches
        + hierarchy.gender_prior_matches,
        hierarchy.quality.opponent_diversity_score, recency, source_quality,
        hierarchy.quality.source_conflict_count, deduplicated_count, 0,
        "HIGH" if hierarchy.quality.opponent_diversity_score < 0.3 else "MEDIUM",
        hierarchy.quality.earliest_match, hierarchy.quality.latest_match,
        median_age, span,
        "HIGH" if hierarchy.quality.roster_continuity_unknown else "LOW")
