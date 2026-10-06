"""Audited fail-closed production trial orchestration."""

from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
import sys
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

from erguoyuan_football.app.input.match_input import MatchInputParserV2, MatchRequest
from erguoyuan_football.contracts.common import Availability
from erguoyuan_football.data.availability import (
    AvailabilityItem,
    DataAvailabilityReport,
)
from erguoyuan_football.data.schemas import Fixture
from erguoyuan_football.data.snapshots import PredictionSnapshot
from erguoyuan_football.knowledge.competitions.competition_database import (
    CompetitionDatabase,
)
from erguoyuan_football.knowledge.competitions.competition_resolver import (
    CompetitionResolver,
)
from erguoyuan_football.knowledge.competitions.competition_schema import (
    CompetitionIdentity,
)
from erguoyuan_football.knowledge.entities.entity_fingerprint import (
    canonical_entity_id,
    entity_fingerprint,
)
from erguoyuan_football.knowledge.entities.openligadb_discovery import (
    OpenLigaDBTeamDiscovery,
)
from erguoyuan_football.knowledge.entities.universal_team_resolver import (
    UniversalTeamResolver,
)
from erguoyuan_football.knowledge.entities.verified_entity_store import (
    VerifiedEntityStore,
)
from erguoyuan_football.knowledge.match_identity import GlobalKnowledgeResolver
from erguoyuan_football.knowledge.teams.team_database import TeamDatabase
from erguoyuan_football.knowledge.teams.team_resolver import TeamResolver
from erguoyuan_football.knowledge.teams.team_schema import TeamIdentity
from erguoyuan_football.match_source.match_source import MatchSourceType
from erguoyuan_football.match_source.match_source_classifier import (
    MatchSourceClassifier,
)
from erguoyuan_football.models.config import ModelConfig
from erguoyuan_football.prediction.execution_records import ModelExecutionStore
from erguoyuan_football.prediction.existing_model_executor import ExistingModelExecutor
from erguoyuan_football.prediction.model_execution_engine import ModelExecutionEngine
from erguoyuan_football.prediction.model_execution_planner import ModelExecutionPlanner
from erguoyuan_football.research.global_registry_factory import (
    build_global_provider_registry,
)
from erguoyuan_football.research.live_data.model_requirements import (
    load_model_requirements,
)
from erguoyuan_football.research.live_data.openligadb import (
    OpenLigaDBConfig,
    OpenLigaDBProvider,
    SeasonBatch,
    TeamBinding,
)
from erguoyuan_football.research.live_data.openligadb_directory import (
    DirectoryTeam,
    OpenLigaDBDirectoryProvider,
)
from erguoyuan_football.research.live_data.readiness import (
    LiveDataReadinessGate,
    ReadinessConfig,
)
from erguoyuan_football.research.pipeline_execution_record import PipelineExecutionStore
from erguoyuan_football.research.production_match_pipeline import (
    ProductionMatchPipeline,
)
from erguoyuan_football.web_research.evidence.evidence_store import EvidenceStore
from production.association_sources import AssociationSourceRegistry
from production.config import ProductionConfig
from production.fixture_discovery import (
    FixtureDiscoveryRequest,
    FixtureDiscoveryResult,
    FixtureDiscoveryRouter,
)
from production.global_fixture_research import (
    GlobalFixtureResearch,
    GlobalFixtureResearchProvider,
    build_research_source_registry,
)
from production.jc_metadata import JCMetadataParser
from production.official_source_registry import OfficialFootballSourceRegistry
from production.trial_record import TrialPredictionRecord, TrialRecordStore
from production.v7_renderer import V7TrialRenderer

LOGGER = logging.getLogger(__name__)
StageStatus = Literal["READY", "WARNING", "FAILED"]
_ISO_DATE_HINT = re.compile(
    r"(?<!\d)(20\d{2}-\d{2}-\d{2})(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:?\d{2})?)?"
)


@dataclass(frozen=True)
class StageResult:
    """One named pipeline stage, status, and explicit outcome or reason."""

    stage: str
    status: StageStatus
    detail: str


@dataclass(frozen=True)
class ProductionRunResult:
    """Full trial path outcome, including the immutable saved record ID."""

    status: Literal["READY", "WARNING", "FAILED"]
    request: MatchRequest
    source_type: MatchSourceType
    stages: tuple[StageResult, ...]
    rendered_output: str
    prediction_id: str | None


class ProductionRunner:
    """Run every declared stage and stop probability generation at missing gates."""

    def __init__(self, config: ProductionConfig) -> None:
        self.config = config
        self.parser = MatchInputParserV2()
        self.classifier = MatchSourceClassifier()
        self.entity_store = VerifiedEntityStore(
            config.record_database.parent / "dynamic_entity_store.sqlite"
        )
        self.identity_resolver = GlobalKnowledgeResolver(entity_store=self.entity_store)
        self.renderer = V7TrialRenderer()
        self.discovery_provider: OpenLigaDBProvider | None = None
        self.discovery_evidence: EvidenceStore | None = None

    def close(self) -> None:
        """Release the shared provider-verified identity cache."""
        if self.discovery_evidence is not None:
            self.discovery_evidence.close()
        if self.discovery_provider is not None:
            self.discovery_provider.close()
        self.entity_store.close()

    def _enable_reviewed_club_discovery(self, competition: str | None) -> None:
        """Use only the configured legal provider scope; unknown scopes remain explicit."""
        if (
            not self.config.enable_live_data
            or not competition
            or (competition.casefold().strip() not in {"premier league", "epl", "英超"})
            or self.discovery_provider is not None
        ):
            return
        root = self._project_root()
        provider = OpenLigaDBProvider(
            OpenLigaDBConfig.from_yaml(root / "config/phase13_5_provider.yaml")
        )
        evidence = EvidenceStore(
            root / "data/production_entity_evidence.sqlite", provider.sources
        )
        self.discovery_provider = provider
        self.discovery_evidence = evidence
        self.identity_resolver.entity_resolver.discovery = (
            OpenLigaDBTeamDiscovery(provider, evidence),
        )

    def run(
        self,
        raw_text: str,
        *,
        competition: str | None = None,
        jc_confirmed: bool = False,
        simulation: bool = False,
        research_test: bool = False,
        date_hint: date | None = None,
        mode: Literal["AUTO_RESEARCH", "JC_PRODUCTION"] | None = None,
        jc_fields: dict[str, str] | None = None,
    ) -> ProductionRunResult:
        """Execute preflight, explicitly block unavailable stages, and append a trial row."""
        if mode == "JC_PRODUCTION":
            return self._run_jc(raw_text, jc_fields or {})
        parse_text, parsed_date_hint = _remove_iso_date_hint(raw_text)
        effective_date_hint = date_hint or parsed_date_hint
        competition = (
            competition or self.parser.parse_structure(parse_text).competition_raw
        )
        if (not competition and not simulation and self.config.enable_live_data
                and self.config.enable_official_fixture_research):
            competition = self._infer_german_competition(parse_text)
        if (
            not simulation
            and self.config.enable_live_data
            and competition
            and _german_scope_for(competition) is not None
        ):
            return self._run_live_german(
                raw_text,
                competition=competition,
                jc_confirmed=jc_confirmed,
                research_test=research_test,
                date_hint=effective_date_hint,
                force_auto_research=mode == "AUTO_RESEARCH",
            )
        stages: list[StageResult] = []
        source_audit = self.classifier.classify(
            raw_text,
            jc_confirmed=jc_confirmed,
            declared_type=(MatchSourceType.AUTO_DISCOVERY if mode == "AUTO_RESEARCH"
                           else MatchSourceType.RESEARCH_TEST if research_test else None),
        )
        parsed = self.parser.parse(self.classifier.clean_match_text(parse_text))
        request = replace(
            parsed,
            competition=competition or parsed.competition,
            date=effective_date_hint or parsed.date,
            raw_text=raw_text,
            match_source_type=source_audit.source_type,
            jc_confirmed=source_audit.source_type == MatchSourceType.USER_JC_CONFIRMED,
        )
        if parsed.validation_status != "VALID":
            stages.append(
                StageResult("INPUT", "FAILED", parsed.reason or "MATCH_SYNTAX_INVALID")
            )
        else:
            stages.append(StageResult("INPUT", "READY", "INPUT_PARSER_V2"))

        stages.append(
            StageResult("SOURCE_CLASSIFIER", "READY", source_audit.source_type.value)
        )
        identity = None
        if (
            request.validation_status == "VALID"
            and request.home_team
            and request.away_team
        ):
            if not simulation:
                self._enable_reviewed_club_discovery(competition)
            identity = self.identity_resolver.resolve_names(
                request.home_team,
                request.away_team,
                competition,
                allow_discovery=not simulation,
            )
        resolved_home = identity.home_team if identity is not None else None
        resolved_away = identity.away_team if identity is not None else None
        resolved_competition = identity.competition if identity is not None else None
        if resolved_home is not None and resolved_away is not None:
            stages.append(
                StageResult(
                    "ENTITY_RESOLVER",
                    "READY",
                    f"{resolved_home.team_id};{resolved_away.team_id}",
                )
            )
        else:
            outcomes = (
                (identity.home_resolution, identity.away_resolution) if identity else ()
            )
            statuses = {
                outcome.resolution_status for outcome in outcomes if outcome is not None
            }
            reason = (
                "ENTITY_AMBIGUOUS"
                if "AMBIGUOUS" in statuses
                else "ENTITY_DISCOVERY_REQUIRED"
                if "DISCOVERY_REQUIRED" in statuses
                else "TEAM_NOT_FOUND"
            )
            stages.append(StageResult("ENTITY_RESOLVER", "WARNING", reason))
        if resolved_competition is not None:
            stages.append(
                StageResult(
                    "COMPETITION_RESOLVER", "READY", resolved_competition.competition_id
                )
            )
        elif competition:
            stages.append(
                StageResult(
                    "COMPETITION_RESOLVER",
                    "WARNING",
                    "COMPETITION_HINT_UNRESOLVED;FIXTURE_DISCOVERY_CONTINUES",
                )
            )
        else:
            stages.append(
                StageResult(
                    "COMPETITION_RESOLVER", "WARNING", "COMPETITION_DISCOVERY_REQUIRED"
                )
            )
        if source_audit.source_type == MatchSourceType.USER_JC_CONFIRMED:
            stages.append(
                StageResult(
                    "JC_VERIFICATION",
                    "READY",
                    "USER_CONFIRMED; OFFICIAL_LOOKUP_BYPASSED",
                )
            )
        else:
            stages.append(
                StageResult(
                    "JC_VERIFICATION", "READY", "NOT_REQUIRED_FOR_AUTO_RESEARCH"
                )
            )

        fixture_discovery: FixtureDiscoveryResult | None = None
        if not simulation and resolved_home is not None and resolved_away is not None:
            fixture_discovery = self._discover_fixture(
                resolved_home,
                resolved_away,
                date_hint=effective_date_hint
                or (request.date if request.date else None),
                competition_hint=competition or request.competition,
            )
            fixture_status: StageStatus = (
                "READY"
                if fixture_discovery.status in {"FOUND", "RESEARCH_FOUND"}
                else "WARNING"
            )
            fixture_detail = (
                f"{fixture_discovery.status};{fixture_discovery.reason};"
                f"structured={fixture_discovery.structured_status};"
                f"global_research={'CALLED' if fixture_discovery.research_called else 'NOT_CALLED'};"
                f"research_status={fixture_discovery.research_status};"
                f"providers_queried={','.join(fixture_discovery.providers_queried) or 'NONE'};"
                f"research_sources={','.join(fixture_discovery.research_sources_attempted) or 'NONE'};"
                f"cache_candidates={fixture_discovery.cache_candidates}"
            )
            if fixture_discovery.candidates:
                candidate_details = ";".join(
                    f"{item.kickoff_time.isoformat()}|{item.competition_name}|{item.fixture_id}"
                    for item in fixture_discovery.candidates
                )
                fixture_detail += f";candidates={candidate_details}"
            stages.append(
                StageResult("FIXTURE_DISCOVERY", fixture_status, fixture_detail)
            )

        research_candidate = (
            fixture_discovery.candidates[0]
            if fixture_discovery is not None
            and fixture_discovery.status == "RESEARCH_FOUND"
            and len(fixture_discovery.candidates) == 1
            else None
        )
        current_prediction_time = datetime.now(UTC)
        pit_blocked = bool(
            research_candidate
            and research_candidate.kickoff_time.astimezone(UTC)
            <= current_prediction_time
        )
        if research_candidate is not None:
            request = replace(request, competition=research_candidate.competition_name)
            stages.append(
                StageResult(
                    "COMPETITION_RESOLVER",
                    "READY",
                    f"DISCOVERED_FROM_FIXTURE:{research_candidate.competition_id}",
                )
            )
        if pit_blocked:
            stages.append(
                StageResult(
                    "DATA_PIPELINE",
                    "WARNING",
                    "PIT_LIVE_PREDICTION_BLOCKED:KICKOFF_ALREADY_PASSED",
                )
            )
        else:
            stages.append(
                StageResult(
                    "DATA_PIPELINE",
                    "WARNING",
                    "VERIFIED_FIXTURE_AND_PIT_SNAPSHOT_UNAVAILABLE",
                )
            )
        stages.append(
            StageResult(
                "MODEL_PLANNER", "WARNING", "REQUIRED_PRODUCTION_INPUTS_UNAVAILABLE"
            )
        )
        stages.append(
            StageResult(
                "MODEL_EXECUTION", "WARNING", "NO_READY_MODEL_PLAN; NO_MODEL_WAS_CALLED"
            )
        )
        stages.append(
            StageResult("ENSEMBLE", "WARNING", "FINAL_CORE_PROMOTION_GATE_CLOSED")
        )
        if simulation:
            primary_blocker = None
            secondary_consequences: tuple[str, ...] = ()
            blocked_reasons = tuple(
                stage.detail for stage in stages if stage.status != "READY"
            )
        elif fixture_discovery is None:
            primary_blocker = "ENTITY_RESOLUTION_FAILED"
            secondary_consequences = (
                "FIXTURE_DISCOVERY_NOT_RUN",
                "SNAPSHOT_NOT_CREATED",
                "MODELS_NOT_CALLED",
                "CORE_UNAVAILABLE",
            )
            blocked_reasons = ()
        elif fixture_discovery.status in {"FOUND", "RESEARCH_FOUND"} and pit_blocked:
            primary_blocker = "PIT_LIVE_PREDICTION_BLOCKED"
            secondary_consequences = (
                "POST_MATCH_FIXTURE_EVIDENCE_NOT_USED_AS_PREMATCH_FEATURES",
                "SNAPSHOT_NOT_CREATED",
                "MODELS_NOT_CALLED",
                "CORE_UNAVAILABLE",
            )
            blocked_reasons = ()
        elif fixture_discovery.status == "RESEARCH_FOUND":
            primary_blocker = "HISTORY_PROVIDER_UNAVAILABLE"
            secondary_consequences = (
                "SNAPSHOT_NOT_CREATED",
                "MODELS_NOT_CALLED",
                "CORE_UNAVAILABLE",
            )
            blocked_reasons = ()
        elif fixture_discovery.status in {
            "AMBIGUOUS_FIXTURE",
            "OFFICIAL_FIXTURE_AMBIGUOUS",
            "FIXTURE_EVIDENCE_CONFLICT",
        }:
            primary_blocker = fixture_discovery.status
            secondary_consequences = (
                "FIXTURE_NOT_UNIQUELY_RESOLVED",
                "SNAPSHOT_NOT_CREATED",
                "MODELS_NOT_CALLED",
                "CORE_UNAVAILABLE",
            )
            blocked_reasons = ()
        elif fixture_discovery.status == "KICKOFF_TIME_UNVERIFIED":
            primary_blocker = "KICKOFF_TIME_UNVERIFIED"
            secondary_consequences = (
                "SNAPSHOT_NOT_CREATED",
                "MODELS_NOT_CALLED",
                "CORE_UNAVAILABLE",
            )
            blocked_reasons = ()
        elif fixture_discovery.status == "FOUND":
            primary_blocker = "FIXTURE_FOUND_BUT_NO_SUPPORTED_PRODUCTION_PIPELINE"
            secondary_consequences = (
                "SNAPSHOT_NOT_CREATED",
                "MODELS_NOT_CALLED",
                "CORE_UNAVAILABLE",
            )
            blocked_reasons = ()
        else:
            explicit_research_failures = {
                "OFFICIAL_SOURCE_COVERAGE_MISSING",
                "OFFICIAL_SOURCE_UNREACHABLE",
                "OFFICIAL_FIXTURE_NOT_FOUND",
                "OFFICIAL_NETWORK_BLOCKED",
                "OFFICIAL_FIXTURE_ENTRYPOINT_NOT_CONFIGURED",
                "OFFICIAL_ADAPTER_UNSUPPORTED_PAGE",
            }
            primary_blocker = (
                fixture_discovery.research_status
                if fixture_discovery.research_called
                and fixture_discovery.research_status in explicit_research_failures
                else f"NO_VERIFIED_FIXTURE_FOUND:{fixture_discovery.reason}"
            )
            consequence_list = [
                "SNAPSHOT_NOT_CREATED",
                "MODELS_NOT_CALLED",
                "CORE_UNAVAILABLE",
            ]
            if resolved_competition is None:
                consequence_list.insert(0, "COMPETITION_UNRESOLVED")
            secondary_consequences = tuple(consequence_list)
            blocked_reasons = ()
        rendered = self.renderer.render(
            request,
            source_type=source_audit.source_type,
            competition=request.competition or competition,
            blocked_reasons=blocked_reasons,
            simulation=simulation,
            primary_blocker=primary_blocker,
            secondary_consequences=secondary_consequences,
            pipeline_diagnostics=_pipeline_diagnostics(
                stages,
                fixture_discovery,
                resolved_competition,
                research_candidate,
                pit_blocked,
            ),
            run_mode="AUTO_RESEARCH" if mode == "AUTO_RESEARCH" and not simulation else None,
        )
        stages.append(
            StageResult("V7_RENDER", "WARNING", "V7_DIAGNOSTIC_NOT_FINAL_PREDICTION")
        )

        prediction_id: str | None = None
        if self.config.enable_record:
            try:
                store = TrialRecordStore(self.config.record_database)
                try:
                    record = TrialPredictionRecord.create(
                        match_name=f"{request.home_team or ''} VS {request.away_team or ''}".strip(),
                        competition=request.competition or competition,
                        source_type=source_audit.source_type.value,
                        models_used=(),
                        input_snapshot={
                            "home_team": request.home_team,
                            "away_team": request.away_team,
                            "home_team_id": (
                                resolved_home.team_id if resolved_home else None
                            ),
                            "away_team_id": (
                                resolved_away.team_id if resolved_away else None
                            ),
                            "competition_id": (
                                resolved_competition.competition_id
                                if resolved_competition
                                else None
                            ),
                            "competition": competition,
                            "simulation_only": simulation,
                            "fixture_discovery": (
                                {
                                    "status": fixture_discovery.status,
                                    "reason": fixture_discovery.reason,
                                    "research_called": fixture_discovery.research_called,
                                    "research_status": fixture_discovery.research_status,
                                    "structured_status": fixture_discovery.structured_status,
                                    "research_sources_attempted": list(
                                        fixture_discovery.research_sources_attempted
                                    ),
                                    "research_queries": list(
                                        fixture_discovery.research_queries
                                    ),
                                    "research_evidence_ids": list(
                                        fixture_discovery.research_evidence_ids
                                    ),
                                    "pit_classifications": list(
                                        fixture_discovery.pit_classifications
                                    ),
                                    "providers_registered": list(
                                        fixture_discovery.providers_registered
                                    ),
                                    "providers_queried": list(
                                        fixture_discovery.providers_queried
                                    ),
                                    "cache_candidates": fixture_discovery.cache_candidates,
                                    "candidates": [
                                        {
                                            "fixture_id": item.fixture_id,
                                            "competition_id": item.competition_id,
                                            "competition": item.competition_name,
                                            "kickoff_time": item.kickoff_time.isoformat(),
                                            "source": item.source,
                                            "evidence_id": item.evidence_id,
                                        }
                                        for item in fixture_discovery.candidates
                                    ],
                                }
                                if fixture_discovery
                                else None
                            ),
                            "primary_blocker": primary_blocker,
                            "secondary_consequences": list(secondary_consequences),
                            "source_evidence": list(source_audit.evidence),
                            "captured_at": datetime.now(UTC).isoformat(),
                        },
                        prediction_output={
                            "status": "UNAVAILABLE",
                            "steps": [asdict(stage) for stage in stages],
                            "v7_text": rendered,
                        },
                    )
                    store.append(record)
                    prediction_id = record.prediction_id
                finally:
                    store.close()
                stages.append(StageResult("SAVE_RECORD", "READY", prediction_id))
            except (OSError, ValueError, RuntimeError, sqlite3.Error) as error:
                LOGGER.exception("trial record persistence failed")
                stages.append(
                    StageResult(
                        "SAVE_RECORD", "FAILED", f"{type(error).__name__}:{error}"
                    )
                )
        else:
            stages.append(StageResult("SAVE_RECORD", "WARNING", "RECORDING_DISABLED"))

        overall: Literal["READY", "WARNING", "FAILED"]
        if any(stage.status == "FAILED" for stage in stages):
            overall = "FAILED"
        elif all(stage.status == "READY" for stage in stages):
            overall = "READY"
        else:
            overall = "WARNING"
        for stage in stages:
            LOGGER.info(
                "production_stage=%s status=%s detail=%s",
                stage.stage,
                stage.status,
                stage.detail,
            )
        return ProductionRunResult(
            overall,
            request,
            source_audit.source_type,
            tuple(stages),
            rendered,
            prediction_id,
        )

    def _run_jc(self, raw_text: str, fields: dict[str, str]) -> ProductionRunResult:
        """Trust only user-declared fixture metadata; keep PIT and history gates."""
        request_enrichment = fields.get("official_enrichment") == "true"
        metadata_fields = {key: value for key, value in fields.items()
                           if key != "official_enrichment"}
        metadata = JCMetadataParser().parse(raw_text, **metadata_fields)
        cutoff = datetime.now(UTC)
        complete = not metadata.missing_fields and metadata.kickoff_utc is not None
        request = MatchRequest(
            None, metadata.home_team, metadata.away_team,
            metadata.competition_canonical,
            date.fromisoformat(metadata.kickoff_original[:10])
            if metadata.kickoff_utc and metadata.kickoff_original else None,
            "USER_INPUT", "VALID" if complete else "INVALID",
            None if complete else "JC_FIXTURE_METADATA_INCOMPLETE",
            raw_text, MatchSourceType.USER_JC_CONFIRMED, True,
        )
        stages = [StageResult("INPUT_MODE", "READY", "JC_PRODUCTION"),
                  StageResult("SOURCE_CLASSIFIER", "READY", "USER_JC_CONFIRMED")]
        home_id: str | None = None
        away_id: str | None = None
        official_stage = StageResult("OFFICIAL_RESEARCH", "READY", "NOT_REQUIRED")
        if not complete:
            blocker = "JC_FIXTURE_METADATA_INCOMPLETE"
            stages.append(StageResult("JC_FIXTURE", "WARNING",
                                      f"{blocker}:{','.join(metadata.missing_fields)}"))
        else:
            assert metadata.kickoff_utc is not None
            stages.append(StageResult("JC_FIXTURE", "READY", "USER_CONFIRMED_FIXTURE"))
            assert metadata.home_team is not None and metadata.away_team is not None
            identity = self.identity_resolver.resolve_names(
                metadata.home_team, metadata.away_team,
                metadata.competition_canonical, allow_discovery=False,
            )
            home = identity.home_team
            away = identity.away_team
            if home is None or away is None or home.team_id == away.team_id:
                blocker = "ENTITY_RESOLUTION_FAILED"
                stages.append(StageResult("ENTITY", "WARNING", blocker))
            else:
                home_id, away_id = home.team_id, away.team_id
                stages.append(StageResult("ENTITY", "READY", f"{home_id};{away_id}"))
                stages.append(StageResult("COMPETITION", "READY",
                                          f"USER_CONFIRMED:{metadata.competition_canonical}"))
                stages.append(StageResult("KICKOFF", "READY",
                                          f"USER_CONFIRMED:{metadata.kickoff_utc.isoformat()}"))
                if cutoff >= metadata.kickoff_utc:
                    blocker = "PIT_LIVE_PREDICTION_BLOCKED"
                    stages.append(StageResult("PIT", "WARNING", blocker))
                else:
                    stages.append(StageResult("PIT", "READY", "PASS"))
                    if request_enrichment:
                        try:
                            discovery = self._discover_fixture(
                                home, away, date_hint=metadata.kickoff_utc.date(),
                                competition_hint=metadata.competition_canonical,
                            )
                            matching = (discovery.status in {"FOUND", "RESEARCH_FOUND"}
                                        and len(discovery.candidates) == 1
                                        and discovery.candidates[0].kickoff_time.astimezone(UTC)
                                        == metadata.kickoff_utc)
                            official_stage = StageResult(
                                "OFFICIAL_RESEARCH",
                                "READY" if matching else "WARNING",
                                "OPTIONAL_ENRICHMENT_AVAILABLE" if matching else
                                "OFFICIAL_ENRICHMENT_UNAVAILABLE",
                            )
                        except (OSError, ValueError, RuntimeError) as error:
                            LOGGER.warning("optional JC enrichment unavailable: %s", error)
                            official_stage = StageResult(
                                "OFFICIAL_RESEARCH", "WARNING",
                                "OFFICIAL_ENRICHMENT_UNAVAILABLE",
                            )
                    blocker = "HISTORY_PROVIDER_UNAVAILABLE"
                    stages.append(StageResult("HISTORY", "WARNING", blocker))
        history_position = next((index for index, stage in enumerate(stages)
                                 if stage.stage == "HISTORY"), len(stages))
        stages.insert(history_position, official_stage)
        stages.append(StageResult("SNAPSHOT", "WARNING", "NOT_CREATED"))
        stages.append(StageResult("MODELS", "WARNING", "NOT_EXECUTED"))
        pipeline = tuple(f"{stage.stage:<20} {stage.detail}" for stage in stages
                         if stage.stage not in {"INPUT_MODE", "SOURCE_CLASSIFIER"})
        rendered = self.renderer.render(
            request, source_type=MatchSourceType.USER_JC_CONFIRMED,
            competition=metadata.competition_canonical,
            blocked_reasons=(f"本次预测未完成：{blocker}",),
            kickoff_time=metadata.kickoff_utc,
            primary_blocker=blocker, pipeline_diagnostics=pipeline,
            run_mode="JC_PRODUCTION",
            fixture_origin="USER_CONFIRMED_CHINA_SPORTTERY" if complete else None,
        )
        prediction_id = None
        if self.config.enable_record:
            record = TrialPredictionRecord.create(
                match_name=f"{metadata.home_team or 'UNAVAILABLE'} VS {metadata.away_team or 'UNAVAILABLE'}",
                competition=metadata.competition_canonical,
                source_type=MatchSourceType.USER_JC_CONFIRMED.value,
                models_used=(),
                input_snapshot={
                    "mode": "JC_PRODUCTION", "source_type": "USER_JC_CONFIRMED",
                    "fixture_metadata_origin": (
                        "USER_CONFIRMED_CHINA_SPORTTERY" if complete else None),
                    "user_confirmed_fixture": complete, "fixture_verified": False,
                    "real_snapshot_created": False, "synthetic_data": False,
                    "simulation_only": False,
                    "jc_match_code": metadata.jc_match_code,
                    "competition_original": metadata.competition_original,
                    "competition_canonical": metadata.competition_canonical,
                    "home": metadata.home_team, "away": metadata.away_team,
                    "home_team_id": home_id, "away_team_id": away_id,
                    "kickoff_original": metadata.kickoff_original,
                    "timezone": metadata.source_timezone,
                    "kickoff_utc": (metadata.kickoff_utc.isoformat()
                                    if metadata.kickoff_utc else None),
                    "prediction_as_of": cutoff.isoformat(),
                    "history_status": ("UNAVAILABLE" if blocker == "HISTORY_PROVIDER_UNAVAILABLE"
                                       else "NOT_RUN"),
                    "snapshot_id": None, "models_executed": [],
                    "warnings": ([official_stage.detail]
                                 if official_stage.status == "WARNING" else []),
                    "final_status": "UNAVAILABLE",
                },
                prediction_output={"status": "UNAVAILABLE", "blocking_reason": blocker,
                                   "steps": [asdict(stage) for stage in stages],
                                   "v7_text": rendered},
            )
            store = TrialRecordStore(self.config.record_database)
            try:
                store.append(record)
                prediction_id = record.prediction_id
            finally:
                store.close()
            stages.append(StageResult("SAVE_RECORD", "READY", prediction_id))
        return ProductionRunResult("WARNING", request,
                                   MatchSourceType.USER_JC_CONFIRMED,
                                   tuple(stages), rendered, prediction_id)

    def _discover_fixture(
        self,
        home: TeamIdentity,
        away: TeamIdentity,
        *,
        date_hint: date | None,
        competition_hint: str | None,
    ) -> FixtureDiscoveryResult:
        """Try structured routes, then always invoke global research as final fallback."""
        root = self._project_root()
        from erguoyuan_football.research.global_provider_registry import (
            GlobalProviderRegistry,
        )

        registry = GlobalProviderRegistry()
        try:
            if self.config.enable_live_data:
                registry = build_global_provider_registry(
                    root / "config/phase13_7_openligadb.yaml"
                )
        except (OSError, ValueError, TypeError, KeyError) as error:
            LOGGER.warning(
                "structured fixture registry unavailable: %s", type(error).__name__
            )
        try:
            associations = AssociationSourceRegistry()
            evidence_path = self.config.record_database.parent / "global_fixture_evidence.sqlite"
            official_registry = OfficialFootballSourceRegistry()
            providers: tuple[GlobalFixtureResearchProvider, ...] = ()
            if self.config.enable_official_fixture_research:
                from production.official_live_adapters import OfficialLiveFixtureAdapter

                relevant = list(official_registry.for_national_teams(home.country, away.country))
                relevant.extend(official_registry.for_competition(competition_hint))
                relevant.extend(item for item in official_registry.all()
                                if item.source_id == "UEFA" and home.federation == away.federation == "UEFA")
                providers = tuple({item.source_id: OfficialLiveFixtureAdapter(item, home, away)
                                   for item in relevant if "FIXTURES" in item.capabilities}.values())
            sources = build_research_source_registry(associations, providers)
            evidence = EvidenceStore(evidence_path, sources)
            research = GlobalFixtureResearch(
                associations=associations, providers=providers, evidence_store=evidence,
                fixture_sources_enabled=self.config.enable_official_fixture_research,
            )
            router = FixtureDiscoveryRouter(
                registry,
                root / "data/production_evidence.sqlite",
                global_research=research,
            )
            return router.discover(
                FixtureDiscoveryRequest(
                    home, away, datetime.now(UTC), date_hint, competition_hint
                )
            )
        finally:
            if "evidence" in locals():
                evidence.close()
            for registered in registry.list():
                close = getattr(registered.provider, "close", None)
                if callable(close):
                    close()

    def _project_root(self) -> Path:
        """Resolve runtime data and provider config beside the configured record store."""
        configured = self.config.record_database.parent.parent.resolve()
        provider_config = configured / "config/phase13_7_openligadb.yaml"
        if provider_config.is_file() or getattr(sys, "frozen", False):
            return configured
        return Path(__file__).resolve().parents[2]

    def _run_live_german(
        self,
        raw_text: str,
        *,
        competition: str,
        jc_confirmed: bool,
        research_test: bool,
        date_hint: date | None = None,
        force_auto_research: bool = False,
    ) -> ProductionRunResult:
        """Run the existing German OpenLigaDB pipeline and real ready models."""
        stages: list[StageResult] = []
        source_audit = self.classifier.classify(
            raw_text,
            jc_confirmed=jc_confirmed,
            declared_type=(MatchSourceType.AUTO_DISCOVERY if force_auto_research
                           else MatchSourceType.RESEARCH_TEST if research_test else None),
        )
        parse_text, parsed_date_hint = _remove_iso_date_hint(raw_text)
        parsed = self.parser.parse(self.classifier.clean_match_text(parse_text))
        request = replace(
            parsed,
            competition=competition,
            date=date_hint or parsed_date_hint or parsed.date,
            raw_text=raw_text,
            match_source_type=source_audit.source_type,
            jc_confirmed=source_audit.source_type == MatchSourceType.USER_JC_CONFIRMED,
        )
        if (
            parsed.validation_status != "VALID"
            or not parsed.home_team
            or not parsed.away_team
        ):
            raise ValueError(parsed.error_code or "MATCH_SYNTAX_INVALID")
        stages.extend(
            (
                StageResult("INPUT", "READY", "INPUT_PARSER_V2"),
                StageResult(
                    "SOURCE_CLASSIFIER", "READY", source_audit.source_type.value
                ),
            )
        )

        scope_id = _german_scope_for(competition)
        if scope_id is None:
            raise ValueError("PROVIDER_COVERAGE_MISSING")
        project_root = self._project_root()
        directory = OpenLigaDBDirectoryProvider(
            project_root / "config/phase13_7_openligadb.yaml"
        )
        provider: _PrefetchedOpenLigaDBProvider | None = None
        pipeline_audit: PipelineExecutionStore | None = None
        evidence: EvidenceStore | None = None
        model_store: ModelExecutionStore | None = None
        try:
            directory_result = directory.fetch_teams(scope_id)
            if directory_result.status.value != "VERIFIED" or not directory_result.rows:
                raise ValueError(
                    f"TEAM_DIRECTORY_UNAVAILABLE:{directory_result.reason}"
                )
            scope = directory.scopes[scope_id]
            bindings: dict[str, TeamBinding] = {}
            identities: list[TeamIdentity] = []
            for row in directory_result.rows:
                team_id = _canonical_live_team_id(
                    row, "OPENLIGADB_PHASE16_REAL_ACCEPTANCE"
                )
                bindings[team_id] = TeamBinding(row.provider_team_id, row.provider_name)
                identity = TeamIdentity(
                    team_id,
                    row.provider_name,
                    scope.country,
                    scope.federation,
                    [row.provider_name],
                    scope.entity_type,
                    scope.gender,
                    scope.age_group,
                    scope.squad_level,
                    provider_ids={directory.provider_id: str(row.provider_team_id)},
                    identity_status="PROVIDER_VERIFIED",
                    verification_evidence_ids=[row.evidence_id],
                )
                identities.append(identity)
                self.entity_store.save(
                    identity,
                    provider_id=directory.provider_id,
                    provider_team_id=str(row.provider_team_id),
                    verified_at=row.retrieved_at,
                )
            directory_team_resolver = TeamResolver(TeamDatabase(identities))
            team_entity_resolver = UniversalTeamResolver(
                static=directory_team_resolver, store=self.entity_store
            )
            if request.home_team is None or request.away_team is None:
                raise ValueError("MATCH_SYNTAX_INVALID")
            home = team_entity_resolver.resolve(request.home_team).identity
            away = team_entity_resolver.resolve(request.away_team).identity
            if home is None or away is None or home.team_id == away.team_id:
                stages.append(
                    StageResult(
                        "ENTITY_RESOLUTION", "FAILED", "TEAM_NOT_FOUND_OR_AMBIGUOUS"
                    )
                )
                raise ValueError("TEAM_NOT_FOUND_OR_AMBIGUOUS")
            comp_identity = CompetitionIdentity(
                scope.competition_id,
                scope.competition_name,
                scope.federation,
                scope.country,
                "CLUB",
            )
            comp_resolver = CompetitionResolver(
                CompetitionDatabase(
                    (comp_identity,),
                    {
                        scope.competition_id: (
                            scope.competition_name,
                            "German Bundesliga"
                            if scope.shortcut == "bl1"
                            else "German 2. Bundesliga",
                        )
                    },
                )
            )
            resolved_competition = comp_resolver.resolve(competition)
            if resolved_competition is None:
                raise ValueError("COMPETITION_NOT_FOUND")
            stages.extend(
                (
                    StageResult(
                        "ENTITY_RESOLUTION", "READY", f"{home.team_id};{away.team_id}"
                    ),
                    StageResult(
                        "COMPETITION_RESOLUTION", "READY", scope.competition_id
                    ),
                )
            )

            config = OpenLigaDBConfig(
                provider_id="OPENLIGADB_PHASE16_REAL_ACCEPTANCE",
                base_url=directory.base_url,
                league_shortcut=scope.shortcut,
                league_season=scope.season,
                competition_id=scope.competition_id,
                competition_name=scope.competition_name,
                team_bindings=bindings,
                enabled=True,
                license="ODbL-1.0",
                license_url="https://openligadb.de/lizenz",
                team_country=scope.country,
                team_federation=scope.federation,
                competition_gender=scope.gender,
                competition_age_group=scope.age_group,
                competition_entity_type=scope.entity_type,
                history_seasons=(2025, 2024),
                competition_type=scope.competition_type,
                competition_aliases=(
                    "German Bundesliga"
                    if scope.shortcut == "bl1"
                    else "German 2. Bundesliga",
                ),
            )
            live_provider = OpenLigaDBProvider(config)
            try:
                batches = {
                    season: live_provider.fetch_season(season, bypass_cache=True)
                    for season in (2026, 2025, 2024)
                }
                cutoff = datetime.now(UTC)
                current_batch = batches[scope.season]
                verified = live_provider.verify_fixture(
                    current_batch,
                    home.team_id,
                    away.team_id,
                    date_hint=date_hint or parsed_date_hint,
                    cutoff=cutoff,
                )
            finally:
                live_provider.close()
            if verified.fixture is None:
                raise ValueError(f"FIXTURE_NOT_VERIFIED:{verified.status}")
            stages.append(
                StageResult(
                    "FIXTURE_VERIFICATION",
                    "READY",
                    f"fixture_verified=true;provider_match_id={verified.fixture.provider_match_id};"
                    f"kickoff={verified.fixture.kickoff_at.isoformat()}",
                )
            )
            provider = _PrefetchedOpenLigaDBProvider(config, batches)
            evidence = EvidenceStore(
                project_root / "data/production_evidence.sqlite", provider.sources
            )
            pipeline_audit = PipelineExecutionStore(
                project_root / "data/production_pipeline.sqlite"
            )
            readiness = LiveDataReadinessGate(
                ReadinessConfig.from_yaml(
                    project_root / "config/phase13_5_readiness.yaml"
                ),
                load_model_requirements(project_root / "config/model_registry.yaml"),
            )
            planner = ModelExecutionPlanner(
                project_root / "config/phase14_model_requirements.yaml",
                project_root / "config/model_registry.yaml",
            )
            pipeline = ProductionMatchPipeline(
                provider,
                evidence,
                readiness,
                planner=planner,
                audit_store=pipeline_audit,
                team_resolver=directory_team_resolver,
            )
            parsed = replace(request, date=verified.fixture.kickoff_at.date())
            data_bundle = pipeline.build(
                parsed, prediction_cutoff=cutoff, mode="LIVE", fetch_live=True
            )
            stages.append(
                StageResult(
                    "DATA_PIPELINE",
                    "READY",
                    f"pipeline_id={data_bundle.research_session_id};historical_rows="
                    f"{len(data_bundle.historical_matches)};provider={config.provider_id}",
                )
            )
            if not data_bundle.canonical_match.verified:
                raise ValueError("FIXTURE_NOT_VERIFIED")

            venue_context = data_bundle.canonical_match.venue_type
            neutral = verified.fixture.neutral_venue
            venue_basis = "PROVIDER_EXPLICIT"
            if neutral is None and venue_context == "HOME_AWAY":
                neutral = False
                venue_basis = "VERIFIED_DOMESTIC_LEAGUE_HOME_AWAY_INFERENCE"
            if neutral is None:
                venue_basis = "UNKNOWN"
            fixture = Fixture(
                match_id=data_bundle.canonical_match.match_id,
                competition_id=scope.competition_id,
                home_team_id=home.team_id,
                away_team_id=away.team_id,
                kickoff_time=verified.fixture.kickoff_at,
                source=config.provider_id,
                retrieved_at=current_batch.retrieved_at,
                as_of_time=current_batch.retrieved_at,
                data_version=_fixture_version(
                    verified.fixture.provider_match_id, current_batch.retrieved_at
                ),
                season=str(scope.season),
                neutral_venue=neutral,
            )
            fixture_evidence = tuple(
                record.evidence_id
                for record in data_bundle.research_package.available_data.get(
                    "fixture", []
                )
            )
            history_evidence = tuple(
                record.evidence_id
                for record in data_bundle.research_package.available_data.get(
                    "historical_results", []
                )
            )
            snapshot = PredictionSnapshot(
                match_id=fixture.match_id,
                prediction_time=cutoff,
                match_data_snapshot=fixture,
                data_completeness=DataAvailabilityReport(
                    match_id=fixture.match_id,
                    items={
                        "fixture": AvailabilityItem(
                            availability=Availability.AVAILABLE,
                            reason="LIVE_PROVIDER_FIXTURE_VERIFIED",
                            evidence_ids=fixture_evidence,
                        ),
                        "historical_results": AvailabilityItem(
                            availability=Availability.AVAILABLE
                            if history_evidence
                            else Availability.UNAVAILABLE,
                            reason="LIVE_PROVIDER_PIT_HISTORY"
                            if history_evidence
                            else "NO_HISTORICAL_RESULTS",
                            evidence_ids=history_evidence,
                        ),
                    },
                ),
            )
            stages.append(
                StageResult(
                    "REAL_MATCH_SNAPSHOT",
                    "READY",
                    f"real_snapshot_created=true;snapshot_id={snapshot.prediction_snapshot_id};"
                    f"as_of={snapshot.prediction_time.isoformat()};synthetic_data=false;"
                    f"venue_basis={venue_basis}",
                )
            )
            stages.extend(
                (
                    StageResult(
                        "HISTORICAL_DATA",
                        "READY",
                        f"pit_rows={len(data_bundle.historical_matches)};seasons=2024,2025,2026",
                    ),
                    StageResult(
                        "SAMPLE_WINDOWS",
                        "READY",
                        f"competition_window={len(data_bundle.sample_windows.competition_history) if data_bundle.sample_windows else 0};"
                        f"hierarchy_competition={data_bundle.sample_hierarchy.competition_matches}",
                    ),
                )
            )
            entry_status = ",".join(
                f"{entry.model_id}={entry.action}"
                for entry in data_bundle.execution_plan.entries
            )
            stages.append(StageResult("MODEL_PLANNER", "READY", entry_status))
            execute_plan = replace(
                data_bundle.execution_plan, dry_run=False, execution_mode="REAL_EXECUTE"
            )
            executor = ExistingModelExecutor(fixture, snapshot, cutoff, ModelConfig())
            model_store = ModelExecutionStore(
                project_root / "data/model_execution.sqlite"
            )
            execution_results = ModelExecutionEngine(model_store).execute(
                execute_plan,
                data_bundle.model_input_bundles,
                {entry.model_id: executor for entry in execute_plan.entries},
                execute_ready_models=self.config.enable_prediction,
            )
            successes = [
                result
                for result in execution_results
                if result.status in {"EXECUTED", "DEGRADED_EXECUTED"}
                and result.probabilities is not None
            ]
            bayesian = next(
                (
                    result
                    for result in execution_results
                    if result.model_name == "BAYESIAN_HIERARCHICAL_V1"
                ),
                None,
            )
            stages.append(
                StageResult(
                    "MODEL_EXECUTION",
                    "READY" if successes else "WARNING",
                    ";".join(
                        f"{result.model_name}={result.status}"
                        + (f"({result.error})" if result.error else "")
                        for result in execution_results
                    ),
                )
            )
            stages.append(
                StageResult(
                    "ENSEMBLE",
                    "WARNING",
                    "BLOCKED: PHASE9_NOT_PROMOTED; BASE_MODEL_OUTPUTS_REMAIN_INDEPENDENT",
                )
            )
            prediction_payload = []
            for result in execution_results:
                prediction_payload.append(
                    {
                        "model_id": result.model_name,
                        "execution_status": result.status,
                        "execution_record_id": result.execution_record_id,
                        "reason": result.error,
                        "probability": (
                            {
                                "p_home": result.probabilities.p_home,
                                "p_draw": result.probabilities.p_draw,
                                "p_away": result.probabilities.p_away,
                            }
                            if result.probabilities
                            else None
                        ),
                        "raw_output": dict(result.raw_output)
                        if result.raw_output
                        else None,
                    }
                )
            rendered = self.renderer.render(
                request,
                source_type=source_audit.source_type,
                competition=scope.competition_name,
                blocked_reasons=(
                    "BAYESIAN_SAMPLING_DIAGNOSTICS_REVIEW_REQUIRED"
                    if bayesian and bayesian.status == "FAILED"
                    else "BAYESIAN_NOT_READY_OR_NOT_EXECUTED",
                    "ODDS_XG_LINEUP_OPTA_ML_ARTIFACT_UNAVAILABLE",
                    "PHASE9_GOLDEN_REAL_OOS_INCOMPLETE",
                    "LONG_TERM_LIVE_VALIDATION_INCOMPLETE",
                ),
                model_predictions=tuple(prediction_payload),
                fixture_verified=True,
                snapshot_id=snapshot.prediction_snapshot_id,
                kickoff_time=verified.fixture.kickoff_at,
            )
            stages.append(
                StageResult(
                    "V7_CORE_OUTPUT",
                    "WARNING",
                    "BASE_MODEL_HDA_AVAILABLE; CORE_META_CALIBRATION_AND_DATA_DEPENDENT_MARKETS_UNAVAILABLE",
                )
            )
            prediction_id: str | None = None
            if self.config.enable_record:
                store = TrialRecordStore(self.config.record_database)
                try:
                    record = TrialPredictionRecord.create(
                        match_name=f"{request.home_team} VS {request.away_team}",
                        competition=scope.competition_name,
                        source_type=source_audit.source_type.value,
                        models_used=tuple(result.model_name for result in successes),
                        input_snapshot={
                            "match_id": fixture.match_id,
                            "provider_match_id": verified.fixture.provider_match_id,
                            "fixture_verified": True,
                            "real_snapshot_created": True,
                            "synthetic_data": False,
                            "prediction_snapshot": snapshot.model_dump(mode="json"),
                            "source_evidence": list(source_audit.evidence),
                            "provider_evidence_ids": list(data_bundle.evidence_ids),
                            "prediction_cutoff": cutoff.isoformat(),
                            "kickoff_time": verified.fixture.kickoff_at.isoformat(),
                        },
                        prediction_output={
                            "status": "BASE_MODELS_EXECUTED_WITH_WARNINGS"
                            if successes
                            else "UNAVAILABLE",
                            "models": prediction_payload,
                            "core_probability": None,
                            "core_status": "BLOCKED_NOT_PROMOTED",
                            "v7_text": rendered,
                            "warning_codes": [
                                "BAYESIAN_DIAGNOSTICS_REVIEW_REQUIRED",
                                "ODDS_XG_LINEUP_OPTA_ML_ARTIFACT_UNAVAILABLE",
                                "PROBABILITIES_NOT_LONG_TERM_VALIDATED",
                                "PHASE15_GOLDEN_REAL_OOS_INCOMPLETE",
                            ],
                        },
                    )
                    store.append(record)
                    prediction_id = record.prediction_id
                finally:
                    store.close()
                stages.append(StageResult("SAVE_RECORD", "READY", prediction_id))
            else:
                stages.append(
                    StageResult("SAVE_RECORD", "WARNING", "RECORDING_DISABLED")
                )
            overall: Literal["READY", "WARNING", "FAILED"] = (
                "WARNING" if successes and prediction_id else "FAILED"
            )
            return ProductionRunResult(
                overall,
                request,
                source_audit.source_type,
                tuple(stages),
                rendered,
                prediction_id,
            )
        finally:
            if model_store is not None:
                model_store.close()
            if pipeline_audit is not None:
                pipeline_audit.close()
            if evidence is not None:
                evidence.close()
            if provider is not None:
                provider.close()
            directory.close()

    def _infer_german_competition(self, raw_text: str) -> str | None:
        """Use live season directories only when both exact teams share one scope."""
        parsed = self.parser.parse(self.classifier.clean_match_text(raw_text))
        if parsed.validation_status != "VALID" or not parsed.home_team or not parsed.away_team:
            return None
        offline = self.identity_resolver.resolve_names(
            parsed.home_team, parsed.away_team, None, allow_discovery=False
        )
        if ((offline.home_team is not None and offline.home_team.entity_type == "NATIONAL")
                or (offline.away_team is not None and offline.away_team.entity_type == "NATIONAL")):
            return None
        directory = OpenLigaDBDirectoryProvider(
            self._project_root() / "config/phase13_7_openligadb.yaml"
        )
        matching: list[str] = []
        try:
            for scope_id in (
                "germany_men_bundesliga_2026",
                "germany_men_second_division_2026",
            ):
                try:
                    result = directory.fetch_teams(scope_id)
                except (OSError, ValueError, RuntimeError) as error:
                    LOGGER.warning("live league directory unavailable: %s", error)
                    return None
                if result.status.value != "VERIFIED" or not result.rows:
                    continue
                scope = directory.scopes[scope_id]
                identities = [
                    TeamIdentity(
                        _canonical_live_team_id(row, "OPENLIGADB_PHASE16_REAL_ACCEPTANCE"),
                        row.provider_name,
                        scope.country,
                        scope.federation,
                        [row.provider_name],
                        scope.entity_type,
                        scope.gender,
                        scope.age_group,
                        scope.squad_level,
                    )
                    for row in result.rows
                ]
                resolver = TeamResolver(TeamDatabase(identities))
                home = resolver.resolve(parsed.home_team)
                away = resolver.resolve(parsed.away_team)
                if home is not None and away is not None and home.team_id != away.team_id:
                    matching.append(scope.competition_name)
        finally:
            directory.close()
        return matching[0] if len(matching) == 1 else None


class _PrefetchedOpenLigaDBProvider(OpenLigaDBProvider):
    """Expose live-fetched batches with their original retrieval times to Phase 14."""

    def __init__(
        self, config: OpenLigaDBConfig, batches: dict[int, SeasonBatch]
    ) -> None:
        super().__init__(config)
        self._batches = dict(batches)

    def fetch_season(
        self, season: int | None = None, *, bypass_cache: bool = True
    ) -> SeasonBatch:
        selected = self.config.league_season if season is None else season
        if selected in self._batches:
            return self._batches[selected]
        return super().fetch_season(selected, bypass_cache=bypass_cache)


def _german_scope_for(competition: str) -> str | None:
    aliases = {
        "bundesliga": "germany_men_bundesliga_2026",
        "german bundesliga": "germany_men_bundesliga_2026",
        "1. bundesliga": "germany_men_bundesliga_2026",
        "2. bundesliga": "germany_men_second_division_2026",
        "2nd bundesliga": "germany_men_second_division_2026",
        "german 2. bundesliga": "germany_men_second_division_2026",
    }
    return aliases.get(" ".join(competition.casefold().split()))


def _canonical_live_team_id(row: DirectoryTeam, provider_id: str) -> str:
    scope = row.competition
    fingerprint = entity_fingerprint(
        country=scope.country,
        entity_type=scope.entity_type,
        gender=scope.gender,
        age_group=scope.age_group,
        squad_level=scope.squad_level,
        name=row.provider_name,
        provider_id=provider_id,
        provider_team_id=str(row.provider_team_id),
    )
    return canonical_entity_id(
        country=scope.country,
        federation=scope.federation,
        entity_type=scope.entity_type,
        gender=scope.gender,
        age_group=scope.age_group,
        fingerprint=fingerprint,
        provider_id=provider_id,
        provider_team_id=str(row.provider_team_id),
    )


def _fixture_version(provider_match_id: str, retrieved_at: datetime) -> str:
    return (
        "OPENLIGADB_"
        + hashlib.sha256(
            f"{provider_match_id}|{retrieved_at.isoformat()}".encode()
        ).hexdigest()[:20]
    )


def _remove_iso_date_hint(raw_text: str) -> tuple[str, date | None]:
    """Extract one unambiguous ISO calendar date without treating it as a team name."""
    matches = tuple(_ISO_DATE_HINT.finditer(raw_text))
    if not matches:
        return raw_text, None
    values = {match.group(1) for match in matches}
    if len(values) != 1:
        return raw_text, None
    try:
        hint = date.fromisoformat(next(iter(values)))
    except ValueError:
        return raw_text, None
    return _ISO_DATE_HINT.sub(" ", raw_text).strip(), hint


def _pipeline_diagnostics(
    stages: list[StageResult],
    fixture: FixtureDiscoveryResult | None,
    resolved_competition: CompetitionIdentity | None,
    candidate: object | None,
    pit_blocked: bool,
) -> tuple[str, ...]:
    """Describe actual stage outcomes for the V7 diagnostic, without inference."""
    by_name = {stage.stage: stage for stage in stages}
    entity = by_name.get("ENTITY_RESOLVER")
    if fixture is None:
        structured, research = "NOT_RUN", "NOT_CALLED"
        competition, kickoff, history, snapshot, models = (
            "NOT_RUN",
            "NOT_RUN",
            "BLOCKED",
            "NOT_CREATED",
            "NOT_CALLED",
        )
    else:
        structured = fixture.structured_status
        research = fixture.research_status if fixture.research_called else "NOT_CALLED"
        competition = (
            "DISCOVERED"
            if candidate is not None
            else "READY"
            if resolved_competition is not None
            else "UNRESOLVED"
        )
        kickoff = (
            "VERIFIED"
            if candidate is not None
            else "UNAVAILABLE"
            if fixture.status
            in {
                "UNAVAILABLE",
                "NO_VERIFIED_FIXTURE_FOUND",
                "PROVIDER_REGISTRY_UNAVAILABLE",
            }
            else "UNVERIFIED"
        )
        history = "BLOCKED"
        snapshot, models = "NOT_CREATED", "NOT_CALLED"
        if pit_blocked:
            kickoff = "PAST_KICKOFF_PIT_BLOCKED"
    return (
        f"Entity             {entity.status if entity else 'NOT_RUN'}",
        f"Structured Fixture {structured}",
        f"Research Fixture   {research}",
        f"Competition        {competition}",
        f"Kickoff            {kickoff}",
        f"History            {history}",
        f"Snapshot           {snapshot}",
        f"Models             {models}",
        *(
            f"Research source attempt: {attempt}"
            for attempt in (fixture.research_sources_attempted if fixture else ())
        ),
    )
