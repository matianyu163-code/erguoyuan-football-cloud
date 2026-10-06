"""Single CLI and Python entry for external research packets into existing CORE."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from erguoyuan_football.app.input.match_input import MatchRequest
from erguoyuan_football.app.installer.version import runtime_build_info
from erguoyuan_football.contracts.common import Availability
from erguoyuan_football.data.availability import (
    AvailabilityItem,
    DataAvailabilityReport,
)
from erguoyuan_football.data.schemas import Fixture
from erguoyuan_football.data.snapshots import PredictionSnapshot
from erguoyuan_football.knowledge.entities.verified_entity_store import (
    VerifiedEntityStore,
)
from erguoyuan_football.knowledge.match_identity import GlobalKnowledgeResolver
from erguoyuan_football.knowledge.teams.alias_matcher import normalize_alias
from erguoyuan_football.knowledge.teams.team_database import TeamDatabase
from erguoyuan_football.knowledge.teams.team_resolver import TeamResolver
from erguoyuan_football.match_source.match_source import MatchSourceType
from erguoyuan_football.models.config import ModelConfig
from erguoyuan_football.prediction.execution_records import ModelExecutionStore
from erguoyuan_football.prediction.existing_model_executor import ExistingModelExecutor
from erguoyuan_football.prediction.model_execution_engine import ModelExecutionEngine
from erguoyuan_football.prediction.model_execution_planner import ModelExecutionPlanner
from erguoyuan_football.prediction.model_input_adapter import (
    default_model_input_adapters,
)
from erguoyuan_football.research.live_data.model_readiness import (
    ModelReadinessEvaluator,
)
from erguoyuan_football.research.live_data.model_requirements import (
    load_model_requirements,
)
from erguoyuan_football.research.live_data.readiness import (
    DataAvailability,
    DataReadinessItem,
    LiveDataReadinessReport,
)
from erguoyuan_football.research.samples.match_deduplicator import HistoricalMatchSample
from erguoyuan_football.research.samples.sample_builder import SampleBuilder
from production.bridge.contracts import CoreDataPacketV1, CoreResultPacketV1
from production.bridge.exporter import archive_packet, export_result
from production.bridge.importer import bridge_competition, import_history
from production.bridge.validator import BridgeRejected, packet_hash, validate_packet
from production.config import ProductionConfig, default_config_path
from production.trial_record import TrialPredictionRecord, TrialRecordStore
from production.v7_renderer import V7TrialRenderer

LOGGER = logging.getLogger(__name__)
_BRIDGE_MODELS = frozenset({
    "DIXON_COLES_V1", "BIVARIATE_POISSON_V1", "ELO_V1", "PI_RATING_V1",
    "DYNAMIC_BAYESIAN_POISSON_V1", "CORE_SPI_LIKE_V1",
    "BAYESIAN_HIERARCHICAL_V1",
})


def _availability(history_count: int, evidence: tuple[str, ...],
                  fixture_evidence: tuple[str, ...], match_id: str) -> DataAvailabilityReport:
    fixture_present = AvailabilityItem(availability=Availability.AVAILABLE,
        reason="BRIDGE_FIXTURE_SOURCE_CITATION", evidence_ids=fixture_evidence)
    absent = AvailabilityItem(availability=Availability.UNAVAILABLE,
                              reason="NO_BRIDGE_HISTORY")
    present = (AvailabilityItem(availability=Availability.AVAILABLE,
               reason="BRIDGE_IMPORTED_PIT_HISTORY", evidence_ids=evidence)
               if evidence else absent)
    return DataAvailabilityReport(match_id=match_id, items={
        "fixture": fixture_present,
        "historical_results": present if history_count else absent,
        "historical_goals": present if history_count else absent,
    })


def _readiness(hierarchy: Any, history_count: int, outcomes: set[int],
               cutoff: datetime, registry_path: Path) -> LiveDataReadinessReport:
    requirements = load_model_requirements(registry_path)
    required_items = {name for requirement in requirements for name in requirement.inputs}
    required_items.update(("FIXTURE", "HISTORICAL_RESULTS", "MATCH_TIMESTAMPS",
                           "HOME_AWAY_SPLIT", "THREE_WAY_MAPPER_TRAINING",
                           "COMPETITION_CONTEXT", "ODDS", "XG", "ML_ARTIFACT"))
    available = {"FIXTURE", "COMPETITION_CONTEXT"}
    if history_count:
        available.update(("HISTORICAL_RESULTS", "MATCH_TIMESTAMPS", "HOME_AWAY_SPLIT"))
    if history_count >= 12 and len(outcomes) == 3:
        available.add("THREE_WAY_MAPPER_TRAINING")
    items = {name: DataReadinessItem(name, DataAvailability.AVAILABLE
             if name in available else DataAvailability.MISSING)
             for name in required_items}
    verdicts = tuple(ModelReadinessEvaluator().evaluate(
        requirement, items, hierarchy, fixture_verified=True)
        for requirement in requirements)
    return LiveDataReadinessReport(
        True, "READY" if any(row.ready for row in verdicts) else "BLOCKED",
        tuple(items.values()), verdicts,
        tuple(row.model_name for row in verdicts if row.ready),
        tuple(row.model_name for row in verdicts if not row.ready),
        tuple(row.model_name for row in verdicts if row.degraded), (), cutoff,
        entity_resolution_status="BRIDGE_CANONICAL_RESOLVED",
        sample_hierarchy=hierarchy, sample_quality=hierarchy.quality,
    )


class BridgeRunner:
    """Revalidate external research before calling existing model implementations."""

    def __init__(self, config: ProductionConfig, *, project_root: Path | None = None,
                 synthetic_test: bool = False,
                 execute_synthetic_models: bool = False) -> None:
        self.config = config
        self.root = (project_root or Path(__file__).resolve().parents[3]).resolve()
        self.synthetic_test = synthetic_test
        self.execute_synthetic_models = execute_synthetic_models
        self.store = VerifiedEntityStore(
            config.record_database.parent / "dynamic_entity_store.sqlite")
        self.resolver = GlobalKnowledgeResolver(entity_store=self.store)

    def close(self) -> None:
        """Close the shared entity cache."""
        self.store.close()

    def _resolver_for_packet(self, packet: CoreDataPacketV1) -> GlobalKnowledgeResolver:
        """Prefer unique existing provider-verified identities over older local seeds."""
        names = {packet.match.home, packet.match.away}
        names.update(name for row in packet.history.all_matches()
                     for name in (row.home, row.away))
        verified = {}
        cutoff = packet.research_as_of.astimezone(UTC)
        for name in names:
            matches = tuple(row for row in self.store.find(name, as_of=cutoff)
                            if row.status == "VERIFIED")
            if len(matches) > 1:
                raise BridgeRejected("ENTITY_RESOLUTION_FAILED",
                                     f"AMBIGUOUS_PROVIDER_ENTITY:{name}")
            if matches:
                verified[matches[0].identity.team_id] = matches[0].identity
        if not verified:
            return self.resolver
        occupied = {normalize_alias(alias) for team in verified.values()
                    for alias in (team.official_name, *team.aliases)}
        local = {team.team_id: team for team in self.resolver.teams.database.all()
                 if not any(normalize_alias(alias) in occupied
                            for alias in (team.official_name, *team.aliases))}
        local.update(verified)
        return GlobalKnowledgeResolver(
            team_resolver=TeamResolver(TeamDatabase(local.values())),
            entity_store=self.store)

    def run(self, packet: CoreDataPacketV1, *,
            local_history: tuple[HistoricalMatchSample, ...] = ()) -> CoreResultPacketV1:
        """Execute a packet and persist both success and structured refusal."""
        digest = packet_hash(packet)
        build_id = runtime_build_info().get("build_id", "UNKNOWN")
        request = MatchRequest(None, packet.match.home, packet.match.away,
            packet.match.competition, None, "CHATGPT_WORK_BRIDGE", "VALID",
            None, f"{packet.match.home} VS {packet.match.away}",
            MatchSourceType.RESEARCH_TEST, False)
        snapshot: PredictionSnapshot | None = None
        models_planned: tuple[str, ...] = ()
        executed: tuple[str, ...] = ()
        failed: tuple[str, ...] = ()
        versions: dict[str, str] = {}
        probabilities: dict[str, dict[str, float]] = {}
        source_hashes: tuple[str, ...] = ()
        history_count = 0
        sample_windows: dict[str, object] = {}
        warnings: list[str] = ["CORE_META_CALIBRATION_NOT_PROMOTED",
                               "OPTIONAL_DATA_NOT_PROMOTED"]
        blocker: str | None = None
        fixture: Fixture | None = None
        try:
            if not self.config.enable_record:
                raise BridgeRejected("BRIDGE_INPUT_INVALID", "RECORDING_DISABLED")
            if packet.request_id.startswith("SYNTHETIC_TEST") and not self.synthetic_test:
                raise BridgeRejected("BRIDGE_INPUT_INVALID", "SYNTHETIC_PACKET_IN_PRODUCTION")
            archive_packet(packet, self.root)
            validate_packet(packet)
            resolver = self._resolver_for_packet(packet)
            identity = resolver.resolve_names(
                packet.match.home, packet.match.away, packet.match.competition,
                allow_discovery=False)
            home, away = identity.home_team, identity.away_team
            if home is None or away is None:
                raise BridgeRejected("ENTITY_RESOLUTION_FAILED", identity.status)
            competition = bridge_competition(packet.match.competition,
                                             home.federation, home.entity_type)
            if (home.team_id != packet.entities.home_entity
                    or away.team_id != packet.entities.away_entity):
                raise BridgeRejected("ENTITY_RESOLUTION_FAILED", "TARGET_ENTITY_ID_CONFLICT")
            imported = import_history(packet, resolver, local_history=local_history)
            fixture_hashes = tuple(hashlib.sha256(
                f"{row.source}|{row.source_url}|{row.fetched_at.isoformat()}".encode()
                ).hexdigest() for row in packet.fixture_evidence.sources)
            source_hashes = tuple(sorted(set(imported.source_hashes + fixture_hashes)))
            history_count = imported.deduplicated_count
            cutoff = packet.research_as_of.astimezone(UTC)
            kickoff = packet.match.kickoff_utc.astimezone(UTC)
            match_id = hashlib.sha256((competition.competition_id + "|"
                + home.team_id + "|" + away.team_id + "|" + kickoff.isoformat())
                .encode()).hexdigest()[:32]
            fixture_fetched = max(row.fetched_at.astimezone(UTC)
                                  for row in packet.fixture_evidence.sources)
            fixture = Fixture(match_id=match_id,
                lottery_match_no=packet.match.jc_match_code,
                competition_id=competition.competition_id,
                home_team_id=home.team_id, away_team_id=away.team_id,
                kickoff_time=kickoff, source="CHATGPT_WORK_BRIDGE",
                retrieved_at=fixture_fetched, as_of_time=fixture_fetched,
                data_version=digest, season=str(kickoff.year),
                neutral_venue=packet.match.neutral_venue)
            samples = imported.repository.samples
            outcomes = {0 if row.home_goals > row.away_goals else
                        1 if row.home_goals == row.away_goals else 2 for row in samples}
            evidence = tuple(sorted({value for row in samples for value in row.evidence_ids}))
            fixture_evidence = tuple(hashlib.sha256(
                row.source_url.encode()).hexdigest()
                for row in packet.fixture_evidence.sources)
            snapshot = PredictionSnapshot(match_id=match_id, prediction_time=cutoff,
                match_data_snapshot=fixture,
                data_completeness=_availability(history_count, evidence,
                                                 fixture_evidence, match_id))
            hierarchy = SampleBuilder(imported.repository).build(
                home, away, competition, cutoff=cutoff,
                competition_type="LEAGUE" if home.entity_type == "CLUB"
                                 else "INTERNATIONAL")
            sample_windows = {
                "home_direct_5": min(len(hierarchy.home_direct), 5),
                "home_direct_10": min(len(hierarchy.home_direct), 10),
                "home_direct_20": min(len(hierarchy.home_direct), 20),
                "away_direct_5": min(len(hierarchy.away_direct), 5),
                "away_direct_10": min(len(hierarchy.away_direct), 10),
                "away_direct_20": min(len(hierarchy.away_direct), 20),
                "competition_samples": len(hierarchy.competition),
                "direct_samples": len({row.match_id for row in
                                       (*hierarchy.home_direct, *hierarchy.away_direct)}),
            }
            readiness = _readiness(hierarchy, history_count, outcomes, cutoff,
                                   self.root / "config/model_registry.yaml")
            planner = ModelExecutionPlanner(
                self.root / "config/phase14_model_requirements.yaml",
                self.root / "config/model_registry.yaml")
            plan = planner.plan(readiness, prediction_time=cutoff,
                                training_cutoff=cutoff,
                                neutral_venue_known=fixture.neutral_venue is not None)
            entries = tuple(replace(row, action="BLOCK", mode="NONE",
                                    reasons=("BRIDGE_V1_SCOPE",))
                            if row.model_id not in _BRIDGE_MODELS else row
                            for row in plan.entries)
            plan = replace(plan, entries=entries, dry_run=False,
                           execution_mode="REAL_EXECUTE")
            models_planned = tuple(row.model_id for row in entries
                                   if row.model_id in _BRIDGE_MODELS and row.action != "BLOCK")
            if not models_planned:
                raise BridgeRejected("HISTORY_INSUFFICIENT", "NO_READY_MODEL")
            adapters = default_model_input_adapters()
            bundles = {}
            for entry in entries:
                if entry.action == "BLOCK":
                    continue
                try:
                    bundles[entry.model_id] = adapters[entry.model_id].build_input(
                        imported.repository, hierarchy, cutoff,
                        degraded=entry.action == "RUN_DEGRADED")
                except ValueError as error:
                    LOGGER.warning("bridge model input blocked %s: %s", entry.model_id, error)
                    warnings.append(f"MODEL_INPUT_BLOCKED:{entry.model_id}:{error}")
            model_store = ModelExecutionStore(self.root / "data/model_execution.sqlite")
            try:
                executor = ExistingModelExecutor(
                    fixture, snapshot, cutoff,
                    ModelConfig(allow_test_data=self.synthetic_test))
                results = ModelExecutionEngine(model_store).execute(
                    plan, bundles, {model_id: executor for model_id in models_planned},
                    execute_ready_models=self.config.enable_prediction
                                         and (not self.synthetic_test
                                              or self.execute_synthetic_models))
            finally:
                model_store.close()
            executed = tuple(row.model_name for row in results
                             if row.status in {"EXECUTED", "DEGRADED_EXECUTED"}
                             and row.probabilities is not None)
            failed = tuple(row.model_name for row in results if row.status == "FAILED")
            versions = {entry.model_id: entry.requirements.model_version
                        for entry in entries if entry.model_id in _BRIDGE_MODELS}
            probabilities = {row.model_name: {
                "p_home": row.probabilities.p_home,
                "p_draw": row.probabilities.p_draw,
                "p_away": row.probabilities.p_away,
            } for row in results if row.model_name in executed
                and row.probabilities is not None}
            if not executed:
                raise BridgeRejected("MODEL_EXECUTION_FAILED", "NO_VALID_MODEL_OUTPUT")
            warnings.extend(f"MODEL_FAILED:{model_id}" for model_id in failed)
        except BridgeRejected as error:
            blocker = error.code
            warnings.append(str(error))
        except (OSError, ValueError, TypeError, KeyError) as error:
            LOGGER.exception("bridge run failed")
            blocker = "SNAPSHOT_FAILED" if snapshot is None else "MODEL_EXECUTION_FAILED"
            warnings.append(f"{type(error).__name__}:{error}")
        model_outputs: tuple[dict[str, object], ...] = tuple({
            "model_id": name, "execution_status": "EXECUTED",
            "probability": values,
        } for name, values in probabilities.items())
        v7 = V7TrialRenderer().render(request,
            source_type=MatchSourceType.RESEARCH_TEST,
            competition=packet.match.competition,
            blocked_reasons=tuple(warnings), model_predictions=model_outputs,
            fixture_verified=False, snapshot_id=(snapshot.prediction_snapshot_id
                                                 if snapshot else None),
            kickoff_time=packet.match.kickoff_utc,
            primary_blocker=blocker, run_mode="CHATGPT_WORK_BRIDGE",
            fixture_origin="BRIDGE_SOURCE_CITATION_VALIDATED" if fixture else None)
        status = ("BASE_MODELS_EXECUTED_WITH_WARNINGS" if executed else "BLOCKED")
        record = TrialPredictionRecord.create(
            match_name=f"{packet.match.home} VS {packet.match.away}",
            competition=packet.match.competition, source_type="CHATGPT_WORK_BRIDGE",
            models_used=executed,
            input_snapshot={"input_origin": "CHATGPT_WORK_BRIDGE",
                "request_id": packet.request_id, "packet_hash": digest,
                "bridge_version": "CORE_BRIDGE_V1",
                "source_hashes": source_hashes,
                "source_urls": [row.source_url for row in packet.fixture_evidence.sources],
                "prediction_cutoff": packet.research_as_of.astimezone(UTC).isoformat(),
                "kickoff_time": packet.match.kickoff_utc.astimezone(UTC).isoformat(),
                "snapshot_id": snapshot.prediction_snapshot_id if snapshot else None,
                "synthetic_data": self.synthetic_test,
                "eligible_history_count": history_count,
                "sample_windows": sample_windows,
                "provider_provenance": sorted({row.source for row in
                    packet.fixture_evidence.sources} | {row.source for row in
                    packet.history.all_matches()}),
                "models_planned": models_planned,
                "build_id": build_id,
                "fixture_evidence_status": "SOURCE_CITATION_VALIDATED_UNVERIFIED_CONTENT"},
            prediction_output={"status": status, "primary_blocker": blocker,
                "models_executed": executed, "models_failed": failed,
                "model_versions": versions, "raw_probabilities": probabilities,
                "core_probability": None, "v7_text": v7, "warnings": warnings})
        trial_store = TrialRecordStore(self.config.record_database)
        try:
            trial_store.append(record)
        finally:
            trial_store.close()
        result = CoreResultPacketV1(build_id=build_id, request_id=packet.request_id,
            prediction_id=record.prediction_id, status=status,
            primary_blocker=blocker,
            snapshot_id=snapshot.prediction_snapshot_id if snapshot else None,
            models_planned=models_planned, models_executed=executed,
            models_failed=failed, model_versions=versions,
            raw_probabilities=probabilities, core_probability=None,
            v7_output=v7, warnings=tuple(warnings), packet_hash=digest)
        export_result(result, self.root)
        return result


def main(argv: list[str] | None = None) -> int:
    """Read one JSON packet and emit exactly one result packet on stdout."""
    parser = argparse.ArgumentParser(description="CORE Bridge V1")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--project-root", type=Path,
                        help="Explicit data/config root for packaged desktop builds")
    args = parser.parse_args(argv)
    try:
        if args.input.stat().st_size > 5_000_000:
            raise BridgeRejected("BRIDGE_INPUT_INVALID", "PACKET_TOO_LARGE")
        packet = CoreDataPacketV1.model_validate_json(args.input.read_text(encoding="utf-8"))
        runner = BridgeRunner(ProductionConfig.load(args.config),
                              project_root=args.project_root)
        try:
            result = runner.run(packet)
        finally:
            runner.close()
    except (OSError, ValidationError, BridgeRejected, ValueError) as error:
        print(json.dumps({"schema_version": "CORE_RESULT_PACKET_V1",
                          "status": "BLOCKED", "primary_blocker": "BRIDGE_INPUT_INVALID",
                          "detail": str(error)}, ensure_ascii=False))
        return 2
    print(result.model_dump_json())
    return 0 if result.models_executed else 3


if __name__ == "__main__":
    raise SystemExit(main())
