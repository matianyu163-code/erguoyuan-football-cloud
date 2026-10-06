"""Opt-in live prediction probe; emits raw model results, never final V7 output."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import replace
from pathlib import Path

from erguoyuan_football.data.schemas import Fixture
from erguoyuan_football.data.snapshots import PredictionSnapshot
from erguoyuan_football.models.config import load_config
from erguoyuan_football.prediction.execution_records import ModelExecutionStore
from erguoyuan_football.prediction.existing_model_executor import ExistingModelExecutor
from erguoyuan_football.prediction.model_execution_engine import ModelExecutionEngine
from erguoyuan_football.prediction.model_execution_planner import ModelExecutionPlanner
from erguoyuan_football.research.as_of_evidence_loader import (
    AsOfEvidenceLoader,
    ReplayEvidenceUnavailable,
)
from erguoyuan_football.research.live_data.competition_provider_factory import (
    bind_verified_competition,
)
from erguoyuan_football.research.live_data.model_requirements import (
    load_model_requirements,
)
from erguoyuan_football.research.live_data.openligadb import (
    OpenLigaDBConfig,
    OpenLigaDBProvider,
)
from erguoyuan_football.research.live_data.openligadb_directory import (
    OpenLigaDBDirectoryProvider,
)
from erguoyuan_football.research.live_data.readiness import (
    LiveDataReadinessGate,
    ReadinessConfig,
)
from erguoyuan_football.research.match_universe import MatchUniverse
from erguoyuan_football.research.pipeline_execution_record import PipelineExecutionStore
from erguoyuan_football.research.production_match_pipeline import (
    ProductionMatchPipeline,
    ProductionMatchRequest,
)
from erguoyuan_football.web_research.evidence.evidence_store import EvidenceStore
from erguoyuan_football.web_research.time_utils import parse_utc

ROOT = Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> int:
    """Collect PIT data and execute only when the explicit flag is supplied."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("match", help='e.g. "Arsenal vs Liverpool"')
    parser.add_argument("--competition", default="Premier League")
    parser.add_argument("--cutoff", required=True, help="Explicit UTC ISO-8601 cutoff")
    parser.add_argument("--universe", choices=[item.value for item in MatchUniverse],
                        default=MatchUniverse.GLOBAL_RESEARCH.value)
    parser.add_argument("--mode", choices=("LIVE", "REPLAY"), default="LIVE")
    parser.add_argument("--match-id", help="Provider match ID required for persisted replay lookup")
    parser.add_argument("--execute-ready-models", action="store_true")
    parser.add_argument("--profile", choices=("development", "production"), default="development")
    parser.add_argument("--execution-db", type=Path, default=ROOT / "data/model_execution.sqlite")
    parser.add_argument("--evidence-db", type=Path, default=ROOT / "data/research_evidence.sqlite")
    parser.add_argument("--pipeline-audit-db", type=Path,
                        default=ROOT / "data/production_pipeline_audit.sqlite")
    args = parser.parse_args(argv)
    cutoff = parse_utc(args.cutoff)
    if args.mode == "REPLAY":
        if not args.match_id:
            parser.error("--match-id is required for REPLAY")
        return _replay_probe(args.evidence_db, args.match_id, cutoff)

    directory = None
    team_resolver = None
    directory_scopes = {
        "2. bundesliga": "germany_men_second_division_2026",
        "bundesliga 2": "germany_men_second_division_2026",
        "德乙": "germany_men_second_division_2026",
        "bundesliga": "germany_men_bundesliga_2026",
        "german bundesliga": "germany_men_bundesliga_2026",
        "德甲": "germany_men_bundesliga_2026",
    }
    scope_id = directory_scopes.get(args.competition.casefold())
    if scope_id is not None:
        directory = OpenLigaDBDirectoryProvider(ROOT / "config/phase13_7_openligadb.yaml")
        team_result = directory.fetch_teams(scope_id)
        bound = bind_verified_competition(directory, scope_id, team_result)
        provider = bound.provider
        team_resolver = bound.team_resolver
        evidence = EvidenceStore(args.evidence_db, directory.sources)
        for evidence_id in bound.team_evidence_ids:
            record = directory.evidence_store.get(evidence_id)
            if record is not None and evidence.get(evidence_id) is None:
                evidence.save(record)
    elif args.competition.casefold() in {"premier league", "epl", "英超"}:
        config = OpenLigaDBConfig.from_yaml(ROOT / "config/phase13_5_provider.yaml")
        provider = OpenLigaDBProvider(config)
        evidence = EvidenceStore(args.evidence_db, provider.sources)
    else:
        print("DATA STATUS: PROVIDER_COVERAGE_MISSING")
        return 2
    pipeline_audit = PipelineExecutionStore(args.pipeline_audit_db)
    execution_store = ModelExecutionStore(args.execution_db)
    try:
        gate = LiveDataReadinessGate(
            ReadinessConfig.from_yaml(ROOT / "config/phase13_5_readiness.yaml"),
            load_model_requirements(ROOT / "config/model_registry.yaml"),
        )
        planner = ModelExecutionPlanner(ROOT / "config/phase14_model_requirements.yaml",
                                        ROOT / "config/model_registry.yaml")
        pipeline = ProductionMatchPipeline(provider, evidence, gate, planner=planner,
                                           audit_store=pipeline_audit,
                                           team_resolver=team_resolver)
        request = ProductionMatchRequest(args.match, args.competition,
            match_universe=MatchUniverse(args.universe))
        result = pipeline.build(request, prediction_cutoff=cutoff)
        _print_data_status(result)
        if not args.execute_ready_models:
            print("MODE: REAL_DRY_RUN")
            print("PREDICTION: NOT_EXECUTED (use --execute-ready-models explicitly)")
            return 0
        verified = result.research_package.verified_fixture
        if verified is None:
            print("PREDICTION: NOT_EXECUTED (fixture unavailable)")
            return 2
        neutral = (True if result.canonical_match.venue_type == "NEUTRAL" else
                   False if result.canonical_match.venue_type == "HOME_AWAY" else None)
        data_version = hashlib.sha256((verified.provider_id + "|" +
            verified.provider_match_id + "|" + verified.kickoff_at.isoformat()).encode()).hexdigest()
        fixture = Fixture(match_id=verified.provider_match_id,
            competition_id=verified.competition_id, home_team_id=verified.home_team_id,
            away_team_id=verified.away_team_id, kickoff_time=verified.kickoff_at,
            source=verified.provider_id, retrieved_at=verified.verified_at,
            as_of_time=verified.verified_at, data_version=data_version,
            neutral_venue=neutral)
        snapshot = PredictionSnapshot(match_id=fixture.match_id,
            prediction_time=cutoff, match_data_snapshot=fixture)
        execution_plan = replace(result.execution_plan, dry_run=False,
                                 execution_mode="REAL_EXECUTE")
        executor = ExistingModelExecutor(fixture, snapshot, cutoff,
            load_config(ROOT / "configs/models.yaml", profile=args.profile))
        outputs = ModelExecutionEngine(execution_store).execute(
            execution_plan, result.model_input_bundles,
            {model_id: executor for model_id in planner.policies},
            execute_ready_models=True)
        executed = 0
        for row in outputs:
            if row.status in {"EXECUTED", "DEGRADED_EXECUTED"} and row.probabilities:
                executed += 1
                p = row.probabilities
                print(f"{row.model_name}: {row.status} | RAW H={p.p_home:.6f} "
                      f"D={p.p_draw:.6f} A={p.p_away:.6f} | CALIBRATED=FALSE "
                      f"| execution_record={row.execution_record_id}")
            elif row.status in {"BLOCKED", "FAILED"}:
                print(f"{row.model_name}: {row.status} | {row.error or 'NO_OUTPUT'}")
        print(f"MODEL EXECUTION: {executed} valid raw outputs")
        print("CALIBRATED: FALSE")
        print("FINAL V7: NOT_READY (NOT FINAL V7 OUTPUT)")
        return 0 if executed else 2
    except (ValueError, OSError, RuntimeError) as error:
        print(f"PIPELINE: FAILED ({type(error).__name__}: {error})")
        print("PREDICTION: NOT_EXECUTED")
        return 2
    finally:
        execution_store.close()
        pipeline_audit.close()
        evidence.close()
        provider.close()
        if directory is not None:
            directory.close()


def _print_data_status(result) -> None:
    """Print observed windows and independent model readiness before probabilities."""
    fixture = result.canonical_match
    windows = result.sample_windows
    source_audit = result.match_source_audit
    if source_audit is not None:
        print(f"MATCH SOURCE: {source_audit.source_type.value} "
              f"| confidence={source_audit.confidence:.0%} "
              f"| audit={source_audit.audit_id}")
    if result.jc_verification is not None:
        print(f"JC STATUS: {result.jc_verification.status.value} "
              f"| source={result.jc_verification.source or 'UNAVAILABLE'} "
              f"| confidence={result.jc_verification.confidence_score:.0%} "
              f"| reason={result.jc_verification.reason}")
    print(f"FIXTURE: {fixture.home_team_id} vs {fixture.away_team_id} "
          f"| {fixture.competition_id} | {fixture.kickoff_at.isoformat()}")
    print(f"CUT-OFF: {result.prediction_cutoff.isoformat()}")
    if windows is not None:
        print(f"LAST 5: home={len(windows.short_home)} away={len(windows.short_away)}")
        print(f"LAST 10: home={len(windows.recent_home)} away={len(windows.recent_away)}")
        print(f"LAST 20: home={len(windows.standard_home)} away={len(windows.standard_away)}")
        print(f"LONG TERM: home={len(windows.long_term_home)} away={len(windows.long_term_away)}")
    print(f"COMPETITION SAMPLE: {result.data_quality_report.competition_count}")
    bayesian_bundle = result.model_input_bundles.get("BAYESIAN_HIERARCHICAL_V1")
    bayesian_prior_count = (bayesian_bundle.bayesian_prior.sample_count
                            if bayesian_bundle and bayesian_bundle.bayesian_prior else 0)
    print(f"PRIOR SAMPLE: hierarchy={result.data_quality_report.prior_count} "
          f"bayesian_rate_prior={bayesian_prior_count}")
    print("MODEL READINESS:")
    for row in result.readiness_report.model_readiness:
        stages = row.stages or {}
        print(f"  {row.model_name}: training={stages.get('TRAINING_READY', 'UNKNOWN')} "
              f"artifact={stages.get('ARTIFACT_READY', 'UNKNOWN')} "
              f"input={stages.get('INPUT_READY', 'UNKNOWN')} "
              f"execution={row.status} output=NOT_RUN"
              + (f" reasons={';'.join(row.reasons)}" if row.reasons else ""))


def _replay_probe(evidence_db: Path, match_id: str, cutoff) -> int:
    """Validate stored as-of evidence without live fetching or probability generation."""
    config = OpenLigaDBConfig.from_yaml(ROOT / "config/phase13_5_provider.yaml")
    provider = OpenLigaDBProvider(config)
    store = EvidenceStore(evidence_db, provider.sources)
    try:
        bundle = AsOfEvidenceLoader(store).load_as_of(match_id, cutoff)
    except ReplayEvidenceUnavailable as error:
        print(f"REPLAY: {error}")
        return 2
    finally:
        store.close()
        provider.close()
    print(f"REPLAY EVIDENCE: fixture={len(bundle.fixture)} "
          f"historical_results={len(bundle.historical_results)} "
          f"other={len(bundle.other_evidence)}")
    print("REPLAY EXECUTION: NOT_IMPLEMENTED (evidence lookup only)")
    print("FINAL V7: NOT_READY (NOT FINAL V7 OUTPUT)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
