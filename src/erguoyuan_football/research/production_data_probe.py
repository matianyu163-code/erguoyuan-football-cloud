"""Read-only Phase 14.1 production data probe; model execution is prohibited."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from erguoyuan_football.prediction.model_execution_planner import (
    ModelExecutionPlanner,
)
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
from erguoyuan_football.research.pipeline_execution_record import (
    PipelineExecutionStore,
)
from erguoyuan_football.research.production_match_pipeline import (
    ProductionMatchPipeline,
    ProductionMatchRequest,
)
from erguoyuan_football.web_research.evidence.evidence_store import EvidenceStore
from erguoyuan_football.web_research.time_utils import parse_utc, utc_iso

ROOT = Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> int:
    """Print pipeline data status; no probability or V5.1 formatter is imported."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("match", help='e.g. "Arsenal vs Liverpool"')
    parser.add_argument("--competition", default="Premier League")
    parser.add_argument("--cutoff", required=True, help="Explicit UTC ISO-8601 cutoff")
    parser.add_argument("--mode", choices=("LIVE", "REPLAY"), default="LIVE")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--audit-db", type=Path,
                        default=ROOT / "data/production_pipeline_audit.sqlite")
    args = parser.parse_args(argv)
    try:
        cutoff = parse_utc(args.cutoff)
    except ValueError as error:
        parser.error(f"invalid UTC cutoff: {error}")

    config = OpenLigaDBConfig.from_yaml(ROOT / "config/phase13_5_provider.yaml")
    provider = OpenLigaDBProvider(config)
    evidence_store = EvidenceStore(ROOT / "data/research_evidence.sqlite",
                                   provider.sources)
    audit_store = PipelineExecutionStore(args.audit_db)
    try:
        gate = LiveDataReadinessGate(
            ReadinessConfig.from_yaml(ROOT / "config/phase13_5_readiness.yaml"),
            load_model_requirements(ROOT / "config/model_registry.yaml"),
        )
        pipeline = ProductionMatchPipeline(provider, evidence_store, gate,
            planner=ModelExecutionPlanner(ROOT / "config/phase14_model_requirements.yaml",
                                          ROOT / "config/model_registry.yaml"),
            audit_store=audit_store)
        result = pipeline.build(ProductionMatchRequest(args.match, args.competition),
                                prediction_cutoff=cutoff, mode=args.mode)
    except (ValueError, OSError, RuntimeError) as error:
        if args.as_json:
            print(json.dumps({"status": "FAILED", "error": str(error),
                              "prediction_executed": False}, ensure_ascii=False))
        else:
            print(f"PIPELINE: FAILED ({error})")
            print("PREDICTION: NOT EXECUTED")
        return 2
    finally:
        audit_store.close()
        evidence_store.close()
        provider.close()

    if args.as_json:
        print(json.dumps(_json_result(result), ensure_ascii=False, default=str))
        return 0
    if result.match_source_audit is not None:
        print(f"MATCH SOURCE: {result.match_source_audit.source_type.value} "
              f"| confidence={result.match_source_audit.confidence:.0%}")
    if result.jc_verification is not None:
        print(f"JC STATUS: {result.jc_verification.status.value} "
              f"| source={result.jc_verification.source or 'UNAVAILABLE'} "
              f"| confidence={result.jc_verification.confidence_score:.0%}")
    print(f"MATCH: {result.canonical_match.match_id}")
    print("ENTITY STATUS: VERIFIED")
    print(f"FIXTURE STATUS: VERIFIED ({result.canonical_match.provider_match_ids})")
    print(f"HISTORY: {len(result.historical_matches)} accepted final results")
    print("ACQUISITION: " + ", ".join(
        f"{key}={value}" for key, value in result.acquisition_metrics.items()))
    print(f"DIRECT: home={result.data_quality_report.direct_home_count}, "
          f"away={result.data_quality_report.direct_away_count}")
    print(f"COMPETITION: {result.data_quality_report.competition_count}")
    print(f"COMPARABLE: {result.data_quality_report.comparable_count}")
    print(f"PRIOR: {result.data_quality_report.prior_count}")
    print(f"DEDUPLICATED: {result.data_quality_report.deduplicated_count}; "
          f"CONFLICTS: {result.data_quality_report.conflict_count}; "
          f"STALE: {result.data_quality_report.stale_count}")
    print("MODEL INPUTS: " + ", ".join(
        f"{row.model_id}={row.input_status}/{row.execution_status}"
        + (f"[{row.reason}]" if row.reason else "")
        for row in result.model_input_readiness))
    print("EXECUTION PLAN: " + ", ".join(
        f"{row.model_id}={row.action}"
        + (f"[{';'.join(row.reasons)}]" if row.reasons else "")
        for row in result.execution_plan.entries))
    print(f"PRODUCTION DATA STATUS: {result.production_readiness.status}")
    print("PREDICTION: NOT EXECUTED")
    return 0


def _json_result(result) -> dict[str, object]:
    """Serialize only stable diagnostic fields, retaining null/unavailable states."""
    return {
        "research_session_id": result.research_session_id,
        "match": asdict(result.canonical_match),
        "match_source": (asdict(result.match_source_audit)
                         if result.match_source_audit is not None else None),
        "jc_verification": (asdict(result.jc_verification)
                            if result.jc_verification is not None else None),
        "prediction_cutoff": utc_iso(result.prediction_cutoff),
        "mode": result.mode,
        "history_count": len(result.historical_matches),
        "sample_counts": asdict(result.data_quality_report),
        "model_inputs": [asdict(row) for row in result.model_input_readiness],
        "execution_plan": [asdict(row) for row in result.execution_plan.entries],
        "production_data_status": asdict(result.production_readiness),
        "evidence_ids": result.evidence_ids,
        "provider_ids": result.provider_ids,
        "acquisition_metrics": result.acquisition_metrics,
        "stages": [asdict(row) for row in result.stages],
        "prediction_executed": result.prediction_executed,
        "generated_at": result.generated_at.isoformat(),
    }


if __name__ == "__main__":
    raise SystemExit(main())
