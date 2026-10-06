"""Run or resume frozen Phase 9 research and generate the CORE V2 preview."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date
from pathlib import Path

import yaml

from erguoyuan_football.meta.artifact import save_artifact
from erguoyuan_football.meta.checkpoint import (
    adopt_existing_phase9_result,
    load_phase9_checkpoint,
    save_phase9_checkpoint,
)
from erguoyuan_football.meta.output import build_development_core_many
from erguoyuan_football.meta.pipeline import run_phase9_development
from erguoyuan_football.meta.promotion import MetaPromotionGate
from erguoyuan_football.meta.runtime import (
    ArtifactIndex,
    CoreRuntimeContext,
    LoadedArtifactCache,
    PerformanceTelemetry,
    peak_working_set_bytes,
)
from erguoyuan_football.ml.schemas import stable_hash
from erguoyuan_football.output_contract.adapters import CoreReportV2PreviewAdapter


def main() -> int:
    """Train once or resume a verified checkpoint; never fit during resume."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--config", default="config/phase9_development.yaml")
    parser.add_argument("--artifact-root", default="artifacts/phase9/meta_no_market")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--resume", action="store_true",
                        help="reuse a compatible research checkpoint without fitting")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[3]
    workspace = Path(os.environ.get("WORKSPACE_ROOT", project_root / ".workspace"))
    index = ArtifactIndex(workspace / "artifacts" / "phase9" / "index.json")
    cache = LoadedArtifactCache()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    config_hash = stable_hash(config)
    checkpoint_path = workspace / "checkpoints" / "phase9_research.json"
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "phase9_development_results.json"

    operation_started = time.perf_counter()
    cache_hit = False
    if args.resume:
        if not checkpoint_path.is_file():
            adopt_existing_phase9_result(
                db_path=args.db, config_path=args.config,
                artifact_root=args.artifact_root, report_path=report_path,
                checkpoint_path=checkpoint_path, artifact_index=index,
                artifact_cache=cache)
        result, artifact, cache_hit = load_phase9_checkpoint(
            db_path=args.db, config_path=args.config, checkpoint_path=checkpoint_path,
            artifact_index=index, artifact_cache=cache)
        report = dict(result.report)
    else:
        result = run_phase9_development(args.db, config_path=args.config)
        artifact = save_artifact(result, root=args.artifact_root)
        index.register(artifact.artifact_id, artifact.path)
        cache.remember_verified(artifact)
        report = dict(result.report)
        report["meta_artifact_id"] = artifact.artifact_id
        report["calibrator_id"] = artifact.manifest["calibrator_id"]
        report["promotion_decision"] = MetaPromotionGate().evaluate(
            report, artifact_verified=True).model_dump(mode="json")

    operation_duration = (time.perf_counter() - operation_started) * 1000
    context = CoreRuntimeContext.create(config=result.config, config_hash=config_hash,
        database_path=args.db, artifact_index=index, artifact_cache=cache,
        meta_artifact=artifact)
    inference_started = time.perf_counter()
    outputs = build_development_core_many(args.db, candidate=result.candidate, artifact=artifact,
        match_ids=[str(value) for value in result.calibration_eval_rows.frame["match_id"]],
        evaluation_start=date(2026, 5, 1), evaluation_end=date(2026, 5, 31))
    inference_duration = (time.perf_counter() - inference_started) * 1000
    if len(outputs) != result.calibration_eval_rows.count:
        raise ValueError("PHASE9_CORE_BATCH_SIZE_MISMATCH")
    report["calibrated_core_count"] = len(outputs)
    report["phase10_software_development_ready"] = bool(outputs) and (
        report["recursive_lineage_validation"]["status"] == "PASS")
    if not args.resume:
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    preview = CoreReportV2PreviewAdapter()
    output = outputs[0]
    (report_dir / "phase9_development_core_example.json").write_text(
        json.dumps({"core": output.core.model_dump(mode="json"),
                    "meta_raw": output.meta_raw.model_dump(mode="json"),
                    "calibrated_core": output.calibrated_core.model_dump(mode="json"),
                    "disagreement": output.disagreement.model_dump(mode="json"),
                    "canonical": preview.render(output.canonical)},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    (report_dir / "phase9_development_core_preview.jsonl").write_text(
        "\n".join(json.dumps(preview.render(item.canonical), ensure_ascii=False)
                  for item in outputs) + "\n", encoding="utf-8")
    if not args.resume:
        save_phase9_checkpoint(result=result, artifact=artifact, report_path=report_path,
            checkpoint_path=checkpoint_path, artifact_index=index)

    memory_peak = peak_working_set_bytes()
    telemetry = PerformanceTelemetry(workspace / "logs" / "phase9_runtime.jsonl")
    telemetry.record(stage="phase9_development_or_resume", duration_ms=operation_duration,
        rows=result.model.training_sample_count, db_queries=0, cache_hit=False,
        artifact_cache_hit=cache_hit, memory_peak_bytes=memory_peak,
        dataset_hash=result.candidate.data_hash, artifact_id=artifact.artifact_id,
        calibrator_id=artifact.manifest["calibrator_id"], run_id=context.run_id)
    telemetry.record(stage="phase9_core_batch", duration_ms=inference_duration,
        rows=len(outputs), db_queries=1, cache_hit=False, artifact_cache_hit=True,
        memory_peak_bytes=memory_peak, dataset_hash=result.candidate.data_hash,
        artifact_id=artifact.artifact_id, calibrator_id=artifact.manifest["calibrator_id"],
        run_id=context.run_id)
    print(json.dumps({"overall_status": report["status"], "reason": report["reason"],
        "artifact_id": artifact.artifact_id, "calibrator_id": artifact.manifest["calibrator_id"],
        "candidate_dataset_id": result.candidate.dataset_id,
        "selected_research_dataset": report["selected_research_dataset"],
        "selected_calibration_method": report["selected_calibration_method"],
        "development_common_sample": report["common_sample_development"]["match_count"],
        "calibrated_core_count": len(outputs), "resumed": args.resume,
        "run_id": context.run_id, "final_holdout_performance": "UNAVAILABLE"},
        ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
