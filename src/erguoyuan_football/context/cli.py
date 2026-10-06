"""DATE_SAFE Phase 10 tournament/context reconstruction command line."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any


# Set all temporary locations before importing modules that transitively load scientific libraries.
def _configure_workspace() -> None:
    root = Path(__file__).resolve().parents[3]
    workspace = root / ".workspace"
    for name in ("temp", "duckdb", "matplotlib", "joblib"):
        (workspace / name).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TEMP", str(workspace / "temp"))
    os.environ.setdefault("TMP", str(workspace / "temp"))
    os.environ.setdefault("MPLCONFIGDIR", str(workspace / "matplotlib"))
    os.environ.setdefault("JOBLIB_TEMP_FOLDER", str(workspace / "joblib"))
    os.environ.setdefault("DUCKDB_TEMP_DIRECTORY", str(workspace / "duckdb"))
    os.environ.setdefault("CORE_DUCKDB_THREADS", "2")


_configure_workspace()

import yaml

from erguoyuan_football.context.artifacts import (
    ContextStore,
    load_real_context_events,
)
from erguoyuan_football.context.competition_rules import CompetitionRuleRegistry
from erguoyuan_football.context.diagnostics import summarize_assessments
from erguoyuan_football.context.features import FEATURE_SCHEMA_HASH
from erguoyuan_football.context.pipeline import ContextPipeline, TargetContextMatch
from erguoyuan_football.meta.runtime import peak_working_set_bytes
from erguoyuan_football.ml.schemas import stable_hash
from erguoyuan_football.output_contract.schemas import CanonicalPredictionResult


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 10 DATE_SAFE context evaluation")
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/context.yaml"))
    parser.add_argument("--from-date", type=date.fromisoformat)
    parser.add_argument("--to-date", type=date.fromisoformat)
    parser.add_argument("--match-id", action="append", default=[])
    parser.add_argument("--base-predictions", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def _load_predictions(path: Path, *, start: date, end: date,
                      match_ids: set[str]) -> tuple[CanonicalPredictionResult, ...]:
    rows: list[CanonicalPredictionResult] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = CanonicalPredictionResult.model_validate_json(line)
            if item.match_date is None or not start <= item.match_date <= end:
                continue
            if match_ids and item.match_id not in match_ids:
                continue
            if item.match_date >= date(2026, 8, 1):
                raise ValueError("FINAL_HOLDOUT_ACCESS_BLOCKED")
            rows.append(item)
    if not rows:
        raise ValueError("NO_ELIGIBLE_DEVELOPMENT_BASE_PREDICTIONS")
    return tuple(rows)


def _load_targets(db: Path, predictions: tuple[CanonicalPredictionResult, ...]
                  ) -> dict[str, TargetContextMatch]:
    import duckdb

    connection = duckdb.connect(str(db), read_only=True)
    try:
        ids = [row.match_id for row in predictions]
        marks = ",".join("?" for _ in ids)
        rows = connection.execute(f"""SELECT match_id, competition_id, season_id,
            home_team_id, away_team_id, match_date FROM real_canonical_matches
            WHERE match_id IN ({marks})""", ids).fetchall()
    finally:
        connection.close()
    targets = {row[0]: TargetContextMatch(match_id=row[0], competition_id=row[1],
        season_id=row[2], home_team_id=row[3], away_team_id=row[4], match_date=row[5])
        for row in rows}
    missing = {row.match_id for row in predictions} - targets.keys()
    if missing:
        raise ValueError(f"CONTEXT_TARGETS_MISSING:{len(missing)}")
    return targets


def _canonical_summary(output: Any) -> dict[str, Any]:
    """Return a compact audit view while the complete CORE V2 payload is persisted."""
    item = output.canonical
    return {
        "prediction_id": item.prediction_id,
        "match_id": item.match_id,
        "prediction_snapshot_id": item.prediction_snapshot_id,
        "probability_stage": item.probability_stage.value if item.probability_stage else None,
        "context_status": item.context_status,
        "production_status": item.production_status,
        "base_probability": {"home": item.p_home, "draw": item.p_draw, "away": item.p_away},
        "final_probability": output.assessment.ledger.final_probability,
        "audit_status": output.audit.get("status"),
    }


def run(argv: list[str] | None = None) -> dict[str, Any]:
    """Execute a real-data development reconstruction and return an audit summary."""
    _configure_workspace()
    args = _parser().parse_args(argv)
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    window = config["development_window"]
    start = args.from_date or date.fromisoformat(str(window["start"]))
    end = args.to_date or date.fromisoformat(str(window["end"]))
    if start < date.fromisoformat(str(window["start"])) or end > date.fromisoformat(str(window["end"])):
        raise ValueError("REQUEST_OUTSIDE_PHASE10_DEVELOPMENT_WINDOW")
    if start > end:
        raise ValueError("INVALID_DATE_RANGE")
    if args.resume and args.dry_run:
        raise ValueError("RESUME_AND_DRY_RUN_ARE_MUTUALLY_EXCLUSIVE")
    base_path = (args.base_predictions or Path(config["base_predictions"])).resolve()
    predictions = _load_predictions(base_path, start=start, end=end,
                                    match_ids=set(args.match_id))
    targets = _load_targets(args.db.resolve(), predictions)
    dataset = load_real_context_events(args.db, before_date=end)
    rules_path = (config_path.parent.parent / str(config["rules"])).resolve()
    rules = CompetitionRuleRegistry.from_yaml(rules_path)
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    rule_hash = hashlib.sha256(rules_path.read_bytes()).hexdigest()
    run_id = stable_hash({"phase": 10, "config_hash": config_hash,
        "pipeline_version": str(config["version"]),
        "feature_schema_hash": FEATURE_SCHEMA_HASH,
        "rule_hash": rule_hash, "dataset_hash": dataset.data_hash,
        "prediction_ids": [item.prediction_id for item in predictions]})[:32]
    if args.resume and not args.dry_run:
        with ContextStore(args.db) as store:
            existing = store.load_run(run_id)
            if existing is not None:
                return {**existing[0], "resumed": True,
                        "canonical_output_count": len(existing[1])}
    pipeline = ContextPipeline(rules=rules,
        context_model_status=str(config["context_adjustment"]["model_version"]))
    started = time.perf_counter()
    outputs = pipeline.build_many(base_predictions=predictions, matches=targets,
                                  events=dataset.events)
    elapsed_ms = (time.perf_counter() - started) * 1000
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_hash": dataset.data_hash,
        "base_predictions_hash": stable_hash([item.model_dump(mode="json") for item in predictions]),
        "config_hash": config_hash,
        "pipeline_version": str(config["version"]),
        "feature_schema_hash": FEATURE_SCHEMA_HASH,
        "rule_registry_hash": rule_hash,
        "source": dataset.source,
        "temporal_mode": "DATE_SAFE_BATCH",
        "target_count": len(outputs),
        "historical_event_count": len(dataset.events),
        "db_query_count": 2,
        "elapsed_ms": round(elapsed_ms, 3),
        "peak_working_set_bytes": peak_working_set_bytes(),
        "status": "DEVELOPMENT_ONLY_NOT_PROMOTED",
        "diagnostics": summarize_assessments(output.assessment for output in outputs),
        "audit_pass_count": sum(output.audit.get("status") == "PASS" for output in outputs),
        "audit_fail_count": sum(output.audit.get("status") != "PASS" for output in outputs),
        "phase9_final_holdout_rows_read": 0,
        "production_promoted": False,
    }
    if args.dry_run:
        return {**manifest, "dry_run": True,
                "canonical_outputs": [_canonical_summary(output) for output in outputs]}
    args.db.resolve().parent.joinpath(".workspace").mkdir(parents=True, exist_ok=True)
    with ContextStore(args.db) as store:
        store.transaction()
        try:
            for output in outputs:
                store.save_assessment(output.assessment,
                    output.canonical.model_dump(mode="json"), run_id=run_id)
            store.save_run(run_id=run_id, created_at=datetime.now(UTC),
                dataset_hash=dataset.data_hash, target_count=len(outputs),
                status="DEVELOPMENT_ONLY_NOT_PROMOTED", payload=manifest)
            store.commit()
        except Exception:
            store.rollback()
            raise
        manifest["table_counts"] = store.table_counts()
    manifest["canonical_outputs"] = [_canonical_summary(output) for output in outputs]
    return manifest


def main(argv: list[str] | None = None) -> int:
    """Console entry point with structured errors and nonzero failure exit."""
    try:
        result = run(argv)
    except (OSError, ValueError, KeyError, yaml.YAMLError) as error:
        print(json.dumps({"status": "FAILED", "reason": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
