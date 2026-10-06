"""Development-only Phase 11 batch from real Phase 9/10 inputs."""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import duckdb

from erguoyuan_football.context.artifacts import load_real_context_events
from erguoyuan_football.context.competition_rules import CompetitionRuleRegistry
from erguoyuan_football.context.pipeline import ContextPipeline, TargetContextMatch
from erguoyuan_football.output_contract.schemas import CanonicalPredictionResult
from erguoyuan_football.prediction_heads.core import EffectiveCoreProbabilityResolver
from erguoyuan_football.prediction_heads.heads import score_top2, total_top2
from erguoyuan_football.prediction_heads.readiness import (
    ScoreMatrixReadinessAudit,
    ScoreMatrixSourceSelector,
)
from erguoyuan_football.prediction_heads.reconcile import OutcomeMassReconciler
from erguoyuan_football.report.assembler import CoreReportV2Assembler
from erguoyuan_football.report.schemas import CoreReportV2Schema
from erguoyuan_football.selection.schemas import MatchHeads


def load_development_predictions(path: Path, count: int) -> tuple[CanonicalPredictionResult, ...]:
    """Read at most the requested real May development rows; never enter final holdout."""
    rows: list[CanonicalPredictionResult] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = CanonicalPredictionResult.model_validate_json(line)
            if row.match_date is None or row.match_date >= date(2026, 8, 1):
                raise ValueError("FINAL_HOLDOUT_ACCESS_BLOCKED")
            if row.data_origin != "REAL" or row.production_status != "NOT_PROMOTED" or (
                    row.validation_status != "DEVELOPMENT_ONLY" or
                    row.prediction_temporal_mode != "DATE_SAFE_BATCH"):
                raise ValueError("PHASE11_DEVELOPMENT_STATUS_REQUIRED")
            rows.append(row)
            if len(rows) == count:
                break
    if len(rows) != count:
        raise ValueError("INSUFFICIENT_REAL_CANONICAL_PREDICTIONS")
    return tuple(rows)


def run_development_batch(*, db_path: Path, predictions_path: Path, rules_path: Path,
                          count: int, required_ev: float = 0.03
                          ) -> tuple[CoreReportV2Schema, dict[str, float | int]]:
    """Execute the full four-layer software path without live market or promotion."""
    if not 1 <= count <= 167:
        raise ValueError("PHASE11_COUNT_OUT_OF_RANGE")
    timings: dict[str, float | int] = {}
    started = time.perf_counter()
    bases = load_development_predictions(predictions_path, count)
    ids = [item.match_id for item in bases]
    with duckdb.connect(str(db_path), read_only=True) as db:
        db.execute("SET TimeZone='UTC'")
        rows = db.execute("""SELECT m.match_id,m.competition_id,m.season_id,
            m.home_team_id,m.away_team_id,m.match_date,h.team_name,a.team_name
            FROM real_canonical_matches m
            LEFT JOIN real_canonical_teams h ON h.team_id=m.home_team_id
            LEFT JOIN real_canonical_teams a ON a.team_id=m.away_team_id
            WHERE m.match_id IN (SELECT unnest(?)) AND m.match_date < DATE '2026-08-01'""",
            [ids]).fetchall()
    indexed = {r[0]: r for r in rows}
    if set(ids) != set(indexed):
        raise ValueError("PHASE11_TARGET_MATCH_MISSING")
    targets = {r[0]: TargetContextMatch(match_id=r[0], competition_id=r[1], season_id=r[2],
        home_team_id=r[3], away_team_id=r[4], match_date=r[5]) for r in rows}
    dataset = load_real_context_events(db_path, before_date=max(base.match_date for base in bases
                                                                  if base.match_date is not None))
    context = ContextPipeline(rules=CompetitionRuleRegistry.from_yaml(rules_path))
    outputs = context.build_many(base_predictions=bases, matches=targets, events=dataset.events)
    canonicals = tuple(output.canonical for output in outputs)
    selection = ScoreMatrixSourceSelector(db_path).select(
        evaluation_start=date(2026, 1, 1), evaluation_end=date(2026, 5, 1),
        target_start=min(base.match_date for base in bases if base.match_date is not None))
    matrices = ScoreMatrixReadinessAudit(db_path).load_many(canonicals, selection)
    resolver = EffectiveCoreProbabilityResolver()
    reconciler = OutcomeMassReconciler()
    heads: list[MatchHeads] = []
    for canonical in canonicals:
        assert canonical.match_date is not None
        row = indexed[canonical.match_id]
        effective = resolver.resolve(canonical)
        source = matrices.get(canonical.match_id)
        score = total = None
        final = None
        matrix_status = "UNAVAILABLE"
        reasons = []
        if source is not None:
            try:
                final = reconciler.reconcile(source.matrix, effective.probability)
                score = score_top2(final.final_matrix)
                total = total_top2(final.final_matrix)
                matrix_status = "AVAILABLE_RECONCILED"
            except ValueError as error:
                final = None
                matrix_status = "RECONCILIATION_FAILED"
                reasons.append(str(error))
        else:
            reasons.append("REAL_OOS_SCORE_MATRIX_UNAVAILABLE")
        heads.append(MatchHeads(match_id=canonical.match_id, competition_id=canonical.competition_id,
            match_date=canonical.match_date, kickoff_time=canonical.kickoff_time,
            kickoff_status="EXACT_UTC_VERIFIED" if canonical.kickoff_time else
                "DATE_SAFE_BATCH_EXACT_KICKOFF_UNAVAILABLE",
            home_team_id=row[3], away_team_id=row[4],
            home_team_name=row[6] or row[3], away_team_name=row[7] or row[4],
            prediction_snapshot_id=canonical.prediction_snapshot_id,
            probability=effective.probability, probability_source_stage=effective.source_stage,
            probability_source_id=effective.source_id, score_top2=score, totals_top2=total,
            matrix_source=source, matrix_status=matrix_status,
            reconciliation=final,
            context_status=canonical.context_status, models_failed=canonical.models_failed,
            models_unavailable=canonical.models_unavailable,
            reason_codes=tuple(reasons), validation_status=effective.validation_status,
            production_status=effective.production_status, temporal_mode=effective.temporal_mode,
            data_origin=canonical.data_origin))
    timings["prediction_heads_ms"] = round((time.perf_counter() - started) * 1000, 3)
    status = {
        "probability_source": "CALIBRATED_CORE_OR_VERIFIED_CONTEXT_ADJUSTMENT",
        "matrix_source_model": selection.model_id,
        "matrix_source_evaluated_until": selection.evaluated_until.isoformat(),
        "matrix_source_selection_sample_count": selection.sample_count,
        "matrix_source_selection_metrics": selection.metrics,
        "validation_status": "DEVELOPMENT_ONLY", "production_status": "NOT_PROMOTED",
        "temporal_mode": "DATE_SAFE_BATCH", "live_production_ready": False,
        "market_status": "UNAVAILABLE", "score_matrix_ready_rows": sum(
            item.matrix_status == "AVAILABLE_RECONCILED" for item in heads),
        "htft_status": "UNAVAILABLE", "official_handicap_status": "UNAVAILABLE",
        "context_status": "EVALUATED_NO_ADJUSTMENT",
        "lineup_status": "UNAVAILABLE", "injury_status": "UNAVAILABLE",
        "models_failed": sorted({model for item in heads for model in item.models_failed}),
        "models_unavailable": sorted({model for item in heads for model in item.models_unavailable}),
        "matches_excluded": [], "reason_codes": sorted({reason for item in heads
            for reason in item.reason_codes} | {"MARKET_UNAVAILABLE", "OFFICIAL_HANDICAP_UNAVAILABLE",
                "HTFT_INDEPENDENT_MODEL_UNAVAILABLE"}), "final_holdout_rows_read": 0,
    }
    # No historical market snapshots exist for DATE_SAFE_BATCH rows.
    report, assembly_timings = CoreReportV2Assembler(required_ev=required_ev).build(
        heads=tuple(heads), status=status)
    timings.update(assembly_timings)
    timings["db_query_count"] = 6
    return report, timings
