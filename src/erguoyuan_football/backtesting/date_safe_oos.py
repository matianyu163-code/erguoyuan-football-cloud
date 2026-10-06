"""Real DATE_SAFE_BATCH walk-forward for immutable match events only."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import duckdb

from erguoyuan_football.backtesting.base_model_backtest import (
    independent_poisson_benchmark,
    naive_league_frequency,
)
from erguoyuan_football.contracts.common import (
    Availability,
    ExecutionStatus,
    ImplementationType,
)
from erguoyuan_football.contracts.predictions import ModelPrediction
from erguoyuan_football.data.kickoff_enrichment import prepare_phase8_1_migration
from erguoyuan_football.data.schemas import Fixture
from erguoyuan_football.models.config import ModelConfig
from erguoyuan_football.models.dixon_coles import CoreDixonColesModel
from erguoyuan_football.models.elo import CoreEloModel
from erguoyuan_football.models.pi_rating import CorePiRatingModel
from erguoyuan_football.models.training import TrainingDataset, TrainingMatch
from erguoyuan_football.output_contract.adapters import CoreReportV2PreviewAdapter
from erguoyuan_football.output_contract.builder import CanonicalPredictionResultBuilder

LOGGER = logging.getLogger(__name__)
NEUTRAL_POLICY = "DOMESTIC_LEAGUE_HOME_ROLE_ASSUMED_NON_NEUTRAL"
PIPELINE_VERSION = "DATE_SAFE_BATCH_V1"


@dataclass(frozen=True)
class DateSafeOOSReport:
    status: str
    temporal_mode: str
    evaluation_start: date
    evaluation_end: date
    real_oos_rows_by_model: dict[str, int]
    unavailable_by_model: dict[str, int]
    metrics: tuple[dict[str, Any], ...]
    canonical_result_count: int
    canonical_preview: dict[str, Any] | None
    exact_utc_enriched_matches: int
    date_safe_eligible_matches: int
    ambiguous_enrichments: int
    unresolved_time_matches: int
    reasons: tuple[str, ...]


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _day_boundary(day: date) -> datetime:
    return datetime.combine(day, time.min, UTC)


def _training_row(row: tuple) -> TrainingMatch:
    match_id, competition_id, season_id, match_date, home_id, away_id, home_goals, away_goals, source, retrieved = row[:10]
    data_version = row[10]
    kickoff = _day_boundary(match_date)
    completed = kickoff + timedelta(days=1) - timedelta(microseconds=1)
    return TrainingMatch(
        match_id=match_id, competition_id=competition_id, season=season_id,
        kickoff_time=kickoff, home_team_id=home_id, away_team_id=away_id,
        home_goals=home_goals, away_goals=away_goals, neutral_venue=False,
        source=source, completed_at=completed, as_of_time=completed,
        retrieved_at=retrieved, data_version=data_version,
    )


def _prior_history_rows(rows: tuple[TrainingMatch, ...], target_date: date) -> tuple[TrainingMatch, ...]:
    """Return only event-date batches strictly before the target date."""
    return tuple(row for row in rows if row.kickoff_time.date() < target_date)


def _dataset(rows: tuple[TrainingMatch, ...], team_ids: frozenset[str]) -> TrainingDataset:
    return TrainingDataset(matches=rows, known_team_ids=team_ids, dataset_kind="REAL",
        temporal_mode="DATE_SAFE_BATCH", assumptions=(NEUTRAL_POLICY,))


def _predict_record(model_id: str, model_version: str, *, match_id: str,
                    competition_id: str, season_id: str, target_date: date,
                    training_data: TrainingDataset, cutoff: datetime,
                    snapshot_id: str, config_hash: str, sources: tuple[str, ...],
                    values: dict[str, Any],
                    prediction_time: datetime | None = None,
                    temporal_mode: str = "DATE_SAFE_BATCH") -> ModelPrediction:
    metadata = dict(values.pop("metadata", {}))
    matrix = values.pop("score_matrix", None)
    metadata.update({
        "data_origin": "REAL", "prediction_temporal_mode": temporal_mode,
        "match_date": target_date.isoformat(), "training_match_count": len(training_data.matches),
        "training_data_hash": training_data.data_hash,
        "training_match_ids_hash": _digest("|".join(sorted(row.match_id for row in training_data.matches))),
        "config_hash": config_hash, "neutral_venue_policy": NEUTRAL_POLICY,
        "score_matrix_tail_policy": "CONDITIONAL_RENORMALIZATION" if matrix is not None else None,
    })
    actual_prediction_time = prediction_time or _day_boundary(target_date)
    prediction_id = _digest(
        f"{model_id}|{model_version}|{match_id}|{snapshot_id}|{config_hash}|{training_data.data_hash}"
    )[:32]
    return ModelPrediction(
        prediction_id=prediction_id, match_id=match_id, prediction_snapshot_id=snapshot_id,
        model_id=model_id, model_version=model_version,
        implementation_type=ImplementationType.REAL_IMPLEMENTATION,
        training_end_time=cutoff, trained_until=cutoff, prediction_time=actual_prediction_time,
        input_data_version=training_data.data_hash,
        p_home=values["p_home"], p_draw=values["p_draw"], p_away=values["p_away"],
        lambda_home=values.get("lambda_home"), lambda_away=values.get("lambda_away"),
        expected_home_goals=values.get("expected_home_goals"),
        expected_away_goals=values.get("expected_away_goals"),
        score_matrix=matrix,
        data_source=sources or ("OPENFOOTBALL",), data_status=Availability.AVAILABLE,
        execution_status=ExecutionStatus.SUCCESS, is_oos=True,
        metadata=metadata,
    )


def _metrics(rows: list[tuple[float, float, float, int]]) -> dict[str, float | int]:
    if not rows:
        return {"sample_size": 0}
    logloss = brier = rps = correct = 0.0
    confidence_bins: dict[int, list[tuple[float, bool]]] = defaultdict(list)
    for home, draw, away, outcome in rows:
        probs = (home, draw, away)
        logloss -= math.log(max(probs[outcome], 1e-15))
        brier += sum((prob - int(index == outcome)) ** 2 for index, prob in enumerate(probs))
        cdf_home = home
        cdf_home_draw = home + draw
        rps += ((cdf_home - int(outcome <= 0)) ** 2 +
                (cdf_home_draw - int(outcome <= 1)) ** 2) / 2
        chosen = max(range(3), key=probs.__getitem__)
        confidence_bins[min(9, int(max(probs) * 10))].append((max(probs), chosen == outcome))
        correct += chosen == outcome
    n = len(rows)
    ece = sum(len(items) / n * abs(sum(p for p, _ in items) / len(items) -
                                   sum(ok for _, ok in items) / len(items))
              for items in confidence_bins.values())
    return {"sample_size": n, "log_loss": logloss / n, "brier": brier / n,
            "rps": rps / n, "accuracy": correct / n, "ece_top_label": ece}


def _build_models() -> tuple[tuple[str, type[Any]], ...]:
    return ((CoreEloModel.model_id, CoreEloModel),
            (CorePiRatingModel.model_id, CorePiRatingModel),
            (CoreDixonColesModel.model_id, CoreDixonColesModel))


def run_real_date_safe_oos(db_path: str | Path, *, evaluation_start: date = date(2025, 10, 1),
                           evaluation_end: date = date(2026, 6, 30),
                           minimum_history_matches: int = 80,
                           minimum_metric_sample: int = 100,
                           competitions: tuple[str, ...] = ("EPL", "LALIGA", "BUNDESLIGA", "SERIE_A", "LIGUE_1"),
                           ) -> DateSafeOOSReport:
    """Refit Elo/Pi daily; refit Dixon-Coles monthly; predict each date as one batch."""
    if evaluation_end < evaluation_start:
        raise ValueError("evaluation_end precedes evaluation_start")
    prepare_phase8_1_migration(db_path)
    connection = duckdb.connect(str(db_path))
    try:
        exact_count = connection.execute("SELECT count(*) FROM kickoff_enrichment_records WHERE verified").fetchone()[0]
        ambiguous_count = connection.execute("SELECT count(*) FROM kickoff_enrichment_review_queue "
                                              "WHERE reason='AMBIGUOUS_MATCH'").fetchone()[0]
        unresolved_count = connection.execute("""SELECT count(DISTINCT m.match_id)
            FROM real_canonical_matches m LEFT JOIN kickoff_enrichment_records e
              ON e.match_id=m.match_id AND e.verified AND e.enriched_kickoff_utc IS NOT NULL
            WHERE e.match_id IS NULL""").fetchone()[0]
        model_counts: dict[str, int] = defaultdict(int)
        unavailable: dict[str, int] = defaultdict(int)
        canonical_preview: dict[str, Any] | None = None
        builder = CanonicalPredictionResultBuilder()
        config = ModelConfig(min_matches=minimum_history_matches, min_team_matches=3,
                             training_window=1095, profile="production", allow_test_data=False)
        all_competitions = tuple(competition for competition in competitions if connection.execute(
            "SELECT 1 FROM real_canonical_matches WHERE competition_id=? AND match_date BETWEEN ? AND ? "
            "AND status='FINISHED' LIMIT 1", [competition, evaluation_start, evaluation_end]).fetchone())
        for competition_id in all_competitions:
            source_rows = connection.execute("""
                SELECT m.match_id, m.competition_id, m.season_id, m.match_date,
                       m.home_team_id, m.away_team_id, m.home_goals, m.away_goals,
                       'OPENFOOTBALL' AS source, m.retrieved_at, m.raw_hash, m.source_commit,
                       e.enriched_kickoff_utc, m.timestamp_precision
                FROM real_canonical_matches AS m
                LEFT JOIN (
                    SELECT match_id, min(enriched_kickoff_utc) AS enriched_kickoff_utc
                    FROM kickoff_enrichment_records
                    WHERE verified AND enriched_kickoff_utc IS NOT NULL
                    GROUP BY match_id
                    HAVING count(DISTINCT enriched_kickoff_utc)=1
                ) AS e USING(match_id)
                WHERE m.competition_id=? AND m.match_date<=? AND m.status='FINISHED'
                ORDER BY match_date, match_id
            """, [competition_id, evaluation_end]).fetchall()
            canonical_rows = tuple(_training_row(tuple(row)) for row in source_rows)
            teams = frozenset(row[0] for row in connection.execute(
                "SELECT team_id FROM real_canonical_teams WHERE competition_id=?", [competition_id]).fetchall())
            rows_by_date: dict[date, list[tuple]] = defaultdict(list)
            for raw in source_rows:
                if evaluation_start <= raw[3] <= evaluation_end:
                    rows_by_date[raw[3]].append(tuple(raw))
            eval_dates = [day for day in sorted(rows_by_date)
                          if evaluation_start <= day <= evaluation_end]
            if not eval_dates:
                continue
            month_dc: dict[tuple[int, int], tuple[CoreDixonColesModel | None, TrainingDataset | None,
                                                   datetime | None, str, tuple[str, ...]]] = {}
            for target_date in eval_dates:
                boundary = _day_boundary(target_date)
                target_rows = rows_by_date[target_date]
                target_ids = {row[0] for row in target_rows}
                prior = _prior_history_rows(canonical_rows, target_date)
                if any(row.match_id in target_ids for row in prior):
                    raise ValueError("SAME_DATE_TARGET_LEAKAGE")
                if len(prior) < minimum_history_matches:
                    for model_id, _ in (*_build_models(), ("NAIVE_LEAGUE_FREQUENCY_V1", None),
                                        ("SIMPLE_INDEPENDENT_POISSON_V1", None)):
                        unavailable[model_id] += len(target_rows)
                    continue
                daily_data = _dataset(prior, teams)
                day_hash = daily_data.data_hash
                exact_batch = all(row[12] is not None and
                                  row[12].astimezone(UTC) > boundary for row in target_rows)
                temporal_mode = "EXACT_UTC" if exact_batch else "DATE_SAFE_BATCH"
                snapshot_id = _digest(f"{temporal_mode}|{competition_id}|{target_date}|{day_hash}|"+
                                      "|".join(sorted(target_ids)))[:32]
                if not exact_batch:
                    connection.execute("""INSERT INTO real_date_safe_snapshots VALUES
                        (?, ?, 'DATE_SAFE_BATCH', ?, ?, ?, ?) ON CONFLICT DO NOTHING""",
                        [snapshot_id, target_date, source_rows[0][11], day_hash, datetime.now(UTC),
                         json.dumps({"source": "OPENFOOTBALL", "same_date_excluded": True,
                                     "training_data_hash": day_hash,
                                     "target_ids_hash": _digest("|".join(sorted(target_ids)))})])
                connection.execute("""INSERT INTO real_oos_prediction_snapshots VALUES
                    (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING""",
                    [snapshot_id, competition_id, target_date, temporal_mode,
                     _day_boundary(target_date), day_hash, datetime.now(UTC),
                     json.dumps({"same_date_batch": True, "training_data_hash": day_hash,
                                 "target_ids_hash": _digest("|".join(sorted(target_ids)))} )])
                for model_id, model_type in _build_models():
                    month = (target_date.year, target_date.month)
                    if model_id == CoreDixonColesModel.model_id:
                        cached = month_dc.get(month)
                        if cached is None:
                            month_start = date(target_date.year, target_date.month, 1)
                            monthly_prior = _prior_history_rows(canonical_rows, month_start)
                            if len(monthly_prior) < minimum_history_matches:
                                month_dc[month] = (None, None, None, "", ())
                            else:
                                data = _dataset(monthly_prior, teams)
                                cutoff = _day_boundary(month_start)
                                config_hash = _digest(config.config_hash + "|" + PIPELINE_VERSION + "|" + NEUTRAL_POLICY)
                                instance = CoreDixonColesModel()
                                try:
                                    instance.fit(data, cutoff, config)
                                    fitted_data = data.window(cutoff, config.training_window)
                                    month_dc[month] = (instance, fitted_data, cutoff, config_hash, instance.sources)
                                except (ValueError, RuntimeError, ArithmeticError) as error:
                                    LOGGER.warning("%s date-safe month fit unavailable: %s", model_id, error)
                                    month_dc[month] = (None, None, None, config_hash, ())
                        instance, fit_data, cutoff, config_hash, sources = month_dc[month]
                    else:
                        instance = model_type()
                        cutoff = boundary
                        fit_data = daily_data
                        config_hash = _digest(config.config_hash + "|" + PIPELINE_VERSION + "|" + NEUTRAL_POLICY)
                        try:
                            instance.fit(fit_data, cutoff, config)
                            fit_data = fit_data.window(cutoff, config.training_window)
                            sources = instance.sources
                        except (ValueError, RuntimeError, ArithmeticError) as error:
                            LOGGER.warning("%s date-safe fit unavailable for %s: %s", model_id, target_date, error)
                            instance = None
                            sources = ()
                    for raw in target_rows:
                        if instance is None or fit_data is None or cutoff is None:
                            unavailable[model_id] += 1
                            continue
                        if raw[0] in {row.match_id for row in fit_data.matches}:
                            raise ValueError("TARGET_MATCH_IN_TRAINING")
                        # Use the explicit UTC start-of-day cutoff for all rows in the batch.
                        # This is earlier than every verified same-day kickoff and ensures
                        # the full batch shares one auditable point-in-time snapshot.
                        prediction_time = boundary
                        if cutoff > prediction_time:
                            raise ValueError("TRAINING_CUTOFF_AFTER_PREDICTION")
                        # All rows from the date are predicted before any same-date result is admitted.
                        target_fixture = Fixture(
                            match_id=raw[0], competition_id=competition_id,
                            home_team_id=raw[4], away_team_id=raw[5],
                            kickoff_time=(raw[12].astimezone(UTC) if temporal_mode == "EXACT_UTC"
                                          else boundary + timedelta(days=1)),
                            source="OPENFOOTBALL", retrieved_at=boundary, as_of_time=boundary,
                            data_version=str(raw[10]), season=raw[2], neutral_venue=False)
                        try:
                            values = instance._predict_values(target_fixture)
                            record = _predict_record(model_id, instance.model_version,
                                match_id=raw[0], competition_id=competition_id, season_id=raw[2],
                                target_date=target_date, training_data=fit_data, cutoff=cutoff,
                                snapshot_id=snapshot_id, config_hash=config_hash,
                                sources=sources, values=values, prediction_time=prediction_time,
                                temporal_mode=temporal_mode)
                        except (KeyError, ValueError, RuntimeError, ArithmeticError) as error:
                            LOGGER.info("%s unavailable for %s: %s", model_id, raw[0], error)
                            unavailable[model_id] += 1
                            continue
                        _persist_oos(connection, record, competition_id=competition_id,
                                     season_id=raw[2], match_date=target_date,
                                     training_count=len(fit_data.matches), config_hash=config_hash,
                                     timestamp_precision=temporal_mode, kickoff_time=raw[12])
                        model_counts[model_id] += 1
                        if canonical_preview is None:
                            if temporal_mode == "EXACT_UTC":
                                canonical = builder.from_exact_oos_prediction(record,
                                    competition_id=competition_id, match_date=target_date,
                                    kickoff_time=raw[12].astimezone(UTC))
                            else:
                                canonical = builder.from_date_safe_prediction(record,
                                    competition_id=competition_id, match_date=target_date)
                            connection.execute("""INSERT INTO real_canonical_predictions_v2 VALUES
                                (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING""",
                                [canonical.prediction_id, canonical.match_id,
                                 canonical.prediction_snapshot_id, canonical.probability_stage.value,
                                 canonical.prediction_temporal_mode, canonical.data_origin, canonical.created_at,
                                 canonical.model_dump_json()])
                            canonical_preview = CoreReportV2PreviewAdapter().render(canonical)
                # Baselines use only the same prior-date history; no target-date update until loop completes.
                _run_baselines(connection, target_rows, daily_data, target_date, snapshot_id,
                               competition_id, config, temporal_mode=temporal_mode)
        metrics = _persist_metrics(connection, evaluation_start, evaluation_end, minimum_metric_sample)
        canonical_count = connection.execute("SELECT count(*) FROM real_canonical_predictions_v2 "
            "WHERE data_origin='REAL'").fetchone()[0]
        total_rows = connection.execute("SELECT count(*) FROM real_oos_predictions_v2 "
                                        "WHERE data_origin='REAL' AND is_oos AND match_date BETWEEN ? AND ?",
                                        [evaluation_start, evaluation_end]).fetchone()[0]
        reasons = []
        if total_rows == 0:
            reasons.append("NO_REAL_OOS_ROWS")
        if sum(model_counts.get(model_id, 0) >= 100 for model_id, _ in _build_models()) < 3:
            reasons.append("FEWER_THAN_THREE_MODELS_HAVE_100_OOS_ROWS")
        status = ("COMPLETE_WITH_DATE_SAFE_LIMITATION" if not reasons and exact_count == 0
                  else "PASS_WITH_WARNINGS" if not reasons else "PARTIAL")
        persisted_counts = dict(connection.execute("SELECT model_id, count(*) FROM real_oos_predictions_v2 "
            "WHERE data_origin='REAL' AND is_oos AND match_date BETWEEN ? AND ? GROUP BY model_id",
            [evaluation_start, evaluation_end]).fetchall())
        eligible = connection.execute("SELECT count(DISTINCT match_id) FROM real_oos_predictions_v2 "
            "WHERE prediction_temporal_mode='DATE_SAFE_BATCH' AND data_origin='REAL' AND is_oos "
            "AND match_date BETWEEN ? AND ?",
            [evaluation_start, evaluation_end]).fetchone()[0]
        temporal_modes = connection.execute("SELECT DISTINCT prediction_temporal_mode FROM real_oos_predictions_v2 "
            "WHERE data_origin='REAL' AND is_oos AND match_date BETWEEN ? AND ?",
            [evaluation_start, evaluation_end]).fetchall()
        reported_mode = temporal_modes[0][0] if len(temporal_modes) == 1 else "MIXED"
        return DateSafeOOSReport(status, reported_mode, evaluation_start, evaluation_end,
            persisted_counts, dict(unavailable), tuple(metrics), canonical_count,
            canonical_preview, exact_count, eligible, ambiguous_count, unresolved_count,
            tuple(reasons))
    finally:
        connection.close()


def _persist_oos(connection, prediction: ModelPrediction, *, competition_id: str,
                  season_id: str, match_date: date, training_count: int,
                  config_hash: str, timestamp_precision: str,
                  kickoff_time: datetime | None = None) -> None:
    connection.execute("""INSERT INTO real_oos_predictions_v2 VALUES
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, TRUE, 'REAL', ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING""",
        [prediction.prediction_id, prediction.match_id, competition_id, season_id,
         prediction.model_id, prediction.model_version, prediction.trained_until,
         prediction.prediction_time, timestamp_precision, kickoff_time, match_date,
         prediction.p_home, prediction.p_draw, prediction.p_away, training_count,
         prediction.input_data_version, config_hash, prediction.prediction_snapshot_id,
         timestamp_precision, prediction.model_dump_json()])


def _run_baselines(connection, target_rows: list[tuple], data: TrainingDataset,
                   target_date: date, snapshot_id: str, competition_id: str,
                   config: ModelConfig, *, temporal_mode: str) -> None:
    if not data.matches:
        return
    cutoff = _day_boundary(target_date)
    frequencies = naive_league_frequency(data.matches)
    poisson, matrix = independent_poisson_benchmark(data.matches)
    baseline_inputs = (
        ("NAIVE_LEAGUE_FREQUENCY_V1", {
            "p_home": frequencies.p_home, "p_draw": frequencies.p_draw, "p_away": frequencies.p_away}),
        ("SIMPLE_INDEPENDENT_POISSON_V1", {
            "p_home": poisson.p_home, "p_draw": poisson.p_draw, "p_away": poisson.p_away,
            "score_matrix": matrix}),
    )
    target_ids = {row[0] for row in target_rows}
    config_hash = _digest(config.config_hash + "|" + PIPELINE_VERSION + "|" + NEUTRAL_POLICY)
    for model_id, values in baseline_inputs:
        for raw in target_rows:
            record = _predict_record(model_id, PIPELINE_VERSION,
                match_id=raw[0], competition_id=competition_id, season_id=raw[2],
                target_date=target_date, training_data=data, cutoff=cutoff,
                snapshot_id=snapshot_id, config_hash=config_hash,
                sources=("OPENFOOTBALL",), values=dict(values), temporal_mode=temporal_mode)
            if raw[0] in target_ids and raw[0] in {row.match_id for row in data.matches}:
                raise ValueError("TARGET_MATCH_IN_TRAINING")
            _persist_oos(connection, record, competition_id=competition_id,
                         season_id=raw[2], match_date=target_date,
                         training_count=len(data.matches), config_hash=config_hash,
                         timestamp_precision=temporal_mode, kickoff_time=raw[12])


def _persist_metrics(connection, start: date, end: date,
                     minimum_sample: int) -> list[dict[str, Any]]:
    groups = connection.execute("""
        SELECT p.model_id, p.prediction_temporal_mode, p.competition_id, p.season_id,
               p.p_home, p.p_draw, p.p_away, m.home_goals, m.away_goals,
               p.prediction_id, p.model_version, p.config_hash
        FROM real_oos_predictions_v2 p
        JOIN real_canonical_matches m USING(match_id)
        WHERE p.data_origin='REAL' AND p.is_oos AND p.match_date BETWEEN ? AND ?
        ORDER BY p.model_id, p.competition_id, p.match_date, p.match_id
    """, [start, end]).fetchall()
    buckets: dict[tuple[str, str, str, str], list[tuple[float, float, float, int]]] = defaultdict(list)
    versions: dict[tuple[str, str, str, str], tuple[str, str]] = {}
    for row in groups:
        model_id, temporal_mode, comp, season, home, draw, away, hg, ag, _pid, version, config_hash = row
        outcome = 0 if hg > ag else 1 if hg == ag else 2
        for key in ((model_id, temporal_mode, comp, season),
                    (model_id, temporal_mode, "ALL", "ALL")):
            buckets[key].append((home, draw, away, outcome))
            versions[key] = (version, config_hash)
    reports = []
    for key, values in sorted(buckets.items()):
        scores = _metrics(values)
        model_id, mode, comp, season = key
        status = "PASS" if len(values) >= minimum_sample else "INSUFFICIENT_SAMPLE"
        record_id = _digest("|".join(key + (versions[key][1],)))[:32]
        payload = {**scores, "model_version": versions[key][0], "config_hash": versions[key][1],
                   "evaluation_window": [start.isoformat(), end.isoformat()]}
        connection.execute("""INSERT INTO real_oos_metrics VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(model_id, temporal_mode, competition_id, season_id)
            DO UPDATE SET sample_size=excluded.sample_size,status=excluded.status,
                          metrics=excluded.metrics,created_at=excluded.created_at""",
            [record_id, model_id, mode, comp, season, len(values), status,
             json.dumps(payload, allow_nan=False), datetime.now(UTC)])
        reports.append({"model_id": model_id, "temporal_mode": mode, "competition": comp,
                        "season": season, "status": status, **scores})
    return reports


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Real date-safe chronological OOS; no market/forecast snapshots")
    parser.add_argument("--db", type=Path, default=Path("data/football.duckdb"))
    parser.add_argument("--from-date", type=date.fromisoformat, default=date(2025, 10, 1))
    parser.add_argument("--to-date", type=date.fromisoformat, default=date(2026, 6, 30))
    parser.add_argument("--minimum-history-matches", type=int, default=80)
    parser.add_argument("--minimum-metric-sample", type=int, default=100)
    parser.add_argument("--competition", action="append")
    args = parser.parse_args(argv)
    report = run_real_date_safe_oos(args.db, evaluation_start=args.from_date,
        evaluation_end=args.to_date, minimum_history_matches=args.minimum_history_matches,
        minimum_metric_sample=args.minimum_metric_sample,
        competitions=tuple(args.competition) if args.competition else
            ("EPL", "LALIGA", "BUNDESLIGA", "SERIE_A", "LIGUE_1"))
    print(json.dumps(asdict(report), default=str, ensure_ascii=False, indent=2))
    return 0 if report.real_oos_rows_by_model else 2


if __name__ == "__main__":
    raise SystemExit(main())
