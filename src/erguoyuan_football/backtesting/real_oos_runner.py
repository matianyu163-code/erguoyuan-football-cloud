"""Registry-driven, resumable DATE_SAFE_BATCH execution for existing base models."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, cast

import duckdb

from erguoyuan_football.backtesting.date_safe_oos import (
    NEUTRAL_POLICY,
    _dataset,
    _day_boundary,
    _digest,
    _persist_oos,
    _predict_record,
    _prior_history_rows,
    _training_row,
)
from erguoyuan_football.backtesting.phase8_2_store import prepare_phase8_2_migration
from erguoyuan_football.backtesting.real_oos_matrix import BASE_MODEL_IDS, lineage_hash
from erguoyuan_football.contracts.common import ImplementationType
from erguoyuan_football.contracts.predictions import ModelPrediction
from erguoyuan_football.data.schemas import Fixture
from erguoyuan_football.models.artifact_index import ArtifactIndex
from erguoyuan_football.models.base import BaseFootballModel
from erguoyuan_football.models.config import ModelConfig
from erguoyuan_football.models.registry import ModelRegistry
from erguoyuan_football.models.training import InsufficientData, TrainingDataset

LOGGER = logging.getLogger(__name__)
RUNNER_VERSION = "UNIFIED_REAL_OOS_V1"
RefitPolicy = Literal["DAILY", "MONTHLY", "QUARTERLY"]


@dataclass(frozen=True)
class ModelCapability:
    """Historical-data requirements and legal execution mode for a registered model."""

    model_id: str
    supports_date_safe: bool
    supports_exact_utc: bool
    requires_market: bool
    requires_xg: bool
    requires_external: bool
    update_mode: str
    refit_policy: RefitPolicy
    feature_mode: str
    requires_active_prediction_time: bool = False
    heavy: bool = False


CAPABILITIES: tuple[ModelCapability, ...] = (
    ModelCapability("DIXON_COLES_V1", True, True, False, False, False, "SCHEDULED_REFIT", "MONTHLY", "GOALS_ONLY"),
    ModelCapability("BIVARIATE_POISSON_V1", True, True, False, False, False, "SCHEDULED_REFIT", "MONTHLY", "GOALS_ONLY"),
    ModelCapability("BAYESIAN_HIERARCHICAL_V1", True, True, False, False, False, "SCHEDULED_REFIT", "QUARTERLY", "GOALS_ONLY", False, True),
    ModelCapability("ELO_V1", True, True, False, False, False, "CHRONOLOGICAL_STATE_FIT", "DAILY", "RESULT_ONLY"),
    ModelCapability("PI_RATING_V1", True, True, False, False, False, "CHRONOLOGICAL_STATE_FIT", "DAILY", "RESULT_ONLY"),
    ModelCapability("DYNAMIC_BAYESIAN_POISSON_V1", False, True, False, False, False, "CHRONOLOGICAL_STATE_FIT", "DAILY", "GOALS_ONLY", True),
    ModelCapability("CORE_SPI_LIKE_V1", True, True, False, False, False, "CHRONOLOGICAL_STATE_FIT", "DAILY", "GOALS_ONLY"),
    ModelCapability("CORE_OPTA_XG_ELO_LIKE_V1", True, True, False, False, False, "CHRONOLOGICAL_STATE_FIT", "DAILY", "RESULT_ONLY_NO_XG", True),
)


@dataclass(frozen=True)
class RealOOSRunResult:
    run_id: str
    status: str
    start_date: date
    end_date: date
    completed_rows: int
    unavailable_rows: int
    failed_rows: int
    artifact_count: int
    model_oos_start_date: dict[str, date | None]


def _fit_boundary(day: date, policy: RefitPolicy) -> date:
    if policy == "DAILY":
        return day
    if policy == "MONTHLY":
        return date(day.year, day.month, 1)
    return date(day.year, 3 * ((day.month - 1) // 3) + 1, 1)


def _config_hash(config: ModelConfig, capability: ModelCapability) -> str:
    return _digest(f"{config.config_hash}|{RUNNER_VERSION}|{NEUTRAL_POLICY}|"
                   f"{capability.feature_mode}|{capability.refit_policy}")


def _artifact_id(model: BaseFootballModel, fitted_data: TrainingDataset,
                 cutoff: datetime) -> str:
    return _digest(f"{model.model_id}|{model.model_version}|{cutoff.isoformat()}|"
                   f"{fitted_data.data_hash}|{model.config.config_hash}")[:32]


class UnifiedRealOOSRunner:
    """Read immutable results in date order and persist each model independently."""

    def __init__(self, db_path: str | Path, *, artifact_root: str | Path) -> None:
        self.db_path = Path(db_path)
        self.artifact_root = Path(artifact_root)
        self.registry = ModelRegistry()
        self.capabilities = {item.model_id: item for item in CAPABILITIES}

    def run(self, start: date, end: date, *, model_ids: tuple[str, ...],
            minimum_history_matches: int = 80,
            competitions: tuple[str, ...] = ("EPL", "LALIGA", "BUNDESLIGA", "SERIE_A", "LIGUE_1"),
            max_dates: int | None = None, enable_heavy: bool = False) -> RealOOSRunResult:
        if end < start or start >= date(2026, 8, 1):
            raise ValueError("INVALID_OOS_WINDOW_OR_FINAL_HOLDOUT")
        if end >= date(2026, 8, 1):
            raise ValueError("FINAL_HOLDOUT_LOCKED")
        if not model_ids or len(set(model_ids)) != len(model_ids):
            raise ValueError("MODEL_IDS_REQUIRED_AND_UNIQUE")
        for model_id in model_ids:
            if model_id not in BASE_MODEL_IDS or model_id not in self.registry.list_models():
                raise ValueError(f"MODEL_NOT_REGISTERED_FOR_REAL_OOS:{model_id}")
            cap = self.capabilities[model_id]
            if cap.requires_market or cap.requires_xg or cap.requires_external:
                raise ValueError(f"MODEL_DATA_REQUIREMENT_UNAVAILABLE:{model_id}")
        prepare_phase8_2_migration(self.db_path)
        config = ModelConfig(min_matches=minimum_history_matches, min_team_matches=3,
                             training_window=1095, profile="production", allow_test_data=False)
        run_config_hash = _digest(config.config_hash + RUNNER_VERSION + repr(model_ids) + repr(competitions))
        run_id = _digest(f"{start}|{end}|{run_config_hash}")[:32]
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        counts = {"SUCCESS": 0, "UNAVAILABLE": 0, "FAILED": 0}
        artifact_count = 0
        starts: dict[str, date | None] = {model_id: None for model_id in model_ids}
        with duckdb.connect(str(self.db_path)) as connection, ArtifactIndex(
                self.artifact_root / "artifact_index.duckdb") as index:
            connection.execute("""INSERT INTO real_oos_run_manifest
                (run_id,model_ids,start_date,end_date,run_status,config_hash)
                VALUES (?,?,?,?,?,?) ON CONFLICT(run_id) DO UPDATE SET run_status='RUNNING'""",
                [run_id, json.dumps(model_ids), start, end, "RUNNING", run_config_hash])
            processed_dates = 0
            for competition_id in competitions:
                raw_rows = connection.execute("""
                    SELECT m.match_id,m.competition_id,m.season_id,m.match_date,
                           m.home_team_id,m.away_team_id,m.home_goals,m.away_goals,
                           'OPENFOOTBALL' AS source,m.retrieved_at,m.raw_hash,m.source_commit,
                           e.enriched_kickoff_utc,m.timestamp_precision
                    FROM real_canonical_matches m
                    LEFT JOIN (SELECT match_id,min(enriched_kickoff_utc) enriched_kickoff_utc
                        FROM kickoff_enrichment_records WHERE verified
                        GROUP BY match_id HAVING count(DISTINCT enriched_kickoff_utc)=1) e
                        USING(match_id)
                    WHERE m.competition_id=? AND m.match_date<=? AND m.status='FINISHED'
                    ORDER BY m.match_date,m.match_id
                """, [competition_id, end]).fetchall()
                if not raw_rows:
                    continue
                history = tuple(_training_row(tuple(row)) for row in raw_rows)
                team_ids = frozenset(row[0] for row in connection.execute(
                    "SELECT team_id FROM real_canonical_teams WHERE competition_id=?",
                    [competition_id]).fetchall())
                by_date: dict[date, list[tuple]] = defaultdict(list)
                for raw in raw_rows:
                    if start <= raw[3] <= end:
                        by_date[raw[3]].append(tuple(raw))
                cache: dict[tuple[str, date], tuple[BaseFootballModel, TrainingDataset,
                                                  datetime, str, str]] = {}
                for day in sorted(by_date):
                    if max_dates is not None and processed_dates >= max_dates:
                        break
                    processed_dates += 1
                    artifacts_before_day = artifact_count
                    targets = by_date[day]
                    boundary = _day_boundary(day)
                    target_ids = {row[0] for row in targets}
                    prior = _prior_history_rows(history, day)
                    if target_ids & {row.match_id for row in prior}:
                        raise ValueError("SAME_DAY_RESULT_ENTERED_TRAINING")
                    exact = all(row[12] is not None and row[12].astimezone(UTC) > boundary
                                for row in targets)
                    mode = "EXACT_UTC" if exact else "DATE_SAFE_BATCH"
                    daily_data = _dataset(prior, team_ids)
                    snapshot_id = _digest("|".join((RUNNER_VERSION, competition_id,
                        day.isoformat(), daily_data.data_hash, *sorted(target_ids))))[:32]
                    connection.execute("""INSERT INTO real_oos_prediction_snapshots VALUES
                        (?,?,?,?,?,?,?,?) ON CONFLICT DO NOTHING""",
                        [snapshot_id, competition_id, day, mode, boundary,
                         daily_data.data_hash, datetime.now(UTC),
                         json.dumps({"same_date_batch": True, "target_ids_hash":
                                     _digest("|".join(sorted(target_ids)))})])
                    for model_id in model_ids:
                        cap = self.capabilities[model_id]
                        existing = {row[0] for row in connection.execute(
                            "SELECT match_id FROM real_oos_predictions_v2 WHERE model_id=? "
                            "AND match_id IN (SELECT match_id FROM real_canonical_matches "
                            "WHERE competition_id=? AND match_date=?)",
                            [model_id, competition_id, day]).fetchall()}
                        pending = [raw for raw in targets if raw[0] not in existing]
                        if not pending:
                            continue
                        if not cap.supports_date_safe or (cap.heavy and not enable_heavy):
                            reason = ("DATE_SAFE_ADAPTER_NOT_AVAILABLE" if not cap.supports_date_safe
                                      else "RESOURCE_BUDGET_HEAVY_MODEL_DEFERRED")
                            for raw in pending:
                                self._status(connection, raw[0], model_id, day, mode,
                                    "UNAVAILABLE", reason, None,
                                    None, _config_hash(config, cap), run_id)
                                counts["UNAVAILABLE"] += 1
                            continue
                        fit_day = _fit_boundary(day, cap.refit_policy)
                        fit_cutoff = _day_boundary(fit_day)
                        fit_prior = _prior_history_rows(history, fit_day)
                        dataset = _dataset(fit_prior, team_ids)
                        windowed = dataset.window(fit_cutoff, config.training_window)
                        previous = {row[0]: row[1:] for row in connection.execute("""
                            SELECT match_id,execution_status,training_data_hash,config_hash
                            FROM real_oos_execution_status WHERE model_id=? AND prediction_date=?
                        """, [model_id, day]).fetchall()}
                        pending = [raw for raw in pending if not (
                            raw[0] in previous and previous[raw[0]][0] in {"UNAVAILABLE", "FAILED"}
                            and previous[raw[0]][1] == windowed.data_hash
                            and previous[raw[0]][2] == _config_hash(config, cap))]
                        if not pending:
                            continue
                        if len(windowed.matches) < minimum_history_matches:
                            for raw in pending:
                                self._status(connection, raw[0], model_id, day, mode,
                                    "UNAVAILABLE", "INSUFFICIENT_HISTORY", None,
                                    windowed.data_hash, _config_hash(config, cap), run_id)
                                counts["UNAVAILABLE"] += 1
                            continue
                        key = (model_id, fit_day)
                        fitted = cache.get(key)
                        if fitted is None:
                            model = self.registry.get_model(model_id)
                            if model.model_id != model_id:
                                raise ValueError("MODEL_REGISTRY_IDENTITY_MISMATCH")
                            try:
                                cached_path = index.find_model(model_id, model.model_version,
                                    fit_cutoff, model.prepare_config(config).config_hash,
                                    windowed.data_hash)
                                if cached_path is not None:
                                    model = type(model).load(cached_path)
                                    artifact_path = cached_path
                                else:
                                    model.fit(dataset, fit_cutoff, config)
                                    artifact_path = str(self.artifact_root / model_id /
                                        _artifact_id(model, windowed, fit_cutoff))
                                    artifact_manifest = model.save(artifact_path)
                                    index.register_model(artifact_manifest)
                                    artifact_count += 1
                                fitted = (model, windowed, fit_cutoff,
                                          _config_hash(model.config, cap), artifact_path)
                                cache[key] = fitted
                            except (ValueError, RuntimeError, ArithmeticError, MemoryError) as error:
                                LOGGER.exception("real OOS fit failed: %s %s", model_id, fit_day)
                                status = "UNAVAILABLE" if isinstance(error, InsufficientData) else "FAILED"
                                for raw in pending:
                                    self._status(connection, raw[0], model_id, day, mode,
                                        status, f"{type(error).__name__}:{error}", None,
                                        windowed.data_hash, _config_hash(config, cap), run_id)
                                    counts[status] += 1
                                continue
                        model, fit_data, cutoff, config_hash, artifact_path = fitted
                        if cutoff > boundary or any(row.kickoff_time.date() >= day for row in fit_data.matches):
                            raise ValueError("FUTURE_ARTIFACT_OR_SAME_DAY_TRAINING")
                        artifact_id = _artifact_id(model, fit_data, cutoff)
                        for raw in pending:
                            if raw[0] in model.training_ids:
                                raise ValueError("TARGET_MATCH_IN_ARTIFACT_TRAINING")
                            fixture = Fixture(match_id=raw[0], competition_id=competition_id,
                                home_team_id=raw[4], away_team_id=raw[5],
                                kickoff_time=(raw[12].astimezone(UTC) if exact else
                                              boundary + timedelta(days=1)),
                                source="OPENFOOTBALL", retrieved_at=boundary,
                                as_of_time=boundary, data_version=str(raw[10]),
                                season=raw[2], neutral_venue=False)
                            try:
                                if cap.requires_active_prediction_time:
                                    cast(Any, model)._active_prediction_time = boundary
                                values = model._predict_values(fixture)
                                record = _predict_record(model_id, model.model_version,
                                    match_id=raw[0], competition_id=competition_id,
                                    season_id=raw[2], target_date=day,
                                    training_data=fit_data, cutoff=cutoff,
                                    snapshot_id=snapshot_id, config_hash=config_hash,
                                    sources=model.sources, values=values,
                                    prediction_time=boundary, temporal_mode=mode)
                                digest = lineage_hash((record.prediction_id, raw[0], model_id,
                                    artifact_id, fit_data.data_hash, config_hash, snapshot_id,
                                    cutoff.isoformat(), boundary.isoformat()))
                                metadata = {**record.metadata, "artifact_id": artifact_id,
                                    "artifact_path": artifact_path, "lineage_hash": digest,
                                    "feature_mode": cap.feature_mode,
                                    "training_match_ids_hash": hashlib.sha256(json.dumps(
                                        sorted(model.training_ids), separators=(",", ":")).encode()).hexdigest()}
                                record = ModelPrediction.model_validate({**record.model_dump(),
                                    "implementation_type": ImplementationType(model.implementation_type),
                                    "metadata": metadata})
                                _persist_oos(connection, record, competition_id=competition_id,
                                    season_id=raw[2], match_date=day,
                                    training_count=len(fit_data.matches), config_hash=config_hash,
                                    timestamp_precision=mode, kickoff_time=raw[12])
                                connection.execute("""INSERT INTO real_oos_lineage_v2 VALUES
                                    (?,?,?,?,?,?,?) ON CONFLICT DO NOTHING""",
                                    [record.prediction_id, artifact_id, digest,
                                     metadata["training_match_ids_hash"], cap.feature_mode,
                                     json.dumps([]), datetime.now(UTC)])
                                self._status(connection, raw[0], model_id, day, mode,
                                    "SUCCESS", None, record.prediction_id,
                                    fit_data.data_hash, config_hash, run_id)
                                counts["SUCCESS"] += 1
                                current_start = starts[model_id]
                                if current_start is None or day < current_start:
                                    starts[model_id] = day
                            except Exception as error:
                                status = "UNAVAILABLE" if isinstance(error, (KeyError, InsufficientData)) else "FAILED"
                                LOGGER.exception("real OOS prediction %s %s failed", model_id, raw[0])
                                self._status(connection, raw[0], model_id, day, mode,
                                    status, f"{type(error).__name__}:{error}", None,
                                    fit_data.data_hash, config_hash, run_id)
                                counts[status] += 1
                    connection.execute("""UPDATE real_oos_run_manifest SET current_date=?,
                        completed_rows=completed_rows+?,failed_rows=failed_rows+?,
                        unavailable_rows=unavailable_rows+?,artifact_count=artifact_count+?,
                        last_checkpoint=? WHERE run_id=?""",
                        [day, counts["SUCCESS"], counts["FAILED"], counts["UNAVAILABLE"],
                         artifact_count - artifacts_before_day, datetime.now(UTC), run_id])
                    counts = {"SUCCESS": 0, "UNAVAILABLE": 0, "FAILED": 0}
            artifact_row = connection.execute("""SELECT count(DISTINCT l.artifact_id)
                FROM real_oos_execution_status s JOIN real_oos_lineage_v2 l
                ON l.prediction_id=s.prediction_id WHERE s.run_id=?""", [run_id]).fetchone()
            assert artifact_row is not None
            used_artifacts = artifact_row[0]
            actual = connection.execute("""SELECT
                count(*) FILTER (WHERE execution_status='SUCCESS'),
                count(*) FILTER (WHERE execution_status='UNAVAILABLE'),
                count(*) FILTER (WHERE execution_status='FAILED')
                FROM real_oos_execution_status WHERE run_id=?""", [run_id]).fetchone()
            assert actual is not None
            connection.execute("""UPDATE real_oos_run_manifest SET artifact_count=?,
                completed_rows=?,unavailable_rows=?,failed_rows=? WHERE run_id=?""",
                [used_artifacts, *actual, run_id])
            run_totals = connection.execute("SELECT completed_rows,unavailable_rows,failed_rows,"
                                           "artifact_count FROM real_oos_run_manifest WHERE run_id=?",
                                           [run_id]).fetchone()
            assert run_totals is not None
            status = "PARTIAL" if max_dates is not None else "COMPLETE"
            connection.execute("UPDATE real_oos_run_manifest SET run_status=? WHERE run_id=?",
                               [status, run_id])
            for model_id in model_ids:
                first_row = connection.execute("""SELECT min(match_date)
                    FROM real_oos_predictions_v2
                    WHERE model_id=? AND match_date BETWEEN ? AND ?""",
                    [model_id, start, end]).fetchone()
                starts[model_id] = first_row[0] if first_row else None
        return RealOOSRunResult(run_id, status, start, end, run_totals[0], run_totals[1],
                                run_totals[2], run_totals[3], starts)

    @staticmethod
    def _status(connection: duckdb.DuckDBPyConnection, match_id: str, model_id: str,
                day: date, mode: str, status: str, reason: str | None,
                prediction_id: str | None, data_hash: str | None,
                config_hash: str, run_id: str) -> None:
        connection.execute("""INSERT INTO real_oos_execution_status VALUES
            (?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(match_id,model_id) DO UPDATE SET
            execution_status=excluded.execution_status,reason=excluded.reason,
            prediction_id=excluded.prediction_id,training_data_hash=excluded.training_data_hash,
            config_hash=excluded.config_hash,run_id=excluded.run_id,updated_at=excluded.updated_at""",
            [match_id, model_id, day, mode, status, reason, prediction_id,
             data_hash, config_hash, run_id, datetime.now(UTC)])
