"""Matched-target benchmark scoring from independently produced OOS records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from erguoyuan_football.contracts.common import ExecutionStatus
from erguoyuan_football.contracts.predictions import ModelPrediction
from erguoyuan_football.ml.backtest import (
    MLBacktestReport,
    MLMetricSet,
    MLOOSObservation,
    metrics_for,
)

BENCHMARK_IDS = (
    "NAIVE_LEAGUE_FREQUENCY", "SIMPLE_INDEPENDENT_POISSON", "DIXON_COLES_V1",
    "DYNAMIC_BAYESIAN_POISSON_V1", "ELO_V1", "CORE_SPI_LIKE_V1",
    "HISTORICAL_MARKET_BAYESIAN_POISSON_V1", "MARKET_CONSENSUS",
    "CORE_XGBOOST_V1", "CORE_CATBOOST_V1",
)


@dataclass(frozen=True)
class BenchmarkResult:
    benchmark_id: str
    status: Literal["AVAILABLE", "UNAVAILABLE"]
    metrics: MLMetricSet | None
    reason: str | None


def compare_model_benchmarks(report: MLBacktestReport,
                             candidates: dict[str, tuple[ModelPrediction, ...]]) -> tuple[BenchmarkResult, ...]:
    """Score only complete, time-audited predictions for the identical OOS targets."""
    results = []
    targets = {item.row.vector.match_id: item for item in report.observations}
    for benchmark_id in BENCHMARK_IDS:
        if benchmark_id == report.model_id:
            results.append(BenchmarkResult(benchmark_id, "AVAILABLE", report.overall, None))
            continue
        rows = candidates.get(benchmark_id)
        if not rows:
            results.append(BenchmarkResult(benchmark_id, "UNAVAILABLE", None,
                                           "NO_MATCHED_OOS_PREDICTIONS"))
            continue
        by_match = {item.match_id: item for item in rows}
        if len(by_match) != len(rows) or set(by_match) != set(targets):
            results.append(BenchmarkResult(benchmark_id, "UNAVAILABLE", None,
                                           "INCOMPLETE_OR_DUPLICATED_TARGET_COVERAGE"))
            continue
        comparable: list[MLOOSObservation] = []
        for match_id, target in targets.items():
            prediction = by_match[match_id]
            vector = target.row.vector
            if (prediction.model_id != benchmark_id or
                    (prediction.prediction_snapshot_id, prediction.prediction_time,
                     prediction.input_data_version) != (
                         vector.prediction_snapshot_id, vector.prediction_time,
                         vector.input_data_version) or
                    prediction.execution_status != ExecutionStatus.SUCCESS or not prediction.is_oos or
                    prediction.training_end_time is None or
                    prediction.training_end_time > prediction.prediction_time or
                    prediction.prediction_time >= vector.kickoff_time):
                comparable = []
                break
            comparable.append(MLOOSObservation(row=target.row, prediction=prediction,
                                               window_index=target.window_index))
        if not comparable:
            results.append(BenchmarkResult(benchmark_id, "UNAVAILABLE", None,
                                           "BENCHMARK_OOS_OR_SNAPSHOT_AUDIT_FAILED"))
        else:
            results.append(BenchmarkResult(benchmark_id, "AVAILABLE", metrics_for(tuple(comparable)), None))
    return tuple(results)
