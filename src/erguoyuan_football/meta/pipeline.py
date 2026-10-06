"""Chronological Phase 9 NO_MARKET development; final holdout remains sealed."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from erguoyuan_football.backtesting.date_safe_oos import _metrics
from erguoyuan_football.meta.calibration import (
    MulticlassCalibrator,
    multiclass_log_loss,
)
from erguoyuan_football.meta.candidate import (
    MODEL_PREFIXES,
    VerifiedCandidate,
    load_verified_candidate,
)
from erguoyuan_football.meta.lineage import PredictionLineageValidator
from erguoyuan_football.meta.model import MetaRows, NoMarketMetaModel, select_rows
from erguoyuan_football.meta.promotion import MetaPromotionGate
from erguoyuan_football.ml.schemas import stable_hash


def _window(values: list[date]) -> tuple[date, date]:
    if len(values) != 2:
        raise ValueError("INVALID_PHASE9_WINDOW")
    return date.fromisoformat(str(values[0])), date.fromisoformat(str(values[1]))


def _scores(probabilities: np.ndarray, rows: MetaRows) -> dict[str, float | int]:
    if len(probabilities) != rows.count:
        raise ValueError("METRIC_ROW_COUNT_MISMATCH")
    return _metrics([(float(p[0]), float(p[1]), float(p[2]), int(label))
                     for p, label in zip(probabilities, rows.labels, strict=True)])


def _baseline_scores(rows: MetaRows) -> dict[str, dict[str, float | int]]:
    scores = {}
    for prefix in MODEL_PREFIXES:
        probabilities = rows.frame[[f"{prefix}_home", f"{prefix}_draw",
                                    f"{prefix}_away"]].to_numpy(dtype=float)
        if not np.isfinite(probabilities).all():
            raise ValueError("COMMON_SAMPLE_BASE_MODEL_MISSING")
        scores[prefix] = _scores(probabilities, rows)
    return scores


@dataclass(frozen=True)
class Phase9Development:
    """A model, fitted calibrator and inspectable DEVELOPMENT_ONLY findings."""

    candidate: VerifiedCandidate
    model: NoMarketMetaModel
    calibrator: MulticlassCalibrator
    config: dict[str, Any]
    report: dict[str, Any]
    calibration_eval_rows: MetaRows


def run_phase9_development(db_path: str | Path, *, config_path: str | Path) -> Phase9Development:
    """Train only META_NO_MARKET_V1 and fit calibration in isolated date windows."""
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if config.get("model_id") != "META_NO_MARKET_V1" or config.get("temporal_mode") != "DATE_SAFE_BATCH":
        raise ValueError("PHASE9_CONFIG_MODE_REJECTED")
    windows = {key: _window(config[key]) for key in (
        "max_coverage_train", "common_model_era_train", "meta_dev_validation",
        "calibration_fit", "calibration_dev_eval", "final_holdout")}
    max_train = windows["max_coverage_train"]
    common_train = windows["common_model_era_train"]
    dev = windows["meta_dev_validation"]
    calibration_fit = windows["calibration_fit"]
    calibration_eval = windows["calibration_dev_eval"]
    holdout = windows["final_holdout"]
    if not (max_train[0] <= common_train[0] <= common_train[1] <= max_train[1] <
            dev[0] <= dev[1] < calibration_fit[0] <= calibration_fit[1] <
            calibration_eval[0] <= calibration_eval[1] < holdout[0] <= holdout[1]):
        raise ValueError("PHASE9_SPLIT_ORDER_VIOLATION")
    if holdout != (date(2026, 8, 1), date(2027, 6, 30)):
        raise ValueError("FINAL_HOLDOUT_BOUNDARY_CHANGED")
    candidate = load_verified_candidate(db_path, dataset_id=config["candidate_dataset_id"],
        expected_hash=config["candidate_data_hash"],
        expected_part_hashes=config["candidate_part_hashes"])
    source_lineage = PredictionLineageValidator(db_path,
        artifact_root=Path(db_path).resolve().parent.parent / "artifacts" / "phase8_2").validate(candidate)
    audit = candidate.availability_report()
    if audit["candidate_holdout_rows"] != 0 or audit["market_dependency_count"] != 0:
        raise ValueError("PHASE9_HOLDOUT_OR_MARKET_REJECTED")
    min_max = int(config["min_available_models"])
    min_common = int(config["common_era_min_available_models"])
    if not 1 <= min_max <= 8 or min_common != 8:
        raise ValueError("MIN_MODEL_POLICY_REVIEW_REQUIRED")
    dev_common = select_rows(candidate, *dev, min_available_models=8)
    if dev_common.count < 100:
        raise ValueError("INSUFFICIENT_COMMON_DEV_VALIDATION")
    def new_model() -> NoMarketMetaModel:
        return NoMarketMetaModel(regularization_c=float(config["regularization_c"]),
                                 epsilon=float(config["log_ratio_epsilon"]))
    threshold_research: dict[int, dict[str, Any]] = {}
    for threshold in (1, 2, 3):
        threshold_rows = select_rows(candidate, *max_train, min_available_models=threshold)
        threshold_model = new_model().fit(threshold_rows)
        threshold_research[threshold] = {
            "train_rows": threshold_rows.count,
            "dev_validation_rows": dev_common.count,
            "dev_validation_metrics": _scores(threshold_model.predict_proba(dev_common), dev_common),
            "evaluation_status": "DEVELOPMENT_ONLY",
        }
    chosen_threshold = min(threshold_research, key=lambda key: (
        threshold_research[key]["dev_validation_metrics"]["log_loss"], key))
    if chosen_threshold != min_max:
        raise ValueError("CONFIGURED_MIN_MODELS_DISAGREES_WITH_DEV_SELECTION")
    research: dict[str, dict[str, Any]] = {}
    for name, train_window, threshold in (
        ("MAX_COVERAGE", max_train, min_max),
        ("COMMON_MODEL_ERA", common_train, min_common),
    ):
        train_rows = select_rows(candidate, *train_window, min_available_models=threshold)
        model = new_model()
        model.fit(train_rows)
        predictions = model.predict_proba(dev_common)
        research[name] = {
            "train_rows": train_rows.count,
            "dev_validation_rows": dev_common.count,
            "dev_validation_metrics": _scores(predictions, dev_common),
            "trained_until": model.trained_until.isoformat() if model.trained_until else None,
            "min_available_models": threshold,
            "evaluation_status": "DEVELOPMENT_ONLY",
        }
    selected = min(research, key=lambda key: (
        research[key]["dev_validation_metrics"]["log_loss"],
        research[key]["dev_validation_metrics"]["brier"], key))
    selected_start = max_train[0] if selected == "MAX_COVERAGE" else common_train[0]
    threshold = min_max if selected == "MAX_COVERAGE" else min_common
    final_train = select_rows(candidate, selected_start, dev[1], min_available_models=threshold)
    model = new_model().fit(final_train)
    fit_rows = select_rows(candidate, *calibration_fit, min_available_models=8)
    eval_rows = select_rows(candidate, *calibration_eval, min_available_models=8)
    if fit_rows.count < 50 or eval_rows.count < 50:
        raise ValueError("INSUFFICIENT_CALIBRATION_DEVELOPMENT_DATA")
    fit_raw = model.predict_proba(fit_rows)
    eval_raw = model.predict_proba(eval_rows)
    calibration_methods: dict[str, dict[str, Any]] = {}
    calibrators: dict[str, MulticlassCalibrator] = {}
    for method in ("NONE", "TEMPERATURE_SCALING", "MULTICLASS_LOGIT_CALIBRATION"):
        calibrator = MulticlassCalibrator(method).fit(fit_raw, fit_rows.labels)
        fit_probabilities = calibrator.predict_proba(fit_raw)
        eval_probabilities = calibrator.predict_proba(eval_raw)
        calibration_methods[method] = {
            "calibration_fit_rows": fit_rows.count,
            "calibration_fit_log_loss": multiclass_log_loss(fit_probabilities, fit_rows.labels),
            "calibration_dev_eval_rows": eval_rows.count,
            "calibration_dev_eval_metrics": _scores(eval_probabilities, eval_rows),
            "evaluation_status": "DEVELOPMENT_ONLY",
        }
        calibrators[method] = calibrator
    chosen_method = min(calibration_methods, key=lambda key: (
        calibration_methods[key]["calibration_dev_eval_metrics"]["log_loss"],
        calibration_methods[key]["calibration_dev_eval_metrics"]["brier"], key))
    calibrated = calibrators[chosen_method].predict_proba(eval_raw)
    baselines = _baseline_scores(eval_rows)
    best_single = min(baselines, key=lambda key: baselines[key]["log_loss"])
    calibrated_scores = _scores(calibrated, eval_rows)
    calibrated_loss = float(calibrated_scores["log_loss"])
    comparison_deltas = {
        name: {metric: float(calibrated_scores[metric]) - float(baselines[name][metric])
               for metric in ("log_loss", "brier", "rps")}
        for name in ("dc", best_single, "xgb", "cat")
    }
    candidate_audit = audit | {"dataset_id": candidate.dataset_id,
                               "dataset_hash": candidate.data_hash,
                               "part_hashes": candidate.part_hashes}
    candidate_dates = candidate.features["prediction_date"].astype(str)
    candidate_counts = candidate.features[[f"{name}_available" for name in MODEL_PREFIXES]].sum(axis=1)
    candidate_audit["eligible_meta_rows"] = int(((candidate_dates <= "2025-06-30") &
                                                  (candidate_counts >= min_max)).sum())
    candidate_audit["eligible_calibration_rows"] = int(((candidate_dates >= "2025-08-01") &
        (candidate_dates <= "2026-06-30") & (candidate_counts >= min_max)).sum())
    candidate_audit["eligibility_min_available_models"] = min_max
    report: dict[str, Any] = {
        "status": "PARTIAL", "reason": "AWAITING_FINAL_HOLDOUT",
        "engineering_status": "PASS", "data_status": "DATE_SAFE_BATCH_REAL_OOS",
        "meta_training_status": "PASS_NO_MARKET_ONLY", "calibration_status": "DEVELOPMENT_ONLY",
        "meta_full_v1_status": "BLOCKED_MARKET_DATA",
        "final_holdout_status": "LOCKED_AWAITING_DATA", "market_status": "BLOCKED_MARKET_DATA",
        "validation_status": "DEVELOPMENT_ONLY",
        "temporal_mode": "DATE_SAFE_BATCH", "production_compatibility": "RESEARCH_ONLY_NOT_LIVE_VALIDATED",
        "candidate_audit": candidate_audit,
        "recursive_lineage_validation": asdict(source_lineage),
        "base_model_versions": source_lineage.model_versions,
        "feature_schema_version": config["feature_schema_version"],
        "dependency_schema_version": config["dependency_schema_version"],
        "selected_training_data_hash": stable_hash({"candidate_hash": candidate.data_hash,
            "match_ids": final_train.frame["match_id"].tolist(),
            "labels": final_train.labels.tolist()}),
        "calibration_fit_data_hash": stable_hash({"match_ids": fit_rows.frame["match_id"].tolist(),
            "raw_probabilities": fit_raw.tolist(), "labels": fit_rows.labels.tolist()}),
        "min_available_models_review": threshold_research,
        "selected_min_available_models": min_max,
        "split_config_hash": stable_hash(config),
        "development_windows": {key: [value[0].isoformat(), value[1].isoformat()]
                                for key, value in windows.items()},
        "research_datasets": research,
        "selected_research_dataset": selected,
        "selected_meta_train_rows": final_train.count,
        "selected_meta_trained_until": model.trained_until.isoformat() if model.trained_until else None,
        "calibration_methods": calibration_methods,
        "selected_calibration_method": chosen_method,
        "common_sample_development": {
            "match_count": eval_rows.count, "single_model_metrics": baselines,
            "best_single_model": best_single,
            "meta_raw_metrics": _scores(eval_raw, eval_rows),
            "meta_calibrated_metrics": calibrated_scores,
            "calibrated_minus_single_model": comparison_deltas,
            "evaluation_status": "DEVELOPMENT_ONLY"},
        "calibrated_minus_dixon_coles_log_loss": calibrated_loss - float(baselines["dc"]["log_loss"]),
        "calibrated_minus_best_single_log_loss": calibrated_loss - float(baselines[best_single]["log_loss"]),
        "calibrated_minus_catboost_log_loss": calibrated_loss - float(baselines["cat"]["log_loss"]),
        "calibrated_minus_xgboost_log_loss": calibrated_loss - float(baselines["xgb"]["log_loss"]),
        "final_holdout_rows": 0, "final_holdout_performance": "UNAVAILABLE",
        "models_available": 8, "models_deferred": audit["deferred_models"],
        "phase10_software_development_ready": False,
        "live_production_ready": False,
    }
    report["promotion_decision"] = MetaPromotionGate().evaluate(
        report, artifact_verified=False).model_dump(mode="json")
    return Phase9Development(candidate, model, calibrators[chosen_method], config, report, eval_rows)
