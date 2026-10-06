"""Deterministic Phase 9 self-review of splits, source purity and artifacts."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import yaml

from erguoyuan_football.meta.artifact import load_artifact
from erguoyuan_football.meta.candidate import load_verified_candidate
from erguoyuan_football.meta.contracts import (
    CalibratedCoreProbability,
    MetaPromotionDecision,
    MetaRawPrediction,
    ModelDisagreementReport,
)
from erguoyuan_football.meta.lineage import PredictionLineageValidator
from erguoyuan_football.output_contract.schemas import (
    CanonicalPredictionResult,
    ProbabilityStage,
)


def audit_phase9(db_path: str | Path, *, config_path: str | Path,
                 artifact_path: str | Path, report_path: str | Path,
                 example_path: str | Path) -> dict[str, Any]:
    """Fail closed on leakage; report model-quality limitations separately."""
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    example = json.loads(Path(example_path).read_text(encoding="utf-8"))
    candidate = load_verified_candidate(db_path,
        dataset_id=config["candidate_dataset_id"],
        expected_hash=config["candidate_data_hash"],
        expected_part_hashes=config["candidate_part_hashes"])
    artifact = load_artifact(artifact_path, expected_candidate_hash=candidate.data_hash)
    raw = MetaRawPrediction.model_validate(example["meta_raw"])
    calibrated = CalibratedCoreProbability.model_validate(example["calibrated_core"])
    disagreement = ModelDisagreementReport.model_validate(example["disagreement"])
    canonical = CanonicalPredictionResult.model_validate(example["canonical"])
    promotion = MetaPromotionDecision.model_validate(report["promotion_decision"])
    source_lineage = PredictionLineageValidator(db_path,
        artifact_root=Path(db_path).resolve().parent.parent / "artifacts" / "phase8_2").validate(candidate)
    if (source_lineage.status != "PASS" or
            source_lineage.nested_training_vectors_checked == 0 or
            report["recursive_lineage_validation"]["nested_training_vectors_checked"] !=
            source_lineage.nested_training_vectors_checked):
        raise ValueError("PHASE9_RECURSIVE_LINEAGE_NOT_VERIFIED")
    windows = report["development_windows"]
    holdout = windows["final_holdout"]
    if holdout != ["2026-08-01", "2027-06-30"] or report["final_holdout_rows"] != 0:
        raise ValueError("PHASE9_FINAL_HOLDOUT_CHANGED_OR_EVALUATED")
    if any(bounds[1] >= holdout[0] for name, bounds in windows.items() if name != "final_holdout"):
        raise ValueError("PHASE9_DEVELOPMENT_WINDOW_READS_HOLDOUT")
    if any(date.fromisoformat(str(item)) >= date(2026, 8, 1)
           for item in candidate.features["prediction_date"]):
        raise ValueError("PHASE9_CANDIDATE_READS_HOLDOUT")
    with duckdb.connect(str(db_path), read_only=True) as connection:
        counts = connection.execute("""SELECT
            count(*) FILTER (WHERE match_date >= DATE '2026-08-01'),
            count(*) FILTER (WHERE model_id='HISTORICAL_MARKET_BAYESIAN_POISSON_V1')
            FROM real_oos_predictions_v2""").fetchone()
    assert counts is not None
    if counts != (0, 0) or report["candidate_audit"]["market_dependency_count"] != 0:
        raise ValueError("PHASE9_MARKET_OR_HOLDOUT_SOURCE_FOUND")
    research = report["research_datasets"]
    chosen_research = min(research, key=lambda key: (
        research[key]["dev_validation_metrics"]["log_loss"],
        research[key]["dev_validation_metrics"]["brier"], key))
    methods = report["calibration_methods"]
    chosen_calibration = min(methods, key=lambda key: (
        methods[key]["calibration_dev_eval_metrics"]["log_loss"],
        methods[key]["calibration_dev_eval_metrics"]["brier"], key))
    threshold_results = report["min_available_models_review"]
    chosen_threshold = min(threshold_results, key=lambda key: (
        threshold_results[key]["dev_validation_metrics"]["log_loss"], int(key)))
    if (chosen_research != report["selected_research_dataset"] or
            chosen_calibration != report["selected_calibration_method"] or
            int(chosen_threshold) != report["selected_min_available_models"]):
        raise ValueError("PHASE9_SELECTION_DID_NOT_USE_DEV_METRICS")
    if (artifact.manifest["meta_trained_until"] >= windows["calibration_fit"][0] or
            artifact.manifest["calibration_fit_end"] >= windows["calibration_dev_eval"][0]):
        raise ValueError("PHASE9_ARTIFACT_TEMPORAL_LEAKAGE")
    if (artifact.manifest["candidate_part_hashes"] != candidate.part_hashes or
            artifact.manifest["base_model_versions"] != source_lineage.model_versions or
            artifact.manifest["feature_schema_version"] != report["feature_schema_version"] or
            artifact.manifest["calibrator_id"] != calibrated.calibrator_id or
            artifact.manifest["calibration_fit_data_hash"] != report["calibration_fit_data_hash"]):
        raise ValueError("PHASE9_ARTIFACT_SOURCE_INCOMPATIBLE")
    if (artifact.model.pipeline is None or
            "logit" not in artifact.model.pipeline.named_steps or
            artifact.model.training_sample_count != report["selected_meta_train_rows"]):
        raise ValueError("PHASE9_META_FIT_OR_AVERAGE_FALLBACK_INVALID")
    if (canonical.probability_stage != ProbabilityStage.FINAL_CORE_CALIBRATED or
            canonical.validation_status != "DEVELOPMENT_ONLY" or
            canonical.production_status != "NOT_PROMOTED" or
            canonical.data_lineage.get("market_dependency_count") != 0 or
            canonical.prediction_temporal_mode != "DATE_SAFE_BATCH"):
        raise ValueError("PHASE9_CANONICAL_STAGE_OR_SOURCE_INVALID")
    if ("calibrated_core_count" in report and
            report["calibrated_core_count"] != report["common_sample_development"]["match_count"]):
        raise ValueError("PHASE9_CORE_PREVIEW_COUNT_MISMATCH")
    if (not promotion.engineering_ready or not promotion.artifact_ready or
            promotion.production_promoted or promotion.final_holdout_ready or
            promotion.superiority_evidence != "NOT_ESTABLISHED" or
            disagreement.purpose != "DIAGNOSTIC_ONLY" or
            raw.match_id != calibrated.match_id or calibrated.match_id != canonical.match_id or
            (calibrated.calibrated_p_home, calibrated.calibrated_p_draw,
             calibrated.calibrated_p_away) != (canonical.p_home, canonical.p_draw,
                                          canonical.p_away)):
        raise ValueError("PHASE9_OUTPUT_OR_PROMOTION_INVALID")
    availability = candidate.availability_report()
    eight_by_year = {year: sum(count for pattern, count in patterns.items()
                         if len(pattern.split("+")) == 8)
                     for year, patterns in availability["availability_pattern_by_year"].items()}
    confounded = (sum(eight_by_year.get(year, 0) for year in ("2022", "2023", "2024")) == 0 and
                  eight_by_year.get("2025", 0) > 0)
    if not confounded:
        raise ValueError("PHASE9_AVAILABILITY_TIME_PATTERN_UNEXPECTED")
    return {
        "status": "PASS_WITH_LIMITATIONS",
        "candidate_dataset_id": candidate.dataset_id,
        "dataset_and_part_hashes_verified": True,
        "candidate_source_oos_identity_and_cutoff_verified": True,
        "holdout_preservation": "PASS_0_EVALUATED_ROWS",
        "market_purity": "PASS_0_MARKET_ROWS",
        "chronological_split_and_selection": "PASS_DEVELOPMENT_ONLY",
        "artifact_integrity": "PASS",
        "simple_average_fallback": "ABSENT_FITTED_LOGISTIC_META",
        "recursive_lineage": "PASS_ALL_CANDIDATE_AND_ML_TRAINING_VECTORS",
        "nested_training_vectors_checked": source_lineage.nested_training_vectors_checked,
        "canonical_v2_development_stage": "PASS",
        "production_promotion": "BLOCKED",
        "availability_time_confounding": {
            "observed": confounded, "eight_model_rows_by_year": eight_by_year,
            "mitigation": "MAX_COVERAGE_AND_COMMON_MODEL_ERA_COMPARED_ON_IDENTICAL_DEV_MATCHES"},
        "meta_minus_dixon_coles_log_loss": report["calibrated_minus_dixon_coles_log_loss"],
        "quality_limitation": "META_DID_NOT_BEAT_DIXON_COLES_ON_CALIBRATION_DEV_EVAL",
        "final_holdout_performance": "UNAVAILABLE",
    }
