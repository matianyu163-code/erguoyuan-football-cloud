"""Phase 9 NO_MARKET tests; real integration uses pinned local OOS only."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from erguoyuan_football.meta.artifact import load_artifact, save_artifact
from erguoyuan_football.meta.audit import audit_phase9
from erguoyuan_football.meta.calibration import MulticlassCalibrator
from erguoyuan_football.meta.candidate import load_verified_candidate
from erguoyuan_football.meta.contracts import MetaPromotionDecision, MetaRawPrediction
from erguoyuan_football.meta.diagnostics import measure_disagreement
from erguoyuan_football.meta.lineage import assert_acyclic
from erguoyuan_football.meta.model import (
    NoMarketMetaModel,
    encode_features,
    select_rows,
)
from erguoyuan_football.meta.output import build_development_core
from erguoyuan_football.meta.pipeline import run_phase9_development
from erguoyuan_football.meta.promotion import MetaPromotionGate
from erguoyuan_football.output_contract.adapters import CoreReportV2PreviewAdapter
from erguoyuan_football.output_contract.schemas import ProbabilityStage

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "football.duckdb"
CONFIG = ROOT / "config" / "phase9_development.yaml"
DATASET_ID = "10cec92f96327bfb9d2fe0d3f0782ac5"
DATA_HASH = "10cec92f96327bfb9d2fe0d3f0782ac5aa884c14419642c127a479786ccec470"


@pytest.fixture(scope="module")
def real_candidate():
    if not DB.is_file():
        pytest.skip("Pinned Phase 8.2 real OOS warehouse is unavailable")
    return load_verified_candidate(DB, dataset_id=DATASET_ID, expected_hash=DATA_HASH)


@pytest.fixture(scope="module")
def development():
    if not DB.is_file():
        pytest.skip("Pinned Phase 8.2 real OOS warehouse is unavailable")
    return run_phase9_development(DB, config_path=CONFIG)


def test_candidate_immutable_content_and_parts(real_candidate):
    assert real_candidate.data_hash == DATA_HASH
    assert real_candidate.part_hashes == {
        "features": "2a1914818eaf9d7558df2e8dfaeb4e2c9f015e7a0e5532956e356be594021271",
        "labels": "8534a2bba05c5b2cb9a6206ee36113ec66022b1457b04792703f32e5fb19c659",
        "lineage": "99fc6cdf0363807184b79cabab74d32dc0c80909faf8cb48ab3c5b808945a425",
    }
    with pytest.raises(ValueError, match="CANDIDATE_ID_HASH_MISMATCH"):
        load_verified_candidate(DB, dataset_id=DATASET_ID, expected_hash="0" * 64)


def test_candidate_counts_and_availability_time_confounding(real_candidate):
    report = real_candidate.availability_report()
    assert report["candidate_total_rows"] == 6895
    assert report["candidate_meta_train_rows"] == 5175
    assert report["candidate_calibration_rows"] == 1720
    assert report["candidate_holdout_rows"] == 0
    assert report["models_per_row_distribution"][3] == 4871
    assert report["models_per_row_distribution"][8] == 1453
    assert not any("dc+bivariate+elo+pi+spi+opta_like+xgb+cat" in key
                   for key in report["availability_pattern_by_year"]["2022"])
    assert report["market_dependency_count"] == 0


def test_missing_probability_is_null_with_mask(real_candidate):
    row = real_candidate.features.loc[real_candidate.features["bivariate_available"] == 0].iloc[[0]]
    assert np.isnan(float(row.iloc[0]["bivariate_home"]))
    encoded = encode_features(row)
    assert encoded.shape == (1, 24)
    assert np.isfinite(encoded).all()
    # Zero is the internal missing encoding, paired with a zero mask; it is not a forecast.
    assert encoded[0, 3:6].tolist() == [0.0, 0.0, 0.0]


def test_final_holdout_and_in_sample_rejected(real_candidate):
    with pytest.raises(ValueError, match="INVALID_META_WINDOW_OR_MIN_MODELS"):
        select_rows(real_candidate, date(2026, 8, 1), date(2026, 8, 31),
                    min_available_models=8)
    train = select_rows(real_candidate, date(2025, 10, 1), date(2025, 12, 31),
                        min_available_models=8)
    model = NoMarketMetaModel().fit(train)
    with pytest.raises(ValueError, match="META_IN_SAMPLE_PREDICTION_REJECTED"):
        model.predict_proba(train)


def test_development_split_and_model_selection(development):
    report = development.report
    assert report["status"] == "PARTIAL"
    assert report["reason"] == "AWAITING_FINAL_HOLDOUT"
    assert report["selected_research_dataset"] == "MAX_COVERAGE"
    assert report["research_datasets"]["MAX_COVERAGE"]["train_rows"] == 5394
    assert report["research_datasets"]["COMMON_MODEL_ERA"]["train_rows"] == 523
    assert report["research_datasets"]["MAX_COVERAGE"]["dev_validation_rows"] == 416
    assert development.model.trained_until == date(2026, 2, 28)
    assert report["selected_meta_train_rows"] == 5810
    assert report["selected_min_available_models"] == 3
    assert report["min_available_models_review"][3]["dev_validation_metrics"]["log_loss"] < (
        report["min_available_models_review"][2]["dev_validation_metrics"]["log_loss"])
    assert report["final_holdout_rows"] == 0
    assert report["market_status"] == "BLOCKED_MARKET_DATA"
    assert report["recursive_lineage_validation"]["nested_training_vectors_checked"] == 6844
    assert report["recursive_lineage_validation"]["ml_artifacts_checked"] == 6


def test_calibration_development_only_and_honest_benchmark(development):
    report = development.report
    assert set(report["calibration_methods"]) == {
        "NONE", "TEMPERATURE_SCALING", "MULTICLASS_LOGIT_CALIBRATION"}
    assert report["selected_calibration_method"] == "TEMPERATURE_SCALING"
    assert all(value["calibration_fit_rows"] == 347 for value in report["calibration_methods"].values())
    assert report["common_sample_development"]["match_count"] == 167
    assert report["common_sample_development"]["evaluation_status"] == "DEVELOPMENT_ONLY"
    assert report["calibrated_minus_dixon_coles_log_loss"] > 0
    assert report["final_holdout_performance"] == "UNAVAILABLE"
    assert report["common_sample_development"]["calibrated_minus_single_model"]["dc"]["log_loss"] > 0


def test_artifact_save_load_and_weight_checksum(development, tmp_path):
    artifact = save_artifact(development, root=tmp_path)
    loaded = load_artifact(artifact.path, expected_candidate_hash=DATA_HASH)
    row = development.calibration_eval_rows
    raw1, calibrated1 = artifact.predict_proba(row)
    raw2, calibrated2 = loaded.predict_proba(row)
    np.testing.assert_allclose(raw1, raw2, atol=0, rtol=0)
    np.testing.assert_allclose(calibrated1, calibrated2, atol=0, rtol=0)
    assert artifact.manifest["temporal_mode"] == "DATE_SAFE_BATCH"
    assert artifact.manifest["production_compatibility"] == "RESEARCH_ONLY_NOT_LIVE_VALIDATED"
    assert artifact.manifest["feature_schema_version"] == "META_LOG_RATIO_MASK_V1"
    assert (artifact.path / "calibrator" / "manifest.json").is_file()
    with pytest.raises(ValueError, match="META_ARTIFACT_PROVENANCE_REJECTED"):
        load_artifact(artifact.path, expected_candidate_hash="0" * 64)


def test_calibrated_core_v2_development_status(development, tmp_path):
    artifact = save_artifact(development, root=tmp_path)
    match_id = str(development.calibration_eval_rows.frame.iloc[0]["match_id"])
    output = build_development_core(DB, candidate=development.candidate,
        artifact=artifact, match_id=match_id,
        evaluation_start=date(2026, 5, 1), evaluation_end=date(2026, 5, 31))
    assert output.core.validation_status == "DEVELOPMENT_VALIDATED_DATE_SAFE"
    assert output.core.final_core_probability == output.core.calibrated_probability
    assert output.core.meta_artifact_id == artifact.artifact_id
    assert len(output.core.source_prediction_ids) == 8
    assert output.canonical.probability_stage == ProbabilityStage.FINAL_CORE_CALIBRATED
    assert output.canonical.validation_status == "DEVELOPMENT_ONLY"
    assert output.canonical.market_probabilities is None
    assert output.canonical.production_status == "NOT_PROMOTED"
    assert output.meta_raw.probability_stage == "META_RAW"
    assert output.calibrated_core.calibrator_id == artifact.manifest["calibrator_id"]
    assert output.disagreement.available_model_count == 8
    preview = CoreReportV2PreviewAdapter().render(output.canonical)
    assert preview["production_status"] == "NOT_PROMOTED"
    assert not {"ranked_candidate", "bet_advice", "portfolio", "stake"}.intersection(preview)
    assert output.canonical.data_lineage["market_dependency_count"] == 0
    assert len(output.canonical.data_lineage["source_snapshot_ids"]) >= 1
    with pytest.raises(ValueError, match="CORE_DEVELOPMENT_EVAL_WINDOW_REQUIRED"):
        build_development_core(DB, candidate=development.candidate, artifact=artifact,
            match_id=match_id, evaluation_start=date(2026, 4, 1),
            evaluation_end=date(2026, 4, 30))


def test_calibrator_math_synthetic_test():
    """SYNTHETIC_TEST numerical fixture; never enters real results."""
    probabilities = np.array([[0.65, 0.2, 0.15], [0.2, 0.6, 0.2],
                              [0.15, 0.2, 0.65]] * 20)
    labels = np.array([0, 1, 2] * 20)
    for method in ("NONE", "TEMPERATURE_SCALING", "MULTICLASS_LOGIT_CALIBRATION"):
        calibrated = MulticlassCalibrator(method).fit(probabilities, labels).predict_proba(probabilities)
        assert np.isfinite(calibrated).all()
        np.testing.assert_allclose(calibrated.sum(axis=1), 1, atol=1e-12)


def test_config_holdout_lock_and_no_market(development):
    report = development.report
    assert report["development_windows"]["final_holdout"] == ["2026-08-01", "2027-06-30"]
    assert report["candidate_audit"]["market_dependency_count"] == 0


def test_phase9_self_review_is_development_only(development, tmp_path):
    artifact = save_artifact(development, root=tmp_path / "artifacts")
    match_id = str(development.calibration_eval_rows.frame.iloc[0]["match_id"])
    output = build_development_core(DB, candidate=development.candidate,
        artifact=artifact, match_id=match_id,
        evaluation_start=date(2026, 5, 1), evaluation_end=date(2026, 5, 31))
    report_path = tmp_path / "report.json"
    example_path = tmp_path / "example.json"
    report = dict(development.report)
    report["promotion_decision"] = MetaPromotionGate().evaluate(
        report, artifact_verified=True).model_dump(mode="json")
    report_path.write_text(json.dumps(report), encoding="utf-8")
    example_path.write_text(json.dumps({
        "meta_raw": output.meta_raw.model_dump(mode="json"),
        "calibrated_core": output.calibrated_core.model_dump(mode="json"),
        "disagreement": output.disagreement.model_dump(mode="json"),
        "canonical": output.canonical.model_dump(mode="json")}),
                            encoding="utf-8")
    audit = audit_phase9(DB, config_path=CONFIG, artifact_path=artifact.path,
                         report_path=report_path, example_path=example_path)
    assert audit["status"] == "PASS_WITH_LIMITATIONS"
    assert audit["availability_time_confounding"]["observed"]
    assert audit["holdout_preservation"] == "PASS_0_EVALUATED_ROWS"
    assert audit["meta_minus_dixon_coles_log_loss"] > 0


def test_log_ratio_shared_epsilon_and_missing_mask(real_candidate):
    row = real_candidate.features.iloc[[0]]
    encoded = encode_features(row, epsilon=1e-6)
    for index, prefix in enumerate(("dc", "bivariate", "elo", "pi", "spi", "opta_like", "xgb", "cat")):
        if row.iloc[0][f"{prefix}_available"] == 0:
            assert encoded[0, index * 3:index * 3 + 3].tolist() == [0.0, 0.0, 0.0]
        else:
            home = float(row.iloc[0][f"{prefix}_home"])
            draw = float(row.iloc[0][f"{prefix}_draw"])
            assert encoded[0, index * 3] == pytest.approx(np.log((home + 1e-6) / (draw + 1e-6)))
            assert encoded[0, index * 3 + 2] == 1.0


def test_lineage_cycle_rejected():
    with pytest.raises(ValueError, match="LINEAGE_CYCLE"):
        assert_acyclic({"A": ("B",), "B": ("A",)})


def test_promotion_gate_blocks_unearned_production(development):
    decision = MetaPromotionGate().evaluate(development.report, artifact_verified=True)
    assert decision.production_promoted is False
    assert decision.superiority_evidence == "NOT_ESTABLISHED"
    with pytest.raises(ValueError, match="PROMOTION_PREREQUISITES_NOT_MET"):
        MetaPromotionDecision(**(decision.model_dump() | {"production_promoted": True}))


def test_disagreement_is_diagnostic_only():
    result = measure_disagreement("match_1", [(0.5, 0.3, 0.2), (0.4, 0.35, 0.25)])
    assert result.purpose == "DIAGNOSTIC_ONLY"
    assert result.available_model_count == 2
    assert result.probability_range["home"] == pytest.approx(0.1)


def test_meta_raw_contract_rejects_invalid_probability(development, tmp_path):
    artifact = save_artifact(development, root=tmp_path)
    match_id = str(development.calibration_eval_rows.frame.iloc[0]["match_id"])
    output = build_development_core(DB, candidate=development.candidate,
        artifact=artifact, match_id=match_id,
        evaluation_start=date(2026, 5, 1), evaluation_end=date(2026, 5, 31))
    payload = output.meta_raw.model_dump()
    payload["p_home"] = 0.99
    with pytest.raises(ValueError, match="sum to 1"):
        MetaRawPrediction.model_validate(payload)


def test_calibrator_manifest_tamper_rejected(development, tmp_path):
    artifact = save_artifact(development, root=tmp_path)
    manifest_path = artifact.path / "calibrator" / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["fit_end"] = "2027-01-01"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="CALIBRATOR_MANIFEST_HASH_MISMATCH"):
        load_artifact(artifact.path, expected_candidate_hash=DATA_HASH)
