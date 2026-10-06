"""Persist several real intermediate CORE V2 examples without final predictions."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import duckdb

from erguoyuan_football.contracts.predictions import ModelPrediction
from erguoyuan_football.output_contract.builder import CanonicalPredictionResultBuilder


def create_pre_meta_examples(db_path: str | Path, model_ids: tuple[str, ...]) -> dict[str, str]:
    """Use one successful persisted REAL OOS record per requested model."""
    builder = CanonicalPredictionResultBuilder()
    examples: dict[str, str] = {}
    with duckdb.connect(str(db_path)) as connection:
        for model_id in model_ids:
            source = connection.execute("""SELECT payload,competition_id,match_date
                FROM real_oos_predictions_v2 WHERE model_id=? AND data_origin='REAL'
                AND is_oos ORDER BY match_date,match_id LIMIT 1""", [model_id]).fetchone()
            if source is None:
                continue
            payload, competition_id, match_date = source
            prediction = ModelPrediction.model_validate_json(payload)
            canonical = builder.from_date_safe_prediction(prediction,
                competition_id=competition_id, match_date=match_date)
            connection.execute("""INSERT INTO real_canonical_predictions_v2 VALUES
                (?,?,?,?,?,?,?,?) ON CONFLICT DO NOTHING""",
                [canonical.prediction_id, canonical.match_id,
                 canonical.prediction_snapshot_id, canonical.probability_stage.value,
                 canonical.prediction_temporal_mode, canonical.data_origin,
                 datetime.now(UTC), canonical.model_dump_json()])
            examples[model_id] = canonical.prediction_id
    return examples
