"""Immutable labeled rows from frozen feature vectors and subsequently available results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, time, timedelta

from erguoyuan_football.contracts.common import utc
from erguoyuan_football.data.schemas import MatchResult
from erguoyuan_football.ml.feature_contract import validate_feature_values
from erguoyuan_football.ml.labels import ResultClass, label_result
from erguoyuan_football.ml.schemas import (
    FeatureSchema,
    MLDataset,
    MLDatasetManifest,
    MLFeatureVector,
    MLTrainingRow,
    stable_hash,
)


class MLDatasetBuilder:
    """Require one final score per pre-match vector, with label availability after play."""

    def build(self, vectors: Sequence[MLFeatureVector], results: Mapping[str, MatchResult], *,
              schema: FeatureSchema, competition_ids: Mapping[str, str],
              seasons: Mapping[str, str | None], dataset_cutoff: datetime,
              dataset_kind: str) -> MLDataset:
        if dataset_kind not in {"REAL", "SYNTHETIC_TEST"}:
            raise ValueError("UNKNOWN_ML_DATASET_KIND")
        cutoff = utc(dataset_cutoff)
        rows: list[MLTrainingRow] = []
        for vector in sorted(vectors, key=lambda item: (item.kickoff_time, item.match_id)):
            if vector.feature_schema_hash != schema.schema_hash or vector.feature_mode != schema.mode:
                raise ValueError("FEATURE_SCHEMA_MISMATCH")
            validate_feature_values(vector.features, schema)
            result = results.get(vector.match_id)
            if result is None:
                raise ValueError(f"RESULT_UNAVAILABLE:{vector.match_id}")
            available_at = max(result.completed_at, result.as_of_time, result.retrieved_at)
            if available_at > cutoff:
                raise ValueError("FUTURE_LABEL_REJECTED")
            rows.append(MLTrainingRow(vector=vector, label=label_result(result),
                result_id=result.result_id, label_available_at=available_at,
                competition_id=competition_ids[vector.match_id],
                season=seasons.get(vector.match_id)))
        return make_dataset(tuple(rows), schema, dataset_kind=dataset_kind)

    def build_date_safe(self, vectors: Sequence[MLFeatureVector], *,
                        results: Mapping[str, tuple[int, int, date, str]],
                        schema: FeatureSchema, competition_ids: Mapping[str, str],
                        seasons: Mapping[str, str], dataset_cutoff: date) -> MLDataset:
        """Build REAL rows from immutable results with next-day label eligibility.

        The source retrieval timestamp remains in the warehouse. This method
        uses event-date eligibility only and never treats late retrieval as an
        intraday historical snapshot.
        """
        rows: list[MLTrainingRow] = []
        for vector in sorted(vectors, key=lambda item: (item.prediction_time, item.match_id)):
            if vector.feature_schema_hash != schema.schema_hash:
                raise ValueError("DATE_SAFE_FEATURE_SCHEMA_MISMATCH")
            validate_feature_values(vector.features, schema)
            result = results.get(vector.match_id)
            if result is None:
                raise ValueError(f"RESULT_UNAVAILABLE:{vector.match_id}")
            home_goals, away_goals, event_date, result_id = result
            if event_date >= dataset_cutoff or vector.prediction_time.date() != event_date:
                raise ValueError("DATE_SAFE_LABEL_FUTURE_OR_IDENTITY_MISMATCH")
            label = (ResultClass.HOME if home_goals > away_goals else
                     ResultClass.DRAW if home_goals == away_goals else ResultClass.AWAY)
            available_at = datetime.combine(event_date + timedelta(days=1), time.min, UTC)
            rows.append(MLTrainingRow(vector=vector, label=label,
                result_id=result_id, label_available_at=available_at,
                competition_id=competition_ids[vector.match_id],
                season=seasons[vector.match_id]))
        return make_dataset(tuple(rows), schema, dataset_kind="REAL")


def make_dataset(rows: tuple[MLTrainingRow, ...], schema: FeatureSchema, *,
                 dataset_kind: str) -> MLDataset:
    """Create a new content-addressed dataset version; no in-place row mutation."""
    if not rows:
        raise ValueError("EMPTY_ML_DATASET")
    ordered = tuple(sorted(rows, key=lambda row: (row.vector.kickoff_time, row.vector.match_id)))
    data_hash = stable_hash([row.model_dump(mode="json") for row in ordered])
    manifest = MLDatasetManifest(feature_version=ordered[0].vector.feature_version,
        row_count=len(ordered), start_time=ordered[0].vector.kickoff_time,
        end_time=ordered[-1].vector.kickoff_time,
        competitions=tuple(sorted({row.competition_id for row in ordered})),
        feature_schema_hash=schema.schema_hash, data_hash=data_hash)
    return MLDataset(manifest=manifest, feature_schema=schema, rows=ordered,
                     dataset_kind=dataset_kind)  # type: ignore[arg-type]
