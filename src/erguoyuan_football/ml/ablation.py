"""Versioned feature-family ablations for chronological OOS diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from erguoyuan_football.ml.dataset_builder import make_dataset
from erguoyuan_football.ml.feature_contract import (
    FORM_FEATURES,
    GOAL_PREFIXES,
    RATING_PREFIXES,
    XG_FEATURES,
)
from erguoyuan_football.ml.schemas import (
    FeatureMode,
    FeatureSchema,
    MLDataset,
    MLFeatureVector,
    MLTrainingRow,
    stable_hash,
)


class Ablation(StrEnum):
    RATINGS_ONLY = "RATINGS_ONLY"
    RATINGS_GOALS = "RATINGS_GOALS"
    RATINGS_GOALS_FORM_XG = "RATINGS_GOALS_FORM_XG"
    FULL_NO_MARKET = "FULL_NO_MARKET"
    FULL_WITH_MARKET = "FULL_WITH_MARKET"


@dataclass(frozen=True)
class AblationDataset:
    ablation: Ablation
    dataset: MLDataset
    excluded_features: tuple[str, ...]


class MLFeatureAblation:
    """Create a new immutable schema/dataset; never mutate a frozen full dataset."""

    @staticmethod
    def transform(dataset: MLDataset, ablation: Ablation) -> AblationDataset:
        source = dataset.feature_schema
        if ablation == Ablation.FULL_WITH_MARKET and source.mode != FeatureMode.WITH_MARKET:
            raise ValueError("WITH_MARKET_ABLATION_REQUIRES_MARKET_DATASET")
        if ablation == Ablation.FULL_NO_MARKET and source.mode != FeatureMode.NO_MARKET:
            raise ValueError("FULL_NO_MARKET_REQUIRES_NO_MARKET_DATASET")
        if ablation == Ablation.FULL_WITH_MARKET and any(
                row.vector.features.get("market_available") != 1 for row in dataset.rows):
            raise ValueError("MARKET_ABLATION_REQUIRES_AVAILABLE_MARKET")
        if ablation in {Ablation.FULL_NO_MARKET, Ablation.FULL_WITH_MARKET}:
            return AblationDataset(ablation, dataset, ())
        base = {"horizon_seconds", "neutral_venue", "competition_id"}
        prefixes = RATING_PREFIXES if ablation == Ablation.RATINGS_ONLY else RATING_PREFIXES + GOAL_PREFIXES
        selected = base | {name for name in source.feature_names if any(
            name.startswith(prefix + "_") for prefix in prefixes) and not name.startswith("market_bayes_")}
        if ablation == Ablation.RATINGS_GOALS_FORM_XG:
            selected.update((*FORM_FEATURES, *XG_FEATURES, "form_available", "xg_available"))
        names = tuple(name for name in source.feature_names if name in selected)
        if not names:
            raise ValueError("EMPTY_ML_ABLATION")
        categories = tuple(name for name in source.categorical_features if name in selected)
        numeric = tuple(name for name in source.numeric_features if name in selected)
        schema = FeatureSchema(schema_version=f"ML_ABLATION_V1_{ablation.value}",
            feature_names=names, feature_types={name: source.feature_types[name] for name in names},
            categorical_features=categories, numeric_features=numeric,
            nullable_features=tuple(name for name in source.nullable_features if name in selected),
            mode=FeatureMode.NO_MARKET, model_family=source.model_family)
        rows: list[MLTrainingRow] = []
        for row in dataset.rows:
            old = row.vector
            lineage = tuple(item.model_copy(update={"feature_names": tuple(
                name for name in item.feature_names if name in selected)})
                for item in old.feature_lineage if any(name in selected for name in item.feature_names))
            evidence = tuple(item for item in old.base_prediction_evidence if any(
                name.startswith(_prefix_for(item.prediction.model_id) + "_") for name in names))
            draft: dict[str, Any] = old.model_dump(exclude={"feature_data_hash"})
            draft.update(feature_version=f"{old.feature_version}:{ablation.value}",
                feature_schema_hash=schema.schema_hash, feature_mode=FeatureMode.NO_MARKET,
                features={name: old.features[name] for name in names},
                feature_lineage=lineage, base_prediction_evidence=evidence)
            provisional = MLFeatureVector.model_construct(feature_data_hash="PENDING", **draft)
            vector = MLFeatureVector.model_validate({**draft, "feature_data_hash": stable_hash(
                provisional.model_dump(mode="json", exclude={"feature_data_hash"}))})
            rows.append(row.model_copy(update={"vector": vector}))
        return AblationDataset(ablation, make_dataset(tuple(rows), schema, dataset_kind=dataset.dataset_kind),
                               tuple(name for name in source.feature_names if name not in selected))


def _prefix_for(model_id: str) -> str:
    from erguoyuan_football.ml.feature_contract import MODEL_PREFIXES
    return MODEL_PREFIXES.get(model_id, "UNKNOWN")
