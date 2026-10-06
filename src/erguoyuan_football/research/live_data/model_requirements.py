"""Readiness-only requirements for actual registered model IDs; no model import."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import yaml


class RequirementLevel(StrEnum):
    """Blocking, optional, and enhancement-only data dependencies."""

    REQUIRED = "REQUIRED"
    OPTIONAL = "OPTIONAL"
    PRIOR = "PRIOR"
    ENHANCEMENT = "ENHANCEMENT"


@dataclass(frozen=True)
class TrainingRequirement:
    """Fit-time sufficiency; never inferred from target-team recency windows."""

    min_training_rows: int | None = None
    artifact_required: bool = False
    training_cutoff_required: bool = True
    feature_schema_required: bool = False
    sample_levels: tuple[str, ...] = ()


@dataclass(frozen=True)
class InferenceRequirement:
    """Per-match evidence requirements, separate from reusable fit history."""

    min_direct_matches_per_team: int | None = None
    preferred_direct_matches_per_team: int | None = None
    min_competition_matches: int | None = None
    supports_prior: bool = False
    required_features: frozenset[str] = frozenset()
    min_prior_samples: int = 8


@dataclass(frozen=True)
class ModelSamplePolicy:
    """One model's declared current fit threshold, independent of other models."""

    min_total: int
    min_direct_each: int
    min_competition: int
    prior_allowed: bool
    require_three_outcomes: bool = False
    require_xg_for_named_mode: bool = False
    require_current_odds: bool = False
    require_artifact: bool = False
    allow_multi_competition: bool = False
    min_training_team_matches: int = 3
    preferred_direct_each: int = 10
    training_min_matches: int | None = None


@dataclass(frozen=True)
class ModelDataRequirement:
    """One existing model's data needs, independent of execution."""

    model_id: str
    inputs: dict[str, RequirementLevel]
    sample_policy: ModelSamplePolicy | None = None
    training: TrainingRequirement | None = None
    inference: InferenceRequirement | None = None


_REQUIREMENTS: dict[str, dict[str, RequirementLevel]] = {
    "DIXON_COLES_V1": {"HISTORICAL_RESULTS": RequirementLevel.REQUIRED,
                       "HOME_AWAY_SPLIT": RequirementLevel.OPTIONAL},
    "BIVARIATE_POISSON_V1": {"HISTORICAL_RESULTS": RequirementLevel.REQUIRED},
    "BAYESIAN_HIERARCHICAL_V1": {"HISTORICAL_RESULTS": RequirementLevel.REQUIRED,
                                 "COMPETITION_CONTEXT": RequirementLevel.OPTIONAL},
    "ELO_V1": {"HISTORICAL_RESULTS": RequirementLevel.REQUIRED,
               "THREE_WAY_MAPPER_TRAINING": RequirementLevel.REQUIRED},
    "PI_RATING_V1": {"HISTORICAL_RESULTS": RequirementLevel.REQUIRED,
                     "THREE_WAY_MAPPER_TRAINING": RequirementLevel.REQUIRED},
    "DYNAMIC_BAYESIAN_POISSON_V1": {"HISTORICAL_RESULTS": RequirementLevel.REQUIRED,
                                    "MATCH_TIMESTAMPS": RequirementLevel.REQUIRED},
    "CORE_SPI_LIKE_V1": {"HISTORICAL_RESULTS": RequirementLevel.REQUIRED,
                         "XG": RequirementLevel.ENHANCEMENT},
    "CORE_OPTA_XG_ELO_LIKE_V1": {"HISTORICAL_RESULTS": RequirementLevel.REQUIRED,
                                 "THREE_WAY_MAPPER_TRAINING": RequirementLevel.REQUIRED,
                                 "XG": RequirementLevel.ENHANCEMENT,
                                 "XGA": RequirementLevel.ENHANCEMENT},
    "HISTORICAL_MARKET_BAYESIAN_POISSON_V1": {
        "HISTORICAL_RESULTS": RequirementLevel.REQUIRED,
        "ODDS": RequirementLevel.REQUIRED},
    "CORE_XGBOOST_V1": {"ML_FEATURE_VECTOR": RequirementLevel.REQUIRED,
                         "ML_ARTIFACT": RequirementLevel.REQUIRED,
                         "XG": RequirementLevel.ENHANCEMENT},
    "CORE_CATBOOST_V1": {"ML_FEATURE_VECTOR": RequirementLevel.REQUIRED,
                         "ML_ARTIFACT": RequirementLevel.REQUIRED,
                         "XG": RequirementLevel.ENHANCEMENT},
}


def load_model_requirements(registry_path: Path | str) -> tuple[ModelDataRequirement, ...]:
    """Refuse drift from the actual model registry instead of inventing names."""
    raw = yaml.safe_load(Path(registry_path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("models"), list):
        raise TypeError("MODEL_REGISTRY_INVALID")
    identifiers: list[str] = []
    for row in raw["models"]:
        if isinstance(row, dict) and row.get("enabled") is True:
            model_id = row.get("model_id")
            if not isinstance(model_id, str):
                raise TypeError("MODEL_ID_REQUIRED")
            identifiers.append(model_id)
    if len(identifiers) != len(set(identifiers)) or set(identifiers) != set(_REQUIREMENTS):
        raise ValueError("MODEL_READINESS_REGISTRY_DRIFT")
    policy_path = Path(registry_path).parent / "phase13_6_model_readiness.yaml"
    policy_raw = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    model_policies = policy_raw.get("models") if isinstance(policy_raw, dict) else None
    if not isinstance(model_policies, dict) or set(model_policies) != set(identifiers):
        raise ValueError("MODEL_SAMPLE_POLICY_REGISTRY_DRIFT")
    policies = {model_id: ModelSamplePolicy(**model_policies[model_id])
                for model_id in identifiers}
    if any(min(policy.min_total, policy.min_direct_each,
               policy.min_competition) < 0 for policy in policies.values()):
        raise ValueError("INVALID_MODEL_SAMPLE_THRESHOLD")
    training_raw = policy_raw.get("training_requirements", {})
    inference_raw = policy_raw.get("inference_requirements", {})
    return tuple(ModelDataRequirement(
        model_id, dict(_REQUIREMENTS[model_id]), policies[model_id],
        TrainingRequirement(**{
            **training_raw.get(model_id, {}),
            "sample_levels": tuple(training_raw.get(model_id, {}).get("sample_levels", ())),
        }),
        InferenceRequirement(**{
            **inference_raw.get(model_id, {}),
            "required_features": frozenset(inference_raw.get(model_id, {}).get(
                "required_features", ())),
        }),
    )
                 for model_id in identifiers)
