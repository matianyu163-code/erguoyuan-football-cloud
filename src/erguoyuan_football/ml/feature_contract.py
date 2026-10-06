"""Fixed feature vocabulary and explicit prohibition of outcome/future columns."""

from __future__ import annotations

from erguoyuan_football.ml.schemas import FeatureMode, FeatureSchema, ModelFamily

MODEL_PREFIXES = {
    "DIXON_COLES_V1": "dc",
    "BIVARIATE_POISSON_V1": "bivariate",
    "BAYESIAN_HIERARCHICAL_V1": "hier_bayes",
    "ELO_V1": "elo",
    "PI_RATING_V1": "pi",
    "DYNAMIC_BAYESIAN_POISSON_V1": "dynamic_bayes",
    "CORE_SPI_LIKE_V1": "spi",
    "CORE_OPTA_XG_ELO_LIKE_V1": "opta_like",
    "HISTORICAL_MARKET_BAYESIAN_POISSON_V1": "market_bayes",
}
GOAL_PREFIXES = ("dc", "bivariate", "hier_bayes", "dynamic_bayes", "market_bayes")
RATING_PREFIXES = ("elo", "pi", "spi", "opta_like")
FORM_FEATURES = (
    "home_goals_for_avg", "home_goals_against_avg", "home_points_avg", "home_rest_days",
    "away_goals_for_avg", "away_goals_against_avg", "away_points_avg", "away_rest_days",
)
XG_FEATURES = ("home_xg", "home_xga", "away_xg", "away_xga")
MARKET_FEATURES = (
    "market_consensus_home", "market_consensus_draw", "market_consensus_away",
    "market_overround", "market_freshness_seconds", "market_available",
)
FORBIDDEN_TOKENS = ("post_match", "final_score", "future", "closing", "actual_result",
                    "target", "home_team_id", "away_team_id", "final_xg")


def build_feature_schema(mode: FeatureMode, family: ModelFamily) -> FeatureSchema:
    """One deterministic ordered schema per mode/library; never learn columns from test data."""
    prefixes = tuple(prefix for prefix in MODEL_PREFIXES.values()
                     if mode == FeatureMode.WITH_MARKET or prefix != "market_bayes")
    names: list[str] = ["horizon_seconds", "neutral_venue"]
    for prefix in prefixes:
        names.extend((f"{prefix}_p_home", f"{prefix}_p_draw", f"{prefix}_p_away",
                      f"{prefix}_available"))
        if prefix in GOAL_PREFIXES:
            names.extend((f"{prefix}_lambda_home", f"{prefix}_lambda_away"))
    names.extend(FORM_FEATURES)
    names.extend(XG_FEATURES)
    names.extend(("xg_available", "form_available"))
    if mode == FeatureMode.WITH_MARKET:
        names.extend(MARKET_FEATURES)
    categories = ("competition_id",) if family == ModelFamily.CATBOOST else ()
    names.extend(categories)
    if any(any(token in name for token in FORBIDDEN_TOKENS) for name in names):
        raise ValueError("FORBIDDEN_FEATURE_NAME")
    numeric = tuple(name for name in names if name not in categories)
    nonnull = {"horizon_seconds", "xg_available", "form_available", "market_available"}
    nonnull.update(name for name in names if name.endswith("_available"))
    return FeatureSchema(feature_names=tuple(names), feature_types={name: "categorical" if name in categories
        else "numeric" for name in names}, categorical_features=categories, numeric_features=numeric,
        nullable_features=tuple(name for name in names if name not in nonnull), mode=mode, model_family=family)


def validate_feature_values(features: dict[str, float | str | None], schema: FeatureSchema) -> None:
    """Reject changed names/order/types and unknown missingness, never silently fill columns."""
    if tuple(features) != schema.feature_names:
        raise ValueError("FEATURE_SCHEMA_MISMATCH")
    for name, value in features.items():
        if value is None:
            if name not in schema.nullable_features:
                raise ValueError(f"FEATURE_REQUIRED_MISSING:{name}")
        elif name in schema.numeric_features and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ValueError(f"FEATURE_NUMERIC_TYPE_MISMATCH:{name}")
        elif name in schema.categorical_features and not isinstance(value, str):
            raise ValueError(f"FEATURE_CATEGORY_TYPE_MISMATCH:{name}")
