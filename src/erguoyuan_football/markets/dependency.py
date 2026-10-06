"""Explicit market-information lineage for models and future prediction bundles."""

from __future__ import annotations

from erguoyuan_football.contracts.common import Contract
from erguoyuan_football.markets.schemas import DependencyTag


class FeatureDependency(Contract):
    feature_id: str
    dependency_tags: tuple[DependencyTag, ...]
    parent_feature_ids: tuple[str, ...] = ()
    source_quote_ids: tuple[str, ...] = ()
    contains_market_information: bool = False

    @classmethod
    def market_derived(cls, feature_id: str, *, source_quote_ids: tuple[str, ...],
                       parents: tuple[str, ...] = ()) -> FeatureDependency:
        return cls(feature_id=feature_id, dependency_tags=(
            DependencyTag.MARKET_RAW, DependencyTag.MARKET_DEVIG, DependencyTag.MARKET_CONSENSUS,
        ), parent_feature_ids=parents, source_quote_ids=source_quote_ids, contains_market_information=True)


class ModelDependencyDeclaration(Contract):
    model_id: str
    uses_historical_goals: bool = False
    uses_historical_results: bool = False
    uses_xg: bool = False
    uses_market: bool = False
    market_dependency_tags: tuple[DependencyTag, ...] = ()


class FeatureDependencyGraph(Contract):
    """DAG with cycle/missing-parent checks and stable market dependency propagation."""

    features: tuple[FeatureDependency, ...] = ()

    def validate_graph(self) -> None:
        by_id = {feature.feature_id: feature for feature in self.features}
        if len(by_id) != len(self.features):
            raise ValueError("DUPLICATE_FEATURE_ID")
        for feature in self.features:
            missing = set(feature.parent_feature_ids) - set(by_id)
            if missing:
                raise ValueError("MISSING_FEATURE_PARENT:" + ",".join(sorted(missing)))
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(feature_id: str) -> None:
            if feature_id in visiting:
                raise ValueError("FEATURE_DEPENDENCY_CYCLE")
            if feature_id in visited:
                return
            visiting.add(feature_id)
            for parent in by_id[feature_id].parent_feature_ids:
                visit(parent)
            visiting.remove(feature_id)
            visited.add(feature_id)

        for feature_id in by_id:
            visit(feature_id)

    def lineage_for(self, feature_id: str) -> tuple[FeatureDependency, ...]:
        self.validate_graph()
        by_id = {feature.feature_id: feature for feature in self.features}
        if feature_id not in by_id:
            raise KeyError(feature_id)
        found: dict[str, FeatureDependency] = {}

        def collect(item: str) -> None:
            feature = by_id[item]
            found[item] = feature
            for parent in feature.parent_feature_ids:
                collect(parent)

        collect(feature_id)
        return tuple(found[key] for key in sorted(found))

    def tags_for(self, feature_id: str) -> tuple[DependencyTag, ...]:
        return tuple(sorted({tag for item in self.lineage_for(feature_id)
                             for tag in item.dependency_tags}, key=lambda item: item.value))


MODEL_DEPENDENCIES: dict[str, ModelDependencyDeclaration] = {
    "DIXON_COLES_V1": ModelDependencyDeclaration(model_id="DIXON_COLES_V1", uses_historical_goals=True),
    "BIVARIATE_POISSON_V1": ModelDependencyDeclaration(model_id="BIVARIATE_POISSON_V1", uses_historical_goals=True),
    "BAYESIAN_HIERARCHICAL_V1": ModelDependencyDeclaration(model_id="BAYESIAN_HIERARCHICAL_V1", uses_historical_goals=True),
    "ELO_V1": ModelDependencyDeclaration(model_id="ELO_V1", uses_historical_results=True),
    "PI_RATING_V1": ModelDependencyDeclaration(model_id="PI_RATING_V1", uses_historical_goals=True),
    "DYNAMIC_BAYESIAN_POISSON_V1": ModelDependencyDeclaration(model_id="DYNAMIC_BAYESIAN_POISSON_V1", uses_historical_goals=True),
    "CORE_SPI_LIKE_V1": ModelDependencyDeclaration(model_id="CORE_SPI_LIKE_V1", uses_historical_results=True),
    "CORE_OPTA_XG_ELO_LIKE_V1": ModelDependencyDeclaration(model_id="CORE_OPTA_XG_ELO_LIKE_V1", uses_historical_results=True),
    "HISTORICAL_MARKET_BAYESIAN_POISSON_V1": ModelDependencyDeclaration(
        model_id="HISTORICAL_MARKET_BAYESIAN_POISSON_V1", uses_historical_goals=True,
        uses_market=True, market_dependency_tags=(
            DependencyTag.MARKET_RAW, DependencyTag.MARKET_DEVIG, DependencyTag.MARKET_CONSENSUS,
            DependencyTag.MARKET_IMPLIED_GOALS, DependencyTag.MARKET_BAYES,
        ),
    ),
}


def declaration_for(model_id: str) -> ModelDependencyDeclaration:
    """Return a typed dependency declaration; unknown models do not inherit defaults."""
    try:
        return MODEL_DEPENDENCIES[model_id]
    except KeyError as error:
        raise KeyError(f"MODEL_DEPENDENCY_UNDECLARED:{model_id}") from error
