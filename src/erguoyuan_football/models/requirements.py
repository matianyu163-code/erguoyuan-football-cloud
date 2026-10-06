"""Declarative minimum dependencies only; no model implementations."""

from typing import Literal

from erguoyuan_football.contracts.common import Availability, Contract, ExecutionStatus
from erguoyuan_football.data.availability import DataAvailabilityReport


class ModelRequirements(Contract):
    model_id: str
    required: tuple[str, ...]
    optional: tuple[str, ...] = ()
    role: Literal["BASE_MODEL", "SIMULATION_ENGINE"] = "BASE_MODEL"
    eligible_for_meta: bool = True


class RequirementDecision(Contract):
    match_id: str
    model_id: str
    execution_status: ExecutionStatus
    missing_required: tuple[str, ...]
    reason: str


REQUIREMENTS = {
    item.model_id: item for item in (
        ModelRequirements(model_id="DIXON_COLES", required=("historical_goals",), optional=("market_odds", "xg")),
        ModelRequirements(model_id="BIVARIATE_POISSON", required=("historical_goals",)),
        ModelRequirements(model_id="DYNAMIC_BAYESIAN_POISSON", required=("historical_goals",)),
        ModelRequirements(model_id="BAYESIAN_HIERARCHICAL", required=("historical_goals", "league_hierarchy")),
        ModelRequirements(model_id="ELO", required=("historical_results",)),
        ModelRequirements(model_id="PI_RATING", required=("historical_goals",)),
        ModelRequirements(model_id="CORE_SPI_LIKE_V1", required=("historical_results",),
                          optional=("xg", "league_hierarchy")),
        ModelRequirements(model_id="CORE_OPTA_XG_ELO_LIKE_V1", required=("historical_results",),
                          optional=("xg", "league_hierarchy")),
        ModelRequirements(model_id="OPTA_SUPERCOMPUTER_LIKE", required=("final_probability",),
                          role="SIMULATION_ENGINE", eligible_for_meta=False),
        ModelRequirements(model_id="HISTORICAL_MARKET_BAYESIAN_POISSON_V1",
                          required=("historical_goals", "market_odds"),
                          optional=("opening_odds", "asian_handicap", "totals")),
        ModelRequirements(model_id="XGBOOST", required=("historical_results", "team_stats"), optional=("market_odds", "xg", "lineup")),
        ModelRequirements(model_id="CATBOOST", required=("historical_results", "team_stats"), optional=("market_odds", "xg", "lineup")),
        ModelRequirements(model_id="CORE_XGBOOST_V1", required=("ml_feature_vector",),
                          optional=("market_odds", "xg")),
        ModelRequirements(model_id="CORE_CATBOOST_V1", required=("ml_feature_vector",),
                          optional=("market_odds", "xg")),
    )
}


def check_requirements(model_id: str, report: DataAvailabilityReport) -> RequirementDecision:
    requirement = REQUIREMENTS[model_id]
    missing = tuple(name for name in requirement.required if name not in report.items
                    or report.items[name].availability != Availability.AVAILABLE)
    return RequirementDecision(match_id=report.match_id, model_id=model_id, missing_required=missing,
                               execution_status=ExecutionStatus.UNAVAILABLE if missing else ExecutionStatus.SKIPPED,
                               reason="MISSING_REQUIRED_DATA" if missing else "NOT_IMPLEMENTED")
