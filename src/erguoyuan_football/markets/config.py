"""Validated market-engine controls loaded from the standalone market YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import Field, model_validator

from erguoyuan_football.contracts.common import Contract
from erguoyuan_football.markets.schemas import (
    ConsensusMethod,
    DeVigMethod,
    DeVigPolicy,
    MarketType,
)


class MarketConfig(Contract):
    """Versioned provisional v1 policies; defaults are not OOS-validated claims."""

    config_version: str = "MARKET_ENGINE_V1_PROVISIONAL"
    consensus_method: ConsensusMethod = ConsensusMethod.MEDIAN
    minimum_bookmakers: int = Field(default=2, ge=1)
    trim_fraction: float = Field(default=0.1, ge=0, lt=0.5)
    maximum_overround: dict[MarketType, float] = Field(default_factory=dict)
    freshness_seconds_by_horizon: dict[str, int] = Field(default_factory=lambda: {
        "T_MINUS_24H": 86_400, "T_MINUS_6H": 21_600, "T_MINUS_3H": 10_800,
        "T_MINUS_60M": 3_600, "T_MINUS_15M": 900, "CUSTOM": 21_600,
    })
    # Provisional classification windows pending frozen real-data horizon validation.
    horizon_tolerance_seconds: dict[str, int] = Field(default_factory=lambda: {
        "T_MINUS_24H": 10_800, "T_MINUS_6H": 5_400, "T_MINUS_3H": 2_700,
        "T_MINUS_60M": 900, "T_MINUS_15M": 300,
    })
    allowed_quality_statuses: tuple[Literal["GOOD", "ACCEPTABLE"], ...] = ("GOOD", "ACCEPTABLE")
    implied_goals_max_score: int = Field(default=12, ge=4, le=30)
    maximum_goal_fit_error: float = Field(default=0.12, gt=0)
    maximum_market_freshness_seconds: int = Field(default=86_400, ge=1)
    require_native_timestamp_for_backtest: bool = True
    devig_policy: dict[str, Any] = Field(default_factory=dict)
    bookmaker_weights: dict[str, float] = Field(default_factory=dict)
    weight_evidence_hash: str | None = None

    @model_validator(mode="after")
    def validate_market_controls(self) -> MarketConfig:
        if any(value < 0 for value in self.horizon_tolerance_seconds.values()):
            raise ValueError("MARKET_HORIZON_TOLERANCES_MUST_BE_NONNEGATIVE")
        if self.consensus_method == ConsensusMethod.WEIGHTED_MEAN:
            evidence = self.weight_evidence_hash or ""
            if len(evidence) != 64 or any(character not in "0123456789abcdef" for character in evidence.lower()):
                raise ValueError("WEIGHTED_CONSENSUS_REQUIRES_SHA256_OOS_EVIDENCE")
            if not self.bookmaker_weights:
                raise ValueError("WEIGHTED_CONSENSUS_REQUIRES_VALIDATED_WEIGHTS")
        return self


def load_market_config(path: str | Path) -> MarketConfig:
    """Load explicit market configuration without silently filling malformed YAML."""
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return MarketConfig.model_validate(payload)


def load_devig_policy(path: str | Path) -> DeVigPolicy:
    """Read the fixed-but-provisional devig mapping from market configuration."""
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    policy = payload.get("devig_policy")
    if not isinstance(policy, dict):
        raise TypeError("DEVIG_POLICY_NOT_CONFIGURED")
    methods = {MarketType(key): DeVigMethod(value) for key, value in policy.get("method_by_market", {}).items()}
    policy_status = cast(Literal["PROVISIONAL_NOT_VALIDATED", "VALIDATED"],
                          policy.get("status", "PROVISIONAL_NOT_VALIDATED"))
    return DeVigPolicy(policy_version=policy["policy_version"], method_by_market=methods,
                       status=policy_status,
                       validation_evidence_hash=policy.get("validation_evidence_hash"))
