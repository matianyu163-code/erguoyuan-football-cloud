"""PIT market interpretation alongside, never inside, frozen R3 model output."""

from __future__ import annotations

import hashlib
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from erguoyuan_football.blind_test_r3.market_providers import (
    ManualMarketProvider,
    MarketProviderRegistry,
)
from erguoyuan_football.blind_test_r3.store import canonical_bytes
from erguoyuan_football.markets.config import MarketConfig
from erguoyuan_football.markets.consensus import MarketConsensusEngine
from erguoyuan_football.markets.movement import MarketMovementEngine
from erguoyuan_football.markets.schemas import (
    ConsensusMethod,
    DeVigMethod,
    DeVigPolicy,
    MarketQualityStatus,
    MarketType,
    OddsQuote,
)
from erguoyuan_football.markets.snapshot import build_market_snapshot


def _triple(values: dict[str, Any]) -> dict[str, float]:
    keys = ("HOME", "DRAW", "AWAY")
    if set(values) != set(keys) or any(isinstance(values[key], bool) or
        not isinstance(values[key], (int, float)) or
        not math.isfinite(values[key]) or not 0 <= values[key] <= 1 for key in keys):
        raise ValueError("MARKET_INTELLIGENCE_PROBABILITY_INVALID")
    result = {key: float(values[key]) for key in keys}
    if not math.isclose(math.fsum(result.values()), 1.0, abs_tol=1e-8):
        raise ValueError("MARKET_INTELLIGENCE_PROBABILITY_NOT_NORMALIZED")
    return result


def load_r3_market_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("version") != "R3_MARKET_INTELLIGENCE_V1" or (
        config.get("devig_method") != "MULTIPLICATIVE" or
        config.get("consensus_method") != "MEDIAN" or
        config["fusion"].get("version") != "MODEL_MARKET_FUSION_V1" or
        config["fusion"].get("dynamic_weight_training") is not False
    ):
        raise ValueError("R3_MARKET_CONFIG_VERSION_OR_METHOD_INVALID")
    warning = float(config["divergence"]["warning_pp"])
    severe = float(config["divergence"]["severe_pp"])
    if not 0 < warning < severe:
        raise ValueError("R3_MARKET_DIVERGENCE_THRESHOLDS_INVALID")
    model_weight = float(config["fusion"]["model_weight"])
    market_weight = float(config["fusion"]["market_weight"])
    if min(model_weight, market_weight) < 0 or model_weight + market_weight <= 0:
        raise ValueError("R3_FUSION_WEIGHTS_INVALID")
    return config


def fuse_model_market(model: dict[str, Any], market: dict[str, Any] | None,
                      config: dict[str, Any]) -> dict[str, Any]:
    """Fixed V1 weights; missing market preserves the original model triple."""
    model_values = _triple(model)
    weights = config["fusion"]
    if market is None:
        return {"mode": "MODEL_ONLY", "version": weights["version"],
                "model_weight": 1.0, "market_weight": 0.0,
                "configured_model_weight": weights["model_weight"],
                "configured_market_weight": weights["market_weight"],
                "probabilities": model_values}
    market_values = _triple(market)
    total = float(weights["model_weight"] + weights["market_weight"])
    model_weight = float(weights["model_weight"]) / total
    market_weight = float(weights["market_weight"]) / total
    result = {key: model_values[key] * model_weight +
              market_values[key] * market_weight for key in model_values}
    normalized = {key: value / math.fsum(result.values()) for key, value in result.items()}
    return {"mode": "MODEL_MARKET", "version": weights["version"],
            "model_weight": model_weight, "market_weight": market_weight,
            "probabilities": _triple(normalized)}


def r5_market_only(market: dict[str, Any] | None) -> dict[str, Any]:
    """Keep challenge route R5 independent of all R3 model and fusion inputs."""
    return {"mode": "MARKET_ONLY", "probabilities": _triple(market)} if market is not None else (
        {"mode": "MARKET_ONLY", "status": "UNAVAILABLE", "probabilities": None})


def build_market_intelligence(*, fixture_id: str, home_team: str, away_team: str,
                              kickoff_utc: datetime, prediction_time: datetime,
                              model_probabilities: dict[str, Any],
                              registry: MarketProviderRegistry,
                              config: dict[str, Any]) -> dict[str, Any]:
    """Fetch independent providers, reuse canonical PIT snapshot/consensus/movement."""
    if prediction_time.tzinfo is None or kickoff_utc.tzinfo is None or (
        prediction_time >= kickoff_utc):
        raise ValueError("R3_MARKET_NOT_PREMATCH")
    model = _triple(model_probabilities)
    quotes: list[OddsQuote] = []
    provider_status: dict[str, dict[str, str]] = {}
    evidence: dict[str, dict[str, str]] = {}
    for provider in registry.all():
        try:
            result = provider.fetch_1x2(fixture_id, home_team, away_team,
                                         kickoff_utc, prediction_time)
            provider_status[provider.provider_name] = {"status":
                "NOT_CONFIGURED" if provider.status == "NOT_CONFIGURED" else result.status,
                "reason": result.reason or provider.status}
            if result.status == "AVAILABLE":
                if result.event_identity is None or (
                    result.event_identity.home_team_name != home_team or
                    result.event_identity.away_team_name != away_team or
                    result.event_identity.kickoff_time != kickoff_utc):
                    raise ValueError("R3_MARKET_PROVIDER_FIXTURE_MISMATCH")
                if any(quote.match_id != fixture_id or quote.provider_id != provider.provider_name
                       or quote.as_of_time is None or quote.retrieved_at > prediction_time
                       or quote.as_of_time > prediction_time for quote in result.quotes):
                    raise ValueError("R3_MARKET_PROVIDER_QUOTE_PIT_OR_IDENTITY_INVALID")
                quotes.extend(result.quotes)
                if isinstance(provider, ManualMarketProvider) and provider.last_evidence:
                    evidence[provider.provider_name] = provider.last_evidence
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
            provider_status[provider.provider_name] = {
                "status": "ERROR", "reason": f"{type(error).__name__}:{error}"}
    quality = MarketQualityStatus.ACCEPTABLE if quotes else MarketQualityStatus.UNAVAILABLE
    snapshot = build_market_snapshot(tuple(quotes), match_id=fixture_id,
        prediction_time=prediction_time, kickoff_time=kickoff_utc,
        max_age_seconds=int(config["maximum_quote_age_seconds"]), quality_status=quality)
    market_config = MarketConfig(config_version="R3_MARKET_INTELLIGENCE_V1",
        consensus_method=ConsensusMethod.MEDIAN,
        minimum_bookmakers=int(config["minimum_bookmakers"]),
        maximum_overround={MarketType.MATCH_1X2: float(config["maximum_overround"])},
        maximum_market_freshness_seconds=int(config["maximum_quote_age_seconds"]))
    policy = DeVigPolicy(policy_version="R3_BASIC_PROPORTIONAL_V1",
        method_by_market={MarketType.MATCH_1X2: DeVigMethod.MULTIPLICATIVE})
    consensuses = MarketConsensusEngine(market_config, policy).build(snapshot)
    consensus = next((item for item in consensuses
                      if item.market_type == MarketType.MATCH_1X2), None)
    current = _triple(consensus.probabilities) if consensus else None
    opening = tuple(quote for quote in quotes if quote.is_opening_confirmed)
    movements = MarketMovementEngine().build(opening, tuple(quotes),
        match_id=fixture_id, prediction_time=prediction_time, policy=policy)
    movement_pp = None
    if movements:
        by_selection = {item.selection.value: item.delta_probability * 100
            for item in movements if item.delta_probability is not None}
        if set(by_selection) == {"HOME", "DRAW", "AWAY"}:
            movement_pp = {key + "_MOVE_PP": value for key, value in by_selection.items()}
            movement_pp["MAX_ABS_MOVE_PP"] = max(abs(value) for value in by_selection.values())
    edge = None
    divergence = "UNAVAILABLE"
    research_recommended = False
    if current is not None:
        edge = {key + "_EDGE_PP": (model[key] - current[key]) * 100 for key in model}
        max_edge = max(abs(value) for value in edge.values())
        edge["MAX_ABS_EDGE_PP"] = max_edge
        divergence = ("SEVERE" if max_edge >= config["divergence"]["severe_pp"] else
                      "WARNING" if max_edge >= config["divergence"]["warning_pp"] else "NORMAL")
        research_recommended = divergence == "SEVERE"
    market_status = "AVAILABLE" if current is not None else (
        "PARTIAL" if snapshot.quotes else "UNAVAILABLE")
    source_quality = ("USER_CONFIRMED" if evidence and len(evidence) == len({
        quote.provider_id for quote in snapshot.quotes}) else
        "UNKNOWN")
    fusion = fuse_model_market(model, current, config)
    snapshot_payload = snapshot.model_dump(mode="json")
    snapshot_hash = hashlib.sha256(canonical_bytes(snapshot_payload)).hexdigest()
    return {"fixture_id": fixture_id, "prediction_time": prediction_time.isoformat(),
            "kickoff_time": kickoff_utc.isoformat(),
            "market_status": market_status, "market_source_quality": source_quality,
            "provider_status": provider_status, "source_evidence": evidence,
            "market_snapshot_id": snapshot.market_snapshot_id,
            "market_snapshot_sha256": snapshot_hash,
            "market_snapshot": snapshot_payload,
            "quote_count": len(snapshot.quotes),
            "sources": sorted({quote.provider_id for quote in snapshot.quotes}),
            "bookmaker_count": consensus.bookmaker_count if consensus else 0,
            "no_vig_method": policy.method_by_market[MarketType.MATCH_1X2].value,
            "consensus_method": market_config.consensus_method.value,
            "market_no_vig": current, "market_movement_pp": movement_pp,
            "model_probabilities": model, "model_market_edge_pp": edge,
            "market_divergence": divergence,
            "research_recommended": research_recommended,
            "fusion": fusion, "r5_market_only": r5_market_only(current),
            "config_version": config["version"],
            "config_sha256": hashlib.sha256(canonical_bytes(config)).hexdigest(),
            "quote_payloads": [quote.model_dump(mode="json") for quote in snapshot.quotes],
            "movement_payloads": [item.model_dump(mode="json") for item in movements]}
