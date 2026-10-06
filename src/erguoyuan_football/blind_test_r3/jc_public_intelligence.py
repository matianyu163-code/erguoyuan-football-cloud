"""Deterministic, separate JC/Global interpretation of frozen model probabilities."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

from erguoyuan_football.blind_test_r3.jc_public_parser import JCParsedMatch
from erguoyuan_football.blind_test_r3.market_intelligence import fuse_model_market
from erguoyuan_football.blind_test_r3.odds_api_provider import match_fixture
from erguoyuan_football.blind_test_r3.store import canonical_bytes, sha256

SIDES = ("HOME", "DRAW", "AWAY")


def match_jc_fixture(matches: tuple[JCParsedMatch, ...], *, publication: dict[str, Any],
                     tolerance_minutes: int = 15) -> tuple[JCParsedMatch, str] | None:
    """Require unique oriented team/time match and, when known, JC number."""
    expected = publication.get("input_snapshot", {}).get("fixture_evidence", {}).get("jc_code")
    events = [{"home_team": item.home_team, "away_team": item.away_team,
               "commence_time": item.kickoff_utc.isoformat(), "match": item}
              for item in matches if expected is None or item.jc_match_number == expected]
    try:
        found = match_fixture(events, home_team=publication["home_team"],
            away_team=publication["away_team"],
            kickoff_utc=datetime.fromisoformat(publication["kickoff_time"]).astimezone(UTC),
            tolerance_minutes=tolerance_minutes, normalized_tolerance_minutes=0)
    except ValueError as error:
        if str(error) == "AMBIGUOUS_FIXTURE_MATCH":
            raise ValueError("AMBIGUOUS_JC_FIXTURE") from error
        raise
    return (found[0]["match"], found[1]) if found else None


def fixed_bonus_market(bonus: dict[str, float] | None, *, handicap: int | None = None,
                       maximum_overround: float = 0.20) -> dict[str, Any] | None:
    """Interpret three fixed bonuses as decimal-price-like, without mixing markets."""
    if bonus is None:
        return None
    if set(bonus) != set(SIDES) or any(not math.isfinite(bonus[k]) or
            bonus[k] <= 1 for k in SIDES):
        raise ValueError("JC_BONUS_INVALID")
    raw = {side: 1 / bonus[side] for side in SIDES}
    total = math.fsum(raw.values())
    if not 1 <= total <= 1 + maximum_overround:
        raise ValueError("JC_BONUS_OVERROUND_INVALID")
    return {"handicap": handicap, "raw_bonus": bonus, "raw_implied": raw,
            "overround": total - 1, "no_vig": {side: raw[side] / total for side in SIDES}}


def movement(previous: dict[str, Any] | None, current: dict[str, Any]) -> dict[str, Any]:
    if previous is None:
        return {"status": "INSUFFICIENT_HISTORY"}
    result: dict[str, Any] = {"status": "PREVIOUS_OBSERVED_TO_CURRENT",
                              "previous_snapshot_id": previous["snapshot_id"]}
    old = previous["payload"]
    for market in ("spf", "rqspf"):
        earlier, latest = old.get(market), current.get(market)
        if earlier is None or latest is None:
            result[market] = {"status": "UNAVAILABLE"}
        elif market == "rqspf" and earlier["handicap"] != latest["handicap"]:
            result[market] = {"status": "HANDICAP_CHANGED",
                              "previous_handicap": earlier["handicap"],
                              "current_handicap": latest["handicap"]}
        else:
            result[market] = {"status": "AVAILABLE",
                "bonus_delta": {k: latest["raw_bonus"][k] - earlier["raw_bonus"][k]
                                for k in SIDES},
                "no_vig_move_pp": {k: 100 * (latest["no_vig"][k] - earlier["no_vig"][k])
                                   for k in SIDES}}
    return result


def select_market_sources(model: dict[str, float], jc: dict[str, float] | None,
                          global_market: dict[str, float] | None,
                          fusion_config: dict[str, Any]) -> dict[str, Any]:
    """One R5 route; JC primary when both markets exist, Global a benchmark."""
    primary = jc if jc is not None else global_market
    source = ("JC_ONLY" if jc is not None and global_market is None else
              "GLOBAL_ONLY" if jc is None and global_market is not None else
              "DUAL_MARKET_AVAILABLE" if jc is not None else "MODEL_ONLY")
    edge = lambda market: ({k: 100 * (model[k] - market[k]) for k in SIDES}
                           if market is not None else None)
    return {"yycore_model": model, "jc_market_no_vig": jc,
            "global_market_no_vig": global_market,
            "model_jc_edge_pp": edge(jc), "model_global_edge_pp": edge(global_market),
            "jc_global_edge_pp": ({k: 100 * (jc[k] - global_market[k]) for k in SIDES}
                                  if jc is not None and global_market is not None else None),
            "fusion_market_source": source,
            "fusion_market_provider": "JC_OFFICIAL_PUBLIC" if jc is not None else
                "GLOBAL_ODDS_PROVIDER" if global_market is not None else None,
            "fusion": fuse_model_market(model, primary, fusion_config),
            "r5_primary": primary, "r5_global_benchmark": global_market}


def snapshot_id(payload: dict[str, Any]) -> str:
    return "JC-" + sha256(canonical_bytes(payload))[:32]
