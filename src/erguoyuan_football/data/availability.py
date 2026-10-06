"""Evidence-based availability; this is not model sufficiency or model accuracy."""

from pydantic import model_validator

from erguoyuan_football.contracts.common import Availability, Contract
from erguoyuan_football.data.schemas import OddsSnapshot
from erguoyuan_football.markets.schemas import MarketType, OddsQuote, QuoteQuality

FEATURES = (
    "historical_results", "historical_goals", "current_odds", "opening_odds", "closing_odds",
    "market_odds", "asian_handicap", "over_under", "sports_lottery", "team_stats", "xg",
    "lineup", "injuries", "league_hierarchy", "opta_power_ranking", "opta_xg", "opta_xga",
    "opta_team_stats", "opta_player_stats", "opta_prediction", "final_probability",
)


class AvailabilityItem(Contract):
    availability: Availability
    reason: str
    evidence_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def evidence_required(self):
        if not self.reason:
            raise ValueError("availability requires reason")
        if self.availability == Availability.AVAILABLE and not self.evidence_ids:
            raise ValueError("available data requires evidence IDs")
        return self


class DataAvailabilityReport(Contract):
    match_id: str
    items: dict[str, AvailabilityItem]


def build_report(snapshot, *, league_hierarchy_ids=()) -> DataAvailabilityReport:
    evidence: dict[str, list[str]] = {name: [] for name in FEATURES}
    teams = {snapshot.match_data_snapshot.home_team_id, snapshot.match_data_snapshot.away_team_id}
    covered = set()
    for history in snapshot.historical_results:
        covered.update({history.fixture.home_team_id, history.fixture.away_team_id} & teams)
    if covered == teams:
        ids = [h.result.result_id for h in snapshot.historical_results]
        evidence["historical_results"] = ids
        evidence["historical_goals"] = ids
    # Each available quote set must be complete, from one source/bookmaker/time/version/line.
    quote_sets: dict[tuple[str, ...], list[OddsSnapshot]] = {}
    for quote in snapshot.market_snapshot:
        legacy_key = (quote.source, quote.bookmaker, quote.market_type, quote.phase,
                      str(quote.line), quote.as_of_time.isoformat(), quote.data_version)
        if quote.availability == Availability.AVAILABLE:
            quote_sets.setdefault(legacy_key, []).append(quote)
    for legacy_group_key, legacy_group_quotes in quote_sets.items():
        market, phase = legacy_group_key[2], legacy_group_key[3]
        needed = {"HOME", "DRAW", "AWAY"} if market in {"1X2", "SPORTS_LOTTERY"} else (
            {"HOME", "AWAY"} if market == "ASIAN_HANDICAP" else {"OVER", "UNDER"})
        if {q.selection for q in legacy_group_quotes} != needed:
            continue
        ids = [q.snapshot_id for q in legacy_group_quotes]
        if market == "1X2":
            evidence[{"OPEN": "opening_odds", "CURRENT": "current_odds", "CLOSE": "closing_odds"}[phase]].extend(ids)
            evidence["market_odds"].extend(ids)
        else:
            evidence[{"ASIAN_HANDICAP": "asian_handicap", "OVER_UNDER": "over_under",
                      "SPORTS_LOTTERY": "sports_lottery"}[market]].extend(ids)
    # Canonical Phase 6 consensus is frozen at one explicit prediction cutoff.
    # It is the authoritative source for current international and Sporttery markets.
    for consensus in snapshot.canonical_market_consensus:
        if consensus.quality_status.value not in {"GOOD", "ACCEPTABLE"}:
            continue
        quote_ids = list(consensus.quote_ids)
        if consensus.market_type == MarketType.MATCH_1X2:
            evidence["current_odds"].extend(quote_ids)
            evidence["market_odds"].extend(quote_ids)
        elif consensus.market_type == MarketType.ASIAN_HANDICAP:
            evidence["asian_handicap"].extend(quote_ids)
            evidence["market_odds"].extend(quote_ids)
        elif consensus.market_type == MarketType.TOTALS:
            evidence["over_under"].extend(quote_ids)
            evidence["market_odds"].extend(quote_ids)
        elif consensus.market_type in {MarketType.SPORTTERY_1X2, MarketType.SPORTTERY_HANDICAP_1X2}:
            evidence["sports_lottery"].extend(quote_ids)
    # Opening is only available when a source explicitly marked its native
    # pre-match opening quotes; the database's first observed late quote is not opening.
    opening_sets: dict[tuple[str, ...], list[OddsQuote]] = {}
    for quote in snapshot.opening_market_quotes:
        if (quote.is_opening_confirmed and quote.quality_status == QuoteQuality.GOOD
                and not quote.is_live and not quote.is_suspended and quote.as_of_time is not None):
            opening_key = (quote.provider_id, quote.bookmaker_id, quote.market_type.value,
                           str(quote.line_quarters), quote.as_of_time.isoformat(), quote.schema_version)
            opening_sets.setdefault(opening_key, []).append(quote)
    for opening_group_key, opening_group_quotes in opening_sets.items():
        market = MarketType(opening_group_key[2])
        needed = ({"HOME", "DRAW", "AWAY"} if market in {
            MarketType.MATCH_1X2, MarketType.SPORTTERY_1X2, MarketType.SPORTTERY_HANDICAP_1X2
        } else {"HOME", "AWAY"} if market == MarketType.ASIAN_HANDICAP else {"OVER", "UNDER"})
        if {quote.selection.value for quote in opening_group_quotes} == needed:
            ids = [quote.quote_id for quote in opening_group_quotes]
            if market == MarketType.MATCH_1X2:
                evidence["opening_odds"].extend(ids)
                evidence["market_odds"].extend(ids)
            elif market in {MarketType.SPORTTERY_1X2, MarketType.SPORTTERY_HANDICAP_1X2}:
                evidence["sports_lottery"].extend(ids)
    for feature, records in (("team_stats", snapshot.team_stats_snapshot), ("xg", snapshot.xg_snapshot),
                             ("lineup", snapshot.lineup_snapshot), ("injuries", snapshot.injury_snapshot)):
        valid = [r for r in records if r.availability == Availability.AVAILABLE
                 and (feature != "lineup" or r.confirmed)
                 and (feature != "xg" or {"xg", "xga"}.issubset(r.metrics))]
        if {r.team_id for r in valid} >= teams:
            evidence[feature] = [r.snapshot_id for r in valid]
    for external in snapshot.external_snapshot:
        if external.availability == Availability.AVAILABLE:
            evidence["opta_" + external.product.lower()].append(external.snapshot_id)
    evidence["league_hierarchy"] = list(league_hierarchy_ids)
    goal_features = getattr(snapshot, "market_goal_features", None)
    if goal_features is not None and goal_features.availability == Availability.AVAILABLE:
        evidence["market_odds"].extend(goal_features.dependency_ids)
    return DataAvailabilityReport(match_id=snapshot.match_id, items={
        key: AvailabilityItem(availability=Availability.AVAILABLE if ids else Availability.UNAVAILABLE,
                              reason="PIT_EVIDENCE_PRESENT" if ids else "MISSING_OR_INCOMPLETE_PIT_EVIDENCE",
                              evidence_ids=tuple(sorted(set(ids)))) for key, ids in evidence.items()})
