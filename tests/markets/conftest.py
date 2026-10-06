"""Explicit SYNTHETIC_TEST fixtures for offline market-engine verification."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import numpy as np
import pytest

from erguoyuan_football.data.schemas import Fixture
from erguoyuan_football.markets.config import MarketConfig
from erguoyuan_football.markets.devig import DeVigPolicy
from erguoyuan_football.markets.engine import MarketEngine
from erguoyuan_football.markets.schemas import (
    DeVigMethod,
    MarketType,
    OddsQuote,
    QuoteQuality,
    Selection,
    SourceType,
    TimestampQuality,
)
from erguoyuan_football.markets.settlement import TotalSettlementEngine
from erguoyuan_football.models.dynamic_bayes.likelihood import score_matrix_from_lambdas


@pytest.fixture
def market_fixture(at: object) -> Fixture:
    match_time = at
    return Fixture(match_id="market_test_match", competition_id="test_league",
        home_team_id="test_home", away_team_id="test_away", kickoff_time=match_time + timedelta(hours=24),
        source="SYNTHETIC_TEST", retrieved_at=match_time - timedelta(days=1),
        as_of_time=match_time - timedelta(days=1), data_version="synthetic-fixture-v1",
        season="2025", neutral_venue=False)


@pytest.fixture
def quote_factory(at: object, market_fixture: Fixture):
    matrix = score_matrix_from_lambdas(np.asarray([1.45]), np.asarray([1.05]), max_goals=12)
    outcome = matrix.outcome()
    total = TotalSettlementEngine.probabilities(matrix, 2.5, Selection.OVER)
    probabilities = {
        MarketType.MATCH_1X2: {
            Selection.HOME: outcome.p_home, Selection.DRAW: outcome.p_draw, Selection.AWAY: outcome.p_away,
        },
        MarketType.TOTALS: {
            Selection.OVER: total.expected_win_equivalent,
            Selection.UNDER: total.expected_loss_equivalent,
        },
    }

    def make(*, opening: bool = False):
        rows = []
        for bookmaker_index, bookmaker in enumerate(("TEST_BOOK_A", "TEST_BOOK_B")):
            margin = 1.045 + bookmaker_index * 0.015
            source_time = at - (timedelta(hours=12) if opening else timedelta(minutes=10))
            retrieved_at = source_time + timedelta(minutes=1)
            for market_type, selections in probabilities.items():
                for selection, probability in selections.items():
                    odds = Decimal(str(round(1.0 / (probability * margin), 6)))
                    identity = f"{bookmaker}:{market_type.value}:{selection.value}:{opening}"
                    rows.append(OddsQuote(
                        quote_id=identity, match_id=market_fixture.match_id,
                        provider_id="synthetic_market_feed", bookmaker_id=bookmaker,
                        market_type=market_type, selection=selection,
                        line_quarters=10 if market_type == MarketType.TOTALS else None,
                        odds_decimal=odds, original_odds=str(odds), original_format="DECIMAL",
                        source_type=SourceType.SYNTHETIC_TEST, source_event_id="test_event",
                        source_quote_id=identity, source_time=source_time, retrieved_at=retrieved_at,
                        as_of_time=source_time, timestamp_quality=TimestampQuality.SOURCE_NATIVE,
                        is_opening_confirmed=opening, schema_version="market-schema-v1",
                        content_hash=f"hash-{identity}", quality_status=QuoteQuality.GOOD,
                    ))
        return tuple(rows)

    return make


@pytest.fixture
def provisional_market_engine() -> MarketEngine:
    config = MarketConfig(minimum_bookmakers=2, maximum_overround={
        MarketType.MATCH_1X2: 0.2, MarketType.TOTALS: 0.2,
    }, freshness_seconds_by_horizon={
        "T_MINUS_24H": 86_400, "T_MINUS_6H": 21_600, "T_MINUS_3H": 10_800,
        "T_MINUS_60M": 3_600, "T_MINUS_15M": 900, "CUSTOM": 86_400,
    }, maximum_market_freshness_seconds=86_400)
    policy = DeVigPolicy(policy_version="SYNTHETIC_TEST_POLICY_V1", status="PROVISIONAL_NOT_VALIDATED",
        method_by_market={
            MarketType.MATCH_1X2: DeVigMethod.POWER,
            MarketType.ASIAN_HANDICAP: DeVigMethod.MULTIPLICATIVE,
            MarketType.TOTALS: DeVigMethod.MULTIPLICATIVE,
            MarketType.SPORTTERY_1X2: DeVigMethod.MULTIPLICATIVE,
            MarketType.SPORTTERY_HANDICAP_1X2: DeVigMethod.MULTIPLICATIVE,
        })
    return MarketEngine(config, policy)
