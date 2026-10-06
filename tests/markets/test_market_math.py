"""Market normalization, de-vig, settlement and point-in-time regression tests."""

from datetime import timedelta
from decimal import Decimal

import numpy as np
import pytest

from erguoyuan_football.markets.devig import (
    DeVigError,
    DeVigValidationCase,
    calculate_devig,
    devig_validation_report,
)
from erguoyuan_football.markets.normalizer import (
    BookmakerRegistry,
    QuoteNormalizer,
    decimal_odds,
    quarter_units,
    swap_home_away_quote,
)
from erguoyuan_football.markets.schemas import (
    Bookmaker,
    DeVigMethod,
    MarketQualityStatus,
    MarketType,
    Selection,
    SourceType,
    TimestampQuality,
)
from erguoyuan_football.markets.settlement import (
    AsianSettlementEngine,
    SettlementOutcome,
    TotalSettlementEngine,
)
from erguoyuan_football.markets.snapshot import (
    FutureClosingOddsLeakageGuard,
    build_market_snapshot,
    closing_quote,
    current_quotes_at,
    opening_quote,
)
from erguoyuan_football.models.dynamic_bayes.likelihood import score_matrix_from_lambdas


@pytest.mark.parametrize("method", list(DeVigMethod))
def test_devig_methods_produce_complete_probability_vector(method: DeVigMethod) -> None:
    result = calculate_devig("market-1", {"HOME": 2.6, "DRAW": 3.4, "AWAY": 3.1}, method)
    assert result.success
    assert sum(result.devig_probabilities.values()) == pytest.approx(1.0, abs=1e-8)
    assert result.overround > 0


def test_devig_shin_rejects_two_way_markets() -> None:
    with pytest.raises(DeVigError, match="SHIN_REQUIRES_THREE_OR_MORE_OUTCOMES"):
        calculate_devig("two-way", {"OVER": 1.9, "UNDER": 1.9}, DeVigMethod.SHIN)


def test_odds_formats_and_quarter_line_identity() -> None:
    assert decimal_odds("-110", "AMERICAN") == Decimal(1) + Decimal(100) / Decimal(110)
    assert decimal_odds("5/4", "FRACTIONAL") == Decimal("2.25")
    assert decimal_odds("0.8", "HONG_KONG") == Decimal("1.8")
    assert decimal_odds("-0.8", "MALAY") == Decimal("2.25")
    assert quarter_units("-0.25") == -1
    assert quarter_units("2.75") == 11
    with pytest.raises(ValueError, match="LINE_NOT_QUARTER_COMPATIBLE"):
        quarter_units("2.6")


def test_normalizer_resolves_canonical_bookmaker_and_rejects_unknown(at) -> None:
    registry = BookmakerRegistry((Bookmaker(bookmaker_id="book_a", canonical_name="Book A",
        aliases=("A Bookie",), provider_mapping={"provider": "provider-label-a"}),))
    normalizer = QuoteNormalizer("provider", SourceType.AUTHORIZED_PROVIDER, registry,
                                 schema_version="schema-v1")
    quote = normalizer.normalize({"match_id": "m1", "bookmaker": "provider-label-a", "market_type": "AH",
        "selection": "HOME", "line": 0.25, "line_perspective": "AWAY", "odds": "-0.8",
        "odds_format": "MALAY", "source_time": at.isoformat()}, retrieved_at=at + timedelta(minutes=1))
    assert quote.bookmaker_id == "book_a"
    assert quote.line_quarters == -1
    assert quote.odds_decimal == Decimal("2.25")
    assert quote.timestamp_quality == TimestampQuality.SOURCE_NATIVE
    with pytest.raises(ValueError, match="UNKNOWN_BOOKMAKER"):
        normalizer.normalize({"match_id": "m1", "bookmaker": "unknown", "market_type": "1X2",
            "selection": "HOME", "odds": 2, "source_time": at.isoformat()}, retrieved_at=at)


def test_team_swap_reverses_handicap_line_and_side(quote_factory) -> None:
    quote = next(item for item in quote_factory() if item.market_type == MarketType.TOTALS)
    quote = quote.model_copy(update={"market_type": MarketType.ASIAN_HANDICAP,
        "selection": Selection.HOME, "line_quarters": -1})
    swapped = swap_home_away_quote(quote)
    assert swapped.selection == Selection.AWAY
    assert swapped.line_quarters == 1


@pytest.mark.parametrize(("line", "score", "selection", "expected"), [
    ("-1.0", (1, 0), Selection.HOME, SettlementOutcome.PUSH),
    ("-0.5", (1, 0), Selection.HOME, SettlementOutcome.WIN),
    ("-0.25", (0, 0), Selection.HOME, SettlementOutcome.HALF_LOSS),
    ("+0.25", (0, 0), Selection.HOME, SettlementOutcome.HALF_WIN),
    ("-0.25", (0, 0), Selection.AWAY, SettlementOutcome.HALF_WIN),
])
def test_asian_handicap_whole_half_quarter_settlement(line, score, selection, expected) -> None:
    assert AsianSettlementEngine.settle(*score, line, selection).outcome == expected


@pytest.mark.parametrize(("line", "total", "selection", "expected"), [
    ("2.0", 2, Selection.OVER, SettlementOutcome.PUSH),
    ("2.5", 2, Selection.OVER, SettlementOutcome.LOSS),
    ("2.25", 2, Selection.OVER, SettlementOutcome.HALF_LOSS),
    ("2.75", 3, Selection.OVER, SettlementOutcome.HALF_WIN),
    ("3.0", 3, Selection.UNDER, SettlementOutcome.PUSH),
])
def test_totals_whole_half_quarter_settlement(line, total, selection, expected) -> None:
    assert TotalSettlementEngine.settle(total, 0, line, selection).outcome == expected


def test_current_opening_closing_are_separate_and_closing_leakage_is_rejected(
        market_fixture, quote_factory, at) -> None:
    current = quote_factory()
    opening = quote_factory(opening=True)
    all_quotes = opening + current
    selected = current_quotes_at(all_quotes, match_id=market_fixture.match_id, prediction_time=at)
    assert {item.quote_id for item in selected} == {item.quote_id for item in current}
    open_result = opening_quote(all_quotes, match_id=market_fixture.match_id,
        bookmaker_id="TEST_BOOK_A", market_type=MarketType.MATCH_1X2, selection=Selection.HOME,
        prediction_time=at)
    assert open_result.quote is not None and open_result.quote.is_opening_confirmed
    close_time = market_fixture.kickoff_time - timedelta(minutes=5)
    closing = current[0].model_copy(update={"quote_id": "future-close", "source_time": close_time,
        "as_of_time": close_time, "retrieved_at": close_time + timedelta(seconds=1)})
    close_result = closing_quote((closing,), match_id=market_fixture.match_id,
        kickoff_time=market_fixture.kickoff_time, bookmaker_id=closing.bookmaker_id,
        market_type=closing.market_type, selection=closing.selection)
    assert close_result.quote is closing
    with pytest.raises(ValueError, match="DATA_LEAKAGE_BLOCKED"):
        FutureClosingOddsLeakageGuard().validate((closing.quote_id,), (closing,), prediction_time=at)
    snapshot = build_market_snapshot(all_quotes, match_id=market_fixture.match_id, prediction_time=at,
        kickoff_time=market_fixture.kickoff_time, max_age_seconds=86_400,
        quality_status=MarketQualityStatus.ACCEPTABLE)
    assert closing.quote_id not in snapshot.included_quote_ids


def test_market_goal_features_are_fitted_from_consensus_quotes(
        provisional_market_engine, market_fixture, quote_factory, at) -> None:
    result = provisional_market_engine.run(market_fixture, quote_factory(), prediction_time=at)
    assert result.market_goal_features.availability.value == "AVAILABLE"
    assert result.market_goal_features.inference_mode == "ONE_X_TWO_AND_TOTALS_2_5"
    assert result.market_goal_features.market_implied_lambda_home > 0
    assert result.market_goal_features.market_implied_lambda_away > 0
    assert result.market_goal_features.fit_error < 0.12
    assert result.quality_report.status.value == "POOR"  # provisional policy never passes production gate
    assert result.production_gate_status == "DEVELOPMENT_ONLY"


def test_devig_validation_selects_fixed_method_by_market_type(at) -> None:
    cases = tuple(DeVigValidationCase(market_id=f"m-{index}", market_type=MarketType.MATCH_1X2,
        prediction_time=at + timedelta(days=index), kickoff_time=at + timedelta(days=index, hours=4),
        odds={"HOME": 2.5, "DRAW": 3.4, "AWAY": 3.2},
        outcome=("HOME", "DRAW", "AWAY")[index % 3], competition_id="league") for index in range(12))
    report = devig_validation_report(cases, validation_start=at, final_test_start=at + timedelta(days=20))
    assert report["final_test_used"] is False
    assert set(report["selected_method_by_market"]) == {MarketType.MATCH_1X2.value}
    assert report["policy"]["status"] == "VALIDATED"


def test_score_matrix_market_settlement_probability_mass() -> None:
    matrix = score_matrix_from_lambdas(np.array([1.2]), np.array([1.0]), max_goals=12)
    asian = AsianSettlementEngine.probabilities(matrix, -0.25, Selection.HOME)
    totals = TotalSettlementEngine.probabilities(matrix, 2.75, Selection.OVER)
    assert sum((asian.win, asian.half_win, asian.push, asian.half_loss, asian.loss)) == pytest.approx(1)
    assert sum((totals.win, totals.half_win, totals.push, totals.half_loss, totals.loss)) == pytest.approx(1)
