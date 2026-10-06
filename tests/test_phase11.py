"""Phase 11 source, mathematical, selection, advice, portfolio and report gates."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from erguoyuan_football.contracts.predictions import ProbabilityVector
from erguoyuan_football.markets.schemas import (
    MarketType,
    OddsQuote,
    QuoteQuality,
    Selection,
    SourceType,
    TimestampQuality,
)
from erguoyuan_football.models.score_matrix import ScoreMatrix
from erguoyuan_football.output_contract.schemas import CanonicalPredictionResult
from erguoyuan_football.portfolio.engine import PortfolioEngine
from erguoyuan_football.portfolio.schemas import SettlementLedger
from erguoyuan_football.portfolio.weekly import weekly_metrics
from erguoyuan_football.prediction_heads.evidence import (
    HalfFullTimeEvidence,
    OfficialHandicapEvidence,
)
from erguoyuan_football.prediction_heads.heads import (
    htft_top2,
    official_handicap_1x2,
    score_top2,
    total_goals,
    total_top2,
)
from erguoyuan_football.prediction_heads.readiness import (
    MatrixSource,
    ScoreMatrixReadinessAudit,
    ScoreMatrixSourceSelector,
)
from erguoyuan_football.prediction_heads.reconcile import OutcomeMassReconciler
from erguoyuan_football.recommendation.engine import (
    RecommendationEngine,
    RecommendationPolicy,
)
from erguoyuan_football.recommendation.market_value import MarketValueEngine
from erguoyuan_football.recommendation.schemas import BetAdvice, MarketPriceEvidence
from erguoyuan_football.report.assembler import CoreReportV2Assembler
from erguoyuan_football.report.audit import audit_phase11
from erguoyuan_football.report.pipeline import run_development_batch
from erguoyuan_football.report.renderer import CoreReportV2Renderer
from erguoyuan_football.report.schemas import SECTION_ORDER, CoreReportV2Schema
from erguoyuan_football.selection.engine import (
    ACCOUNT_400_ORDER,
    HIGHEST_HIT_ORDER,
    SelectionEngine,
)
from erguoyuan_football.selection.schemas import MatchHeads, RankedCandidate
from erguoyuan_football.selection.value import ValueSelectionEngine

ROOT = Path(__file__).resolve().parents[1]


def matrix() -> ScoreMatrix:
    raw = np.array([[0.1, 0.1, 0.05], [0.2, 0.15, 0.05], [0.2, 0.1, 0.05]], dtype=float)
    return ScoreMatrix.from_raw(raw)


def match(key: str, *, p: tuple[float, float, float] = (0.5, 0.3, 0.2),
          with_matrix: bool = True) -> MatchHeads:
    source = matrix()
    reconciliation = OutcomeMassReconciler().reconcile(source, ProbabilityVector(
        p_home=p[0], p_draw=p[1], p_away=p[2]))
    final = reconciliation.final_matrix
    return MatchHeads(match_id=key, competition_id="TEST", match_date=date(2026, 5, 1),
        home_team_id=f"{key}H", away_team_id=f"{key}A",
        home_team_name=f"Home {key}", away_team_name=f"Away {key}",
        prediction_snapshot_id=f"snapshot-{key}",
        probability=ProbabilityVector(p_home=p[0], p_draw=p[1], p_away=p[2]),
        probability_source_stage="FINAL_CORE_CALIBRATED", probability_source_id=f"core-{key}",
        data_origin="TEST_FIXTURE",
        score_top2=score_top2(final) if with_matrix else None,
        totals_top2=total_top2(final) if with_matrix else None,
        matrix_source=MatrixSource(source, "TEST_MODEL", "TEST", None, f"oos-{key}",
            f"snapshot-{key}", reconciliation.reference_matrix_hash, "2026-04-30", 1,
            "test-training-hash", "test-ids-hash", "test-config-hash")
            if with_matrix else None,
        reconciliation=reconciliation if with_matrix else None,
        matrix_status="AVAILABLE_RECONCILED" if with_matrix else "UNAVAILABLE")


def candidate():
    return SelectionEngine().highest_hit((match("a"), match("b")))[0].candidate


def price(candidate_id: str, *, odds: float = 2.0) -> MarketPriceEvidence:
    prediction = datetime(2026, 5, 1, 10, tzinfo=UTC)
    return MarketPriceEvidence(candidate_id=candidate_id, decimal_odds=odds,
        market_probability=0.45, quote_ids=("quote-1", "quote-2"), source="SPORTTERY_OFFICIAL_PUBLIC",
        provider_id="controlled-provider", bookmaker_id="controlled-bookmaker",
        market_type="MATCH_1X2_PAIR", as_of_time=prediction - timedelta(hours=1),
        retrieved_at=prediction - timedelta(minutes=30), prediction_time=prediction,
        kickoff_times=(prediction + timedelta(hours=2), prediction + timedelta(hours=3)),
        quality_status="GOOD", devig_method="MULTIPLICATIVE",
        devig_evidence_id="real-devig-1", purchasable=True)


def test_outcome_mass_reconciliation() -> None:
    target = ProbabilityVector(p_home=0.4, p_draw=0.35, p_away=0.25)
    result = OutcomeMassReconciler().reconcile(matrix(), target)
    observed = result.final_matrix.outcome()
    assert (observed.p_home, observed.p_draw, observed.p_away) == pytest.approx((0.4, 0.35, 0.25))
    assert sum(map(sum, result.final_matrix.values)) == pytest.approx(1)
    assert result.reference_matrix_hash != result.final_matrix_hash


def test_zero_mass_reconciliation_failure() -> None:
    source = ScoreMatrix.from_raw(np.array([[1.0, 0.0], [0.0, 0.0]]))
    with pytest.raises(ValueError, match="RECONCILIATION_FAILED"):
        OutcomeMassReconciler().reconcile(source, ProbabilityVector(
            p_home=0.5, p_draw=0.3, p_away=0.2))


def test_score_and_totals_same_matrix() -> None:
    source = matrix()
    score = score_top2(source)
    totals = total_goals(source)
    assert score.selections[0] == "1-0" or score.selections[0] == "2-0"
    assert sum(totals.values()) == pytest.approx(1)
    assert total_top2(source).coverage == pytest.approx(sum(
        sorted(totals.values(), reverse=True)[:2]))


@pytest.mark.parametrize("line,expected", [(-1, {"HOME": 0.2, "DRAW": 0.3, "AWAY": 0.5}),
                                             (1, {"HOME": 0.8, "DRAW": 0.15, "AWAY": 0.05})])
def test_home_handicap_direction(line: int, expected: dict[str, float]) -> None:
    outcome = official_handicap_1x2(matrix(), line, verified_source=True)
    assert sum(outcome.values()) == pytest.approx(1)
    assert outcome == pytest.approx(expected)


def test_handicap_requires_official_line() -> None:
    with pytest.raises(ValueError, match="OFFICIAL_HANDICAP_UNAVAILABLE"):
        official_handicap_1x2(matrix(), None, verified_source=False)
    with pytest.raises(ValueError, match="OFFICIAL_HANDICAP_UNAVAILABLE"):
        official_handicap_1x2(matrix(), -1, verified_source=False)


def test_htft_independent_model_required() -> None:
    with pytest.raises(ValueError, match="HTFT_UNAVAILABLE"):
        htft_top2(None, independent_oos_model=False)
    distribution = {a + b: 1 / 9 for a in "HDA" for b in "HDA"}
    with pytest.raises(ValueError, match="HTFT_UNAVAILABLE"):
        htft_top2(distribution, independent_oos_model=False)
    assert htft_top2(distribution, independent_oos_model=True).coverage == pytest.approx(2 / 9)


def test_official_handicap_evidence_and_head_gate() -> None:
    original = match("a")
    prediction = datetime(2026, 5, 1, 10, tzinfo=UTC)
    evidence = OfficialHandicapEvidence(match_id="a", home_handicap=-1, quote_id="official-quote",
        source="SPORTTERY_OFFICIAL_PUBLIC", source_time=prediction - timedelta(minutes=10),
        retrieved_at=prediction - timedelta(minutes=5), prediction_time=prediction,
        kickoff_time=prediction + timedelta(hours=2))
    assert original.reconciliation is not None
    derived = official_handicap_1x2(original.reconciliation.final_matrix, -1,
                                    verified_source=True)
    enriched = MatchHeads.model_validate({**original.model_dump(),
        "official_home_handicap": -1, "official_handicap_evidence": evidence.model_dump(),
        "handicap_probabilities": derived, "handicap_status": "AVAILABLE"})
    assert enriched.handicap_probabilities == pytest.approx(derived)
    with pytest.raises(ValueError, match="OFFICIAL_HANDICAP_EVIDENCE_MISSING"):
        MatchHeads.model_validate({**original.model_dump(), "official_home_handicap": -1,
            "handicap_probabilities": derived, "handicap_status": "AVAILABLE"})


def test_htft_head_requires_independent_oos_lineage() -> None:
    original = match("a")
    prediction = datetime(2026, 5, 1, 10, tzinfo=UTC)
    distribution = {a + b: 1 / 9 for a in "HDA" for b in "HDA"}
    evidence = HalfFullTimeEvidence(match_id="a", model_id="TEST_HTFT", model_version="TEST",
        prediction_id="oos-htft-a", prediction_snapshot_id="snapshot-a",
        trained_until=prediction - timedelta(days=1), prediction_time=prediction,
        kickoff_time=prediction + timedelta(hours=2), historical_half_time_source="TEST_FIXTURE",
        training_match_ids_hash="test-hash", target_excluded_from_training=True,
        is_oos=True, data_origin="REAL", distribution=distribution)
    enriched = MatchHeads.model_validate({**original.model_dump(), "htft_status": "AVAILABLE",
        "htft_evidence": evidence.model_dump(),
        "htft_top2": htft_top2(distribution, independent_oos_model=True).__dict__})
    assert enriched.htft_top2 is not None
    with pytest.raises(ValueError, match="HTFT_INDEPENDENT_MODEL_EVIDENCE_MISSING"):
        MatchHeads.model_validate({**original.model_dump(), "htft_status": "AVAILABLE",
            "htft_top2": htft_top2(distribution, independent_oos_model=True).__dict__})


def test_highest_hit_pairs_and_absence() -> None:
    slots = SelectionEngine().highest_hit((match("a"), match("b", p=(0.6, 0.2, 0.2))))
    assert tuple(slot.play_type for slot in slots) == HIGHEST_HIT_ORDER
    pair = slots[0].candidate
    assert pair is not None and pair.joint_probability == pytest.approx(0.3)
    assert pair.match_ids == ("a", "b")
    assert pair.probability_method == "INDEPENDENCE_ASSUMPTION"
    assert slots[1].status == "UNAVAILABLE"
    assert slots[3].status == "UNAVAILABLE"
    assert len(slots[4].candidate.selections) == 4


def test_same_match_never_independent_pair() -> None:
    slots = SelectionEngine().highest_hit((match("a"), match("a")))
    assert slots[0].candidate is None
    assert slots[4].candidate is None


def test_best_totals_coverage() -> None:
    slots = SelectionEngine().highest_hit((match("a"), match("b")))
    assert slots[2].candidate is not None
    assert slots[2].candidate.coverage_probability == pytest.approx(match("a").totals_top2.coverage)


def test_missing_price_is_unavailable_not_no_bet() -> None:
    selected = candidate()
    assert selected is not None
    advice = RecommendationEngine(RecommendationPolicy(0.03)).advise(selected, None)
    assert advice.advice_status == "UNAVAILABLE" and advice.recommendation is None
    assert advice.expected_value is None and advice.recommended_stake == 0
    assert "PRICE_UNAVAILABLE" in advice.reason_codes


def test_fair_odds_ev_and_no_bet_candidate_persists() -> None:
    selected = candidate()
    assert selected is not None
    recommendation = RecommendationEngine(RecommendationPolicy(0.03))
    advice = recommendation.advise(selected, price(selected.candidate_id), reference_allocation=50)
    assert advice.minimum_acceptable_price == pytest.approx(1.03 / selected.joint_probability)
    assert advice.expected_value == pytest.approx(selected.joint_probability * 2 - 1)
    assert advice.recommendation == "NO_BET"
    assert selected.candidate_id == advice.candidate_id
    assert advice.recommended_stake == 0


def test_available_bet_with_sufficient_price() -> None:
    selected = candidate()
    assert selected is not None
    advice = RecommendationEngine(RecommendationPolicy(0.03)).advise(
        selected, price(selected.candidate_id, odds=5), reference_allocation=25)
    assert advice.recommendation == "BET" and advice.recommended_stake == 25


def test_stale_market_price_blocked() -> None:
    selected = candidate()
    assert selected is not None
    evidence = price(selected.candidate_id)
    stale = MarketPriceEvidence.model_validate({**evidence.model_dump(),
        "as_of_time": evidence.prediction_time - timedelta(hours=3)})
    advice = RecommendationEngine(RecommendationPolicy(0.03)).advise(selected, stale)
    assert advice.advice_status == "BLOCKED" and advice.recommended_stake == 0


def test_phase6_devig_market_value_controlled_fixture() -> None:
    prediction = datetime(2026, 5, 1, 10, tzinfo=UTC)
    selected = RankedCandidate(candidate_id="controlled-home", prediction_snapshot_ids=("snap",),
        match_ids=("controlled",), objective="VALUE", play_type="MATCH_1X2",
        selections=("HOME",), component_probabilities=(0.5,), joint_probability=0.5,
        coverage_probability=0.5, probability_method="DIRECT_EFFECTIVE_CORE",
        data_quality="TEST_FIXTURE", context_quality="UNAVAILABLE",
        correlation_status="NOT_APPLICABLE", rank=1,
        validation_status="DEVELOPMENT_ONLY", production_status="NOT_PROMOTED")
    source_time = prediction - timedelta(minutes=30)
    quotes = tuple(OddsQuote(quote_id=f"q-{direction.value}", match_id="controlled",
        provider_id="controlled", bookmaker_id="controlled", market_type=MarketType.SPORTTERY_1X2,
        selection=direction, odds_decimal=Decimal(str(odds)),
        source_type=SourceType.SYNTHETIC_TEST, source_time=source_time,
        as_of_time=source_time, retrieved_at=source_time,
        timestamp_quality=TimestampQuality.SOURCE_NATIVE,
        schema_version="TEST", content_hash=f"hash-{direction.value}",
        quality_status=QuoteQuality.GOOD)
        for direction, odds in ((Selection.HOME, 2.5), (Selection.DRAW, 3.2), (Selection.AWAY, 3.3)))
    market = MarketValueEngine().evidence(selected, quotes, prediction_time=prediction,
        kickoff_times=(prediction + timedelta(hours=2),), synthetic_test_only=True)
    assert market.market_probability == pytest.approx((1 / 2.5) / (1 / 2.5 + 1 / 3.2 + 1 / 3.3))
    assert market.decimal_odds == 2.5
    assert RecommendationEngine(RecommendationPolicy(0.03)).advise(selected, market).advice_status == "BLOCKED"
    controlled = RecommendationEngine(RecommendationPolicy(0.03), allow_synthetic_test=True)
    assert controlled.advise(selected, market, reference_allocation=10).expected_value == pytest.approx(0.25)


def test_parlay_rejects_prices_from_different_bookmakers() -> None:
    selected = candidate()
    assert selected is not None
    prediction = datetime(2026, 5, 1, 10, tzinfo=UTC)
    source_time = prediction - timedelta(minutes=20)
    quotes = tuple(OddsQuote(quote_id=f"{match_id}-{direction.value}", match_id=match_id,
        provider_id="controlled-provider", bookmaker_id=f"controlled-{match_id}",
        market_type=MarketType.SPORTTERY_1X2, selection=direction,
        odds_decimal=Decimal("2.5"), source_type=SourceType.SYNTHETIC_TEST,
        source_time=source_time, as_of_time=source_time, retrieved_at=source_time,
        timestamp_quality=TimestampQuality.SOURCE_NATIVE, schema_version="TEST",
        content_hash=f"hash-{match_id}-{direction.value}")
        for match_id in ("a", "b") for direction in (Selection.HOME, Selection.DRAW, Selection.AWAY))
    with pytest.raises(ValueError, match="PARLAY_NOT_PURCHASABLE_ACROSS_BOOKMAKERS"):
        MarketValueEngine().evidence(selected, quotes, prediction_time=prediction,
            kickoff_times=(prediction + timedelta(hours=2), prediction + timedelta(hours=3)),
            synthetic_test_only=True)


def test_value_requires_price_and_longshot_not_just_high_odds() -> None:
    matches = (match("a"),)
    assert ValueSelectionEngine().candidates(matches, {}, objective="VALUE") == ()
    assert ValueSelectionEngine().candidates(matches, {}, objective="LONGSHOT") == ()


def test_longshot_lexicographic_and_priced_account_caps_controlled_fixture() -> None:
    one = match("a", p=(0.5, 0.3, 0.2))
    prediction = datetime(2026, 5, 1, 10, tzinfo=UTC)
    evidence = {}
    for direction, odds in (("HOME", 5.0), ("AWAY", 10.0)):
        identity = sha256(json.dumps(["LONGSHOT", "MATCH_1X2", ["a"], [direction], []],
                                     sort_keys=True).encode()).hexdigest()[:24]
        evidence[("a", direction)] = MarketPriceEvidence(candidate_id=identity,
            decimal_odds=odds, market_probability=0.05,
            quote_ids=(f"controlled-{direction}",), source="SYNTHETIC_TEST",
            provider_id="controlled-provider", bookmaker_id="controlled-bookmaker",
            market_type="MATCH_1X2", as_of_time=prediction - timedelta(minutes=10),
            retrieved_at=prediction - timedelta(minutes=5), prediction_time=prediction,
            kickoff_times=(prediction + timedelta(hours=2),), quality_status="GOOD",
            devig_method="MULTIPLICATIVE", devig_evidence_id=f"controlled-{direction}",
            purchasable=True, synthetic_test_only=True)
    ranked = ValueSelectionEngine().candidates((one,), evidence, objective="LONGSHOT",
        allow_synthetic_test=True)
    assert len(ranked) == 2
    assert ranked[0].selections == ("HOME",)  # EV=1.5 beats higher-odds AWAY (EV=1.0).
    prices = {item.candidate_id: evidence[("a", item.selections[0])] for item in ranked}
    engine = PortfolioEngine(RecommendationEngine(RecommendationPolicy(0.03),
                                                  allow_synthetic_test=True))
    account100 = engine.priced_account("VALUE_100", ranked, prices)
    account20 = engine.priced_account("LONGSHOT_20", ranked, prices)
    assert account100.recommended_total <= 100
    assert account20.recommended_total <= 20
    assert all(entry.candidate.selections in {("HOME",), ("AWAY",)} for entry in account20.entries)


def test_report_assembler_price_gate_controlled_fixture() -> None:
    one = match("a")
    two = match("b")
    selected = candidate()
    assert selected is not None
    source = price(selected.candidate_id, odds=5)
    with pytest.raises(ValueError, match="DATE_SAFE_BATCH_MARKET_PRICE_BLOCKED"):
        CoreReportV2Assembler(allow_synthetic_test=True).build(
            heads=(one, two), status={}, highest_hit_prices={selected.candidate_id: source})
    prediction = source.prediction_time
    exact = tuple(MatchHeads.model_validate({**item.model_dump(),
        "temporal_mode": "EXACT_UTC", "kickoff_time": prediction + timedelta(hours=2)})
        for item in (one, two))
    report, _ = CoreReportV2Assembler(allow_synthetic_test=True).build(
        heads=exact, status={"market_status": "TEST_FIXTURE"},
        highest_hit_prices={selected.candidate_id: source})
    assert report.high_hit_400.recommended_total > 0
    assert report.highest_hit[0].candidate is not None
    assert report.value_100.status == "UNAVAILABLE_MARKET_DATA"


def test_report_assembler_value_account_controlled_fixture() -> None:
    prediction = datetime(2026, 5, 1, 10, tzinfo=UTC)
    one = MatchHeads.model_validate({**match("a").model_dump(),
        "temporal_mode": "EXACT_UTC", "kickoff_time": prediction + timedelta(hours=2)})
    identity = sha256(json.dumps(["VALUE", "MATCH_1X2", ["a"], ["HOME"], []],
                                 sort_keys=True).encode()).hexdigest()[:24]
    evidence = MarketPriceEvidence(candidate_id=identity, decimal_odds=3,
        market_probability=0.4, quote_ids=("controlled-quote",), source="SYNTHETIC_TEST",
        provider_id="controlled-provider", bookmaker_id="controlled-bookmaker",
        market_type="MATCH_1X2", as_of_time=prediction - timedelta(minutes=10),
        retrieved_at=prediction - timedelta(minutes=5), prediction_time=prediction,
        kickoff_times=(prediction + timedelta(hours=2),), quality_status="GOOD",
        devig_method="MULTIPLICATIVE", devig_evidence_id="controlled-devig",
        purchasable=True, synthetic_test_only=True)
    ranked = ValueSelectionEngine().candidates((one,), {("a", "HOME"): evidence},
        objective="VALUE", allow_synthetic_test=True)
    report, _ = CoreReportV2Assembler(allow_synthetic_test=True).build(
        heads=(one,), status={"market_status": "TEST_FIXTURE"},
        value_candidates=ranked, value_prices={identity: evidence})
    assert len(report.value_100.entries) == 1
    assert report.value_100.recommended_total == 100
    assert report.bet_advice_ledger.advices[0].recommendation == "BET" or any(
        advice.recommendation == "BET" for advice in report.bet_advice_ledger.advices)


def test_invalid_market_time_rejected() -> None:
    selected = candidate()
    assert selected is not None
    evidence = price(selected.candidate_id)
    with pytest.raises(ValueError, match="MARKET_PRICE_PIT_OR_QUALITY_INVALID"):
        MarketPriceEvidence.model_validate({**evidence.model_dump(),
            "as_of_time": evidence.prediction_time + timedelta(seconds=1)})


def test_400_fixed_slots_and_cap() -> None:
    slots = SelectionEngine().highest_hit((match("a"), match("b")))
    account = PortfolioEngine(RecommendationEngine(RecommendationPolicy(0.03))).high_hit_400(slots)
    assert tuple(entry.slot for entry in account.entries) == ACCOUNT_400_ORDER
    assert account.reference_total <= 400 and account.recommended_total == 0
    assert all(entry.advice is None or entry.advice.recommendation is None for entry in account.entries)


def test_100_20_market_blockers() -> None:
    engine = PortfolioEngine(RecommendationEngine(RecommendationPolicy(0.03)))
    for key, cap in (("VALUE_100", 100), ("LONGSHOT_20", 20)):
        account = engine.market_blocked(key)
        assert account.status == "UNAVAILABLE_MARKET_DATA"
        assert account.budget_cap == cap and account.recommended_total == 0


def test_no_bet_stake_contract() -> None:
    selected = candidate()
    assert selected is not None
    with pytest.raises(ValueError, match="NO_BET_STAKE_MUST_BE_ZERO"):
        BetAdvice(candidate_id=selected.candidate_id, advice_status="AVAILABLE",
            recommendation="NO_BET", recommended_stake=1, reference_allocation=1,
            reason_codes=(), market_status="AVAILABLE", risk_level="TEST",
            validation_status="DEVELOPMENT_ONLY", production_status="NOT_PROMOTED")


def test_matrix_selector_rejects_future_evaluation() -> None:
    selector = ScoreMatrixSourceSelector(ROOT / "data" / "football.duckdb")
    with pytest.raises(ValueError, match="MATRIX_SELECTION_TIME_LEAKAGE"):
        selector.select(evaluation_start=date(2026, 5, 1), evaluation_end=date(2026, 6, 1),
                        target_start=date(2026, 5, 1))


@pytest.mark.integration
def test_no_matrix_from_core_1x2_only() -> None:
    db = ROOT / "data" / "football.duckdb"
    if not db.is_file():
        pytest.skip("Real development database is not present")
    with (ROOT / "reports" / "phase9_development_core_preview.jsonl").open(encoding="utf-8") as handle:
        canonical = CanonicalPredictionResult.model_validate_json(next(handle))
    selection = ScoreMatrixSourceSelector(db).select(evaluation_start=date(2026, 1, 1),
        evaluation_end=date(2026, 5, 1), target_start=date(2026, 5, 1))
    no_source = canonical.model_copy(update={"data_lineage": {"source_prediction_ids": {}}})
    assert ScoreMatrixReadinessAudit(db).load_many((no_source,), selection) == {}


@pytest.mark.integration
def test_real_phase11_end_to_end_twenty() -> None:
    db = ROOT / "data" / "football.duckdb"
    if not db.is_file():
        pytest.skip("Real development database is not present")
    report, timing = run_development_batch(db_path=db,
        predictions_path=ROOT / "reports" / "phase9_development_core_preview.jsonl",
        rules_path=ROOT / "config" / "competition_rules.yaml", count=20)
    assert len(report.all_matches) == 20
    assert report.model_data_status["score_matrix_ready_rows"] == 20
    assert report.model_data_status["final_holdout_rows_read"] == 0
    assert report.value_100.status == report.longshot_20.status == "UNAVAILABLE_MARKET_DATA"
    assert all(item.production_status == "NOT_PROMOTED" for item in report.all_matches)
    assert all(item.advice_status == "UNAVAILABLE" for item in report.bet_advice_ledger.advices)
    assert report.section_order == SECTION_ORDER
    assert timing["db_query_count"] == 6
    rendered = CoreReportV2Renderer().render(report)
    headings = ("一、当日全部比赛概率表", "二、当日最高命中率组合", "三、400元高命中率账户",
                "四、100元自由Value实验账户", "五、20元高赔率小博大账户", "六、模型与数据状态")
    assert [rendered.index(heading) for heading in headings] == sorted(
        rendered.index(heading) for heading in headings)
    assert "UNAVAILABLE_MARKET_DATA" in rendered
    assert CoreReportV2Schema.model_validate_json(report.model_dump_json()).section_order == SECTION_ORDER
    assert weekly_metrics(report.candidate_ledger, report.bet_advice_ledger,
                          SettlementLedger())["roi"] is None
    assert all(audit_phase11(report).values())
