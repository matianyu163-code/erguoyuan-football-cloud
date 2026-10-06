"""Deterministic Phase 11 report audit; no final-holdout or source-network access."""

from __future__ import annotations

import math

from erguoyuan_football.portfolio.schemas import ExperimentalAccount
from erguoyuan_football.prediction_heads.heads import score_top2, total_top2
from erguoyuan_football.report.schemas import CoreReportV2Schema


def audit_phase11(report: CoreReportV2Schema) -> dict[str, bool]:
    """Check evidence, visibility, mass, budgets and frozen development status."""
    candidates = {item.candidate_id: item for item in report.candidate_ledger.candidates}
    advice = {item.candidate_id: item for item in report.bet_advice_ledger.advices}
    accounts: tuple[ExperimentalAccount, ...] = report.portfolio_ledger.accounts
    checks = {
        "fake_data": all(item.probability is not None and item.data_origin == "REAL" and
            item.validation_status == "DEVELOPMENT_ONLY"
            and item.production_status == "NOT_PROMOTED" and item.market_status == "UNAVAILABLE"
            for item in report.all_matches),
        "no_bet_hiding": all(item.candidate_id in candidates for item in report.bet_advice_ledger.advices
            if item.recommendation == "NO_BET"),
        "unavailable_distinct": all(item.recommendation is None and item.expected_value is None
            for item in advice.values() if item.advice_status == "UNAVAILABLE"),
        "score_consistency": all(item.reconciliation is None or (
            item.probability is not None and
            all(math.isclose(a, b, abs_tol=1e-8) for a, b in zip(
                (item.reconciliation.final_matrix.outcome().p_home,
                 item.reconciliation.final_matrix.outcome().p_draw,
                 item.reconciliation.final_matrix.outcome().p_away),
                (item.probability.p_home, item.probability.p_draw, item.probability.p_away), strict=True)))
            for item in report.all_matches),
        "derived_markets": all(item.reconciliation is None or (
            item.score_top2 == score_top2(item.reconciliation.final_matrix) and
            item.totals_top2 == total_top2(item.reconciliation.final_matrix))
            for item in report.all_matches),
        "htft_independent": all(item.htft_top2 is None or item.htft_evidence is not None
            for item in report.all_matches),
        "correlation": all(len(item.match_ids) < 2 or
            item.correlation_status == "INDEPENDENCE_ASSUMPTION" for item in candidates.values()),
        "market": report.value_100.status == "UNAVAILABLE_MARKET_DATA" and
            report.longshot_20.status == "UNAVAILABLE_MARKET_DATA" and
            all(item.expected_value is None for item in advice.values()),
        "budget": all(account.reference_total <= account.budget_cap and
            account.recommended_total <= account.budget_cap for account in accounts),
        "production_status": all(item.production_status == "NOT_PROMOTED" for item in report.all_matches),
        "final_holdout": report.model_data_status.get("final_holdout_rows_read") == 0,
        "candidate_advice_separation": all(slot.candidate is None or
            slot.candidate.candidate_id in candidates for slot in report.highest_hit),
    }
    return checks
