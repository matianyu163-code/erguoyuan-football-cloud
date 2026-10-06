"""Independent fail-closed context, lineup, tournament, and promotion gates."""

from __future__ import annotations

from erguoyuan_football.context.schemas import (
    CompetitionRule,
    ContextAssessment,
    LineupEvidenceSnapshot,
    TournamentState,
)
from erguoyuan_football.contracts.common import Contract


class ContextGateResult(Contract):
    gate_id: str
    passed: bool
    status: str
    reason_codes: tuple[str, ...]


class ContextDataGate:
    def evaluate(self, *, point_in_time_passed: bool, rule: CompetitionRule | None,
                 entity_mapping_verified: bool, evidence_fresh: bool,
                 context_complete: bool, conflicts_resolved: bool,
                 post_match_data_present: bool) -> ContextGateResult:
        checks = {
            "PIT_FAILED": point_in_time_passed,
            "COMPETITION_RULE_UNAVAILABLE": rule is not None and rule.verified,
            "ENTITY_MAPPING_UNVERIFIED": entity_mapping_verified,
            "EVIDENCE_STALE": evidence_fresh,
            "CONTEXT_INCOMPLETE": context_complete,
            "EVIDENCE_CONFLICT": conflicts_resolved,
            "POST_MATCH_EVIDENCE_PRESENT": not post_match_data_present,
        }
        failures = tuple(reason for reason, passed in checks.items() if not passed)
        return ContextGateResult(gate_id="CONTEXT_DATA", passed=not failures,
            status="PASS" if not failures else "BLOCKED", reason_codes=failures)


class LineupGate:
    def evaluate(self, evidence: tuple[LineupEvidenceSnapshot, ...], *, temporal_rejections: tuple[str, ...] = ()) -> ContextGateResult:
        passed = bool(evidence) and not temporal_rejections
        reasons = temporal_rejections or (() if passed else ("LINEUP_CONTEXT_UNAVAILABLE",))
        return ContextGateResult(gate_id="LINEUP", passed=passed,
            status="PASS" if passed else "UNAVAILABLE", reason_codes=reasons)


class TournamentGate:
    def evaluate(self, state: TournamentState) -> ContextGateResult:
        passed = state.rule_version is not None and state.standings_state is not None
        reasons = () if passed else state.unavailable_reasons or ("TOURNAMENT_CONTEXT_UNAVAILABLE",)
        return ContextGateResult(gate_id="TOURNAMENT", passed=passed,
            status="PASS" if passed else "UNAVAILABLE", reason_codes=reasons)


class ContextPromotionGate:
    """Context can never promote unpromoted Phase 9 base probabilities."""

    REQUIRED = ("context_engine_ready", "tournament_data_ready", "lineup_data_ready",
                "context_model_ready", "development_validation_ready", "final_holdout_ready",
                "base_probability_promoted", "live_temporal_ready", "production_promoted")

    def evaluate(self, **readiness: bool) -> ContextGateResult:
        failures = tuple(name.upper() for name in self.REQUIRED if not readiness.get(name, False))
        passed = not failures and readiness.get("base_probability_promoted", False)
        return ContextGateResult(gate_id="CONTEXT_PROMOTION", passed=passed,
            status="PROMOTED" if passed else "NOT_PROMOTED", reason_codes=failures)


def promotion_report(assessment: ContextAssessment) -> ContextGateResult:
    return ContextPromotionGate().evaluate(
        context_engine_ready=assessment.context_status.value in {
            "EVALUATED_NO_ADJUSTMENT", "ADJUSTED"},
        tournament_data_ready=assessment.tournament_state.standings_state is not None,
        lineup_data_ready=assessment.lineup_status.value == "AVAILABLE",
        context_model_ready=assessment.context_model_status == "TRAINED_VALIDATED",
        development_validation_ready=False,
        final_holdout_ready=False,
        base_probability_promoted=False,
        live_temporal_ready=False,
        production_promoted=False,
    )
