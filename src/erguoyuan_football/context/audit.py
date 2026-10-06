"""Automated Phase 10 invariants for temporal, probability, and lineage safety."""

from __future__ import annotations

from erguoyuan_football.context.schemas import Availability, ContextAssessment


class ContextAudit:
    def review(self, assessment: ContextAssessment) -> dict[str, object]:
        probability = assessment.adjusted_probability
        no_adjustment_preserves_base = (
            not probability.tournament_adjustment_applied
            and not probability.lineup_adjustment_applied
            and probability.base_p_home == probability.final_p_home
            and probability.base_p_draw == probability.final_p_draw
            and probability.base_p_away == probability.final_p_away
        )
        same_day_sources_excluded = all(
            event_date < assessment.feature_vector.prediction_date
            for event_date in assessment.feature_vector.source_event_dates.values())
        tests = {
            "AVAILABILITY_REPORT_LINEAGE": (
                assessment.data_availability.match_id == assessment.match_id
                and assessment.data_availability.prediction_snapshot_id == assessment.prediction_snapshot_id),
            "LINEUP_AVAILABILITY_MATCHES_EVIDENCE": (
                (assessment.data_availability.component_status.get("lineups") == Availability.AVAILABLE)
                == bool(assessment.lineup_evidence)
            ),
            "INJURY_AVAILABILITY_MATCHES_EVIDENCE": (
                (assessment.data_availability.component_status.get("injuries") == Availability.AVAILABLE)
                == bool(assessment.injury_evidence)
            ),
            "BASE_PROBABILITY_PRESERVED": no_adjustment_preserves_base,
            "POST_MATCH_LINEUP_REJECTED": all(
                item.data_class.value not in {"POST_MATCH_CONTEXT", "UNKNOWN_CONTEXT"}
                for item in assessment.lineup_evidence),
            "UNKNOWN_CONTEXT_NOT_ZERO": (
                assessment.feature_vector.lineup_strength_delta is None
                or assessment.feature_vector.availability_mask.get("player_strength", False)),
            "RULE_VERSION_RECORDED": (
                assessment.tournament_state.rule_version is None
                or bool(assessment.feature_vector.rule_versions)),
            "PRODUCTION_NOT_PROMOTED": (
                probability.production_status == "NOT_PROMOTED"
                and probability.validation_status == "DEVELOPMENT_ONLY"),
            "SAME_DAY_SOURCES_EXCLUDED": same_day_sources_excluded,
            "ADJUSTMENT_LEDGER_PRESENT": bool(assessment.ledger.ledger_id),
        }
        return {"status": "PASS" if all(tests.values()) else "FAIL",
                "checks": tests, "critical_bugs": [],
                "high_bugs": [name for name, passed in tests.items() if not passed]}
