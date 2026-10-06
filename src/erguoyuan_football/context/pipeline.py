"""Base → tournament → schedule/lineup → audited context probability pipeline."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date

from erguoyuan_football.context.adjustment import ContextAdjustmentEngine
from erguoyuan_football.context.audit import ContextAudit
from erguoyuan_football.context.competition_rules import CompetitionRuleRegistry
from erguoyuan_football.context.features import ContextFeatureBuilder
from erguoyuan_football.context.gates import (
    ContextDataGate,
    ContextGateResult,
    LineupGate,
    TournamentGate,
)
from erguoyuan_football.context.incentive import TournamentIncentiveEngine
from erguoyuan_football.context.injury import InjuryEvidenceLayer
from erguoyuan_football.context.lineup import LineupEvidenceLayer
from erguoyuan_football.context.rotation import RotationRiskEngine
from erguoyuan_football.context.schedule_context import ScheduleContextBuilder
from erguoyuan_football.context.schemas import (
    Availability,
    ContextAssessment,
    ContextDataAvailabilityReport,
    ContextStatus,
    HistoricalMatchEvent,
    InjuryEvidence,
    LineupEvidenceSnapshot,
)
from erguoyuan_football.context.temporal import ContextTemporalPolicy
from erguoyuan_football.context.tournament_state import TournamentStateBuilder
from erguoyuan_football.contracts.predictions import ProbabilityVector
from erguoyuan_football.output_contract.schemas import (
    AvailabilityStatus,
    CanonicalPredictionResult,
    ProbabilityStage,
)


@dataclass(frozen=True)
class TargetContextMatch:
    match_id: str
    competition_id: str
    season_id: str
    home_team_id: str
    away_team_id: str
    match_date: date
    round_name: str | None = None
    tie_id: str | None = None
    leg_number: int | None = None


@dataclass(frozen=True)
class ContextPredictionOutput:
    assessment: ContextAssessment
    canonical: CanonicalPredictionResult
    gates: tuple[ContextGateResult, ...]
    audit: dict[str, object]


class ContextPipeline:
    """Build context independently; unavailable lineups never disable standings."""

    def __init__(self, *, rules: CompetitionRuleRegistry,
                 context_model_status: str = "NOT_TRAINED_INSUFFICIENT_PIT_CONTEXT") -> None:
        self.rules = rules
        self.tournament = TournamentStateBuilder(rules)
        self.schedule = ScheduleContextBuilder()
        self.incentive = TournamentIncentiveEngine()
        self.features = ContextFeatureBuilder()
        self.adjustment = ContextAdjustmentEngine()
        self.lineups = LineupEvidenceLayer()
        self.injuries = InjuryEvidenceLayer()
        self.rotation = RotationRiskEngine()
        self.context_model_status = context_model_status

    def build_many(self, *, base_predictions: Iterable[CanonicalPredictionResult],
                   matches: Mapping[str, TargetContextMatch],
                   events: Iterable[HistoricalMatchEvent],
                   lineup_evidence: Mapping[str, tuple[LineupEvidenceSnapshot, ...]] | None = None,
                   injury_evidence: Mapping[str, tuple[InjuryEvidence, ...]] | None = None
                   ) -> tuple[ContextPredictionOutput, ...]:
        """Batch-transform real histories in memory; no per-match database queries."""
        event_rows = tuple(events)
        lineups_by_match = lineup_evidence or {}
        injuries_by_match = injury_evidence or {}
        outputs: list[ContextPredictionOutput] = []
        for base in base_predictions:
            match = matches.get(base.match_id)
            if match is None:
                raise ValueError("CONTEXT_TARGET_MATCH_NOT_FOUND")
            outputs.append(self.build_one(base=base, match=match, events=event_rows,
                lineup_evidence=lineups_by_match.get(base.match_id, ()),
                injury_evidence=injuries_by_match.get(base.match_id, ())))
        return tuple(outputs)

    def build_one(self, *, base: CanonicalPredictionResult,
                  match: TargetContextMatch,
                  events: Iterable[HistoricalMatchEvent],
                  lineup_evidence: tuple[LineupEvidenceSnapshot, ...] = (),
                  injury_evidence: tuple[InjuryEvidence, ...] = ()) -> ContextPredictionOutput:
        self._validate_base(base, match)
        all_events = tuple(events)
        prior: list[HistoricalMatchEvent] = []
        filtered_events = 0
        for event in all_events:
            if event.competition_id != match.competition_id or event.season_id != match.season_id:
                continue
            if event.match_date >= match.match_date or event.match_id == match.match_id:
                filtered_events += 1
                continue
            ContextTemporalPolicy.require_event(event, target_match_id=match.match_id,
                target_date=match.match_date, prediction_time=base.prediction_time, date_safe=True)
            prior.append(event)
        rule = self.rules.resolve(match.competition_id, match.season_id, match.match_date,
                                  prediction_time=base.prediction_time)
        tournament = self.tournament.build(
            match_id=match.match_id, competition_id=match.competition_id,
            season_id=match.season_id, target_date=match.match_date,
            prediction_time=base.prediction_time, home_team_id=match.home_team_id,
            away_team_id=match.away_team_id, events=prior,
            # Provider round metadata is intentionally not consumed unless historical PIT proof exists.
            round_name=None, tie_id=match.tie_id, leg_number=match.leg_number,
        )
        all_prior_events = tuple(event for event in all_events if event.match_date < match.match_date)
        schedule = self.schedule.build(
            match_id=match.match_id, competition_id=match.competition_id,
            season_id=match.season_id, target_date=match.match_date,
            home_team_id=match.home_team_id, away_team_id=match.away_team_id,
            events=all_prior_events, neutral_venue=None, schedule_coverage_verified=False,
        )
        incentive = self.incentive.build(match_id=match.match_id,
            home_team_id=match.home_team_id, away_team_id=match.away_team_id,
            standings=tournament.standings_state, rule=rule, remaining_matches=None)
        accepted_lineups, rejected_lineups = self.lineups.evaluate(
            match_id=match.match_id, prediction_time=base.prediction_time,
            kickoff_time=base.kickoff_time, evidence=lineup_evidence)
        accepted_injuries, rejected_injuries = self.injuries.evaluate(
            match_id=match.match_id, prediction_time=base.prediction_time,
            evidence=injury_evidence)
        rotation = self.rotation.evaluate(historical_lineups=(), current_lineup=accepted_lineups,
            schedule_features_available=schedule.availability == Availability.AVAILABLE)
        event_dates = {event.match_id: event.match_date for event in prior
                       if event.match_id in set(tournament.source_ids)
                       or event.match_id in set(schedule.source_match_ids_home)
                       or event.match_id in set(schedule.source_match_ids_away)}
        feature_vector = self.features.build(
            match_id=match.match_id, prediction_snapshot_id=base.prediction_snapshot_id,
            home_team_id=match.home_team_id, away_team_id=match.away_team_id,
            prediction_date=match.match_date, tournament=tournament, incentive=incentive,
            schedule=schedule, lineup_available=bool(accepted_lineups),
            injury_available=bool(accepted_injuries), player_strength_available=False,
            source_event_dates=event_dates, rotation_risk=rotation.probability,
            lineup_source_ids=tuple(item.source_id for item in accepted_lineups),
            injury_source_ids=tuple(item.source_id for item in accepted_injuries),
            lineup_strength_delta=None, injury_strength_delta=None,
        )
        data_availability = ContextDataAvailabilityReport(
            match_id=match.match_id, prediction_snapshot_id=base.prediction_snapshot_id,
            prediction_time=base.prediction_time,
            component_status={
                "historical_results": Availability.AVAILABLE if prior else Availability.UNAVAILABLE,
                "competition_rules": Availability.AVAILABLE if rule else Availability.UNAVAILABLE,
                "standings": Availability.AVAILABLE if tournament.standings_state
                    and tournament.standings_state.source_match_ids else Availability.UNAVAILABLE,
                "schedule_coverage": schedule.availability,
                "lineups": Availability.AVAILABLE if accepted_lineups else Availability.UNAVAILABLE,
                "injuries": Availability.AVAILABLE if accepted_injuries else Availability.UNAVAILABLE,
                "player_strength": Availability.UNAVAILABLE,
                "rotation_model": rotation.availability,
            },
            source_ids={
                "historical_results": tuple(event.match_id for event in prior),
                "competition_rules": (rule.source,) if rule else (),
                "standings": (tournament.standings_state.source_match_ids
                              if tournament.standings_state else ()),
                "schedule_coverage": tuple(dict.fromkeys((*schedule.source_match_ids_home,
                                                           *schedule.source_match_ids_away)))
                    if schedule.availability == Availability.AVAILABLE else (),
                "lineups": tuple(item.source_id for item in accepted_lineups),
                "injuries": tuple(item.source_id for item in accepted_injuries),
                "player_strength": (), "rotation_model": (),
            },
            reason_codes=tuple(dict.fromkeys((*tournament.unavailable_reasons,
                *schedule.reason_codes, *rejected_lineups, *rejected_injuries,
                rotation.reason))),
        )
        probabilities = (base.p_home, base.p_draw, base.p_away)
        if any(value is None for value in probabilities):
            raise ValueError("PHASE10_BASE_PROBABILITY_UNAVAILABLE")
        assert base.p_home is not None and base.p_draw is not None and base.p_away is not None
        typed_base = (base.p_home, base.p_draw, base.p_away)
        adjusted, ledger, adjust_reasons = self.adjustment.evaluate(
            prediction_id=base.prediction_id,
            prediction_snapshot_id=base.prediction_snapshot_id,
            match_id=base.match_id, base=typed_base, features=feature_vector,
        )
        tournament_gate = TournamentGate().evaluate(tournament)
        lineup_gate = LineupGate().evaluate(accepted_lineups,
                                            temporal_rejections=rejected_lineups)
        context_gate = ContextDataGate().evaluate(
            point_in_time_passed=True, rule=rule, entity_mapping_verified=True,
            evidence_fresh=(not accepted_lineups or not rejected_lineups),
            context_complete=(bool(prior) and tournament_gate.passed and schedule.availability == Availability.AVAILABLE),
            conflicts_resolved=not rejected_injuries,
            post_match_data_present=False,
        )
        status = (ContextStatus.ADJUSTED if adjusted.tournament_adjustment_applied
                  or adjusted.lineup_adjustment_applied else
                  ContextStatus.EVALUATED_NO_ADJUSTMENT if prior
                  else ContextStatus.UNAVAILABLE)
        reason_codes = tuple(dict.fromkeys((
            *tournament.unavailable_reasons, *schedule.reason_codes,
            *rejected_lineups, *rejected_injuries, *adjust_reasons,
            *((f"SAME_DAY_OR_FUTURE_RESULTS_EXCLUDED:{filtered_events}",) if filtered_events else ()),
            *context_gate.reason_codes, *lineup_gate.reason_codes,
        )))
        assessment = ContextAssessment(
            match_id=match.match_id, prediction_snapshot_id=base.prediction_snapshot_id,
            context_status=status, tournament_state=tournament, incentive=incentive,
            schedule=schedule, data_availability=data_availability,
            lineup_status=Availability.AVAILABLE if accepted_lineups else Availability.UNAVAILABLE,
            lineup_evidence=accepted_lineups, injury_evidence=accepted_injuries,
            feature_vector=feature_vector, adjusted_probability=adjusted, ledger=ledger,
            context_model_status=self.context_model_status,
            lineup_model_status=("BLOCKED_DATA" if not accepted_lineups else "NOT_TRAINED"),
            reason_codes=reason_codes,
        )
        audit = ContextAudit().review(assessment)
        gates = (context_gate, tournament_gate, lineup_gate)
        canonical_payload = base.model_dump(mode="json")
        canonical_payload["context_status"] = status.value
        canonical_payload["context_adjustment_status"] = (
            "APPLIED" if status == ContextStatus.ADJUSTED else
            "NO_ADJUSTMENT_DUE_TO_INSUFFICIENT_EVIDENCE")
        canonical_payload["context"] = {
            "assessment": assessment.model_dump(mode="json"),
            "gates": [gate.model_dump(mode="json") for gate in gates],
            "audit": audit,
            "probability_chain": {
                "BASE": {"home": adjusted.base_p_home, "draw": adjusted.base_p_draw,
                         "away": adjusted.base_p_away},
                "TOURNAMENT": {"home": adjusted.tournament_p_home, "draw": adjusted.tournament_p_draw,
                               "away": adjusted.tournament_p_away},
                "LINEUP": {"home": adjusted.lineup_p_home, "draw": adjusted.lineup_p_draw,
                           "away": adjusted.lineup_p_away},
                "FINAL": {"home": adjusted.final_p_home, "draw": adjusted.final_p_draw,
                          "away": adjusted.final_p_away},
            },
        }
        if status == ContextStatus.ADJUSTED:
            canonical_payload["probability_stage"] = ProbabilityStage.CONTEXT_ADJUSTED_CORE.value
        canonical = CanonicalPredictionResult.model_validate(canonical_payload)
        return ContextPredictionOutput(assessment, canonical, gates, audit)

    @staticmethod
    def _validate_base(base: CanonicalPredictionResult, match: TargetContextMatch) -> None:
        if base.match_id != match.match_id or base.competition_id != match.competition_id:
            raise ValueError("PHASE10_BASE_TARGET_MISMATCH")
        if (base.status != AvailabilityStatus.AVAILABLE or
                base.probability_stage != ProbabilityStage.FINAL_CORE_CALIBRATED or
                base.validation_status != "DEVELOPMENT_ONLY" or
                base.production_status != "NOT_PROMOTED"):
            raise ValueError("PHASE10_REQUIRES_DEVELOPMENT_CALIBRATED_BASE")
        if base.data_origin != "REAL" or base.pit_status != "PASS":
            raise ValueError("PHASE10_REQUIRES_REAL_PIT_BASE")
        if base.match_date is None or base.match_date != match.match_date:
            raise ValueError("PHASE10_MATCH_DATE_MISMATCH")
        if base.prediction_temporal_mode != "DATE_SAFE_BATCH" or base.prediction_time.date() != match.match_date:
            raise ValueError("PHASE10_DATE_SAFE_BOUNDARY_REQUIRED")
        if any(value is None for value in (base.p_home, base.p_draw, base.p_away)):
            raise ValueError("PHASE10_BASE_PROBABILITY_UNAVAILABLE")
        assert base.p_home is not None and base.p_draw is not None and base.p_away is not None
        ProbabilityVector(p_home=base.p_home, p_draw=base.p_draw, p_away=base.p_away)
