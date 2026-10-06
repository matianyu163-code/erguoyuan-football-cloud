"""Auditable match heads and ranked candidate contracts."""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from erguoyuan_football.contracts.common import Contract, Probability
from erguoyuan_football.contracts.predictions import ProbabilityVector
from erguoyuan_football.prediction_heads.evidence import (
    HalfFullTimeEvidence,
    OfficialHandicapEvidence,
)
from erguoyuan_football.prediction_heads.heads import (
    PlayTop2,
    htft_top2,
    official_handicap_1x2,
    score_top2,
    total_top2,
)
from erguoyuan_football.prediction_heads.readiness import MatrixSource
from erguoyuan_football.prediction_heads.reconcile import ReconciledMatrix, matrix_hash


class MatchHeads(Contract):
    match_id: str
    competition_id: str
    match_date: date
    kickoff_time: datetime | None = None
    kickoff_status: str = "EXACT_KICKOFF_UNAVAILABLE"
    home_team_id: str
    away_team_id: str
    home_team_name: str
    away_team_name: str
    prediction_snapshot_id: str
    probability: ProbabilityVector | None
    probability_source_stage: str | None
    probability_source_id: str | None
    score_top2: PlayTop2 | None = None
    totals_top2: PlayTop2 | None = None
    handicap_probabilities: dict[str, float] | None = None
    official_home_handicap: int | None = None
    official_handicap_evidence: OfficialHandicapEvidence | None = None
    htft_top2: PlayTop2 | None = None
    htft_evidence: HalfFullTimeEvidence | None = None
    matrix_source: MatrixSource | None = None
    reconciliation: ReconciledMatrix | None = None
    matrix_status: str = "UNAVAILABLE"
    handicap_status: str = "UNAVAILABLE"
    htft_status: str = "UNAVAILABLE"
    market_status: str = "UNAVAILABLE"
    context_status: str = "UNAVAILABLE"
    lineup_status: str = "UNAVAILABLE"
    injury_status: str = "UNAVAILABLE"
    models_failed: tuple[str, ...] = ()
    models_unavailable: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    validation_status: str = "DEVELOPMENT_ONLY"
    production_status: str = "NOT_PROMOTED"
    temporal_mode: str = "DATE_SAFE_BATCH"
    data_origin: str

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_head_status(self) -> MatchHeads:
        if self.temporal_mode == "EXACT_UTC" and self.kickoff_time is None:
            raise ValueError("EXACT_UTC_KICKOFF_REQUIRED")
        if self.matrix_status == "AVAILABLE_RECONCILED" and (
                self.matrix_source is None or self.reconciliation is None or
                self.score_top2 is None or self.totals_top2 is None):
            raise ValueError("RECONCILED_MATRIX_EVIDENCE_MISSING")
        if self.reconciliation is not None:
            if self.probability is None:
                raise ValueError("RECONCILED_MATRIX_REQUIRES_CORE")
            if (self.matrix_source is None or
                self.matrix_source.score_matrix_hash != self.reconciliation.reference_matrix_hash or
                matrix_hash(self.matrix_source.matrix) != self.reconciliation.reference_matrix_hash):
                raise ValueError("REFERENCE_SCORE_MATRIX_LINEAGE_MISMATCH")
            observed = self.reconciliation.final_matrix.outcome()
            for actual, expected in zip((observed.p_home, observed.p_draw, observed.p_away),
                (self.probability.p_home, self.probability.p_draw, self.probability.p_away), strict=True):
                if not math.isclose(actual, expected, abs_tol=1e-8, rel_tol=0):
                    raise ValueError("SCORE_MATRIX_CORE_MASS_MISMATCH")
            if self.score_top2 != score_top2(self.reconciliation.final_matrix) or (
                    self.totals_top2 != total_top2(self.reconciliation.final_matrix)):
                raise ValueError("SCORE_OR_TOTALS_NOT_FROM_FINAL_MATRIX")
        if self.matrix_status != "AVAILABLE_RECONCILED" and (
                self.reconciliation is not None or self.score_top2 is not None or
                self.totals_top2 is not None):
            raise ValueError("UNAVAILABLE_MATRIX_DERIVATIVES_PRESENT")
        if self.handicap_status == "UNAVAILABLE" and self.handicap_probabilities is not None:
            raise ValueError("UNAVAILABLE_HANDICAP_PROBABILITIES_PRESENT")
        if self.htft_status == "UNAVAILABLE" and self.htft_top2 is not None:
            raise ValueError("UNAVAILABLE_HTFT_PROBABILITIES_PRESENT")
        if self.handicap_status == "AVAILABLE":
            if (self.reconciliation is None or self.official_handicap_evidence is None or
                self.official_home_handicap != self.official_handicap_evidence.home_handicap or
                self.official_handicap_evidence.match_id != self.match_id or
                self.handicap_probabilities is None):
                raise ValueError("OFFICIAL_HANDICAP_EVIDENCE_MISSING")
            expected_handicap = official_handicap_1x2(self.reconciliation.final_matrix,
                self.official_home_handicap, verified_source=True)
            if any(not math.isclose(self.handicap_probabilities[key], value, abs_tol=1e-8)
                   for key, value in expected_handicap.items()):
                raise ValueError("HANDICAP_NOT_FROM_FINAL_MATRIX")
        if self.htft_status == "AVAILABLE" and (
            self.htft_evidence is None or self.htft_evidence.match_id != self.match_id or
            self.htft_evidence.prediction_snapshot_id != self.prediction_snapshot_id or
            self.htft_top2 != htft_top2(self.htft_evidence.distribution,
                                         independent_oos_model=True)
        ):
            raise ValueError("HTFT_INDEPENDENT_MODEL_EVIDENCE_MISSING")
        return self


class RankedCandidate(Contract):
    """A computed option, independent of price or purchase advice."""

    candidate_id: str
    prediction_snapshot_ids: tuple[str, ...]
    match_ids: tuple[str, ...]
    objective: Literal["HIGHEST_HIT", "VALUE", "LONGSHOT"]
    play_type: str
    selections: tuple[str, ...]
    component_probabilities: tuple[Probability, ...]
    joint_probability: Probability
    coverage_probability: Probability
    probability_method: str
    market_probability: Probability | None = None
    odds: float | None = Field(default=None, gt=1, allow_inf_nan=False)
    fair_odds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    expected_value: float | None = Field(default=None, allow_inf_nan=False)
    uncertainty: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    data_quality: str
    context_quality: str
    correlation_status: str
    rank: int = Field(ge=1)
    validation_status: str
    production_status: str
    market_status: str = "UNAVAILABLE"
    market_quote_ids: tuple[str, ...] = ()
    official_handicap_lines: tuple[int, ...] = ()
    reason_codes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_candidate(self) -> RankedCandidate:
        if not self.match_ids or len(set(self.match_ids)) != len(self.match_ids):
            raise ValueError("CANDIDATE_REQUIRES_DISTINCT_MATCHES")
        if len(self.match_ids) != len(self.prediction_snapshot_ids):
            raise ValueError("CANDIDATE_SNAPSHOT_COUNT_MISMATCH")
        if len(self.selections) != len(self.component_probabilities):
            raise ValueError("CANDIDATE_COMPONENT_MISMATCH")
        if "HANDICAP" in self.play_type and len(self.official_handicap_lines) != len(self.match_ids):
            raise ValueError("OFFICIAL_HANDICAP_LINES_MISSING")
        return self


class CandidateSlot(Contract):
    play_type: str
    status: Literal["AVAILABLE", "UNAVAILABLE"]
    candidate: RankedCandidate | None
    reason: str | None = None

    @model_validator(mode="after")
    def status_consistency(self) -> CandidateSlot:
        if (self.status == "AVAILABLE") != (self.candidate is not None):
            raise ValueError("CANDIDATE_SLOT_STATUS_MISMATCH")
        return self
