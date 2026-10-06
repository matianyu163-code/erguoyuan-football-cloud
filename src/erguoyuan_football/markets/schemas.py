"""Canonical odds, point-in-time market snapshots and dependency-aware features."""

from __future__ import annotations

import math
from decimal import Decimal
from enum import StrEnum
from typing import Literal
from uuid import uuid4

from pydantic import Field, model_validator

from erguoyuan_football.contracts.common import (
    Availability,
    Contract,
    Identifier,
    Probability,
    UTCTime,
    now,
    utc,
)


class MarketType(StrEnum):
    MATCH_1X2 = "MATCH_1X2"
    ASIAN_HANDICAP = "ASIAN_HANDICAP"
    TOTALS = "TOTALS"
    SPORTTERY_1X2 = "SPORTTERY_1X2"
    SPORTTERY_HANDICAP_1X2 = "SPORTTERY_HANDICAP_1X2"


class Selection(StrEnum):
    HOME = "HOME"
    DRAW = "DRAW"
    AWAY = "AWAY"
    OVER = "OVER"
    UNDER = "UNDER"


class SourceType(StrEnum):
    AUTHORIZED_PROVIDER = "AUTHORIZED_PROVIDER"
    JC_OFFICIAL_PUBLIC = "JC_OFFICIAL_PUBLIC"
    USER_CONFIRMED_MARKET = "USER_CONFIRMED_MARKET"
    PUBLIC_PAGE = "PUBLIC_PAGE"
    SPORTTERY_OFFICIAL_PUBLIC = "SPORTTERY_OFFICIAL_PUBLIC"
    USER_SCREENSHOT = "USER_SCREENSHOT"
    SYNTHETIC_TEST = "SYNTHETIC_TEST"
    LEGACY_PHASE2 = "LEGACY_PHASE2"


class TimestampQuality(StrEnum):
    SOURCE_NATIVE = "SOURCE_NATIVE"
    PROVIDER_NATIVE = "PROVIDER_NATIVE"
    RETRIEVAL_ONLY = "RETRIEVAL_ONLY"
    UNKNOWN = "UNKNOWN"


class QuoteQuality(StrEnum):
    GOOD = "GOOD"
    SUSPENDED = "SUSPENDED"
    LIVE_EXCLUDED = "LIVE_EXCLUDED"
    INVALID_QUOTE = "INVALID_QUOTE"
    TIMESTAMP_UNAVAILABLE = "TIMESTAMP_UNAVAILABLE"
    STALE = "STALE"
    HIGH_MARGIN = "HIGH_MARGIN"


class MarketQualityStatus(StrEnum):
    GOOD = "GOOD"
    ACCEPTABLE = "ACCEPTABLE"
    POOR = "POOR"
    UNAVAILABLE = "UNAVAILABLE"


class PredictionHorizon(StrEnum):
    T_MINUS_24H = "T_MINUS_24H"
    T_MINUS_6H = "T_MINUS_6H"
    T_MINUS_3H = "T_MINUS_3H"
    T_MINUS_60M = "T_MINUS_60M"
    T_MINUS_15M = "T_MINUS_15M"
    CUSTOM = "CUSTOM"


class DeVigMethod(StrEnum):
    MULTIPLICATIVE = "MULTIPLICATIVE"
    ADDITIVE = "ADDITIVE"
    POWER = "POWER"
    SHIN = "SHIN"
    ODDS_RATIO = "ODDS_RATIO"


class ConsensusMethod(StrEnum):
    MEDIAN = "MEDIAN"
    TRIMMED_MEAN = "TRIMMED_MEAN"
    WEIGHTED_MEAN = "WEIGHTED_MEAN"


class DependencyTag(StrEnum):
    MARKET_RAW = "MARKET_RAW"
    MARKET_DEVIG = "MARKET_DEVIG"
    MARKET_CONSENSUS = "MARKET_CONSENSUS"
    MARKET_IMPLIED_GOALS = "MARKET_IMPLIED_GOALS"
    MARKET_BAYES = "MARKET_BAYES"


class OddsQuote(Contract):
    """One canonical price, separate from its provider, bookmaker and source clock."""

    quote_id: Identifier = Field(default_factory=lambda: str(uuid4()))
    match_id: Identifier
    provider_id: Identifier
    bookmaker_id: Identifier
    market_type: MarketType
    selection: Selection
    line_quarters: int | None = None
    odds_decimal: Decimal | None = None
    original_odds: str | None = None
    original_format: str = "DECIMAL"
    source_type: SourceType
    source_event_id: str | None = None
    source_quote_id: str | None = None
    source_time: UTCTime | None = None
    retrieved_at: UTCTime
    as_of_time: UTCTime | None = None
    timestamp_quality: TimestampQuality = TimestampQuality.UNKNOWN
    is_live: bool = False
    is_suspended: bool = False
    is_opening_confirmed: bool = False
    currency: str | None = None
    schema_version: Identifier
    content_hash: Identifier
    quality_status: QuoteQuality = QuoteQuality.GOOD
    quality_reason: str | None = None
    screenshot_image_hash: str | None = None
    ocr_confidence: Probability | None = None

    @property
    def line(self) -> Decimal | None:
        """Return the exact line as a Decimal; quarter_units are the stored identity."""
        return Decimal(self.line_quarters) / Decimal(4) if self.line_quarters is not None else None

    @model_validator(mode="after")
    def validate_market_quote(self) -> OddsQuote:
        needs_line = self.market_type in {MarketType.ASIAN_HANDICAP, MarketType.TOTALS,
                                          MarketType.SPORTTERY_HANDICAP_1X2}
        if needs_line != (self.line_quarters is not None):
            raise ValueError("market line presence does not match market type")
        if self.market_type in {MarketType.MATCH_1X2, MarketType.SPORTTERY_1X2,
                                MarketType.SPORTTERY_HANDICAP_1X2} and self.selection not in {
                                    Selection.HOME, Selection.DRAW, Selection.AWAY
                                }:
            raise ValueError("selection invalid for 1X2 market")
        if self.market_type in {MarketType.ASIAN_HANDICAP} and self.selection not in {Selection.HOME, Selection.AWAY}:
            raise ValueError("selection invalid for Asian handicap")
        if self.market_type == MarketType.TOTALS and self.selection not in {Selection.OVER, Selection.UNDER}:
            raise ValueError("selection invalid for totals")
        if self.quality_status == QuoteQuality.INVALID_QUOTE:
            if self.odds_decimal is not None:
                raise ValueError("invalid quote must not expose a usable decimal price")
        elif self.odds_decimal is None or not self.odds_decimal.is_finite() or self.odds_decimal <= 1:
            raise ValueError("valid decimal odds must be finite and greater than 1")
        if self.timestamp_quality in {TimestampQuality.SOURCE_NATIVE, TimestampQuality.PROVIDER_NATIVE}:
            if self.source_time is None or self.as_of_time is None:
                raise ValueError("native timestamp quality requires source_time and as_of_time")
            if utc(self.source_time) != utc(self.as_of_time):
                raise ValueError("as_of_time must reflect the declared source quote time")
        elif self.as_of_time is not None and self.source_time is not None and utc(self.as_of_time) != utc(self.source_time):
            raise ValueError("as_of_time/source_time mismatch")
        if self.source_type == SourceType.USER_SCREENSHOT and not self.screenshot_image_hash:
            raise ValueError("screenshot quote requires the source image hash")
        if self.source_type == SourceType.USER_SCREENSHOT and self.ocr_confidence is None:
            raise ValueError("screenshot quote requires OCR confidence")
        if self.is_suspended and self.quality_status == QuoteQuality.GOOD:
            raise ValueError("suspended quote must be marked SUSPENDED")
        if self.is_live and self.quality_status == QuoteQuality.GOOD:
            raise ValueError("live quote must be marked LIVE_EXCLUDED")
        return self


class Bookmaker(Contract):
    bookmaker_id: Identifier
    canonical_name: Identifier
    aliases: tuple[str, ...] = ()
    provider_mapping: dict[str, str] = Field(default_factory=dict)
    active: bool = True
    region: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class QuoteSelectionResult(Contract):
    availability: Availability
    quote: OddsQuote | None = None
    reason: str

    @model_validator(mode="after")
    def result_consistency(self) -> QuoteSelectionResult:
        if (self.availability == Availability.AVAILABLE) != (self.quote is not None):
            raise ValueError("quote availability must match quote presence")
        return self


class DeVigResult(Contract):
    market_id: Identifier
    method: DeVigMethod
    raw_implied_probabilities: dict[str, Probability]
    devig_probabilities: dict[str, Probability]
    overround: float
    method_parameters: dict[str, float] = Field(default_factory=dict)
    success: bool
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def probability_sum(self) -> DeVigResult:
        if set(self.raw_implied_probabilities) != set(self.devig_probabilities):
            raise ValueError("raw and de-vig selections must align")
        if self.success and (not self.devig_probabilities or not math.isclose(
            math.fsum(self.devig_probabilities.values()), 1.0, rel_tol=0, abs_tol=1e-8
        )):
            raise ValueError("successful de-vig probabilities must sum to one")
        return self


class DeVigPolicy(Contract):
    """Fixed, versioned method selection; never choose a method per fixture."""

    policy_version: Identifier
    method_by_market: dict[MarketType, DeVigMethod]
    status: Literal["PROVISIONAL_NOT_VALIDATED", "VALIDATED"] = "PROVISIONAL_NOT_VALIDATED"
    validation_evidence_hash: str | None = None

    @model_validator(mode="after")
    def validated_policy_evidence(self) -> DeVigPolicy:
        if self.status == "VALIDATED":
            evidence = self.validation_evidence_hash or ""
            if len(evidence) != 64 or any(character not in "0123456789abcdef" for character in evidence.lower()):
                raise ValueError("validated de-vig policy requires SHA256 OOS evidence")
        return self


class MarketQualityReport(Contract):
    report_id: Identifier = Field(default_factory=lambda: str(uuid4()))
    match_id: Identifier
    prediction_time: UTCTime
    source_count: int = Field(ge=0)
    bookmaker_count: int = Field(ge=0)
    freshness_seconds: int | None = Field(default=None, ge=0)
    timestamp_quality: TimestampQuality
    overround: float | None = None
    dispersion: dict[str, float] = Field(default_factory=dict)
    suspended_quote_count: int = Field(ge=0)
    invalid_quote_count: int = Field(ge=0)
    fit_error: float | None = Field(default=None, ge=0)
    markets_available: tuple[MarketType, ...] = ()
    quality_score: float | None = Field(default=None, ge=0, le=1)
    status: MarketQualityStatus
    reason_codes: tuple[str, ...] = ()


class MarketSnapshot(Contract):
    market_snapshot_id: Identifier = Field(default_factory=lambda: str(uuid4()))
    match_id: Identifier
    prediction_time: UTCTime
    created_at: UTCTime = Field(default_factory=now)
    kickoff_time: UTCTime
    prediction_horizon: PredictionHorizon
    horizon_seconds: int = Field(ge=0)
    source_count: int = Field(ge=0)
    bookmaker_count: int = Field(ge=0)
    quote_count: int = Field(ge=0)
    markets_available: tuple[MarketType, ...] = ()
    freshness_seconds: int | None = Field(default=None, ge=0)
    quality_status: MarketQualityStatus
    included_quote_ids: tuple[Identifier, ...] = ()
    quotes: tuple[OddsQuote, ...] = ()

    @model_validator(mode="after")
    def point_in_time(self) -> MarketSnapshot:
        if self.prediction_time >= self.kickoff_time:
            raise ValueError("market prediction time must precede kickoff")
        if self.horizon_seconds != int((utc(self.kickoff_time) - utc(self.prediction_time)).total_seconds()):
            raise ValueError("market horizon_seconds inconsistent with timestamps")
        if len(self.included_quote_ids) != len(set(self.included_quote_ids)):
            raise ValueError("market snapshot quote IDs must be unique")
        if set(self.included_quote_ids) != {quote.quote_id for quote in self.quotes}:
            raise ValueError("market snapshot quote references must match embedded quote records")
        for quote in self.quotes:
            if quote.match_id != self.match_id:
                raise ValueError("cross-match quote in market snapshot")
            if quote.as_of_time is None or utc(quote.as_of_time) > self.prediction_time:
                raise ValueError("quote source time unavailable or after prediction time")
            if quote.retrieved_at > self.prediction_time:
                raise ValueError("quote was retrieved after prediction time")
        return self


class MarketConsensus(Contract):
    consensus_id: Identifier = Field(default_factory=lambda: str(uuid4()))
    match_id: Identifier
    market_snapshot_id: Identifier
    market_type: MarketType
    line_quarters: int | None = None
    prediction_time: UTCTime
    probabilities: dict[str, Probability]
    bookmaker_count: int = Field(ge=1)
    provider_count: int = Field(ge=1)
    overround_mean: float = Field(ge=0)
    dispersion: dict[str, float]
    freshness_seconds: int = Field(ge=0)
    consensus_method: ConsensusMethod
    devig_method: DeVigMethod
    devig_policy_version: Identifier
    quality_status: MarketQualityStatus
    quote_ids: tuple[Identifier, ...]
    dependency_tags: tuple[DependencyTag, ...] = (DependencyTag.MARKET_RAW, DependencyTag.MARKET_DEVIG,
                                                   DependencyTag.MARKET_CONSENSUS)

    @model_validator(mode="after")
    def validate_consensus_shape(self) -> MarketConsensus:
        if self.market_type in {MarketType.MATCH_1X2, MarketType.SPORTTERY_1X2,
                                MarketType.SPORTTERY_HANDICAP_1X2}:
            if set(self.probabilities) != {"HOME", "DRAW", "AWAY"} or not math.isclose(
                math.fsum(self.probabilities.values()), 1, abs_tol=1e-8
            ):
                raise ValueError("1X2 consensus must be a complete probability vector")
        else:
            expected = {"HOME", "AWAY"} if self.market_type == MarketType.ASIAN_HANDICAP else {"OVER", "UNDER"}
            if set(self.probabilities) != expected:
                raise ValueError("two-way consensus requires both canonical selections")
            if not math.isclose(math.fsum(self.probabilities.values()), 1.0, rel_tol=0, abs_tol=1e-8):
                raise ValueError("two-way consensus probabilities must sum to one")
        if not self.quote_ids:
            raise ValueError("consensus requires quote lineage")
        return self


class MarketGoalFeatures(Contract):
    """Market-implied goal rates with full PIT source lineage and fit diagnostics."""

    availability: Availability
    reason: str | None = None
    match_id: Identifier
    market_snapshot_id: Identifier
    prediction_time: UTCTime
    prediction_horizon: PredictionHorizon = PredictionHorizon.CUSTOM
    as_of_time: UTCTime | None = None
    retrieved_at: UTCTime | None = None
    market_implied_lambda_home: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    market_implied_lambda_away: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    market_p_home: Probability | None = None
    market_p_draw: Probability | None = None
    market_p_away: Probability | None = None
    inference_mode: str | None = None
    fit_error: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    residuals: dict[str, float] = Field(default_factory=dict)
    optimizer_success: bool = False
    matrix_mass: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    score_matrix: tuple[tuple[Probability, ...], ...] | None = None
    score_matrix_tail_mass: Probability | None = None
    score_matrix_max_goals: int | None = Field(default=None, ge=1)
    source_count: int = Field(default=0, ge=0)
    bookmaker_count: int = Field(default=0, ge=0)
    source_ids: tuple[Identifier, ...] = ()
    dispersion: dict[str, float] = Field(default_factory=dict)
    dependency_ids: tuple[Identifier, ...] = ()
    dependency_tags: tuple[DependencyTag, ...] = (DependencyTag.MARKET_RAW, DependencyTag.MARKET_DEVIG,
                                                   DependencyTag.MARKET_CONSENSUS, DependencyTag.MARKET_IMPLIED_GOALS)
    devig_policy_version: str | None = None

    @model_validator(mode="after")
    def goal_availability(self) -> MarketGoalFeatures:
        rates = (self.market_implied_lambda_home, self.market_implied_lambda_away)
        market_probabilities = (self.market_p_home, self.market_p_draw, self.market_p_away)
        if (any(value is not None for value in market_probabilities)
                and (any(value is None for value in market_probabilities) or not math.isclose(
                    math.fsum(value for value in market_probabilities if value is not None), 1.0,
                    rel_tol=0, abs_tol=1e-8))):
            raise ValueError("market 1X2 baseline must be a complete probability vector")
        if self.availability == Availability.AVAILABLE:
            if any(value is None for value in rates) or not self.optimizer_success or self.fit_error is None:
                raise ValueError("available market lambdas require a successful, measured fit")
            if self.as_of_time is None or self.retrieved_at is None or not self.dependency_ids:
                raise ValueError("available market lambda requires PIT times and quote lineage")
            if max(utc(self.as_of_time), utc(self.retrieved_at)) > self.prediction_time:
                raise ValueError("market implied goals contain future information")
            if self.score_matrix is None or self.score_matrix_tail_mass is None or self.score_matrix_max_goals is None:
                raise ValueError("available market goal fit requires a score matrix and tail accounting")
            if len(self.score_matrix) != self.score_matrix_max_goals + 1 or any(
                len(row) != self.score_matrix_max_goals + 1 for row in self.score_matrix
            ) or not math.isclose(math.fsum(math.fsum(row) for row in self.score_matrix), 1.0, abs_tol=1e-8):
                raise ValueError("market implied score matrix support or mass is invalid")
        elif any(value is not None for value in rates) or self.reason is None:
            raise ValueError("unavailable market lambdas must be null and include a reason")
        return self


class MarketMovement(Contract):
    movement_id: Identifier = Field(default_factory=lambda: str(uuid4()))
    match_id: Identifier
    market_type: MarketType
    selection: Selection
    line_quarters: int | None = None
    opening_quote_id: Identifier | None = None
    current_quote_id: Identifier
    opening_probability: Probability | None = None
    current_probability: Probability | None = None
    delta_probability: float | None = None
    delta_logit_probability: float | None = None
    opening_line_quarters: int | None = None
    current_line_quarters: int | None = None
    line_delta_quarters: int | None = None
    opening_odds_decimal: Decimal | None = None
    current_odds_decimal: Decimal
    delta_odds: Decimal | None = None
    elapsed_seconds: int | None = Field(default=None, ge=0)
    as_of_time: UTCTime
    dependency_tags: tuple[DependencyTag, ...] = (DependencyTag.MARKET_RAW,)


class MarketEngineResult(Contract):
    market_snapshot: MarketSnapshot
    consensuses: tuple[MarketConsensus, ...] = ()
    quality_report: MarketQualityReport
    market_goal_features: MarketGoalFeatures
    movements: tuple[MarketMovement, ...] = ()
    production_gate_status: str
    reason: str | None = None
