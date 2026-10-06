"""Six fixed sections of CORE REPORT V2."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field, model_validator

from erguoyuan_football.contracts.common import Contract
from erguoyuan_football.portfolio.schemas import (
    BetAdviceLedger,
    CandidateLedger,
    ExperimentalAccount,
    PortfolioLedger,
    SettlementLedger,
)
from erguoyuan_football.selection.schemas import CandidateSlot, MatchHeads

SECTION_ORDER = ("all_matches", "highest_hit", "high_hit_400", "value_100",
                 "longshot_20", "model_data_status")


class CoreReportV2Schema(Contract):
    report_version: str = "CORE_REPORT_V2"
    created_at: datetime
    section_order: tuple[str, ...] = SECTION_ORDER
    all_matches: tuple[MatchHeads, ...]
    highest_hit: tuple[CandidateSlot, ...]
    high_hit_400: ExperimentalAccount
    value_100: ExperimentalAccount
    longshot_20: ExperimentalAccount
    model_data_status: dict[str, object]
    candidate_ledger: CandidateLedger
    bet_advice_ledger: BetAdviceLedger
    portfolio_ledger: PortfolioLedger
    settlement_ledger: SettlementLedger = Field(default_factory=SettlementLedger)
    weekly_metrics: dict[str, object]

    @model_validator(mode="after")
    def fixed_shape(self) -> CoreReportV2Schema:
        if (self.section_order != SECTION_ORDER or len(self.highest_hit) != 5 or
                len(self.high_hit_400.entries) != 4):
            raise ValueError("CORE_REPORT_V2_FIXED_SECTIONS_CHANGED")
        if len({item.match_id for item in self.all_matches}) != len(self.all_matches):
            raise ValueError("DUPLICATE_ALL_MATCH_ROWS")
        shown = {slot.candidate.candidate_id for slot in self.highest_hit if slot.candidate}
        recorded = {item.candidate_id for item in self.candidate_ledger.candidates}
        if not shown <= recorded:
            raise ValueError("RANKED_CANDIDATE_HIDDEN_FROM_LEDGER")
        return self
