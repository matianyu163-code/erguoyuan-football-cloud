"""Frozen V5.1 display order; aliases preserve the phase-1 agreed labels."""

from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from erguoyuan_football.contracts.common import (
    Contract,
    NonNegative,
    Probability,
    UTCTime,
)

SUMMARY_FIELDS = ("编号", "比赛", "模型概率", "MDI", "URS", "Edge", "评级", "结论")
MATCH_FIELDS = ("比赛编码", "联赛", "主队 VS 客队", "玩法", "投注方向", "竞彩赔率", "模型概率", "市场概率", "Edge", "URS", "置信等级")
PARLAY_FIELDS = ("联合命中概率", "组合赔率", "理论EV", "风险")
CATEGORIES = ("胜平负二串一", "胜平负三串一", "总进球数", "半全场", "让球胜平负+普通胜平负混合二串一")
TOP_FIELDS = ("数据冻结时间", "今日比赛筛选总表", *CATEGORIES, "今日投入", "各组金额", "剩余滚存")
NO_BET = "无合格组合 / NO-BET"


class DisplayContract(Contract):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False, populate_by_name=True)


class SummaryRow(DisplayContract):
    number: str = Field(alias="编号")
    match: str = Field(alias="比赛")
    model_probability: Probability | None = Field(default=None, alias="模型概率")
    mdi: float | None = Field(default=None, alias="MDI")
    urs: float | None = Field(default=None, alias="URS")
    edge: float | None = Field(default=None, alias="Edge")
    rating: str | None = Field(default=None, alias="评级")
    conclusion: str = Field(default=NO_BET, alias="结论")


class MatchRow(DisplayContract):
    match_code: str = Field(alias="比赛编码")
    competition: str = Field(alias="联赛")
    matchup: str = Field(alias="主队 VS 客队")
    market: str = Field(alias="玩法")
    direction: str = Field(alias="投注方向")
    lottery_odds: float | None = Field(default=None, gt=1, alias="竞彩赔率")
    model_probability: Probability | None = Field(default=None, alias="模型概率")
    market_probability: Probability | None = Field(default=None, alias="市场概率")
    edge: float | None = Field(default=None, alias="Edge")
    urs: float | None = Field(default=None, alias="URS")
    confidence: str | None = Field(default=None, alias="置信等级")


class ParlayFields(DisplayContract):
    joint_probability: Probability | None = Field(default=None, alias="联合命中概率")
    combined_odds: float | None = Field(default=None, gt=1, alias="组合赔率")
    theoretical_ev: float | None = Field(default=None, alias="理论EV")
    risk: str | None = Field(default=None, alias="风险")


class ResultGroup(DisplayContract):
    conclusion: Literal["无合格组合 / NO-BET", "QUALIFIED"] = Field(default=NO_BET, alias="结论")
    matches: tuple[MatchRow, ...] = Field(default=(), alias="比赛明细")
    parlay: ParlayFields = Field(default_factory=ParlayFields, alias="串关")

    @model_validator(mode="after")
    def no_fake_group(self):
        if self.conclusion == NO_BET:
            if self.matches or any(v is not None for v in self.parlay.model_dump().values()):
                raise ValueError("NO-BET cannot carry a fabricated selection or parlay")
        elif not self.matches:
            raise ValueError("qualified group requires supplied match records")
        return self


class V51Output(DisplayContract):
    freeze_time: UTCTime = Field(alias="数据冻结时间")
    summary: tuple[SummaryRow, ...] = Field(default=(), alias="今日比赛筛选总表")
    two_leg: ResultGroup = Field(default_factory=ResultGroup, alias="胜平负二串一")
    three_leg: ResultGroup = Field(default_factory=ResultGroup, alias="胜平负三串一")
    total_goals: ResultGroup = Field(default_factory=ResultGroup, alias="总进球数")
    half_full: ResultGroup = Field(default_factory=ResultGroup, alias="半全场")
    mixed_two_leg: ResultGroup = Field(default_factory=ResultGroup, alias="让球胜平负+普通胜平负混合二串一")
    daily_stake: NonNegative | None = Field(default=None, alias="今日投入")
    group_amounts: dict[str, NonNegative | None] = Field(
        default_factory=lambda: {category: None for category in CATEGORIES}, alias="各组金额")
    remaining_bankroll: NonNegative | None = Field(default=None, alias="剩余滚存")

    @model_validator(mode="after")
    def fixed_groups(self):
        if tuple(self.group_amounts) != CATEGORIES:
            raise ValueError("group amount labels and order must match V5.1")
        for group, count in ((self.two_leg, 2), (self.three_leg, 3), (self.mixed_two_leg, 2)):
            if group.conclusion == "QUALIFIED" and len(group.matches) != count:
                raise ValueError("incorrect parlay leg count")
        for category, group in zip(CATEGORIES, (self.two_leg, self.three_leg, self.total_goals, self.half_full, self.mixed_two_leg)):
            if group.conclusion == NO_BET and self.group_amounts[category] not in (None, 0):
                raise ValueError("NO-BET group cannot have a positive stake")
        return self
