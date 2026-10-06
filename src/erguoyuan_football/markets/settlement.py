"""Asian handicap and totals settlement, including push and split quarter lines."""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum

from erguoyuan_football.contracts.common import Contract, Probability
from erguoyuan_football.markets.normalizer import quarter_units
from erguoyuan_football.markets.schemas import Selection
from erguoyuan_football.models.score_matrix import ScoreMatrix


class SettlementOutcome(StrEnum):
    WIN = "WIN"
    HALF_WIN = "HALF_WIN"
    PUSH = "PUSH"
    HALF_LOSS = "HALF_LOSS"
    LOSS = "LOSS"


class SettlementFractions(Contract):
    """Economic decomposition across split lines; fractions sum to the stake."""

    win_fraction: Probability
    push_fraction: Probability
    loss_fraction: Probability
    outcome: SettlementOutcome


class SettlementProbabilities(Contract):
    win: Probability
    half_win: Probability
    push: Probability
    half_loss: Probability
    loss: Probability
    expected_win_equivalent: Probability
    expected_loss_equivalent: Probability


def _single_result(adjusted: Decimal) -> tuple[float, float, float]:
    if adjusted > 0:
        return 1.0, 0.0, 0.0
    if adjusted < 0:
        return 0.0, 0.0, 1.0
    return 0.0, 1.0, 0.0


def _split_quarters(units: int) -> tuple[Decimal, ...]:
    line = Decimal(units) / Decimal(4)
    if units % 2 == 0:
        return (line,)
    lower_half_units = units // 2
    return (Decimal(lower_half_units) / Decimal(2), Decimal(lower_half_units + 1) / Decimal(2))


def _combine(parts: tuple[tuple[float, float, float], ...]) -> SettlementFractions:
    win = sum(part[0] for part in parts) / len(parts)
    push = sum(part[1] for part in parts) / len(parts)
    loss = sum(part[2] for part in parts) / len(parts)
    if win > 0 and loss > 0 and abs(win - loss) < 1e-12:
        outcome = SettlementOutcome.PUSH
    elif win == 1:
        outcome = SettlementOutcome.WIN
    elif loss == 1:
        outcome = SettlementOutcome.LOSS
    elif win > 0:
        outcome = SettlementOutcome.HALF_WIN
    elif loss > 0:
        outcome = SettlementOutcome.HALF_LOSS
    else:
        outcome = SettlementOutcome.PUSH
    return SettlementFractions(win_fraction=win, push_fraction=push, loss_fraction=loss, outcome=outcome)


class AsianSettlementEngine:
    """Settle a home-view Asian handicap without collapsing push outcomes."""

    @staticmethod
    def settle(home_goals: int, away_goals: int, line: Decimal | str | float,
               selection: Selection) -> SettlementFractions:
        if home_goals < 0 or away_goals < 0 or selection not in {Selection.HOME, Selection.AWAY}:
            raise ValueError("invalid score or handicap selection")
        units = quarter_units(line)
        components = _split_quarters(units)
        margin = Decimal(home_goals - away_goals)
        parts = tuple(_single_result(margin + handicap if selection == Selection.HOME else -(margin + handicap))
                      for handicap in components)
        return _combine(parts)

    @staticmethod
    def probabilities(matrix: ScoreMatrix, line: Decimal | str | float,
                      selection: Selection) -> SettlementProbabilities:
        return _probabilities(matrix, lambda home, away: AsianSettlementEngine.settle(
            home, away, line, selection))


class TotalSettlementEngine:
    """Settle Over/Under totals with whole, half and quarter line semantics."""

    @staticmethod
    def settle(home_goals: int, away_goals: int, line: Decimal | str | float,
               selection: Selection) -> SettlementFractions:
        if home_goals < 0 or away_goals < 0 or selection not in {Selection.OVER, Selection.UNDER}:
            raise ValueError("invalid score or total selection")
        units = quarter_units(line)
        components = _split_quarters(units)
        total = Decimal(home_goals + away_goals)
        parts = tuple(_single_result(total - threshold if selection == Selection.OVER else threshold - total)
                      for threshold in components)
        return _combine(parts)

    @staticmethod
    def probabilities(matrix: ScoreMatrix, line: Decimal | str | float,
                      selection: Selection) -> SettlementProbabilities:
        return _probabilities(matrix, lambda home, away: TotalSettlementEngine.settle(
            home, away, line, selection))


def _probabilities(matrix: ScoreMatrix, settle) -> SettlementProbabilities:
    outcomes = {key: 0.0 for key in SettlementOutcome}
    for home, row in enumerate(matrix.values):
        for away, probability in enumerate(row):
            outcomes[settle(home, away).outcome] += probability
    win_equivalent = outcomes[SettlementOutcome.WIN] + 0.5 * outcomes[SettlementOutcome.HALF_WIN]
    loss_equivalent = outcomes[SettlementOutcome.LOSS] + 0.5 * outcomes[SettlementOutcome.HALF_LOSS]
    return SettlementProbabilities(
        win=outcomes[SettlementOutcome.WIN], half_win=outcomes[SettlementOutcome.HALF_WIN],
        push=outcomes[SettlementOutcome.PUSH], half_loss=outcomes[SettlementOutcome.HALF_LOSS],
        loss=outcomes[SettlementOutcome.LOSS], expected_win_equivalent=win_equivalent,
        expected_loss_equivalent=loss_equivalent,
    )
