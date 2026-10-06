"""Permanent CORE match-result class mapping."""

from __future__ import annotations

from enum import IntEnum

from erguoyuan_football.data.schemas import MatchResult


class ResultClass(IntEnum):
    HOME = 0
    DRAW = 1
    AWAY = 2


CLASS_MAPPING = {item.name: int(item) for item in ResultClass}
CLASS_ORDER = (0, 1, 2)


def label_result(result: MatchResult) -> ResultClass:
    """Derive one observed outcome only from an available final score."""
    if result.home_goals > result.away_goals:
        return ResultClass.HOME
    if result.home_goals < result.away_goals:
        return ResultClass.AWAY
    return ResultClass.DRAW
