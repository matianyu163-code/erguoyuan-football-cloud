"""Reject missing/reordered display keys instead of silently repairing them."""

from erguoyuan_football.output.v51.schema import (
    CATEGORIES,
    MATCH_FIELDS,
    PARLAY_FIELDS,
    SUMMARY_FIELDS,
    TOP_FIELDS,
    V51Output,
)


def require_keys(value, expected):
    if not isinstance(value, dict) or tuple(value) != expected:
        raise ValueError(f"V5.1 field order/structure mismatch: expected {expected}")


def validate_output(value: dict) -> V51Output:
    require_keys(value, TOP_FIELDS)
    for row in value["今日比赛筛选总表"]:
        require_keys(row, SUMMARY_FIELDS)
    for category in CATEGORIES:
        group = value[category]
        require_keys(group, ("结论", "比赛明细", "串关"))
        for row in group["比赛明细"]:
            require_keys(row, MATCH_FIELDS)
        require_keys(group["串关"], PARLAY_FIELDS)
    require_keys(value["各组金额"], CATEGORIES)
    return V51Output.model_validate(value)
