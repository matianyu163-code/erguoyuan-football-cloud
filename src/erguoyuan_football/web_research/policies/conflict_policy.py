"""Keep contradictory sourced values instead of silently overwriting them."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SourcedValue:
    """One comparable fact with registry-derived source tier."""

    value: Any
    source_id: str
    tier: int


@dataclass(frozen=True)
class ConflictRecord:
    """An explicit contradiction between two sources."""

    first: SourcedValue
    second: SourcedValue
    status: str


@dataclass(frozen=True)
class ConflictDecision:
    """Selection with retained conflict evidence; equal-tier ties stay open."""

    selected: SourcedValue | None
    conflicts: tuple[ConflictRecord, ...]
    status: str


def resolve_conflicts(values: tuple[SourcedValue, ...]) -> ConflictDecision:
    """Prefer the highest tier only across tiers; never break equal-tier ties."""
    if not values:
        return ConflictDecision(None, (), "MISSING")
    if any(item.tier not in {1, 2, 3} for item in values):
        raise ValueError("INVALID_SOURCE_TIER")
    conflicts = tuple(ConflictRecord(left, right, "SOURCE_CONFLICT")
                      for index, left in enumerate(values)
                      for right in values[index + 1:] if left.value != right.value)
    highest = max(item.tier for item in values)
    top = [item for item in values if item.tier == highest]
    if any(item.value != top[0].value for item in top[1:]):
        return ConflictDecision(None, conflicts, "CONFLICT_UNRESOLVED")
    return ConflictDecision(top[0], conflicts,
                            "SOURCE_CONFLICT" if conflicts else "FOUND")
