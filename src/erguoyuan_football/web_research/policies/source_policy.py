"""Explicit source quality tiers; unlisted source types are rejected."""

from __future__ import annotations

SOURCE_TIER: dict[str, int] = {"OFFICIAL": 3, "STRUCTURED": 2, "MEDIA": 1}


def tier_for(source_type: str) -> int:
    """Return a registered tier; Tier 0, blogs and forums are unavailable."""
    try:
        return SOURCE_TIER[source_type]
    except KeyError as error:
        raise ValueError("SOURCE_TYPE_NOT_ALLOWED") from error
