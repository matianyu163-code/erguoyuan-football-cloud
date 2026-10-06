"""Historical sample recency metadata; does not change model weights."""

from __future__ import annotations

from datetime import datetime


def latest_age_days(kickoffs: tuple[datetime, ...], cutoff: datetime) -> int | None:
    """Return nonnegative age of the latest eligible event, or None."""
    eligible = [event for event in kickoffs if event < cutoff]
    return (cutoff - max(eligible)).days if eligible else None
