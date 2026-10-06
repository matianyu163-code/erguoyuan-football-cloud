"""Strict UTC handling for external research timestamps."""

from __future__ import annotations

from datetime import UTC, datetime


def parse_utc(value: str) -> datetime:
    """Parse an aware ISO-8601 timestamp and normalize it to UTC."""
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("INVALID_UTC_TIMESTAMP") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("UTC_TIMESTAMP_REQUIRED")
    return parsed.astimezone(UTC)


def utc_iso(value: datetime) -> str:
    """Serialize an aware datetime in UTC, without fabricating its event time."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("UTC_TIMESTAMP_REQUIRED")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
