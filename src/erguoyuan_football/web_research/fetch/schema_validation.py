"""Strict typed payload checks before a provider response becomes evidence."""

from __future__ import annotations

import math
from typing import Any

from erguoyuan_football.web_research.fetch.fixture_identity import (
    FixtureExpectation,
    FixtureVerification,
    verify_fixture,
)
from erguoyuan_football.web_research.providers.provider_result import ProviderResult
from erguoyuan_football.web_research.search.search_task import SearchTask
from erguoyuan_football.web_research.time_utils import parse_utc


def _nonempty(data: dict[str, Any], *keys: str) -> bool:
    return all(isinstance(data.get(key), str) and bool(data[key].strip()) for key in keys)


def validate_provider_payload(
    task: SearchTask,
    result: ProviderResult,
    *,
    fixture_expectation: FixtureExpectation | None = None,
) -> FixtureVerification | None:
    """Reject ambiguous fixtures and unqualified odds/statistics/news claims."""
    if not result.success or result.task_type != task.task_type:
        raise ValueError("INVALID_PROVIDER_RESPONSE:TASK_MISMATCH")
    data = result.data
    observed = data.get("observed_at")
    if (not isinstance(observed, str) or result.observed_time is None
            or parse_utc(observed) != parse_utc(result.observed_time)):
        raise ValueError("INVALID_PROVIDER_RESPONSE:OBSERVED_TIME_MISMATCH")
    if task.task_type == "FIXTURE":
        if fixture_expectation is None:
            raise ValueError("FIXTURE_EXPECTATION_REQUIRED")
        verification = verify_fixture(data, fixture_expectation)
        if verification.status != "VERIFIED":
            raise ValueError(verification.status)
        return verification
    if task.task_type == "ODDS":
        if not _nonempty(data, "market_type", "selection", "bookmaker", "observed_at"):
            raise ValueError("INVALID_PROVIDER_RESPONSE:ODDS_FIELDS")
        value = data.get("odds")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 1:
            raise ValueError("INVALID_PROVIDER_RESPONSE:ODDS_VALUE")
        parse_utc(data["observed_at"])
        return None
    if task.task_type == "TEAM_STATS":
        if not _nonempty(data, "team_id", "competition", "season", "window",
                         "metric", "observed_at"):
            raise ValueError("INVALID_PROVIDER_RESPONSE:STATS_FIELDS")
        value = data.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("INVALID_PROVIDER_RESPONSE:STATS_VALUE")
        parse_utc(data["observed_at"])
        return None
    if task.task_type in {"NEWS", "INJURY"}:
        if not _nonempty(data, "text", "claim_type", "observed_at"):
            raise ValueError("INVALID_PROVIDER_RESPONSE:NEWS_FIELDS")
        if data["claim_type"] not in {"FACT", "REPORT"}:
            raise ValueError("INVALID_PROVIDER_RESPONSE:NEWS_CLAIM_TYPE")
        if result.source_tier == 1 and data["claim_type"] != "REPORT":
            raise ValueError("MEDIA_CANNOT_ASSERT_OFFICIAL_FACT")
        parse_utc(data["observed_at"])
        return None
    if task.task_type == "LINEUP":
        if (result.source_tier != 3 or data.get("official") is not True
                or not isinstance(data.get("players"), list)
                or not _nonempty(data, "observed_at")):
            raise ValueError("OFFICIAL_LINEUP_UNAVAILABLE")
        parse_utc(data["observed_at"])
        return None
    raise ValueError("PROVIDER_SCHEMA_NOT_IMPLEMENTED")
