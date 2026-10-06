"""Strict, immutable boundary objects. All persisted timestamps are UTC."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field


def utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timezone-aware timestamp required")
    return value.astimezone(UTC)


UTCTime = Annotated[datetime, AfterValidator(utc)]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
NonNegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Identifier = Annotated[str, Field(min_length=1, pattern=r"\S")]


def now() -> datetime:
    return datetime.now(UTC)


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Availability(StrEnum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"


class ExecutionStatus(StrEnum):
    SUCCESS = "SUCCESS"
    UNAVAILABLE = "UNAVAILABLE"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class ImplementationType(StrEnum):
    REAL_IMPLEMENTATION = "REAL_IMPLEMENTATION"
    LIKE_IMPLEMENTATION = "LIKE_IMPLEMENTATION"
    EXTERNAL = "EXTERNAL"
