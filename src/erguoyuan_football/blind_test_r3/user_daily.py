"""Strict format validation for user-authoritative daily JC inputs."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class OddsTriple(BaseModel):
    model_config = ConfigDict(extra="forbid")
    home: float
    draw: float
    away: float

    @field_validator("home", "draw", "away")
    @classmethod
    def valid_odds(cls, value: float) -> float:
        if isinstance(value, bool) or not math.isfinite(value) or value <= 1:
            raise ValueError("JC_ODDS_MUST_BE_FINITE_AND_GREATER_THAN_ONE")
        return value

    def no_vig(self) -> dict[str, float]:
        raw = {"HOME": 1 / self.home, "DRAW": 1 / self.draw, "AWAY": 1 / self.away}
        total = math.fsum(raw.values())
        return {key: value / total for key, value in raw.items()}


class HandicapOdds(OddsTriple):
    handicap: int

    @field_validator("handicap")
    @classmethod
    def integer_handicap(cls, value: int) -> int:
        if isinstance(value, bool):
            raise TypeError("JC_HANDICAP_MUST_BE_INTEGER")
        return value


class UserFixture(BaseModel):
    model_config = ConfigDict(extra="forbid")
    jc_match_number: str = Field(min_length=1)
    competition: str = Field(min_length=1)
    home_team: str = Field(min_length=1)
    away_team: str = Field(min_length=1)
    kickoff: datetime
    neutral_venue: bool | None = None
    spf: OddsTriple
    rqspf: HandicapOdds
    screenshot_path: str | None = None
    metadata_sources: dict[str, str] = Field(default_factory=dict)

    @field_validator("kickoff")
    @classmethod
    def aware_kickoff(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("KICKOFF_TIMEZONE_REQUIRED")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def distinct_teams(self) -> UserFixture:
        if self.home_team.strip() == self.away_team.strip():
            raise ValueError("HOME_AWAY_MUST_DIFFER")
        return self

    def fixture_id(self, slate_date: date) -> str:
        identity = (f"{slate_date.isoformat()}|{self.jc_match_number.strip()}|"
                    f"{self.home_team.strip()}|{self.away_team.strip()}|"
                    f"{self.kickoff.isoformat()}")
        return "YYF-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


class UserDailySlate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    slate_date: date
    source: str
    fixtures: list[UserFixture] = Field(min_length=1)

    @field_validator("source")
    @classmethod
    def authoritative_source(cls, value: str) -> str:
        if value != "USER_AUTHORITATIVE":
            raise ValueError("USER_AUTHORITATIVE_SOURCE_REQUIRED")
        return value

    @model_validator(mode="after")
    def unique_codes(self) -> UserDailySlate:
        codes = [item.jc_match_number for item in self.fixtures]
        if len(codes) != len(set(codes)):
            raise ValueError("DUPLICATE_JC_MATCH_NUMBER")
        return self


def load_slate(path: Path) -> tuple[UserDailySlate, str]:
    raw = path.read_bytes()
    data: Any = json.loads(raw)
    return UserDailySlate.model_validate(data), hashlib.sha256(raw).hexdigest()


def screenshot_hash(root: Path, relative: str | None) -> str | None:
    if relative is None:
        return None
    if Path(relative).is_absolute() or re.match(r"^[A-Za-z]:", relative):
        raise ValueError("SCREENSHOT_PATH_MUST_BE_RELATIVE")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("SCREENSHOT_PATH_OUTSIDE_PROJECT")
    return hashlib.sha256(path.read_bytes()).hexdigest()
