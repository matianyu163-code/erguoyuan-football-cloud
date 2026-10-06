"""Parse user-declared China Sporttery fixture metadata without web inference."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from erguoyuan_football.app.input.match_input import MatchInputParserV2
from erguoyuan_football.match_source.match_source_classifier import (
    MatchSourceClassifier,
)

_LOCAL_TIME = re.compile(r"(?<!\d)(20\d{2}-\d{2}-\d{2})[ T](\d{2}:\d{2})(?!\d)")
_JC_CODE = re.compile(r"周[一二三四五六日天末1-7]?\d{3}(?!\d)")
_KNOWN_COMPETITIONS = {"欧国联": "UEFA Nations League",
                       "UEFA Nations League": "UEFA Nations League"}


@dataclass(frozen=True)
class JCFixtureMetadata:
    """A user assertion; no official fixture or historical result is implied."""

    jc_match_code: str | None
    competition_original: str | None
    competition_canonical: str | None
    home_team: str | None
    away_team: str | None
    kickoff_original: str | None
    source_timezone: str
    kickoff_utc: datetime | None
    missing_fields: tuple[str, ...]


class JCMetadataParser:
    """Parse a single JC form or compact text with an explicit local time."""

    def parse(self, raw_text: str, *, jc_match_code: str | None = None,
              competition: str | None = None, home_team: str | None = None,
              away_team: str | None = None, kickoff_local: str | None = None,
              timezone: str = "Asia/Shanghai") -> JCFixtureMetadata:
        """Preserve missing/invalid fields; never search for or guess a fixture."""
        found_time = _LOCAL_TIME.search(raw_text)
        local = kickoff_local.strip() if kickoff_local else (
            found_time.group() if found_time else None
        )
        match_text = _LOCAL_TIME.sub(" ", raw_text)
        classifier = MatchSourceClassifier()
        parser = MatchInputParserV2()
        structure = parser.parse_structure(classifier.clean_match_text(match_text))
        code = jc_match_code.strip() if jc_match_code else None
        if not code:
            found_code = _JC_CODE.search(raw_text)
            code = found_code.group() if found_code else None
        original_competition = (competition or structure.competition_raw or "").strip() or None
        canonical = (_KNOWN_COMPETITIONS.get(original_competition, original_competition)
                     if original_competition else None)
        home = (home_team or structure.home_raw or "").strip() or None
        away = (away_team or structure.away_raw or "").strip() or None
        missing = [field for field, value in (
            ("competition", original_competition), ("home_team", home),
            ("away_team", away), ("kickoff_local", local),
        ) if not value]
        kickoff_utc = None
        if local:
            try:
                zone = ZoneInfo(timezone)
                day, clock = local.split(" ")
                naive = datetime.combine(date.fromisoformat(day), time.fromisoformat(clock))
                aware = datetime.combine(naive.date(), naive.time(), tzinfo=zone)
                if aware.astimezone(UTC).astimezone(zone).replace(tzinfo=None) != naive:
                    raise ValueError("NONEXISTENT_LOCAL_TIME")
                kickoff_utc = aware.astimezone(UTC)
            except (ValueError, ZoneInfoNotFoundError):
                missing.append("kickoff_local_or_timezone_invalid")
        return JCFixtureMetadata(code, original_competition, canonical, home, away,
                                 local, timezone, kickoff_utc, tuple(missing))
