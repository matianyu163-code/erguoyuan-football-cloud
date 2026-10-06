"""Competition hints are constraints, never independently verified fixtures."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedCompetitionHint:
    """Known competition qualifiers and unchanged raw text."""

    raw_hint: str
    federation: str | None = None
    gender: str | None = None
    age_group: str | None = None
    entity_type: str | None = None
    competition_type: str | None = None
    country: str | None = None
    league_level: int | None = None


class CompetitionHintParser:
    """Parse only explicitly recognizable competition markers."""

    def parse(self, text: str) -> ParsedCompetitionHint:
        """Preserve unknown small-league names instead of guessing region."""
        raw = text.strip()
        lowered = raw.casefold()
        if "女欧" in raw or ("women" in lowered and "uefa" in lowered):
            match = re.search(r"(?i)U[ -]?(1[4-9]|2[0-3])", raw)
            return ParsedCompetitionHint(raw, "UEFA", "WOMEN",
                                         f"U{match.group(1)}" if match else None, "NATIONAL")
        if raw in {"国际友谊赛", "International Friendly"}:
            return ParsedCompetitionHint(raw, competition_type="FRIENDLY")
        if raw == "日乙":
            return ParsedCompetitionHint(raw, country="JPN", entity_type="CLUB",
                                         league_level=2)
        return ParsedCompetitionHint(raw)


def split_competition_hint(text: str) -> tuple[str, ParsedCompetitionHint | None]:
    """Read optional `team vs team | competition` without changing MatchRequest."""
    if "|" not in text:
        return text.strip(), None
    match_text, hint_text = text.rsplit("|", maxsplit=1)
    return match_text.strip(), CompetitionHintParser().parse(hint_text)
