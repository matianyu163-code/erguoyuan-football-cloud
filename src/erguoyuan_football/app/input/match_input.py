"""INPUT_PARSER_V2 syntax boundary; identity is resolved separately."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from importlib.resources import files

from erguoyuan_football.match_source.match_source import MatchSourceType
from erguoyuan_football.match_source.match_source_classifier import (
    MatchSourceClassifier,
)

MATCH_SEPARATOR = "\x1f"
INPUT_PARSER_VERSION = "INPUT_PARSER_V2"

_CJK = "\u3400-\u9fff"
_PATTERNS = (
    re.compile(r"\s+VS\s+"),
    re.compile(r"VS"),
    re.compile(r"\s+vs\s+"),
    re.compile(r"vs"),
    re.compile(r"\s+V\s+"),
    re.compile(r"\s+v\s+"),
    re.compile(rf"(?<=[\w{_CJK}])V(?=[A-Z{_CJK}])"),
    re.compile(r"-(?!\d)"),
    re.compile(r"对"),
    re.compile(r"\s+对\s+"),
)
_EDGE_SYMBOLS = " \t\r\n,，。.;；:：|/\\"
_JC_CODE_ANY = re.compile(r"周\s*[一二三四五六日天末1-7]?\s*\d{3}(?!\d)")
_COMPETITION_PREFIXES: tuple[str, ...] = tuple(json.loads(
    files(__package__).joinpath("competition_prefixes.json").read_text(encoding="utf-8")))


@dataclass(frozen=True)
class ParsedMatchStructure:
    """Syntax-only fields; neither a team nor a fixture is verified here."""

    raw_input: str
    jc_code: str | None
    competition_raw: str | None
    home_raw: str | None
    away_raw: str | None


@dataclass(frozen=True)
class MatchRequest:
    """Unchanged Phase 12 fields; ``reason`` also exposes an error-code alias."""

    match_id: str | None
    home_team: str | None
    away_team: str | None
    competition: str | None
    date: date | None
    input_source: str
    validation_status: str
    reason: str | None = None
    raw_text: str | None = None
    match_source_type: MatchSourceType = MatchSourceType.AUTO_DISCOVERY
    jc_confirmed: bool = False

    @property
    def error_code(self) -> str | None:
        """Expose the existing reason as an error code without changing fields."""
        return self.reason


def normalize_team_name(name: str) -> str:
    """Trim excess edge symbols and CJK internal spacing; retain English word spaces."""
    value = " ".join(name.strip(_EDGE_SYMBOLS).split())
    value = re.sub(rf"(?<=[{_CJK}])\s+(?=[{_CJK}])", "", value)
    return value.strip(_EDGE_SYMBOLS)


class MatchInputParserV2:
    """Split text into a valid home/away request without guessing team identity."""

    version = INPUT_PARSER_VERSION

    def parse_structure(self, text: str) -> ParsedMatchStructure:
        """Separate declared ticket/competition metadata before team syntax."""
        cleaned = self.clean_text(text)
        jc = _JC_CODE_ANY.search(unicodedata.normalize("NFKC", text))
        code = "".join(jc.group().split()) if jc else None
        cleaned = MatchSourceClassifier().clean_match_text(cleaned)
        competition = None
        if "|" in cleaned:
            left, right = (part.strip() for part in cleaned.split("|", maxsplit=1))
            if left and right and any(pattern.search(right) for pattern in _PATTERNS):
                competition, cleaned = left, right
        for prefix in sorted(_COMPETITION_PREFIXES, key=len, reverse=True):
            if cleaned.casefold().startswith(prefix.casefold()):
                rest = cleaned[len(prefix):]
                if rest and (rest[0].isspace() or re.match(rf"[{_CJK}]", rest[0])):
                    competition, cleaned = prefix, rest.strip()
                    break
        for pattern in _PATTERNS:
            separator = pattern.search(cleaned)
            if separator is not None:
                home = normalize_team_name(cleaned[:separator.start()])
                away = normalize_team_name(cleaned[separator.end():])
                return ParsedMatchStructure(text, code, competition,
                                            home or None, away or None)
        return ParsedMatchStructure(text, code, competition, None, None)

    def clean_text(self, text: str) -> str:
        """Trim and collapse whitespace while preserving English team word boundaries."""
        collapsed = " ".join(unicodedata.normalize("NFKC", text).strip().split())
        collapsed = re.sub(r"(?i)\b(vs|v)\.(?=\s)", r"\1", collapsed)
        collapsed = collapsed.replace("对阵", "对")
        return re.sub(r"\s*(VS|vs|对|-)\s*", r"\1", collapsed)

    def parse(self, text: str) -> MatchRequest:
        """Return VALID syntax or MATCH_SYNTAX_INVALID with no invented team."""
        structure = self.parse_structure(text)
        if structure.home_raw and structure.away_raw:
            return MatchRequest(None, structure.home_raw, structure.away_raw,
                                structure.competition_raw, None, "USER_TEXT", "VALID",
                                raw_text=text)
        return MatchRequest(None, None, None, None, None, "USER_TEXT", "INVALID",
                            "MATCH_SYNTAX_INVALID", text)
