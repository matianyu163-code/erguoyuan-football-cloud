"""Syntactic team hints only; parsing never creates a verified identity."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from erguoyuan_football.knowledge.entities.country_alias_registry import (
    CountryAliasRegistry,
)
from erguoyuan_football.knowledge.teams.alias_matcher import normalize_alias

_AGE = re.compile(r"(?i)(?:\b(?:U|Under)[ -]?(1[4-9]|2[0-3])\b|(?:U|Under)[ -]?(1[4-9]|2[0-3])(?=[\u3400-\u9fff]|$))")
_WOMEN = re.compile(r"(?i)(?:\b(?:women|ladies|feminin[ae])\b|女足|女子)")
_MEN = re.compile(r"(?i)(?:\bmen\b|男足|男子)")
_B_TEAM = re.compile(r"(?i)(?:\s+B\b|B队|\s+Castilla\b)")
_SECOND = re.compile(r"(?i)(?:\s+(?:II|2)\b|(?<=[\u3400-\u9fff])(?:II|2|二队)$)")
_RESERVE = re.compile(r"(?i)(?:\bReserves?\b|预备队)")
_ACADEMY = re.compile(r"(?i)(?:\bAcademy\b|青训)")
_YOUTH = re.compile(r"(?i)(?:\bYouth\b|青年队)")


@dataclass(frozen=True)
class ParsedTeamEntityHint:
    """Unverified dimensions extracted conservatively from one team name."""

    raw_name: str
    base_name: str
    country_hint: str | None
    entity_type_hint: str
    gender_hint: str
    age_group_hint: str
    squad_level_hint: str
    city_hint: str | None
    language_hint: str

    @property
    def normalized_base_name(self) -> str:
        """Exact-match key after structural suffixes have been removed."""
        return normalize_alias(self.base_name)

    @property
    def gender(self) -> str:
        return self.gender_hint

    @property
    def age_group(self) -> str:
        return self.age_group_hint

    @property
    def team_level(self) -> str:
        return self.squad_level_hint

    @property
    def team_type_hint(self) -> str:
        return self.entity_type_hint


class UniversalTeamNameParser:
    """Parse a global team name without assuming unspecified gender or country."""

    def __init__(self, countries: CountryAliasRegistry | None = None) -> None:
        self.countries = countries or CountryAliasRegistry()

    def parse(self, raw_name: str) -> ParsedTeamEntityHint:
        """Extract exact suffix/prefix qualifiers, leaving unknowns explicit."""
        name = " ".join(unicodedata.normalize("NFKC", raw_name).strip().split())
        if not name:
            raise ValueError("TEAM_PARSE_FAILED")
        age_match = _AGE.search(name)
        age = f"U{age_match.group(1) or age_match.group(2)}" if age_match else "SENIOR"
        base = _AGE.sub("", name)
        women, men = bool(_WOMEN.search(base)), bool(_MEN.search(base))
        if women and men:
            raise ValueError("TEAM_IDENTITY_CONFLICT:GENDER")
        gender = "WOMEN" if women else "MEN" if men else "UNKNOWN"
        base = _WOMEN.sub("", _MEN.sub("", base))
        squad = "FIRST_TEAM"
        for pattern, level in ((_B_TEAM, "B_TEAM"), (_SECOND, "SECOND_TEAM"),
                               (_RESERVE, "RESERVE"), (_ACADEMY, "ACADEMY"),
                               (_YOUTH, "YOUTH")):
            if pattern.search(base):
                squad = level
                base = pattern.sub("", base)
                break
        base = " ".join(base.strip(" -_队").split())
        if not base:
            raise ValueError("TEAM_PARSE_FAILED")
        country = self.countries.resolve(base)
        if age != "SENIOR" and country is None and squad == "FIRST_TEAM":
            squad = "YOUTH"
        return ParsedTeamEntityHint(
            raw_name=name, base_name=base,
            country_hint=country.iso3 if country else None,
            entity_type_hint="NATIONAL" if country else "CLUB",
            gender_hint=gender, age_group_hint=age,
            squad_level_hint=squad, city_hint=None,
            language_hint="ZH" if re.search(r"[\u3400-\u9fff]", name) else "EN",
        )
