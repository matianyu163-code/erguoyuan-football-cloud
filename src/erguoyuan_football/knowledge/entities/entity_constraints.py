"""Hard dimension constraints precede any candidate scoring."""

from __future__ import annotations

from erguoyuan_football.knowledge.entities.team_entity_parser import (
    ParsedTeamEntityHint,
)


def hard_conflicts(hint: ParsedTeamEntityHint, *, country: str | None,
                   gender: str, age_group: str, entity_type: str,
                   squad_level: str) -> tuple[str, ...]:
    """Reject known disagreements; unknown candidate dimensions cannot verify hints."""
    checks = (
        (hint.country_hint, country, "COUNTRY_CONFLICT"),
        (hint.gender_hint, gender, "GENDER_CONFLICT"),
        (hint.age_group_hint, age_group, "AGE_GROUP_CONFLICT"),
        (hint.entity_type_hint, entity_type, "ENTITY_TYPE_CONFLICT"),
        (hint.squad_level_hint, squad_level, "SQUAD_LEVEL_CONFLICT"),
    )
    return tuple(code for expected, actual, code in checks
                 if expected not in {None, "UNKNOWN"} and actual not in {None, "UNKNOWN"}
                 and expected != actual)
