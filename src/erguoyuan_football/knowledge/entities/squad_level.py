"""Club squad distinction."""

from enum import StrEnum


class SquadLevel(StrEnum):
    """Reserve and youth teams are separate from first teams."""

    FIRST_TEAM = "FIRST_TEAM"
    B_TEAM = "B_TEAM"
    SECOND_TEAM = "SECOND_TEAM"
    RESERVE = "RESERVE"
    ACADEMY = "ACADEMY"
    YOUTH = "YOUTH"
    UNKNOWN = "UNKNOWN"
