"""Auditable categories of non-direct historical prior evidence."""

from enum import StrEnum


class PriorType(StrEnum):
    """Prior rows cannot be counted as target-team direct observations."""

    COMPETITION_PRIOR = "COMPETITION_PRIOR"
    AGE_GROUP_PRIOR = "AGE_GROUP_PRIOR"
    FEDERATION_PRIOR = "FEDERATION_PRIOR"
    NATIONAL_PROGRAM_PRIOR = "NATIONAL_PROGRAM_PRIOR"
    GENDER_PRIOR = "GENDER_PRIOR"
    LEAGUE_LEVEL_PRIOR = "LEAGUE_LEVEL_PRIOR"
