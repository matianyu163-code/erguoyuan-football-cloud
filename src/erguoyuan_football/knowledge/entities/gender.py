"""Football team gender category."""

from enum import StrEnum


class Gender(StrEnum):
    """Do not infer MEN from an unspecified name."""

    MEN = "MEN"
    WOMEN = "WOMEN"
    MIXED = "MIXED"
    UNKNOWN = "UNKNOWN"
