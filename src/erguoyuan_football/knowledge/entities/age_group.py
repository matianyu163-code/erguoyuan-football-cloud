"""Exact youth age categories."""

from enum import StrEnum


class AgeGroup(StrEnum):
    """Adjacent youth levels never represent direct observations."""

    SENIOR = "SENIOR"
    U23 = "U23"
    U22 = "U22"
    U21 = "U21"
    U20 = "U20"
    U19 = "U19"
    U18 = "U18"
    U17 = "U17"
    U16 = "U16"
    U15 = "U15"
    U14 = "U14"
    UNKNOWN = "UNKNOWN"
