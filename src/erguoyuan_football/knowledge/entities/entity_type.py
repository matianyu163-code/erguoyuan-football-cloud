"""Football entity kind."""

from enum import StrEnum


class EntityType(StrEnum):
    """Entity categories; unknown remains explicit."""

    CLUB = "CLUB"
    NATIONAL = "NATIONAL"
    REGIONAL = "REGIONAL"
    UNIVERSITY = "UNIVERSITY"
    ACADEMY = "ACADEMY"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"
