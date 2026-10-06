"""Source categories for user-provided and discovered match requests."""

from enum import StrEnum


class MatchSourceType(StrEnum):
    """How a fixture entered the system; distinct from fixture verification."""

    USER_JC_CONFIRMED = "USER_JC_CONFIRMED"
    AUTO_DISCOVERY = "AUTO_DISCOVERY"
    RESEARCH_TEST = "RESEARCH_TEST"
