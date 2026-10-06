"""Explicit distinction between official lottery and general research matches."""

from enum import StrEnum


class MatchUniverse(StrEnum):
    """The intended operational universe, independent of provider coverage."""

    JC_PRODUCTION = "JC_PRODUCTION"
    GLOBAL_RESEARCH = "GLOBAL_RESEARCH"
