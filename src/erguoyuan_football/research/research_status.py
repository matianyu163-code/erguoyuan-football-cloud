"""Research completeness state, distinct from model execution status."""

from __future__ import annotations

from enum import StrEnum


class ResearchStatus(StrEnum):
    """Whether verified identities and explicitly linked evidence are present."""

    FOUND = "FOUND"
    PARTIAL = "PARTIAL"
    MISSING = "MISSING"
    FAILED = "FAILED"
