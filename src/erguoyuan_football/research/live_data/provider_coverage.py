"""Coverage reflects validated live evidence, never capability declarations alone."""

from __future__ import annotations

from dataclasses import dataclass

COVERAGE_TYPES = ("FIXTURE", "RESULTS", "STATS", "STANDINGS", "ELO", "XG",
                  "ODDS", "INJURY", "LINEUP", "NEWS")


@dataclass(frozen=True)
class ProviderCoverageReport:
    """Per-source researched coverage with verified/unsupported separation."""

    provider_id: str
    statuses: dict[str, str]


def build_coverage(provider_id: str, *, declared: frozenset[str],
                   verified: frozenset[str]) -> ProviderCoverageReport:
    """A declared capability stays UNVERIFIED until actual typed data exists."""
    return ProviderCoverageReport(provider_id, {
        kind: ("VERIFIED" if kind in verified else
               "UNVERIFIED" if kind in declared else "UNSUPPORTED")
        for kind in COVERAGE_TYPES
    })
