"""Deterministic data-fetch ordering for lottery and research match universes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from erguoyuan_football.research.match_universe import MatchUniverse


class DataKind(StrEnum):
    """Fetchable match data families and JC precedence within each match."""

    FIXTURE = "FIXTURE"
    ODDS = "ODDS"
    RECENT_FORM = "RECENT_FORM"
    LINEUP = "LINEUP"
    INJURY = "INJURY"
    HISTORICAL = "HISTORICAL"


_KIND_PRIORITY = {
    DataKind.FIXTURE: 0,
    DataKind.ODDS: 1,
    DataKind.RECENT_FORM: 2,
    DataKind.LINEUP: 3,
    DataKind.INJURY: 4,
    DataKind.HISTORICAL: 5,
}


@dataclass(frozen=True)
class DataFetchTask:
    """A request queued for a bounded fetch worker."""

    match_id: str
    universe: MatchUniverse
    kind: DataKind
    sequence: int


class DataPriorityScheduler:
    """Queue JC production tasks ahead of research without changing data truth."""

    def order(self, tasks: tuple[DataFetchTask, ...]) -> tuple[DataFetchTask, ...]:
        """Stable order: JC universe, declared data priority, then input sequence."""
        if any(not task.match_id or task.sequence < 0 for task in tasks):
            raise ValueError("INVALID_DATA_FETCH_TASK")
        universe_priority = {
            MatchUniverse.JC_PRODUCTION: 0,
            MatchUniverse.GLOBAL_RESEARCH: 1,
        }
        return tuple(sorted(tasks, key=lambda task: (
            universe_priority[task.universe],
            _KIND_PRIORITY[task.kind] if task.universe == MatchUniverse.JC_PRODUCTION else 0,
            task.sequence,
        )))
