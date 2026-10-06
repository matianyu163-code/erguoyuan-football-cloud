"""Pure, auditable search-task contract."""

from __future__ import annotations

from dataclasses import dataclass

from erguoyuan_football.web_research.policies.source_policy import SOURCE_TIER

TASK_TYPES = frozenset({"FIXTURE", "TEAM_STATS", "INJURY", "LINEUP", "ODDS", "NEWS"})


@dataclass(frozen=True)
class SearchTask:
    """A query plan; constructing it performs no network access."""

    task_id: str
    task_type: str
    query: str
    required_sources: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.task_id or self.task_type not in TASK_TYPES or not self.query.strip():
            raise ValueError("INVALID_SEARCH_TASK")
        if not self.required_sources or any(item not in SOURCE_TIER
                                            for item in self.required_sources):
            raise ValueError("INVALID_REQUIRED_SOURCES")
