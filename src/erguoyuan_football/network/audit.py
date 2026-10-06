"""In-memory network audit sink; records metadata, never request credentials."""

from __future__ import annotations

from erguoyuan_football.network.schemas import NetworkAuditRecord


class NetworkAudit:
    """Append-only process-local audit collector."""

    def __init__(self) -> None:
        self._records: list[NetworkAuditRecord] = []

    def record(self, value: NetworkAuditRecord) -> None:
        self._records.append(value)

    def records(self) -> tuple[NetworkAuditRecord, ...]:
        return tuple(self._records)
