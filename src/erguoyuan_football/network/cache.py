"""Small in-memory response cache with explicit freshness and PIT checks."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

from erguoyuan_football.contracts.common import utc
from erguoyuan_football.network.schemas import CachedPayload, CacheStatus


def canonical_hash(value: Any) -> str:
    """Hash canonical JSON request data without storing credentials."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


class NetworkCache:
    """Process-local cache; persistent cache adapters can implement this interface later."""

    def __init__(self) -> None:
        self._items: dict[str, CachedPayload] = {}

    @staticmethod
    def key(source_id: str, endpoint_id: str, params_hash: str, schema_version: str) -> str:
        return canonical_hash((source_id, endpoint_id, params_hash, schema_version))

    def put(self, item: CachedPayload) -> None:
        self._items[item.cache_key] = item

    def get(self, key: str, prediction_time: datetime) -> tuple[CachedPayload, CacheStatus] | None:
        item = self._items.get(key)
        if item is None or item.source_timestamp is None or item.source_timestamp > utc(prediction_time):
            return None
        current = utc(prediction_time)
        status = CacheStatus.FRESH if current < item.expires_at else CacheStatus.EXPIRED
        return item, status

