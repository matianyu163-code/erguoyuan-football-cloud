"""Thread-safe per-source token pacing."""

from __future__ import annotations

import threading
import time

from erguoyuan_football.network.config import RateLimitPolicy


class RateLimiter:
    """A small token bucket that caps burst and average request rate."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: dict[str, tuple[float, float]] = {}
        self._minute_events: dict[str, list[float]] = {}

    def acquire(self, source_id: str, policy: RateLimitPolicy) -> None:
        """Wait only as long as required to stay under the source policy."""
        while True:
            with self._lock:
                current = time.monotonic()
                tokens, last = self._state.get(source_id, (float(policy.burst), current))
                tokens = min(float(policy.burst), tokens + (current - last) * policy.requests_per_second)
                events = [event for event in self._minute_events.get(source_id, []) if current - event < 60]
                self._minute_events[source_id] = events
                second_delay = max(0.0, (1.0 - tokens) / policy.requests_per_second)
                minute_delay = (max(0.0, 60.0 - (current - events[0]))
                                if len(events) >= policy.requests_per_minute else 0.0)
                delay = max(second_delay, minute_delay)
                if delay == 0:
                    self._state[source_id] = (tokens - 1, current)
                    events.append(current)
                    self._minute_events[source_id] = events
                    return
                self._state[source_id] = (tokens, current)
            time.sleep(delay)
