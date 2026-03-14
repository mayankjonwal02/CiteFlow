from __future__ import annotations

import threading
from collections import defaultdict


class MetricsRegistry:
    def __init__(self):
        self._lock = threading.Lock()
        self._counters = defaultdict(int)

    def increment(self, key: str, value: int = 1):
        with self._lock:
            self._counters[key] += value

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counters)


metrics = MetricsRegistry()
