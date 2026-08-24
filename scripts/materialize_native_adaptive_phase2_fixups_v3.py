#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(value: str, old: str, new: str, *, label: str) -> str:
    count = value.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return value.replace(old, new, 1)


def main() -> None:
    path = ROOT / "mtplx/otlp_export.py"
    value = path.read_text(encoding="utf-8")
    value = replace_once(
        value,
        "        self._failed_exports = 0\n        self._last_error_type: str | None = None\n",
        "        self._failed_exports = 0\n        self._active_exports = 0\n        self._last_error_type: str | None = None\n",
        label="OTLP active export counter",
    )
    marker = "    def _send_tracked(self, spans: Sequence[OTLPSpan]) -> None:\n"
    insertion_point = "    def _drain(self, limit: int) -> list[OTLPSpan]:\n"
    if marker in value:
        raise RuntimeError("OTLP tracked sender already exists")
    tracked = '''    def _send_tracked(self, spans: Sequence[OTLPSpan]) -> None:
        with self._lock:
            self._active_exports += 1
        try:
            self._send(spans)
        finally:
            with self._lock:
                self._active_exports = max(0, self._active_exports - 1)

'''
    value = replace_once(
        value,
        insertion_point,
        tracked + insertion_point,
        label="OTLP tracked sender insertion",
    )
    value = value.replace("                self._send(rows)\n", "                self._send_tracked(rows)\n")
    old_flush = '''    def flush(self, *, timeout_s: float = 5.0) -> bool:
        deadline = time.monotonic() + max(0.0, timeout_s)
        while not self._queue.empty() and time.monotonic() < deadline:
            rows = self._drain(self.config.batch_size)
            if rows:
                self._send(rows)
        return self._queue.empty()
'''
    new_flush = '''    def flush(self, *, timeout_s: float = 5.0) -> bool:
        deadline = time.monotonic() + max(0.0, timeout_s)
        self._wake.set()
        while time.monotonic() < deadline:
            rows = self._drain(self.config.batch_size)
            if rows:
                self._send_tracked(rows)
                continue
            with self._lock:
                active = self._active_exports
            if self._queue.empty() and active == 0:
                return True
            time.sleep(0.005)
        with self._lock:
            active = self._active_exports
        return self._queue.empty() and active == 0
'''
    value = replace_once(value, old_flush, new_flush, label="OTLP flush")
    value = replace_once(
        value,
        '                "failed_exports": self._failed_exports,\n',
        '                "failed_exports": self._failed_exports,\n                "active_exports": self._active_exports,\n',
        label="OTLP snapshot active exports",
    )
    path.write_text(value, encoding="utf-8")


if __name__ == "__main__":
    main()
