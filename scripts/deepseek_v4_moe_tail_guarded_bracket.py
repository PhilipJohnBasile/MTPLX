#!/usr/bin/env python3
"""Run the MoE-tail bracket, then attest that the shared service is restored.

``run_guarded.py`` is deliberately the only process that owns the Quality
service lifecycle.  This wrapper only waits for that canonical child to exit
and performs read-only postflight checks; it never starts, stops, or repairs a
service itself.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import tempfile
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


VENV_PYTHON = Path("/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python")
RUN_GUARDED = Path("/Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py")
QUALITY_PLIST = Path("/Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist")
LOCK_PATH = Path("/tmp/mtplx-gpu-exclusive.lock")
ARMS = Path(__file__).with_name("deepseek_v4_moe_tail_arms.sh")
DEFAULT_BENCH_DIR = Path("/Users/davidtai/projects/OpenSourceWTF/bench/deepseek-v4")
QUALITY_MODEL = "mtplx-qwen36-27b-optimized-quality"
WIRED_LIMIT_MB = 114688
WRAPPER_ENV = "MTPLX_DSV4_MOE_TAIL_POSTFLIGHT_WRAPPER"


def _check_lock_free() -> dict[str, Any]:
    try:
        with LOCK_PATH.open("rb") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        return {"ok": True, "path": str(LOCK_PATH)}
    except OSError as error:
        return {"ok": False, "path": str(LOCK_PATH), "error": str(error)}


def _check_wired_limit() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "iogpu.wired_limit_mb"],
            check=False,
            capture_output=True,
            text=True,
        )
        value = completed.stdout.strip()
        if completed.returncode != 0:
            return {
                "ok": False,
                "error": completed.stderr.strip() or "sysctl failed",
                "exit_code": completed.returncode,
            }
        observed = int(value)
        return {"ok": observed == WIRED_LIMIT_MB, "value": observed}
    except (OSError, ValueError) as error:
        return {"ok": False, "error": str(error)}


def _request_json(path: str, *, payload: dict[str, Any] | None, timeout: float) -> Any:
    request = urllib.request.Request(
        f"http://127.0.0.1:8080{path}",
        data=None if payload is None else json.dumps(payload).encode(),
        headers={} if payload is None else {"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _check_quality_models() -> dict[str, Any]:
    try:
        payload = _request_json("/v1/models", payload=None, timeout=10)
        models = [entry["id"] for entry in payload["data"]]
        return {"ok": models == [QUALITY_MODEL], "models": models}
    except (KeyError, TypeError, ValueError, urllib.error.URLError, OSError) as error:
        return {"ok": False, "error": str(error)}


def _check_quality_ready_chat() -> dict[str, Any]:
    try:
        payload = _request_json(
            "/v1/chat/completions",
            payload={
                "model": QUALITY_MODEL,
                "messages": [{"role": "user", "content": "Say READY"}],
                "max_tokens": 8,
                "temperature": 0,
            },
            timeout=60,
        )
        choice = payload["choices"][0]
        content = choice["message"]["content"].strip()
        finish_reason = choice["finish_reason"]
        return {
            "ok": content == "READY" and finish_reason == "stop",
            "content": content,
            "finish_reason": finish_reason,
        }
    except (IndexError, KeyError, TypeError, ValueError, urllib.error.URLError, OSError) as error:
        return {"ok": False, "error": str(error)}


def collect_postflight() -> dict[str, dict[str, Any]]:
    """Read-only checks run after the guard process has already returned."""

    return {
        "lock_free": _check_lock_free(),
        "wired_limit_mb": _check_wired_limit(),
        "quality_models": _check_quality_models(),
        "quality_ready_chat": _check_quality_ready_chat(),
    }


def _write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as receipt_file:
            receipt_file.write(encoded)
            receipt_file.flush()
            os.fsync(receipt_file.fileno())
        os.replace(temporary, path)
    except BaseException:
        os.unlink(temporary)
        raise


def _command(tag: str) -> list[str]:
    return [
        str(VENV_PYTHON),
        str(RUN_GUARDED),
        "--plist",
        str(QUALITY_PLIST),
        "--timeout-seconds",
        "300",
        "--lock-timeout-seconds",
        "3600",
        "--child-timeout-seconds",
        "3600",
        "--",
        "/bin/zsh",
        str(ARMS),
        tag,
    ]


def run(tag: str, *, bench_dir: Path = DEFAULT_BENCH_DIR) -> int:
    """Run the sole service owner, then persist postflight on every outcome."""

    try:
        child_exit_code = subprocess.run(
            _command(tag),
            check=False,
            env={**os.environ, WRAPPER_ENV: "1"},
        ).returncode
    except OSError as error:
        child_exit_code = 127
        child_error: str | None = str(error)
    else:
        child_error = None
    postflight = collect_postflight()
    postflight_ok = all(result.get("ok") is True for result in postflight.values())
    exit_code = child_exit_code if child_exit_code != 0 else (0 if postflight_ok else 1)
    receipt = {
        "schema_version": 1,
        "kind": "deepseek_v4_moe_tail_guarded_postflight",
        "tag": tag,
        "run_guarded_command": _command(tag),
        "child_exit_code": child_exit_code,
        "child_error": child_error,
        "postflight": postflight,
        "postflight_ok": postflight_ok,
        "exit_code": exit_code,
        "completed_utc": datetime.now(UTC).isoformat(),
    }
    _write_receipt(bench_dir / f"{tag}-postflight.json", receipt)
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "tag",
        nargs="?",
        default=f"moe-tail-k3-{datetime.now(UTC):%Y%m%dT%H%M%SZ}",
    )
    parser.add_argument("--bench-dir", type=Path, default=DEFAULT_BENCH_DIR)
    arguments = parser.parse_args()
    return run(arguments.tag, bench_dir=arguments.bench_dir)


if __name__ == "__main__":
    raise SystemExit(main())
