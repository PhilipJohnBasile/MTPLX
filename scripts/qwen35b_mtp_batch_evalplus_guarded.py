#!/usr/bin/env python3
"""Run the fixed-width Qwen B8 EvalPlus generation under the MLX guard."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Any

if __package__:
    from scripts.qwen35b_mtp_batch_numerics_attribution import (
        _QWEN_ROUTE_ENV,
        _assert_no_other_model_runner,
        _validate_qwen_model,
        _verify_parent_guard_attestation,
    )
    from scripts.qwen35b_mtp_batch_numerics_guarded import (
        _json_get,
        _server_command as _benchmark_server_command,
        _wait_health,
        validate_health_profile,
    )
else:
    from qwen35b_mtp_batch_numerics_attribution import (
        _QWEN_ROUTE_ENV,
        _assert_no_other_model_runner,
        _validate_qwen_model,
        _verify_parent_guard_attestation,
    )
    from qwen35b_mtp_batch_numerics_guarded import (
        _json_get,
        _server_command as _benchmark_server_command,
        _wait_health,
        validate_health_profile,
    )


def build_codegen_command(
    *, python: Path, generator: Path, root: Path, port: int
) -> list[str]:
    return [
        str(python),
        "-u",
        str(generator),
        "--arm",
        "b8",
        "--root",
        str(root),
        "--endpoint",
        f"http://127.0.0.1:{port}/v1",
        "--model",
        "qwen35b-mtp-b8-numerics",
        "--datasets",
        "humaneval",
        "mbpp",
        "--max-tokens",
        "768",
        "--no-resume",
    ]


def _server_command(args: argparse.Namespace) -> list[str]:
    """Use a private two-second gather window so every scored group reaches B8."""

    command = _benchmark_server_command(args)
    insert_at = command.index("--depth")
    command[insert_at:insert_at] = ["--batch-wait-ms", "2000"]
    return command


def absolute_launcher_path(path: Path) -> Path:
    """Make a launcher absolute without dereferencing its virtualenv symlink."""

    return path.expanduser().absolute()


def pin_evalplus_site_packages(
    *,
    python: Path,
    cwd: Path,
    env: dict[str, str],
) -> None:
    """Verify EvalPlus and make its package root explicit for the child run."""

    probe = subprocess.run(
        [
            str(python),
            "-c",
            (
                "from pathlib import Path; import evalplus; "
                "print(Path(evalplus.__file__).resolve().parent.parent)"
            ),
        ],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    site_packages = str(probe.stdout or "").strip()
    if probe.returncode != 0 or not site_packages:
        detail = str(probe.stderr or probe.stdout or "import failed").strip()
        raise RuntimeError(f"EvalPlus interpreter preflight failed: {detail}")
    existing = str(env.get("PYTHONPATH") or "").strip()
    env["PYTHONPATH"] = (
        site_packages + os.pathsep + existing if existing else site_packages
    )


def validate_evalplus_b8_receipt(scheduler: dict[str, Any]) -> None:
    """Fail unless the private quality run executed the installed physical B8 lane."""

    mtp_batch = dict(scheduler.get("mtp_batch") or {})
    real = dict(mtp_batch.get("batch_histogram") or {})
    if real != {"8": 69}:
        raise RuntimeError(
            f"EvalPlus run requires exactly 69 real-width-eight cohorts; got {real!r}"
        )
    fixed = dict(mtp_batch.get("fixed_width_histogram") or {})
    if int(fixed.get("8") or 0) <= 0 or any(str(key) != "8" for key in fixed):
        raise RuntimeError(
            f"EvalPlus run did not prove physical B8 execution: {fixed!r}"
        )
    installed_route = str(scheduler.get("mtp_batch_route_id") or "")
    executed_route = str(mtp_batch.get("last_route_id") or "")
    if not installed_route or executed_route != installed_route:
        raise RuntimeError(
            "EvalPlus route mismatch: "
            f"installed={installed_route or 'none'} executed={executed_route or 'none'}"
        )
    if mtp_batch.get("last_error"):
        raise RuntimeError(f"EvalPlus MTP batch error: {mtp_batch['last_error']}")


def _nonempty_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--mtplx", required=True, type=Path)
    parser.add_argument("--chat-template", required=True, type=Path)
    parser.add_argument("--evalplus-python", required=True, type=Path)
    parser.add_argument("--generator", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--lock", default="/tmp/mtplx-gpu-exclusive.lock", type=Path)
    parser.add_argument("--numerics", choices=("throughput", "balanced"), required=True)
    parser.add_argument("--port", default=18080, type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    args.model = args.model.expanduser().resolve()
    args.mtplx = args.mtplx.expanduser().resolve()
    args.chat_template = args.chat_template.expanduser().resolve()
    args.evalplus_python = absolute_launcher_path(args.evalplus_python)
    args.generator = args.generator.expanduser().resolve()
    args.root = args.root.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    _validate_qwen_model(args.model)
    if not _verify_parent_guard_attestation(args.lock):
        raise RuntimeError("EvalPlus generation must run under the attested GPU guard")
    _assert_no_other_model_runner()

    expected = {
        "humaneval": (
            164,
            args.root / "b8/humaneval/b8_default_model_openai_temp_0.0.jsonl",
        ),
        "mbpp": (378, args.root / "b8/mbpp/b8_default_model_openai_temp_0.0.jsonl"),
    }
    occupied = [str(path) for _, path in expected.values() if path.exists()]
    if occupied:
        raise RuntimeError(
            f"refusing to append to existing EvalPlus samples: {occupied}"
        )

    env = os.environ.copy()
    env.update(_QWEN_ROUTE_ENV)
    pin_evalplus_site_packages(
        python=args.evalplus_python,
        cwd=args.generator.parent,
        env=env,
    )
    server_log_path = Path(f"/tmp/qwen35b-mtp-b8-{args.numerics}-evalplus-server.log")
    codegen_log_path = Path(f"/tmp/qwen35b-mtp-b8-{args.numerics}-evalplus-codegen.log")
    server_log = server_log_path.open("w", encoding="utf-8")
    codegen_log = codegen_log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        _server_command(args),
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )
    base_url = f"http://127.0.0.1:{args.port}"
    try:
        startup_health = _wait_health(base_url, process, 240)
        scheduler = validate_health_profile(startup_health, expected=args.numerics)
        command = build_codegen_command(
            python=args.evalplus_python,
            generator=args.generator,
            root=args.root,
            port=args.port,
        )
        completed = subprocess.run(
            command,
            cwd=args.generator.parent,
            env=env,
            stdout=codegen_log,
            stderr=subprocess.STDOUT,
            check=False,
        )
        codegen_log.flush()
        if completed.returncode != 0:
            raise RuntimeError(
                f"EvalPlus generation exited {completed.returncode}; see {codegen_log_path}"
            )
        final_health = _json_get(f"{base_url}/health", timeout=10)
        final_scheduler = validate_health_profile(
            final_health,
            expected=args.numerics,
        )
        validate_evalplus_b8_receipt(final_scheduler)
        mtp_batch = final_scheduler.get("mtp_batch") or {}
        counts = {name: _nonempty_lines(path) for name, (_, path) in expected.items()}
        for name, (count, _) in expected.items():
            if counts[name] != count:
                raise RuntimeError(
                    f"{name} generation produced {counts[name]} rows, expected {count}"
                )
        receipt: dict[str, Any] = {
            "schema_version": 1,
            "profile": args.numerics,
            "route_id": final_scheduler.get("mtp_batch_route_id"),
            "config_fingerprint": final_scheduler.get("mtp_batch_config_fingerprint"),
            "counts": counts,
            "batch_histogram": mtp_batch.get("batch_histogram"),
            "fixed_width_histogram": mtp_batch.get("fixed_width_histogram"),
            "last_route_id": mtp_batch.get("last_route_id"),
            "last_error": mtp_batch.get("last_error"),
            "server_log": str(server_log_path),
            "codegen_log": str(codegen_log_path),
            "startup_route_id": scheduler.get("mtp_batch_route_id"),
        }
        canonical = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
        receipt["receipt_sha256"] = hashlib.sha256(canonical).hexdigest()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(receipt, sort_keys=True), flush=True)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        server_log.close()
        codegen_log.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"guarded EvalPlus failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
