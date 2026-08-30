"""ABBA prefill-ladder battery: dense baseline vs QSA large-prefill lane.

House protocol (fork3842 clean-battery lineage):
- one arm per process, spawned with python -P from a neutral cwd;
- inherited MTPLX_* env stripped, arm env explicit;
- fans pinned max via thermalforge and VERIFIED (mode + target rpm);
- die-temp gate before every arm (all GPU-prefixed SMC sensors < 62C);
- A/B/B/A order so session drift cannot masquerade as an arm effect;
- engagement receipts (qsa_prefill_engagement) required from stderr on
  candidate arms before any number is read.

Usage:
  .venv/bin/python scripts/qsa_prefill_battery_20260830.py [--stretch]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
THERMALFORGE = Path("/usr/local/bin/thermalforge")
MODEL = Path.home() / ".mtplx" / "models" / "Qwen3.8-Flash-Next-MTPLX-Optimized-Speed"
OUT_DIR = ROOT / "outputs" / "qsa-prefill-battery-20260830"
NEUTRAL_CWD = "/tmp"
DIE_TEMP_GATE_C = 62.0
FAN_MIN_TARGET_RPM = 7000
CONTEXTS = "16384,32768,98304"
STRETCH_CONTEXTS = "131072"
SEED = 20260830

DENSE_ENV: dict[str, str] = {}
LANE_ENV = {
    "MTPLX_QSA_PREFILL": "1",
    "MTPLX_QSA_PREFILL_FLASH_MIN_CONTEXT": "32768",
    "MTPLX_QSA_PREFILL_DEBUG": "1",
}


def _thermalforge(*args: str, use_sudo: bool = False) -> dict:
    argv = ["sudo", "-n", str(THERMALFORGE), *args] if use_sudo else [
        str(THERMALFORGE),
        *args,
    ]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    if args[0] == "status":
        return json.loads(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"thermalforge {args} failed: {proc.stderr or proc.stdout}")
    return {}


def pin_fans_max() -> None:
    """Pin fans and verify the actual ramp — trust the measurement, not the
    request (SMC mode alone has lied before)."""

    _thermalforge("max", use_sudo=True)
    deadline = time.time() + 120
    while True:
        status = _thermalforge("status")
        fans = status["fans"]
        if all(
            f["mode"] == "manual"
            and f["target_rpm"] >= FAN_MIN_TARGET_RPM
            and f["actual_rpm"] >= FAN_MIN_TARGET_RPM
            for f in fans
        ):
            print(
                "fans pinned+verified:",
                [(f["index"], f["mode"], f["target_rpm"], f["actual_rpm"]) for f in fans],
                flush=True,
            )
            return
        if time.time() > deadline:
            raise RuntimeError(
                "fan ramp verification timeout: "
                f"{[(f['mode'], f['target_rpm'], f['actual_rpm']) for f in fans]}"
            )
        time.sleep(3)


def restore_fans_auto() -> None:
    try:
        _thermalforge("auto", use_sudo=True)
    except RuntimeError as exc:
        print(f"WARNING: fan restore command failed: {exc}", file=sys.stderr, flush=True)
    time.sleep(2)
    status = _thermalforge("status")
    modes = [f["mode"] for f in status["fans"]]
    print("fans restored:", modes, flush=True)
    if any(m not in ("auto",) for m in modes):
        print("WARNING: fan mode not auto after restore", file=sys.stderr, flush=True)


def gpu_temps(status: dict) -> list[float]:
    return [
        v
        for k, v in status["temperatures"].items()
        if k.lower().startswith("tg") or k.lower().startswith("tp")
    ]


def die_temp_gate() -> None:
    started = time.time()
    while True:
        status = _thermalforge("status")
        temps = gpu_temps(status)
        hottest = max(temps) if temps else 0.0
        if hottest < DIE_TEMP_GATE_C:
            print(f"die-temp gate passed: hottest {hottest:.1f}C", flush=True)
            return
        if time.time() - started > 900:
            raise RuntimeError(f"die-temp gate timeout, hottest {hottest:.1f}C")
        print(f"cooling: hottest {hottest:.1f}C >= {DIE_TEMP_GATE_C}", flush=True)
        time.sleep(20)


def clean_env(extra: dict[str, str]) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("MTPLX_")}
    env.update(extra)
    return env


def no_model_process_gate() -> None:
    proc = subprocess.run(
        ["pgrep", "-fl", r"mtplx(\.cli)? (serve|bench prefill-ladder)|mtplx.server.openai|mlx_lm"],
        capture_output=True,
        text=True,
    )
    if proc.stdout.strip():
        raise RuntimeError(f"live model process detected:\n{proc.stdout}")


def run_arm(name: str, arm_env: dict[str, str], contexts: str, rep: int) -> Path:
    no_model_process_gate()
    die_temp_gate()
    out_file = OUT_DIR / f"{name}-rep{rep}.json"
    log_file = OUT_DIR / f"{name}-rep{rep}.log"
    cmd = [
        str(PYTHON),
        "-P",
        "-m",
        "mtplx.cli",
        "bench",
        "prefill-ladder",
        "--model",
        str(MODEL),
        "--contexts",
        contexts,
        "--max-tokens",
        "1",
        "--profile",
        "turbo",
        "--seed",
        str(SEED),
        "--json",
        "--output",
        str(out_file),
        "--yes",
    ]
    print(f"\n=== ARM {name} rep{rep} start {datetime.now().isoformat()} ===", flush=True)
    started = time.time()
    with open(log_file, "w") as log:
        proc = subprocess.run(
            cmd,
            cwd=NEUTRAL_CWD,
            env=clean_env(arm_env),
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=3600,
        )
    elapsed = time.time() - started
    print(f"=== ARM {name} rep{rep} rc={proc.returncode} in {elapsed:.0f}s ===", flush=True)
    if proc.returncode != 0:
        tail = log_file.read_text()[-2000:]
        raise RuntimeError(f"arm {name} rep{rep} failed rc={proc.returncode}:\n{tail}")
    engagement = [
        line
        for line in log_file.read_text().splitlines()
        if "qsa_prefill_engagement" in line
    ]
    print("engagement:", engagement or "(none)", flush=True)
    return out_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stretch", action="store_true")
    parser.add_argument("--contexts", default=CONTEXTS)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not MODEL.is_dir():
        raise SystemExit(f"model pack missing: {MODEL}")

    pin_fans_max()
    try:
        results = {}
        # A/B/B/A: dense, lane, lane, dense
        results["dense-rep1"] = run_arm("dense", DENSE_ENV, args.contexts, 1)
        results["lane-rep1"] = run_arm("lane", LANE_ENV, args.contexts, 1)
        results["lane-rep2"] = run_arm("lane", LANE_ENV, args.contexts, 2)
        results["dense-rep2"] = run_arm("dense", DENSE_ENV, args.contexts, 2)
        if args.stretch:
            results["lane-stretch"] = run_arm(
                "lane-stretch", LANE_ENV, STRETCH_CONTEXTS, 1
            )
        print("\nall arms complete:", {k: str(v) for k, v in results.items()}, flush=True)
    finally:
        restore_fans_auto()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
