"""Crossover/chunk/262K experiment arms on top of the ABBA battery.

Same discipline as qsa_prefill_battery_20260830 (imports its gates): fans
pinned+verified, die-temp gate per arm, neutral cwd, stripped env,
engagement receipts.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qsa_prefill_battery_20260830 import (  # noqa: E402
    LANE_ENV,
    OUT_DIR,
    pin_fans_max,
    restore_fans_auto,
    run_arm,
)

ARMS = [
    # Aggressive crossover: does the lane pay from 8K history onward?
    (
        "lane-x8k",
        {
            **LANE_ENV,
            "MTPLX_QSA_PREFILL_MIN_CONTEXT": "8192",
            "MTPLX_QSA_PREFILL_FLASH_MIN_CONTEXT": "8192",
        },
        "32768,98304",
    ),
    # Fresh lane-defaults control, back-to-back for drift pairing.
    ("lane-ctl", LANE_ENV, "32768,98304"),
    # 4096-token chunks: mask-free attention removes the big-chunk penalty;
    # do larger chunks now amortize GDN/MoE per-chunk overhead?
    (
        "lane-c4k",
        {**LANE_ENV, "MTPLX_PREFILL_CHUNK_SIZE": "4096"},
        "98304",
    ),
    # The #393 scenario: 262K cold prefill on this 128GB machine.
    ("lane-262k", LANE_ENV, "262144"),
]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pin_fans_max()
    try:
        for name, env, contexts in ARMS:
            run_arm(name, env, contexts, 1)
    finally:
        restore_fans_auto()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
