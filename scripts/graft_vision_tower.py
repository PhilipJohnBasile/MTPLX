#!/usr/bin/env python3
"""Restore a dropped vision tower into a forged MTPLX artifact (issue #263).

Copies the vision tensors byte-for-byte from the original multimodal source
checkpoint into ``model-vision.safetensors``, registers them in the
destination's ``model.safetensors.index.json``, restores ``vision_config``
in ``config.json``, and copies the preprocessor sidecars. Language and MTP
tensors are never touched.

Example:
    python3 scripts/graft_vision_tower.py \
        --source ~/.mtplx/models/Qwen--Qwen3.8-27B \
        --target ~/.mtplx/models/Qwen3.8-27B-MTPLX-Optimized-Speed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mtplx.vision_graft import VisionGraftError, graft_vision_tower  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="original multimodal checkpoint directory (vision donor)",
    )
    parser.add_argument(
        "--target",
        required=True,
        type=Path,
        help="forged artifact directory to repair",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be grafted without writing anything",
    )
    parser.add_argument(
        "--no-verify-load",
        action="store_true",
        help="skip the strict tower load after grafting (saves ~1 GB of RAM)",
    )
    args = parser.parse_args()

    try:
        report = graft_vision_tower(
            args.source.expanduser(),
            args.target.expanduser(),
            dry_run=args.dry_run,
            verify_load=not args.no_verify_load,
        )
    except VisionGraftError as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] in ("grafted", "already-present", "dry-run") else 2


if __name__ == "__main__":
    raise SystemExit(main())
