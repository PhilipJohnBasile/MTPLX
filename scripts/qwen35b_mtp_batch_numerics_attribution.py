#!/usr/bin/env python3
"""Guarded, construction-only B1/B8 numerical attribution for Qwen 35B."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


BOUNDARY_FIELDS = (
    "operator",
    "layer",
    "phase",
    "b1_shape",
    "b8_shape",
    "bitwise",
    "max_abs",
    "max_ulp",
    "argmax_equal",
)


def build_report(
    raw: Mapping[str, Any],
    *,
    model: str,
    route_id: str,
    config_fingerprint: str,
) -> dict[str, Any]:
    """Normalize one startup self-check into a stable attribution receipt."""

    geometry = dict(raw.get("geometry") or {})
    if geometry != {"target": [8, 2], "draft": [8, 1]}:
        raise ValueError(f"unexpected attribution geometry: {geometry!r}")
    source_boundaries = raw.get("boundaries") or raw.get(
        "attribution_boundaries"
    )
    if not isinstance(source_boundaries, list) or not source_boundaries:
        raise ValueError("attribution boundaries are required")
    boundaries = []
    for index, source in enumerate(source_boundaries):
        if not isinstance(source, Mapping):
            raise ValueError(f"boundary {index} is not a mapping")
        missing = [field for field in BOUNDARY_FIELDS if field not in source]
        if missing:
            raise ValueError(
                f"boundary {index} is missing: {', '.join(missing)}"
            )
        boundaries.append({field: source[field] for field in BOUNDARY_FIELDS})
    first = next(
        (
            boundary
            for boundary in boundaries
            if not bool(boundary["bitwise"])
            or float(boundary["max_abs"]) != 0.0
            or int(boundary["max_ulp"]) > 0
        ),
        None,
    )
    payload = {
        "schema_version": 1,
        "model": str(model),
        "route_id": str(route_id),
        "config_fingerprint": str(config_fingerprint),
        "geometry": geometry,
        "row_isolation_parity": bool(raw.get("row_isolation_parity")),
        "boundaries": boundaries,
        "first_material_divergence": first,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["receipt_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


_QWEN_ROUTE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "MTPLX_QWEN_MOE_PACK_GATE_UP": "1",
    "MTPLX_QWEN_ROW_OWNED_ROUTER": "1",
    "MTPLX_QWEN_COMBINE_TAIL": "1",
    "MTPLX_LINEAR_GDN_FROM_CONV_TGY": "4",
    "MTPLX_FUSE_GDN_POST_CONV": "1",
    "MTPLX_A3B_GDN_POSTCONV_IMPL": "headquarter",
    "MTPLX_COMPILED_TARGET_PREFIX": "1",
    "MTPLX_COMPILED_VERIFY": "1",
    "MTPLX_A3B_WHOLE_MOE_FUSION": "0",
    "MTPLX_COMPILED_DRAFT_MTP": "0",
}


def _assert_no_other_model_runner() -> None:
    listing = subprocess.run(
        ["ps", "-axo", "pid=,command="],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    offenders = []
    for line in listing.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, _, command = stripped.partition(" ")
        if int(pid_text) == os.getpid():
            continue
        if "mtplx.server.openai" in command or " mtplx serve " in command:
            offenders.append(stripped)
    if offenders:
        raise RuntimeError(
            "refusing to load Qwen while another model runner is live: "
            + " | ".join(offenders)
        )


def _validate_qwen_model(model_path: Path) -> None:
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    text = config.get("text_config") or {}
    if (
        config.get("model_type") != "qwen3_5_moe"
        or text.get("hidden_size") != 2048
        or text.get("num_hidden_layers") != 40
        or text.get("num_experts") != 256
        or text.get("mtp_num_hidden_layers") != 1
    ):
        raise RuntimeError("attribution accepts only the fixed Qwen 35B A3B model")


def _construct_lane(model_path: Path) -> Any:
    for key, value in _QWEN_ROUTE_ENV.items():
        os.environ[key] = value
    from mtplx.a3b_mtp_batch import install_a3b_mtp_batch_lane
    from mtplx.draft_lm_head import _install_draft_lm_head
    from mtplx.mtp_patch import MTPContract
    from mtplx.runtime import load

    runtime = load(
        model_path,
        mtp=True,
        contract=MTPContract(
            mtp_quant_bits=4,
            mtp_quant_group_size=64,
            mtp_quant_mode="affine",
        ),
    )
    _install_draft_lm_head(runtime, bits=4, group_size=64, mode="affine")
    return install_a3b_mtp_batch_lane(runtime, numerics="throughput")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--lock", default="/tmp/mtplx-gpu-exclusive.lock", type=Path
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    model_path = Path(args.model).expanduser().resolve()
    _validate_qwen_model(model_path)
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    with args.lock.open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"GPU lock is held: {args.lock}") from exc
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(f"pid={os.getpid()} task=qwen35b-mtp-b8-attribution\n")
        lock_file.flush()
        _assert_no_other_model_runner()
        lane = _construct_lane(model_path)
        raw = dict(lane.selfcheck)
        raw["geometry"] = {"target": [8, 2], "draft": [8, 1]}
        report = build_report(
            raw,
            model=str(model_path),
            route_id=lane.route_id,
            config_fingerprint=lane.config_fingerprint,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"attribution failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
