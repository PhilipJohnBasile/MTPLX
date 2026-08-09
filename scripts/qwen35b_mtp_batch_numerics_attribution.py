#!/usr/bin/env python3
"""Guarded, construction-only B1/B8 numerical attribution for Qwen 35B."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
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
    refined_boundaries: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Normalize one startup self-check into a stable attribution receipt."""

    geometry = dict(raw.get("geometry") or {})
    if geometry != {"target": [8, 2], "draft": [8, 1]}:
        raise ValueError(f"unexpected attribution geometry: {geometry!r}")
    source_boundaries = raw.get("boundaries") or raw.get("attribution_boundaries")
    if not isinstance(source_boundaries, list) or not source_boundaries:
        raise ValueError("attribution boundaries are required")
    source_boundaries = [*(refined_boundaries or ()), *source_boundaries]
    boundaries = []
    for index, source in enumerate(source_boundaries):
        if not isinstance(source, Mapping):
            raise ValueError(f"boundary {index} is not a mapping")
        missing = [field for field in BOUNDARY_FIELDS if field not in source]
        if missing:
            raise ValueError(f"boundary {index} is missing: {', '.join(missing)}")
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
    "MTPLX_CONTEXT_COPY": "0",
    "MTPLX_QWEN_STOCK_ORDER_QMM": "0",
    "MTPLX_FUSE_GDN_PROJECTIONS": "0",
    "MTPLX_FUSE_GDN_NORM_GATE": "0",
    "MTPLX_A3B_RESIDUAL_RMSNORM_M2": "0",
    "MTPLX_QWEN_SHARED_EXPERT_SWIGLU": "0",
    "MTPLX_QWEN_SHARED_SWIGLU_DOWN": "0",
    "MTPLX_QWEN_SHARED_COMBINE_TAIL": "0",
    "MTPLX_QWEN_DRAFT_LM_HEAD_SUMMARY": "0",
    "MTPLX_A3B_DRAFT_LOGIT_SUMMARY": "0",
    "MTPLX_A3B_SORTED_EXPERT_M2": "0",
    "MTPLX_A3B_GDN_INPROJ_ROWMAJOR": "0",
    "MTPLX_A3B_LM_HEAD_M2_ROWMAJOR": "0",
    "MTPLX_A3B_TARGET_LOGIT_SUMMARY": "0",
    "MTPLX_A3B_TARGET_LM_HEAD_SUMMARY": "0",
    "MTPLX_COMPILED_DRAFT_MTP": "0",
}

_GUARD_ATTEST_FD = "MTPLX_GUARD_ATTEST_FD"
_GUARD_ATTEST_NONCE = "MTPLX_GUARD_ATTEST_NONCE"
_MAX_ATTESTATION_BYTES = 16 * 1024


def _verify_parent_guard_attestation(expected_lock: Path) -> bool:
    descriptor_text = os.environ.get(_GUARD_ATTEST_FD)
    expected_nonce = os.environ.get(_GUARD_ATTEST_NONCE)
    if descriptor_text is None and expected_nonce is None:
        return False
    if not descriptor_text or not expected_nonce:
        raise RuntimeError("incomplete parent GPU guard attestation")
    descriptor = int(descriptor_text)
    payload_bytes = bytearray()
    while len(payload_bytes) <= _MAX_ATTESTATION_BYTES:
        chunk = os.read(
            descriptor,
            _MAX_ATTESTATION_BYTES + 1 - len(payload_bytes),
        )
        if not chunk:
            break
        payload_bytes.extend(chunk)
    os.close(descriptor)
    if len(payload_bytes) > _MAX_ATTESTATION_BYTES:
        raise RuntimeError("parent GPU guard attestation is too large")
    payload = json.loads(payload_bytes)
    now = time.monotonic_ns()
    required_ints = (
        "guard_pid",
        "child_pid",
        "lock_device",
        "lock_inode",
        "issued_monotonic_ns",
        "expires_monotonic_ns",
    )
    if payload.get("schema_version") != 1 or any(
        isinstance(payload.get(key), bool) or not isinstance(payload.get(key), int)
        for key in required_ints
    ):
        raise RuntimeError("parent GPU guard attestation is malformed")
    if (
        payload.get("nonce") != expected_nonce
        or payload["child_pid"] != os.getpid()
        or payload["guard_pid"] != os.getppid()
        or not (
            payload["issued_monotonic_ns"] <= now <= payload["expires_monotonic_ns"]
        )
        or payload["expires_monotonic_ns"] - payload["issued_monotonic_ns"]
        > 60_000_000_000
    ):
        raise RuntimeError("parent GPU guard attestation identity or expiry failed")
    resolved = expected_lock.resolve(strict=True)
    if Path(str(payload.get("lock_path"))).resolve(strict=True) != resolved:
        raise RuntimeError("parent GPU guard attested a different lock")
    observed = resolved.stat()
    if (observed.st_dev, observed.st_ino) != (
        payload["lock_device"],
        payload["lock_inode"],
    ):
        raise RuntimeError("parent GPU guard lock identity changed")
    probe = resolved.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(probe.fileno(), fcntl.LOCK_UN)
            raise RuntimeError("parent GPU guard no longer holds the lock")
    finally:
        probe.close()
    return True


@contextmanager
def _exclusive_gpu_window(lock_path: Path):
    if _verify_parent_guard_attestation(lock_path):
        yield "attested_parent"
        return
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"GPU lock is held: {lock_path}") from exc
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(f"pid={os.getpid()} task=qwen35b-mtp-b8-attribution\n")
        lock_file.flush()
        try:
            yield "direct"
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


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


def _construct_lane(model_path: Path, *, numerics: str) -> tuple[Any, Any]:
    for key, value in _QWEN_ROUTE_ENV.items():
        os.environ[key] = value
    from mtplx.profiles import apply_profile_env

    apply_profile_env("turbo")
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
    lane = install_a3b_mtp_batch_lane(runtime, numerics=numerics)
    return runtime, lane


def _compare_boundary(
    operator: str,
    b8_value: Any,
    b1_values: list[Any],
) -> dict[str, Any]:
    """Materialize one construction-only B8 versus eight-B1 boundary."""

    import mlx.core as mx
    import numpy as np

    if len(b1_values) != 8:
        raise ValueError("layer attribution requires exactly eight B1 rows")
    mx.eval(b8_value, *b1_values)
    reference = mx.concatenate(b1_values, axis=0)
    bitwise = mx.all(b8_value == reference)
    max_abs = mx.max(mx.abs(b8_value - reference).astype(mx.float32))
    argmax_equal = mx.all(mx.argmax(b8_value, axis=-1) == mx.argmax(reference, axis=-1))
    mx.eval(bitwise, max_abs, argmax_equal)
    return {
        "operator": operator,
        "layer": 0,
        "phase": "decode_verify",
        "b1_shape": [int(value) for value in b1_values[0].shape],
        "b8_shape": [int(value) for value in b8_value.shape],
        "bitwise": bool(np.asarray(bitwise).item()),
        "max_abs": float(np.asarray(max_abs).item()),
        # ULP materialization changed the construction graph in an earlier
        # run.  Keep the production arithmetic untouched and mark it absent.
        "max_ulp": -1,
        "argmax_equal": bool(np.asarray(argmax_equal).item()),
    }


def _attribute_layer_zero(runtime: Any, lane: Any) -> list[dict[str, Any]]:
    """Locate the first B8/M16 divergence before layer-zero postconv."""

    import mlx.core as mx

    from mtplx.gdn_capture import _stock_conv1d_capture

    token = int(getattr(getattr(runtime, "tokenizer", None), "eos_token_id", 1) or 1)
    vocab_size = int(lane.geometry.vocab_size)
    row_tokens = [int((token + row) % vocab_size) for row in range(8)]
    prefills = [
        lane.prefill_request([row_token], abort_check=None) for row_token in row_tokens
    ]
    b1_caches = [item[0] for item in prefills]
    # The merge deliberately releases every scalar source slot once the B8
    # destination is materialized.  Retain only the two layer-zero source
    # references needed by this construction receipt before handing ownership
    # to the merged cache.
    b1_base_conv = [cache[0][0] for cache in b1_caches]
    b1_base_state = [cache[0][1] for cache in b1_caches]
    b8_cache = lane.merge_target_caches(b1_caches)
    verify_tokens = mx.array(
        [
            [
                int((token + 257 + 2 * row) % vocab_size),
                int((token + 258 + 2 * row) % vocab_size),
            ]
            for row in range(8)
        ],
        dtype=mx.int32,
    )
    b1_tokens = [verify_tokens[row : row + 1] for row in range(8)]

    inner = runtime.model.language_model.model
    layer = inner.layers[0]
    gdn = layer.linear_attn
    records: list[dict[str, Any]] = []

    b8_hidden = inner.embed_tokens(verify_tokens)
    b1_hidden = [inner.embed_tokens(value) for value in b1_tokens]
    records.append(_compare_boundary("target.embed_tokens", b8_hidden, b1_hidden))

    b8_normed = layer.input_layernorm(b8_hidden)
    b1_normed = [layer.input_layernorm(value) for value in b1_hidden]
    records.append(
        _compare_boundary("target.layers.0.input_layernorm", b8_normed, b1_normed)
    )

    projections = (
        ("in_proj_qkv", gdn.in_proj_qkv),
        ("in_proj_z", gdn.in_proj_z),
        ("in_proj_b", gdn.in_proj_b),
        ("in_proj_a", gdn.in_proj_a),
    )
    from mtplx.nax_verify import _QLINEAR_PATCH

    stock_qlinear_call = _QLINEAR_PATCH.get("original")
    if not callable(stock_qlinear_call):
        raise RuntimeError("stock QuantizedLinear callable was not retained")
    projected: dict[str, tuple[Any, list[Any]]] = {}
    for name, projection in projections:
        b8_value = projection(b8_normed)
        b1_values = [projection(value) for value in b1_normed]
        projected[name] = (b8_value, b1_values)
        records.append(
            _compare_boundary(
                f"target.layers.0.linear_attn.{name}",
                b8_value,
                b1_values,
            )
        )
        records.append(
            _compare_boundary(
                f"target.layers.0.linear_attn.{name}.stock_m16",
                stock_qlinear_call(projection, b8_normed),
                b1_values,
            )
        )

    b8_base_conv = b8_cache[0][0]
    records.append(
        _compare_boundary(
            "target.layers.0.linear_attn.cache.conv_state_in",
            b8_base_conv,
            b1_base_conv,
        )
    )
    records.append(
        _compare_boundary(
            "target.layers.0.linear_attn.cache.gdn_state_in",
            b8_cache[0][1],
            b1_base_state,
        )
    )

    b8_qkv, b1_qkv = projected["in_proj_qkv"]
    b8_conv_out, b8_conv_states = _stock_conv1d_capture(b8_qkv, b8_base_conv, gdn)
    b1_conv = [
        _stock_conv1d_capture(qkv, base, gdn)
        for qkv, base in zip(b1_qkv, b1_base_conv, strict=True)
    ]
    records.append(
        _compare_boundary(
            "target.layers.0.linear_attn.conv1d.output",
            b8_conv_out,
            [item[0] for item in b1_conv],
        )
    )
    records.append(
        _compare_boundary(
            "target.layers.0.linear_attn.conv1d.conv_state",
            b8_conv_states,
            [item[1] for item in b1_conv],
        )
    )
    return records


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--lock", default="/tmp/mtplx-gpu-exclusive.lock", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--numerics",
        choices=("throughput", "balanced"),
        default="throughput",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    model_path = Path(args.model).expanduser().resolve()
    _validate_qwen_model(model_path)
    with _exclusive_gpu_window(args.lock) as lock_scope:
        _assert_no_other_model_runner()
        runtime, lane = _construct_lane(model_path, numerics=args.numerics)
        refined_boundaries = _attribute_layer_zero(runtime, lane)
        raw = dict(lane.selfcheck)
        raw["geometry"] = {"target": [8, 2], "draft": [8, 1]}
        report = build_report(
            raw,
            model=str(model_path),
            route_id=lane.route_id,
            config_fingerprint=lane.config_fingerprint,
            refined_boundaries=refined_boundaries,
        )
        report["gpu_lock_scope"] = lock_scope
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"attribution failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
