"""Operator-controlled parity gate for production QSA prefill attention.

The custom Metal consumer is compared with stock dense SDPA over the exact
per-row selected-block plus visible-tail mask.  Fixtures use the production
Hq=24/Hkv=2/D=256/K=512 geometry at contexts just beyond the 2048-token
engage boundary.  Adversarial K/V sentinels make both block skipping and each
zero-to-three-token tail non-vacuous.  No model weights are loaded.

Run this only when the operator has released the GPU::

    uv run --no-project --python 3.13 --with mlx --with numpy python \
      scripts/qsa_prefill_flash_numeric_check.py
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

BATCH = 1
Q_HEADS = 24
KV_HEADS = 2
HEAD_DIM = 256
BLOCK_TOPK = 512
COMPRESS_RATIO = 4
SCALE = 0.0625
KV_CAPACITY = 2080
SENSITIVITY_MIN = 5.0e-2

# Explicit output tolerances.  The kernel and stock SDPA both consume the same
# low-precision inputs and return that dtype, but their fp32 accumulation trees
# differ.  Acceptance is elementwise: abs(error) <= atol + rtol*abs(reference).
TOLERANCES = {
    "float16": (1.0e-2, 2.0e-2),
    "bfloat16": (2.0e-2, 2.0e-2),
}


@dataclass(frozen=True)
class FlashCase:
    name: str
    seed: int
    rows: int
    total_tokens: int
    expected_final_tail: int
    kv_layout: str = "contiguous"
    selection_layout: str = "full"


FLASH_CASES = (
    FlashCase("tail0", 101, 6, 2052, 0),
    FlashCase("tail1", 103, 2, 2053, 1),
    FlashCase("tail2", 107, 3, 2054, 2),
    FlashCase("tail3", 109, 4, 2055, 3),
    FlashCase("offset1_tail1", 113, 2, 2053, 1, "offset1"),
    FlashCase("feature_stride2_tail2", 127, 3, 2054, 2, "feature_stride2"),
    FlashCase("token_stride2_tail3", 131, 4, 2055, 3, "token_stride2"),
    FlashCase("active_tile_edges", 137, 4, 2055, 3, "contiguous", "active_edges"),
)
FLASH_DTYPES = ("float16", "bfloat16")


def _command_output(argv: list[str]) -> str:
    result = subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _machine_safety_gate() -> bool:
    """Refuse a GPU gate while a model worker is live."""

    processes = _command_output(
        [
            "pgrep",
            "-fl",
            "mtplx(\\.cli)? (serve|bench prefill-ladder)|mtplx.server.openai|mlx_lm",
        ]
    )
    pressure = _command_output(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"])
    print(
        f"SAFETY pressure={pressure or 'unknown'} "
        f"concurrent_model_process={bool(processes)}",
        flush=True,
    )
    if processes:
        print("SAFETY_REFUSE another model process is live:", flush=True)
        print(processes, flush=True)
        return False
    return True


def _require_active_metal(mx) -> bool:
    try:
        active = mx.metal.is_available() and mx.default_device() == mx.gpu
    except (AttributeError, RuntimeError, TypeError, ValueError):
        active = False
    if not active:
        print("QSA PREFILL FLASH REFUSE active MLX device is not Metal", flush=True)
    return active


def _dtype(mx, name: str):
    return {"float16": mx.float16, "bfloat16": mx.bfloat16}[name]


def _selection_and_masks(np, case: FlashCase):
    """Build chronological selections and their exact dense reference masks."""

    pos_start = case.total_tokens - case.rows
    ids = np.zeros((case.rows, BLOCK_TOPK), dtype=np.int32)
    valid = np.zeros((case.rows, BLOCK_TOPK), dtype=np.bool_)
    selected_mask = np.zeros((case.rows, case.total_tokens), dtype=np.bool_)
    block_only_mask = np.zeros_like(selected_mask)
    causal_mask = np.zeros_like(selected_mask)

    for row in range(case.rows):
        query_pos = pos_start + row
        complete = (query_pos + 1) // COMPRESS_RATIO
        if case.selection_layout == "active_edges":
            selected_count = (7, 8, 9, 2)[row]
            chosen = np.arange(
                complete - selected_count,
                complete,
                dtype=np.int32,
            )
        elif case.selection_layout == "full":
            first = max(0, complete - BLOCK_TOPK)
            chosen = np.arange(first, complete, dtype=np.int32)
        else:
            raise AssertionError(
                f"unknown selection fixture layout {case.selection_layout!r}"
            )
        if chosen.size > BLOCK_TOPK:
            raise AssertionError("fixture produced more than K selected blocks")
        if case.selection_layout == "active_edges" and row == case.rows - 1:
            # Only two entries contribute, but the second lives in slot 17:
            # a valid-count bound would stop after tile zero and miss it.
            ids[row, 0] = chosen[0]
            ids[row, 17] = chosen[1]
            valid[row, 0] = True
            valid[row, 17] = True
        else:
            ids[row, : chosen.size] = chosen
            valid[row, : chosen.size] = True

        for block_id in chosen.tolist():
            token_start = block_id * COMPRESS_RATIO
            token_stop = token_start + COMPRESS_RATIO
            block_only_mask[row, token_start:token_stop] = True
        selected_mask[row] = block_only_mask[row]

        tail_start = complete * COMPRESS_RATIO
        selected_mask[row, tail_start : query_pos + 1] = True
        causal_mask[row, : query_pos + 1] = True

        # The earliest tail0 row has only 511 complete blocks.  Mark its spare
        # slot as metadata-valid but point it at the next, still-incomplete
        # block.  The Metal consumer must apply its own per-row frontier check
        # and ignore this producer-corruption sentinel; the dense oracle above
        # intentionally does not include it.
        if case.selection_layout == "active_edges":
            # A metadata-valid future block in the final slot must neither
            # extend the dynamic tile bound nor enter attention.
            ids[row, -1] = complete
            valid[row, -1] = True
        elif chosen.size < BLOCK_TOPK:
            ids[row, chosen.size] = complete
            valid[row, chosen.size] = True

    final_pos = case.total_tokens - 1
    final_complete = (final_pos + 1) // COMPRESS_RATIO
    final_tail = final_pos - final_complete * COMPRESS_RATIO + 1
    if final_tail != case.expected_final_tail:
        raise AssertionError(
            f"{case.name}: final tail {final_tail} != {case.expected_final_tail}"
        )
    # Once sparse mode engages, deliberately omit block zero.  Its adversarial
    # K/V payload below makes a consumer that accidentally runs full causal
    # attention fail by a wide margin.
    if final_complete <= BLOCK_TOPK or ids[-1, 0] == 0:
        raise AssertionError(f"{case.name}: final selection did not omit block zero")
    return ids, valid, selected_mask, block_only_mask, causal_mask


def _fixture(np, case: FlashCase):
    rng = np.random.default_rng(case.seed)
    # Build the layer-native [B,S,H,D] layout.  The runner transposes this to
    # [B,H,S,D] without making it contiguous, matching Attention.__call__ and
    # proving that the custom kernel consumes the injected Q strides directly.
    q = (0.25 + rng.normal(0.0, 0.01, (BATCH, case.rows, Q_HEADS, HEAD_DIM))).astype(
        np.float32
    )
    k = rng.normal(
        0.0,
        0.05,
        (BATCH, KV_HEADS, KV_CAPACITY, HEAD_DIM),
    ).astype(np.float32)
    v = rng.normal(
        0.0,
        0.10,
        (BATCH, KV_HEADS, KV_CAPACITY, HEAD_DIM),
    ).astype(np.float32)

    # Block zero is visible to full causal attention but intentionally absent
    # from the final sparse selection.  A high logit and negative value make
    # that distinction observable instead of relying on random noise.
    k[:, :, :COMPRESS_RATIO, :] = 1.0
    v[:, :, :COMPRESS_RATIO, :] = -8.0

    # Token 2047 is the future lane of the deliberately corrupt block metadata
    # above. Tail sentinels cover tokens 2052..2054; token 2055 is a future
    # sentinel for the final rows. Their high logits and distinct values make
    # missing, duplicated, or over-read positions observable.
    for token, value in (
        (2047, -24.0),
        (2052, 8.0),
        (2053, 12.0),
        (2054, 16.0),
        (2055, -32.0),
    ):
        k[:, :, token, :] = 1.0
        v[:, :, token, :] = value
    return q, k, v


def _dense_sdpa(mx, q, k, v, mask, *, total_tokens: int):
    return mx.fast.scaled_dot_product_attention(
        q,
        k[:, :, :total_tokens, :],
        v[:, :, :total_tokens, :],
        scale=SCALE,
        mask=mask[None, None],
    )


def _kv_view(mx, np, values, dtype, layout: str):
    """Preserve adversarial backing offsets/strides through MLX conversion."""

    if layout == "contiguous":
        return mx.array(values).astype(dtype)
    if layout == "offset1":
        backing = np.zeros((*values.shape[:-1], HEAD_DIM + 1), dtype=np.float32)
        backing[..., 1:] = values
        return mx.array(backing).astype(dtype)[..., 1:]
    if layout == "feature_stride2":
        backing = np.zeros((*values.shape[:-1], HEAD_DIM * 2), dtype=np.float32)
        backing[..., ::2] = values
        return mx.array(backing).astype(dtype)[..., ::2]
    if layout == "token_stride2":
        backing = np.zeros(
            (BATCH, KV_HEADS, KV_CAPACITY * 2, HEAD_DIM),
            dtype=np.float32,
        )
        backing[:, :, ::2, :] = values
        return mx.array(backing).astype(dtype)[:, :, ::2, :]
    raise AssertionError(f"unknown K/V fixture layout {layout!r}")


def _last_row_max_abs(mx, left, right) -> float:
    delta = mx.abs(
        left[:, :, -1, :].astype(mx.float32) - right[:, :, -1, :].astype(mx.float32)
    )
    return float(mx.max(delta).item())


def _run_case(mx, np, flash_module, case: FlashCase, dtype_name: str):
    q_np, k_np, v_np = _fixture(np, case)
    ids_np, valid_np, selected_np, block_only_np, causal_np = _selection_and_masks(
        np, case
    )
    dtype = _dtype(mx, dtype_name)
    q = mx.array(q_np).astype(dtype).transpose(0, 2, 1, 3)
    k = _kv_view(mx, np, k_np, dtype, case.kv_layout)
    v = _kv_view(mx, np, v_np, dtype, case.kv_layout)
    block_ids = mx.array(ids_np).astype(mx.int32)
    block_valid = mx.array(valid_np).astype(mx.bool_)
    selected_mask = mx.array(selected_np).astype(mx.bool_)
    block_only_mask = mx.array(block_only_np).astype(mx.bool_)
    causal_mask = mx.array(causal_np).astype(mx.bool_)
    pos_start = case.total_tokens - case.rows

    supported = flash_module.qsa_prefill_flash_supported(
        q,
        k,
        v,
        block_ids,
        block_valid,
        pos_start=pos_start,
        total_tokens=case.total_tokens,
        scale=SCALE,
    )
    if not supported:
        raise AssertionError(f"{case.name}/{dtype_name}: production signature rejected")

    actual = flash_module.qsa_prefill_flash(
        q,
        k,
        v,
        block_ids,
        block_valid,
        pos_start=pos_start,
        total_tokens=case.total_tokens,
        scale=SCALE,
    )
    reference = _dense_sdpa(
        mx,
        q,
        k,
        v,
        selected_mask,
        total_tokens=case.total_tokens,
    )
    full_causal = _dense_sdpa(
        mx,
        q,
        k,
        v,
        causal_mask,
        total_tokens=case.total_tokens,
    )
    evaluated = [actual, reference, full_causal]
    block_only = None
    if case.expected_final_tail:
        block_only = _dense_sdpa(
            mx,
            q,
            k,
            v,
            block_only_mask,
            total_tokens=case.total_tokens,
        )
        evaluated.append(block_only)
    mx.eval(*evaluated)

    expected_shape = (BATCH, Q_HEADS, case.rows, HEAD_DIM)
    if tuple(int(x) for x in actual.shape) != expected_shape:
        raise AssertionError(f"{case.name}/{dtype_name}: output shape {actual.shape}")
    if actual.dtype != dtype:
        raise AssertionError(f"{case.name}/{dtype_name}: output dtype {actual.dtype}")
    if not bool(mx.all(mx.isfinite(actual)).item()):
        raise AssertionError(f"{case.name}/{dtype_name}: non-finite kernel output")

    # Guard against a vacuous fixture: selected sparse attention must differ
    # materially from full causal attention on the final row.
    full_gap = _last_row_max_abs(mx, reference, full_causal)
    if full_gap <= SENSITIVITY_MIN:
        raise AssertionError(
            f"{case.name}/{dtype_name}: full-causal sensitivity {full_gap}"
        )
    tail_gap = 0.0
    if block_only is not None:
        tail_gap = _last_row_max_abs(mx, reference, block_only)
        if tail_gap <= SENSITIVITY_MIN:
            raise AssertionError(
                f"{case.name}/{dtype_name}: tail sensitivity {tail_gap}"
            )

    actual_f32 = actual.astype(mx.float32)
    reference_f32 = reference.astype(mx.float32)
    difference = mx.abs(actual_f32 - reference_f32)
    atol, rtol = TOLERANCES[dtype_name]
    limit = atol + rtol * mx.abs(reference_f32)
    passed = bool(mx.all(difference <= limit).item())
    max_abs = float(mx.max(difference).item())
    denominator = mx.maximum(mx.abs(reference_f32), atol)
    max_rel = float(mx.max(difference / denominator).item())
    if not passed:
        raise AssertionError(
            f"{case.name}/{dtype_name}: max_abs={max_abs} max_rel={max_rel} "
            f"outside atol={atol} rtol={rtol}"
        )
    return max_abs, max_rel, full_gap, tail_gap, atol, rtol


def main() -> int:
    if not _machine_safety_gate():
        return 2

    import mlx.core as mx
    import numpy as np

    if not _require_active_metal(mx):
        return 2

    import mtplx.kernels.qsa_prefill_flash as flash_module

    count = 0
    for dtype_name in FLASH_DTYPES:
        for case in FLASH_CASES:
            max_abs, max_rel, full_gap, tail_gap, atol, rtol = _run_case(
                mx,
                np,
                flash_module,
                case,
                dtype_name,
            )
            count += 1
            print(
                f"PASS flash dtype={dtype_name} S={case.rows} "
                f"T={case.total_tokens} tail={case.expected_final_tail} "
                f"layout={case.kv_layout} "
                f"selection={case.selection_layout} "
                f"max_abs={max_abs:.9g} max_rel={max_rel:.9g} "
                f"atol={atol:g} rtol={rtol:g} full_gap={full_gap:.9g} "
                f"tail_gap={tail_gap:.9g}",
                flush=True,
            )

    print(f"QSA PREFILL FLASH EXACTNESS PASS cases={count}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
