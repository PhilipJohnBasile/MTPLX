"""Operator-controlled numeric gate for the large-S QSA indexer backend.

This script compares the complete public prefill path -- its byte-budgeted
float32 score chunks and dedicated Metal top-k -- with the retained v2.10.0
eager expression.  It loads no model weights.  IDs and validity must be
bit-exact; only float32 score arithmetic has a small rounding allowance.

Run this only when the operator has released the GPU::

    uv run --no-project --python 3.13 --with mlx python \
      scripts/qsa_indexer_prefill_numeric_check.py
"""

from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

HEADS = 4
HEAD_DIM = 128
BLOCK_TOPK = 512
COMPRESS_RATIO = 4
SCORE_ATOL = 1.0e-4


@dataclass(frozen=True)
class IndexerCase:
    name: str
    seed: int
    rows: int
    total_tokens: int
    dtype_name: str
    forced_chunk_rows: int
    zero_tie: bool = False
    positive_tie: bool = False


# The first two cases straddle the model's dense-equals-sparse short circuit.
# The next four put the final query at every compress-ratio tail position.  The
# The radix-boundary case crosses 2,048 historical blocks.  The canonical
# 2,048-row production query chunk is deliberately split into sixteen score
# workspaces; final fixtures cover a rounded-away positive tie and the full
# million-token/250,000-block and maximum 1,048,576-token/262,144-block
# production context geometries.
INDEXER_CASES = (
    IndexerCase("dense_boundary", 11, 2, 2048, "float16", 1),
    IndexerCase("boundary_plus_one", 23, 3, 2049, "bfloat16", 1),
    IndexerCase("first_sparse_tail0", 37, 7, 2052, "float32", 2),
    IndexerCase("first_sparse_tail1", 41, 33, 2053, "float16", 3),
    IndexerCase("first_sparse_tail2", 53, 65, 2054, "bfloat16", 4),
    IndexerCase("first_sparse_tail3", 67, 129, 2055, "float32", 32),
    IndexerCase("deeper_sparse", 71, 257, 2304, "bfloat16", 32),
    IndexerCase("selector_radix_boundary", 73, 17, 8196, "float32", 4),
    IndexerCase("large_prefill_chunk", 79, 2048, 4096, "float16", 128),
    IndexerCase("exact_zero_ties", 83, 5, 2056, "float16", 1, True),
    IndexerCase(
        "rounded_positive_ties",
        89,
        3,
        8196,
        "float16",
        1,
        positive_tie=True,
    ),
    IndexerCase("legacy_262k_random", 97, 2, 262144, "bfloat16", 1),
    IndexerCase("half_million_zero_ties", 101, 2, 500_000, "float16", 1, True),
    IndexerCase("binary_512k_zero_ties", 103, 2, 524_288, "float16", 1, True),
    IndexerCase("one_million_random", 107, 2, 1_000_000, "bfloat16", 1),
    IndexerCase(
        "max_context_zero_ties",
        109,
        2,
        1_048_576,
        "float16",
        1,
        True,
    ),
)


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
        print("QSA INDEXER PREFILL REFUSE active MLX device is not Metal", flush=True)
    return active


def _dtype(mx, name: str):
    return {
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
        "float32": mx.float32,
    }[name]


def _fixture(mx, case: IndexerCase):
    logical_blocks = case.total_tokens // COMPRESS_RATIO
    dtype = _dtype(mx, case.dtype_name)
    if case.zero_tie:
        q = mx.zeros((1, case.rows, HEADS, HEAD_DIM), dtype=dtype)
        pooled = mx.zeros((1, logical_blocks, HEAD_DIM), dtype=dtype)
    elif case.positive_tie:
        # At this score magnitude block*1e-12 is below one float32 ULP.  The
        # adjusted scores therefore remain equal and v2.10's stable ascending
        # ArgPartition followed by a final-K slice retains the later IDs.
        q = mx.ones((1, case.rows, HEADS, HEAD_DIM), dtype=dtype)
        pooled = mx.ones((1, logical_blocks, HEAD_DIM), dtype=dtype)
    else:
        mx.random.seed(case.seed)
        q = mx.contiguous(
            mx.random.normal((1, case.rows, HEADS, HEAD_DIM)).astype(dtype)
        )
        pooled = mx.contiguous(
            mx.random.normal((1, logical_blocks, HEAD_DIM)).astype(dtype)
        )
    return q, pooled


def _eager_oracle(mx, q, pooled, *, pos_start: int):
    """Mirror ``QSAIndexer._select_eager`` and its compact block epilogue."""

    rows = int(q.shape[1])
    head_dim = int(q.shape[3])
    blocks = int(pooled.shape[1])

    pooled_t = mx.swapaxes(pooled.astype(mx.float32), 1, 2)[:, None]
    per_head = mx.matmul(q.astype(mx.float32), pooled_t)
    raw_scores = mx.maximum(per_head, 0.0).sum(axis=2) / math.sqrt(head_dim)
    raw_scores = raw_scores[0]

    qpos = mx.arange(pos_start, pos_start + rows, dtype=mx.int32)
    complete = (qpos + 1) // COMPRESS_RATIO
    block = mx.arange(blocks, dtype=mx.int32)
    visible = block[None, :] < complete[:, None]
    adjusted = mx.where(
        visible,
        raw_scores,
        mx.array(-mx.inf, dtype=mx.float32),
    )
    # Load-bearing v2.10.0 tie semantics: the lowest block index wins an
    # exact score tie, including the common all-zero post-ReLU case.
    adjusted = adjusted - block.astype(mx.float32)[None, :] * 1.0e-12

    k_eff = min(BLOCK_TOPK, blocks)
    top_idx = mx.argpartition(
        adjusted,
        kth=blocks - k_eff,
        axis=-1,
    )[:, blocks - k_eff :].astype(mx.int32)
    top_valid = mx.take_along_axis(visible, top_idx.astype(mx.int64), axis=-1)
    top_scores = mx.take_along_axis(adjusted, top_idx.astype(mx.int64), axis=-1)

    if k_eff < BLOCK_TOPK:
        padding = BLOCK_TOPK - k_eff
        top_idx = mx.concatenate(
            [top_idx, mx.zeros((rows, padding), dtype=mx.int32)], axis=1
        )
        top_valid = mx.concatenate(
            [top_valid, mx.zeros((rows, padding), dtype=mx.bool_)], axis=1
        )
        top_scores = mx.concatenate(
            [
                top_scores,
                mx.full((rows, padding), -mx.inf, dtype=mx.float32),
            ],
            axis=1,
        )

    # The production prefill epilogue is chronological, with invalid padding
    # sorted last and canonicalized to id=0/score=-inf.
    sentinel = mx.array(2**31 - 1, dtype=mx.int32)
    order = mx.argsort(mx.where(top_valid, top_idx, sentinel), axis=-1)
    ids = mx.take_along_axis(top_idx, order, axis=-1)
    valid = mx.take_along_axis(top_valid, order, axis=-1)
    selected_scores = mx.take_along_axis(top_scores, order, axis=-1)
    ids = mx.where(valid, ids, mx.array(0, dtype=mx.int32))
    selected_scores = mx.where(
        valid,
        selected_scores,
        mx.array(-mx.inf, dtype=mx.float32),
    )
    return raw_scores, ids, valid, selected_scores


def _max_abs(mx, actual, expected) -> float:
    if int(actual.size) == 0:
        return 0.0
    return float(mx.max(mx.abs(actual - expected)).item())


def _selected_score_max_abs(mx, actual, expected, valid) -> float:
    delta = mx.where(valid, mx.abs(actual - expected), 0.0)
    maximum = 0.0 if int(delta.size) == 0 else float(mx.max(delta).item())
    actual_rows = actual.tolist()
    valid_rows = valid.tolist()
    for score_row, valid_row in zip(actual_rows, valid_rows, strict=True):
        for score, is_valid in zip(score_row, valid_row, strict=True):
            if not is_valid and not (math.isinf(score) and score < 0.0):
                raise AssertionError("invalid selected score is not -inf")
    return maximum


def _expected_chunk_widths(rows: int, chunk_rows: int) -> list[int]:
    return [min(chunk_rows, rows - start) for start in range(0, rows, chunk_rows)]


def _run_case(
    mx,
    prefill_module,
    case: IndexerCase,
) -> tuple[float, float, list[int], list[str]]:
    q, pooled = _fixture(mx, case)
    logical_blocks = int(pooled.shape[1])
    pos_start = case.total_tokens - case.rows
    if pos_start < 0:
        raise AssertionError(f"invalid fixture {case.name}: negative pos_start")

    expected_path = "fallback" if case.dtype_name == "float32" else "mpp"
    producer = "mlx" if expected_path == "fallback" else "mpp"
    score_planes = 1 if producer == "mpp" else HEADS + 1
    bytes_per_row = logical_blocks * 4 * score_planes
    workspace_bytes = bytes_per_row * case.forced_chunk_rows
    planned_chunk = prefill_module.qsa_indexer_prefill_score_chunk_rows(
        case.rows,
        HEADS,
        logical_blocks,
        workspace_bytes,
        producer=producer,
    )
    if planned_chunk >= case.rows:
        raise AssertionError(f"fixture {case.name} did not force a workspace split")

    score_events = []
    original_mpp_scores = prefill_module.qsa_indexer_prefill_scores_mpp
    original_fallback_scores = prefill_module.qsa_indexer_prefill_scores

    def recording_mpp_scores(q_chunk, pooled_chunk):
        scores = original_mpp_scores(q_chunk, pooled_chunk)
        score_events.append(("mpp", scores))
        return scores

    def recording_fallback_scores(q_chunk, pooled_t_float32, *, head_dim):
        scores = original_fallback_scores(
            q_chunk,
            pooled_t_float32,
            head_dim=head_dim,
        )
        score_events.append(("fallback", scores))
        return scores

    prefill_module.qsa_indexer_prefill_scores_mpp = recording_mpp_scores
    prefill_module.qsa_indexer_prefill_scores = recording_fallback_scores
    try:
        actual_ids, actual_valid, actual_selected_scores = (
            prefill_module.qsa_indexer_prefill_blocks_metal(
                q,
                pooled,
                pos_start=pos_start,
                total_tokens=case.total_tokens,
                block_topk=BLOCK_TOPK,
                compress_ratio=COMPRESS_RATIO,
                logical_blocks=logical_blocks,
                score_workspace_bytes=workspace_bytes,
            )
        )
    finally:
        prefill_module.qsa_indexer_prefill_scores_mpp = original_mpp_scores
        prefill_module.qsa_indexer_prefill_scores = original_fallback_scores

    observed_paths = [path for path, _ in score_events]
    recorded_scores = [scores for _, scores in score_events]
    observed_widths = [int(scores.shape[0]) for scores in recorded_scores]
    expected_widths = _expected_chunk_widths(case.rows, planned_chunk)
    if observed_widths != expected_widths:
        raise AssertionError(
            f"{case.name}: score chunks {observed_widths} != {expected_widths}"
        )
    if observed_paths != [expected_path] * len(expected_widths):
        raise AssertionError(
            f"{case.name}: score paths {observed_paths} did not use "
            f"the required {expected_path!r} producer for every chunk"
        )
    actual_raw_scores = mx.concatenate(recorded_scores, axis=0)
    expected_raw_scores, expected_ids, expected_valid, expected_selected_scores = (
        _eager_oracle(mx, q, pooled, pos_start=pos_start)
    )
    mx.eval(
        actual_raw_scores,
        actual_ids,
        actual_valid,
        actual_selected_scores,
        expected_raw_scores,
        expected_ids,
        expected_valid,
        expected_selected_scores,
    )

    if actual_ids.tolist() != expected_ids.tolist():
        raise AssertionError(f"{case.name}: selected block IDs differ")
    if actual_valid.tolist() != expected_valid.tolist():
        raise AssertionError(f"{case.name}: selected block validity differs")

    raw_max_abs = _max_abs(mx, actual_raw_scores, expected_raw_scores)
    selected_max_abs = _selected_score_max_abs(
        mx,
        actual_selected_scores,
        expected_selected_scores,
        expected_valid,
    )
    if raw_max_abs > SCORE_ATOL:
        raise AssertionError(
            f"{case.name}: raw score max abs {raw_max_abs} > {SCORE_ATOL}"
        )
    if selected_max_abs > SCORE_ATOL:
        raise AssertionError(
            f"{case.name}: selected score max abs {selected_max_abs} > {SCORE_ATOL}"
        )

    if case.zero_tie:
        exact_ids = list(range(BLOCK_TOPK))
        expected_rows = [exact_ids for _ in range(case.rows)]
        if actual_ids.tolist() != expected_rows:
            raise AssertionError("exact-zero tie did not select lowest block IDs")
        if actual_valid.tolist() != [[True] * BLOCK_TOPK for _ in range(case.rows)]:
            raise AssertionError("exact-zero tie unexpectedly produced invalid slots")

    if case.positive_tie:
        expected_last = list(range(logical_blocks - BLOCK_TOPK, logical_blocks))
        if actual_ids[-1].tolist() != expected_last:
            raise AssertionError(
                "rounded-away positive tie did not preserve the stable final-K cutoff"
            )

    return raw_max_abs, selected_max_abs, observed_widths, observed_paths


def main() -> int:
    if not _machine_safety_gate():
        return 2

    import mlx.core as mx

    if not _require_active_metal(mx):
        return 2

    import mtplx.kernels.qsa_indexer_prefill as prefill_module

    path_counts = {"mpp": 0, "fallback": 0}
    for case in INDEXER_CASES:
        raw_max_abs, selected_max_abs, chunks, score_paths = _run_case(
            mx,
            prefill_module,
            case,
        )
        for path in score_paths:
            path_counts[path] += 1
        print(
            f"PASS {case.name} seed={case.seed} dtype={case.dtype_name} "
            f"S={case.rows} T={case.total_tokens} chunks={chunks} "
            f"score_paths={score_paths} "
            f"raw_max_abs={raw_max_abs:.9g} "
            f"selected_max_abs={selected_max_abs:.9g}",
            flush=True,
        )

    if path_counts["mpp"] == 0 or path_counts["fallback"] == 0:
        raise AssertionError(
            f"anti-vacuity: both score producers must run: {path_counts}"
        )
    print(
        f"QSA INDEXER PREFILL EXACTNESS PASS cases={len(INDEXER_CASES)} "
        f"score_paths={path_counts}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
