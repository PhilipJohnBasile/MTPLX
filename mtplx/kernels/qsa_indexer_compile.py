"""Retrace-bounded compiled core for the Qwen sparse-attention indexer.

``mx.compile`` only observes array arguments.  Closing over ``QSACache`` and
mutating its Python ``offset``/``pooled_len`` fields therefore records their
first-trace values as constants.  That looks correct on the first decode step
and is stale on replay.  This module keeps the compiled boundary pure instead:
every moving frontier and every mutable cache buffer is an explicit array
input, and every updated buffer/frontier is an explicit output.

The core owns the complete shape-stable indexer suffix::

    hidden -> optional Q/K projection --+
                                      split Q/K -> Q norm/RoPE
    precomputed Q/K rows -------------+       |
                                              +-> fused exact selector
    raw backing -> dynamic slice_update(K) -> pooled block stage/update -+

The projection is skipped only when the attention layer already produced the
same Q/K rows as part of its shared quantized projection.  Query preparation
and pooled-block staging call the dedicated Metal helpers directly.  Their
norm weights and RoPE frequencies are explicit graph inputs, while this graph
manager remains independent of the model module (and cannot form an import
cycle).

Input *values* ``pos_start``, ``total_tokens``, ``logical_blocks`` and
``pooled_len`` are one-element int32 arrays.  They can change on every call
without retracing.  Shapes, dtypes, selector mode, and backing/output
capacities are structural and receive distinct compiled entries.  Array
layout is not part of the Python key: MLX 0.32 does not expose strides there,
its own compile cache keys arrays by shape and dtype, and custom kernels bind
the actual C++ strides on every dispatch.  Backing capacities are power-of-two
buckets, so a 256-token cache does not create a new graph at every subsequent
256-token growth step.

There is deliberately no execution-error fallback here.  Callers may route an
unbucketed/restored cache to the eager path before entering this module, but a
failed compiled/custom-kernel dispatch remains visible instead of silently
turning an A/B into control-versus-control.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Literal, NamedTuple

import mlx.core as mx

from mtplx.kernels.qsa_indexer_prefill import (
    DEFAULT_PREFILL_SCORE_WORKSPACE_BYTES,
    qsa_indexer_prefill_blocks_metal,
    qsa_indexer_prefill_prepared_scores_mpp_supported,
    qsa_indexer_prefill_score_chunk_rows,
)
from mtplx.kernels.qsa_indexer_prepare import (
    qsa_indexer_pool_keys_metal,
    qsa_indexer_prepare_queries_metal,
)
from mtplx.kernels.qsa_indexer_select import (
    qsa_indexer_select_blocks_metal,
    qsa_indexer_select_dense_mask_metal,
    qsa_indexer_select_row_tokens_metal,
)
from mtplx.qsa_mtp_precompute import (
    qsa_indexer_capacity_bucket,
    qsa_indexer_is_bucket_capacity,
)

QSACompiledMode = Literal[
    "blocks",
    "prefill_blocks",
    "dense_mask",
    "row_tokens",
    "update_only",
]
QSACompiledSource = Literal["hidden", "qk_rows"]

ProjectQK = Callable[[mx.array], mx.array]
CompileFactory = Callable[[Callable[..., Any]], Callable[..., Any]]


__all__ = [
    "QSACompileCapacityError",
    "QSACompiledIndexerCore",
    "QSACompiledIndexerResult",
    "QSACompiledMode",
    "qsa_indexer_capacity_bucket",
    "qsa_indexer_dense_output_capacity",
    "qsa_indexer_is_bucket_capacity",
    "qsa_indexer_selector_chunk_rows",
]


class QSACompileCapacityError(ValueError):
    """A mutable indexer leaf does not have a stable bucketed shape."""


class QSACompiledIndexerResult(NamedTuple):
    """One compiled indexer result plus its explicitly threaded state.

    ``selection`` has the existing selector contract for the requested mode:

    * ``blocks``: ``(ids, validity, adjusted_scores)`` from the decode-oriented
      fused selector;
    * ``prefill_blocks``: the same leaves from the production tiled MPP score
      kernel (or statically selected general MLX producer) plus dedicated
      Metal row-top-k prefill backend;
    * ``dense_mask``: a capacity-width bool mask (the caller slices to its
      host-known logical total after the compiled call);
    * ``row_tokens``: ``(token_ids, validity)``.
    * ``update_only``: ``None``; projection, raw cache update, and completed
      pooled-block staging still execute, while Q preparation/selection are
      omitted for the dense-equals-sparse prefix.

    ``raw_keys`` and ``pooled`` replace the corresponding cache leaves.
    ``pooled_len`` and ``offset`` are one-element int32 arrays and can feed the
    next compiled call directly.  A Python cache owner may instead commit its
    already-known host integer values outside the traced region.
    """

    selection: mx.array | tuple[mx.array, ...] | None
    raw_keys: mx.array
    pooled: mx.array
    pooled_len: mx.array
    offset: mx.array


@dataclass(frozen=True)
class _CompileKey:
    source: QSACompiledSource
    mode: QSACompiledMode
    input_shape: tuple[int, ...]
    input_dtype: str
    raw_shape: tuple[int, ...]
    raw_dtype: str
    pooled_shape: tuple[int, ...]
    pooled_dtype: str
    n_heads: int
    kv_heads: int
    head_dim: int
    block_topk: int
    compress_ratio: int
    dense_output_tokens: int
    prefill_score_producer: str
    selector_chunk_rows: int

    @property
    def rows(self) -> int:
        return self.input_shape[1]

    @property
    def label(self) -> str:
        return (
            f"{self.source}:{self.mode}:s{self.rows}:"
            f"raw{self.raw_shape[1]}:pool{self.pooled_shape[1]}:"
            f"out{self.dense_output_tokens}:score{self.prefill_score_producer}:"
            f"chunk{self.selector_chunk_rows}:"
            f"{self.input_dtype}/{self.pooled_dtype}"
        )


def qsa_indexer_dense_output_capacity(
    pooled_capacity: int,
    compress_ratio: int,
) -> int:
    """Static dense-mask width safe for one full pooled backing bucket.

    A full bucket can coexist with an incomplete visible tail of at most
    ``ratio - 1`` tokens.  One extra block is the smallest stable capacity
    that covers that tail without specializing on the current token count.
    """

    blocks = int(pooled_capacity)
    ratio = int(compress_ratio)
    if blocks <= 0:
        raise ValueError(f"pooled_capacity must be positive, got {blocks}")
    if ratio <= 0:
        raise ValueError(f"compress_ratio must be positive, got {ratio}")
    return (blocks + 1) * ratio


def qsa_indexer_selector_chunk_rows(
    rows: int,
    pooled_capacity: int,
    scratch_bytes: int,
) -> int:
    """Rows per selector dispatch under a float32 score-scratch ceiling."""

    query_rows = int(rows)
    blocks = int(pooled_capacity)
    budget = int(scratch_bytes)
    if query_rows <= 0 or blocks <= 0 or budget <= 0:
        raise ValueError("rows, pooled_capacity, and scratch_bytes must be positive")
    return min(query_rows, max(1, budget // (blocks * 4)))


def _i32_scalar(value: int | mx.array, name: str) -> mx.array:
    """Normalize a dynamic scalar to one stable ``[1]`` int32 signature."""

    if isinstance(value, mx.array):
        if value.dtype != mx.int32 or int(value.size) != 1:
            raise TypeError(f"{name} must be one int32 value")
        return value.reshape((1,))
    return mx.array([int(value)], dtype=mx.int32)


def _host_int(value: int | mx.array) -> int | None:
    return None if isinstance(value, mx.array) else int(value)


class QSACompiledIndexerCore:
    """Bank of pure, fixed-signature compiled QSA indexer graphs.

    Parameters are structural and immutable for one QSA indexer instance.
    The dedicated preparation kernels perform Q RMSNorm + partial RoPE and
    stage completed mean->norm->RoPE key blocks.  Their norm weights and RoPE
    frequencies are passed to every compiled graph as explicit array leaves;
    the model-static YaRN attention scale is a structural kernel constant.

    ``project_qk`` is optional because attention can supply shared projected
    Q/K rows.  When present, :meth:`select_hidden` traces it inside the graph;
    :meth:`select_qk_rows` always treats projected rows as an explicit input.
    The two sources receive separate compiled functions so an optional input
    never changes function arity and triggers an accidental retrace.
    """

    def __init__(
        self,
        *,
        n_heads: int,
        kv_heads: int,
        head_dim: int,
        block_topk: int,
        compress_ratio: int,
        q_norm_weight: mx.array,
        k_norm_weight: mx.array,
        inv_freq: mx.array,
        rms_norm_eps: float,
        rope_attention_scaling: float = 1.0,
        project_qk: ProjectQK | None = None,
        minimum_raw_capacity: int = 256,
        minimum_pooled_capacity: int = 256,
        require_bucketed_backing: bool = True,
        selector_scratch_bytes: int = 32 * 1024 * 1024,
        prefill_score_workspace_bytes: int = DEFAULT_PREFILL_SCORE_WORKSPACE_BYTES,
        compile_factory: CompileFactory = mx.compile,
    ) -> None:
        self.n_heads = int(n_heads)
        self.kv_heads = int(kv_heads)
        self.head_dim = int(head_dim)
        self.block_topk = int(block_topk)
        self.compress_ratio = int(compress_ratio)
        if self.n_heads <= 0 or self.head_dim <= 0:
            raise ValueError("n_heads and head_dim must be positive")
        # v2.10's QSA raw-key/cache contract is [1,T,D].  Accepting multiple
        # KV heads here would flatten a different model contract silently.
        if self.kv_heads != 1:
            raise ValueError(
                "compiled QSA indexer requires the v2.10 single indexer KV head"
            )
        if not (1 <= self.block_topk <= 512):
            raise ValueError("block_topk must be in [1,512]")
        if self.compress_ratio <= 0:
            raise ValueError("compress_ratio must be positive")
        if project_qk is not None and not callable(project_qk):
            raise TypeError("project_qk must be callable when provided")
        for name, weight in (
            ("q_norm_weight", q_norm_weight),
            ("k_norm_weight", k_norm_weight),
        ):
            if weight.ndim != 1 or int(weight.shape[0]) != self.head_dim:
                raise ValueError(
                    f"{name} must have shape [{self.head_dim}], got {weight.shape}"
                )
        if inv_freq.ndim != 1 or inv_freq.dtype != mx.float32:
            raise TypeError("inv_freq must be a one-dimensional float32 array")
        rotary_dim = 2 * int(inv_freq.shape[0])
        if rotary_dim <= 0 or rotary_dim > self.head_dim or rotary_dim % 2:
            raise ValueError("2*len(inv_freq) must be positive, even, and <= head_dim")
        # Validate the bucket floors once.
        qsa_indexer_capacity_bucket(1, minimum=minimum_raw_capacity)
        qsa_indexer_capacity_bucket(1, minimum=minimum_pooled_capacity)

        self._q_norm_weight = q_norm_weight
        self._k_norm_weight = k_norm_weight
        self._inv_freq = inv_freq
        self._rms_norm_eps = float(rms_norm_eps)
        self._rope_attention_scaling = float(rope_attention_scaling)
        if (
            not math.isfinite(self._rope_attention_scaling)
            or self._rope_attention_scaling <= 0.0
        ):
            raise ValueError(
                "rope_attention_scaling must be finite and positive; "
                f"got {self._rope_attention_scaling}"
            )
        self._project_qk = project_qk
        self._minimum_raw_capacity = int(minimum_raw_capacity)
        self._minimum_pooled_capacity = int(minimum_pooled_capacity)
        self._require_bucketed_backing = bool(require_bucketed_backing)
        self._selector_scratch_bytes = int(selector_scratch_bytes)
        if self._selector_scratch_bytes <= 0:
            raise ValueError("selector_scratch_bytes must be positive")
        self._prefill_score_workspace_bytes = int(prefill_score_workspace_bytes)
        if self._prefill_score_workspace_bytes <= 0:
            raise ValueError("prefill_score_workspace_bytes must be positive")
        self._compile_factory = compile_factory
        self._compiled: dict[_CompileKey, Callable[..., tuple[mx.array, ...]]] = {}
        self._last_capacity: tuple[int, int] | None = None
        self.stats: dict[str, Any] = {
            "calls": 0,
            "compiled_calls": 0,
            "hidden_calls": 0,
            "qk_rows_calls": 0,
            "traces": 0,
            "selector_dispatches": 0,
            "capacity_transitions": 0,
            "prefill_score_producers": {"mpp": 0, "mlx": 0},
            "modes": {
                "blocks": 0,
                "prefill_blocks": 0,
                "dense_mask": 0,
                "row_tokens": 0,
                "update_only": 0,
            },
            "buckets": {},
        }

    def _validate_backings(
        self,
        raw_keys: mx.array,
        pooled: mx.array,
        rows: int,
    ) -> None:
        if raw_keys.ndim != 3 or tuple(raw_keys.shape[:1]) != (1,):
            raise ValueError(
                f"raw_keys must have shape [1,capacity,D], got {raw_keys.shape}"
            )
        if pooled.ndim != 3 or tuple(pooled.shape[:1]) != (1,):
            raise ValueError(
                f"pooled must have shape [1,capacity,D], got {pooled.shape}"
            )
        if int(raw_keys.shape[2]) != self.head_dim:
            raise ValueError(
                f"raw key dim must be {self.head_dim}, got {raw_keys.shape[2]}"
            )
        if int(pooled.shape[2]) != self.head_dim:
            raise ValueError(
                f"pooled key dim must be {self.head_dim}, got {pooled.shape[2]}"
            )
        raw_capacity = int(raw_keys.shape[1])
        pooled_capacity = int(pooled.shape[1])
        if rows <= 0:
            raise ValueError(f"query rows must be positive, got {rows}")
        if raw_capacity < rows:
            raise QSACompileCapacityError(
                f"raw capacity {raw_capacity} is smaller than query rows {rows}"
            )
        if pooled_capacity <= 0:
            raise QSACompileCapacityError("pooled backing must be non-empty")
        max_new_blocks = (rows + self.compress_ratio - 1) // self.compress_ratio
        if raw_capacity // self.compress_ratio < max_new_blocks:
            raise QSACompileCapacityError(
                "raw backing cannot hold the fixed completed-block staging "
                f"window: raw={raw_capacity}, rows={rows}, "
                f"ratio={self.compress_ratio}"
            )
        if pooled_capacity < max_new_blocks:
            raise QSACompileCapacityError(
                "pooled backing cannot hold the fixed completed-block staging "
                f"window: pooled={pooled_capacity}, blocks={max_new_blocks}"
            )
        if self._require_bucketed_backing:
            if not qsa_indexer_is_bucket_capacity(
                raw_capacity,
                minimum=self._minimum_raw_capacity,
            ):
                raise QSACompileCapacityError(
                    f"raw capacity {raw_capacity} is not a power-of-two bucket"
                )
            if not qsa_indexer_is_bucket_capacity(
                pooled_capacity,
                minimum=self._minimum_pooled_capacity,
            ):
                raise QSACompileCapacityError(
                    f"pooled capacity {pooled_capacity} is not a power-of-two bucket"
                )

    def _key(
        self,
        source: QSACompiledSource,
        mode: QSACompiledMode,
        source_value: mx.array,
        raw_keys: mx.array,
        pooled: mx.array,
        dense_output_tokens: int,
    ) -> _CompileKey:
        prefill_score_producer = "none"
        if mode == "update_only":
            selector_chunk_rows = 0
        elif mode == "prefill_blocks":
            use_mpp_scores = qsa_indexer_prefill_prepared_scores_mpp_supported(
                int(source_value.shape[1]),
                self.n_heads,
                self.head_dim,
                source_value.dtype,
                pooled,
            )
            prefill_score_producer = "mpp" if use_mpp_scores else "mlx"
            selector_chunk_rows = qsa_indexer_prefill_score_chunk_rows(
                int(source_value.shape[1]),
                self.n_heads,
                int(pooled.shape[1]),
                self._prefill_score_workspace_bytes,
                producer=prefill_score_producer,
            )
        else:
            selector_chunk_rows = qsa_indexer_selector_chunk_rows(
                int(source_value.shape[1]),
                int(pooled.shape[1]),
                self._selector_scratch_bytes,
            )
        return _CompileKey(
            source=source,
            mode=mode,
            input_shape=tuple(int(dim) for dim in source_value.shape),
            input_dtype=str(source_value.dtype),
            raw_shape=tuple(int(dim) for dim in raw_keys.shape),
            raw_dtype=str(raw_keys.dtype),
            pooled_shape=tuple(int(dim) for dim in pooled.shape),
            pooled_dtype=str(pooled.dtype),
            n_heads=self.n_heads,
            kv_heads=self.kv_heads,
            head_dim=self.head_dim,
            block_topk=self.block_topk,
            compress_ratio=self.compress_ratio,
            dense_output_tokens=int(dense_output_tokens),
            prefill_score_producer=prefill_score_producer,
            selector_chunk_rows=selector_chunk_rows,
        )

    def _selector(
        self,
        mode: QSACompiledMode,
        dense_output_tokens: int,
        rows: int,
        chunk_rows: int,
    ) -> Callable[..., tuple[mx.array, ...]]:
        common = {
            "block_topk": self.block_topk,
            "compress_ratio": self.compress_ratio,
        }
        if mode == "prefill_blocks":

            def select(q, pooled, pos_start, total_tokens, logical_blocks):
                return qsa_indexer_prefill_blocks_metal(
                    q,
                    pooled,
                    pos_start=pos_start,
                    total_tokens=total_tokens,
                    logical_blocks=logical_blocks,
                    score_workspace_bytes=self._prefill_score_workspace_bytes,
                    **common,
                )

            return select

        if mode == "blocks":

            def select_chunk(q, pooled, pos_start, total_tokens, logical_blocks):
                return qsa_indexer_select_blocks_metal(
                    q,
                    pooled,
                    pos_start=pos_start,
                    total_tokens=total_tokens,
                    logical_blocks=logical_blocks,
                    **common,
                )

        elif mode == "row_tokens":

            def select_chunk(q, pooled, pos_start, total_tokens, logical_blocks):
                return qsa_indexer_select_row_tokens_metal(
                    q,
                    pooled,
                    pos_start=pos_start,
                    total_tokens=total_tokens,
                    logical_blocks=logical_blocks,
                    **common,
                )

        elif mode == "dense_mask":

            def select_chunk(q, pooled, pos_start, total_tokens, logical_blocks):
                mask = qsa_indexer_select_dense_mask_metal(
                    q,
                    pooled,
                    pos_start=pos_start,
                    total_tokens=total_tokens,
                    logical_blocks=logical_blocks,
                    output_total_tokens=dense_output_tokens,
                    **common,
                )
                return (mask,)

        else:
            raise ValueError(f"unknown compiled QSA mode {mode!r}")

        def select(q, pooled, pos_start, total_tokens, logical_blocks):
            chunks = []
            for row_start in range(0, rows, chunk_rows):
                row_stop = min(rows, row_start + chunk_rows)
                chunks.append(
                    select_chunk(
                        q[:, row_start:row_stop],
                        pooled,
                        pos_start + row_start,
                        total_tokens,
                        logical_blocks,
                    )
                )
            if len(chunks) == 1:
                return chunks[0]
            axis = 2 if mode == "dense_mask" else 0
            return tuple(
                mx.concatenate([chunk[leaf] for chunk in chunks], axis=axis)
                for leaf in range(len(chunks[0]))
            )

        return select

    def _make_compiled(
        self,
        key: _CompileKey,
    ) -> Callable[..., tuple[mx.array, ...]]:
        select = (
            None
            if key.mode == "update_only"
            else self._selector(
                key.mode,
                key.dense_output_tokens,
                key.rows,
                key.selector_chunk_rows,
            )
        )
        q_width = self.n_heads * self.head_dim
        k_width = self.kv_heads * self.head_dim
        rows = key.rows
        ratio = self.compress_ratio
        eps = self._rms_norm_eps
        rope_attention_scaling = self._rope_attention_scaling
        max_new_blocks = (rows + ratio - 1) // ratio
        raw_block_capacity = key.raw_shape[1] // ratio
        max_pool_start = min(
            raw_block_capacity - max_new_blocks,
            key.pooled_shape[1] - max_new_blocks,
        )
        raw_window_shape = (1, max_new_blocks * ratio, self.head_dim)

        def from_qk_rows(
            qk_rows,
            raw_keys,
            pooled,
            q_norm_weight,
            k_norm_weight,
            inv_freq,
            pos_start,
            total_tokens,
            logical_blocks,
            pooled_len,
        ):
            k_raw = qk_rows[..., q_width : q_width + k_width].reshape(
                1,
                rows,
                self.head_dim,
            )
            raw_next = mx.slice_update(raw_keys, k_raw, pos_start, axes=(1,))

            # Recompute a fixed trailing window of complete blocks.  The
            # logical frontier is dynamic, while the helper's input/output
            # shapes remain fixed.  Near a backing boundary the start moves
            # backward and harmlessly recomputes already-valid blocks; this
            # avoids an out-of-bounds speculative write when no new block is
            # completed on a decode step.
            block_start = mx.maximum(logical_blocks - max_new_blocks, 0)
            block_start = mx.minimum(block_start, pooled_len)
            block_start = mx.minimum(block_start, max_pool_start)
            raw_start = block_start * ratio
            raw_window = mx.slice(
                raw_next,
                raw_start,
                axes=(1,),
                slice_size=raw_window_shape,
            )
            pooled_window = qsa_indexer_pool_keys_metal(
                raw_window,
                k_norm_weight,
                inv_freq,
                block_start=block_start,
                compress_ratio=ratio,
                eps=eps,
                attention_scaling=rope_attention_scaling,
            )
            pooled_next = mx.slice_update(
                pooled,
                pooled_window,
                block_start,
                axes=(1,),
            )
            if select is None:
                selected = ()
            else:
                q_raw = qk_rows[..., :q_width].reshape(
                    1,
                    rows,
                    self.n_heads,
                    self.head_dim,
                )
                q = qsa_indexer_prepare_queries_metal(
                    q_raw,
                    q_norm_weight,
                    inv_freq,
                    pos_start=pos_start,
                    eps=eps,
                    attention_scaling=rope_attention_scaling,
                )
                selected = select(
                    q,
                    pooled_next,
                    pos_start,
                    total_tokens,
                    logical_blocks,
                )
            return (
                *selected,
                raw_next,
                pooled_next,
                logical_blocks,
                total_tokens,
            )

        if key.source == "qk_rows":
            # ``shapeless=False`` is intentional.  Custom Metal output
            # geometry is shape-dependent; the bank controls the finite set
            # of signatures rather than asking shapeless tracing to guess.
            return self._compile_factory(from_qk_rows)

        project_qk = self._project_qk
        if project_qk is None:
            raise RuntimeError("hidden-source graph requested without project_qk")

        def from_hidden(
            hidden,
            raw_keys,
            pooled,
            q_norm_weight,
            k_norm_weight,
            inv_freq,
            pos_start,
            total_tokens,
            logical_blocks,
            pooled_len,
        ):
            qk_rows = project_qk(hidden)
            return from_qk_rows(
                qk_rows,
                raw_keys,
                pooled,
                q_norm_weight,
                k_norm_weight,
                inv_freq,
                pos_start,
                total_tokens,
                logical_blocks,
                pooled_len,
            )

        return self._compile_factory(from_hidden)

    def _dispatch(
        self,
        source: QSACompiledSource,
        source_value: mx.array,
        raw_keys: mx.array,
        pooled: mx.array,
        *,
        pos_start: int | mx.array,
        total_tokens: int | mx.array,
        logical_blocks: int | mx.array,
        pooled_len: int | mx.array,
        mode: QSACompiledMode,
        dense_output_tokens: int | None,
    ) -> QSACompiledIndexerResult:
        if source_value.ndim != 3 or int(source_value.shape[0]) != 1:
            raise ValueError(
                f"hidden/qk_rows must have shape [1,S,width], got {source_value.shape}"
            )
        rows = int(source_value.shape[1])
        if source == "qk_rows":
            expected_width = (self.n_heads + self.kv_heads) * self.head_dim
            if int(source_value.shape[2]) != expected_width:
                raise ValueError(
                    f"qk_rows width must be {expected_width}, "
                    f"got {source_value.shape[2]}"
                )
            if source_value.dtype != raw_keys.dtype:
                raise TypeError(
                    "qk_rows and raw_keys must have the same dtype for slice_update"
                )
        elif self._project_qk is None:
            raise RuntimeError("select_hidden requires project_qk")
        if raw_keys.dtype != pooled.dtype:
            raise TypeError(
                "raw_keys and pooled must share a dtype for exact pool staging"
            )
        expected_query_dtype = source_value.dtype
        if mode != "update_only" and self._q_norm_weight.dtype != expected_query_dtype:
            raise TypeError(
                "q_norm_weight must share the projected query dtype; got "
                f"{self._q_norm_weight.dtype} and {expected_query_dtype}"
            )
        if self._k_norm_weight.dtype != raw_keys.dtype:
            raise TypeError(
                "k_norm_weight must share the raw-key dtype; got "
                f"{self._k_norm_weight.dtype} and {raw_keys.dtype}"
            )

        if mode not in (
            "blocks",
            "prefill_blocks",
            "dense_mask",
            "row_tokens",
            "update_only",
        ):
            raise ValueError(f"unknown compiled QSA mode {mode!r}")
        if mode == "prefill_blocks" and rows <= 1:
            raise ValueError("compiled prefill_blocks mode requires S > 1")
        self._validate_backings(raw_keys, pooled, rows)
        host_pos = _host_int(pos_start)
        host_total = _host_int(total_tokens)
        host_logical = _host_int(logical_blocks)
        host_pooled_len = _host_int(pooled_len)
        if host_pos is not None:
            if host_pos < 0:
                raise ValueError(f"pos_start must be non-negative, got {host_pos}")
            if host_pos + rows > int(raw_keys.shape[1]):
                raise QSACompileCapacityError(
                    "raw backing must be reserved before compiled dispatch: "
                    f"pos_start={host_pos}, rows={rows}, "
                    f"capacity={raw_keys.shape[1]}"
                )
        if host_total is not None:
            if host_total < 0:
                raise ValueError(f"total_tokens must be non-negative, got {host_total}")
            if host_pos is not None and host_total != host_pos + rows:
                raise ValueError(
                    "total_tokens must equal pos_start + rows; got "
                    f"{host_total} != {host_pos} + {rows}"
                )
        if host_logical is not None:
            if not 0 <= host_logical <= int(pooled.shape[1]):
                raise QSACompileCapacityError(
                    "logical block frontier exceeds pooled backing: "
                    f"logical={host_logical}, capacity={pooled.shape[1]}"
                )
            if (
                host_total is not None
                and host_logical != host_total // self.compress_ratio
            ):
                raise ValueError(
                    "logical_blocks must equal total_tokens // compress_ratio"
                )
        if host_pooled_len is not None:
            if not 0 <= host_pooled_len <= int(pooled.shape[1]):
                raise QSACompileCapacityError(
                    "pooled_len is outside the pooled backing: "
                    f"pooled_len={host_pooled_len}, capacity={pooled.shape[1]}"
                )
            max_new_blocks = (rows + self.compress_ratio - 1) // self.compress_ratio
            if (
                host_logical is not None
                and host_logical - host_pooled_len > max_new_blocks
            ):
                raise QSACompileCapacityError(
                    "compiled pool staging window cannot repair a frontier gap: "
                    f"pooled_len={host_pooled_len}, logical={host_logical}, "
                    f"max_new={max_new_blocks}"
                )
        if dense_output_tokens is None:
            dense_capacity = (
                qsa_indexer_dense_output_capacity(
                    int(pooled.shape[1]),
                    self.compress_ratio,
                )
                if mode == "dense_mask"
                else 0
            )
        else:
            dense_capacity = int(dense_output_tokens)
        if mode == "dense_mask":
            minimum_dense = qsa_indexer_dense_output_capacity(
                int(pooled.shape[1]),
                self.compress_ratio,
            )
            if dense_capacity < minimum_dense:
                raise QSACompileCapacityError(
                    f"dense output capacity {dense_capacity} is smaller than "
                    f"the backing-safe width {minimum_dense}"
                )
        elif dense_capacity:
            raise ValueError("dense_output_tokens is only valid for dense_mask mode")

        pos_value = _i32_scalar(pos_start, "pos_start")
        total_value = _i32_scalar(total_tokens, "total_tokens")
        logical_value = _i32_scalar(logical_blocks, "logical_blocks")
        pooled_len_value = _i32_scalar(pooled_len, "pooled_len")
        key = self._key(
            source,
            mode,
            source_value,
            raw_keys,
            pooled,
            dense_capacity,
        )

        self.stats["calls"] += 1
        self.stats[f"{source}_calls"] += 1
        self.stats["modes"][mode] += 1
        if mode == "prefill_blocks":
            self.stats["prefill_score_producers"][key.prefill_score_producer] += 1
        if key.selector_chunk_rows:
            self.stats["selector_dispatches"] += (
                rows + key.selector_chunk_rows - 1
            ) // key.selector_chunk_rows
        capacity = (int(raw_keys.shape[1]), int(pooled.shape[1]))
        if self._last_capacity is not None and capacity != self._last_capacity:
            self.stats["capacity_transitions"] += 1
        self._last_capacity = capacity
        bucket_label = f"raw{capacity[0]}:pool{capacity[1]}:out{dense_capacity or 0}"
        buckets = self.stats["buckets"]
        buckets[bucket_label] = int(buckets.get(bucket_label, 0)) + 1

        compiled = self._compiled.get(key)
        if compiled is None:
            compiled = self._make_compiled(key)
            self._compiled[key] = compiled
            # One callable per complete MLX input signature.  Because the
            # entry is immediately invoked below, this is also the number of
            # first-trace requests.  No Python mutation occurs inside a
            # compiled body merely to count replays.
            self.stats["traces"] += 1
        flat = compiled(
            source_value,
            raw_keys,
            pooled,
            self._q_norm_weight,
            self._k_norm_weight,
            self._inv_freq,
            pos_value,
            total_value,
            logical_value,
            pooled_len_value,
        )
        self.stats["compiled_calls"] += 1

        selected_leaves = {
            "update_only": 0,
            "dense_mask": 1,
            "row_tokens": 2,
            "blocks": 3,
            "prefill_blocks": 3,
        }[mode]
        if mode == "update_only":
            selection = None
        elif mode == "dense_mask":
            selection = flat[0]
        else:
            selection = tuple(flat[:selected_leaves])
        raw_next, pooled_next, pooled_len_next, offset_next = flat[selected_leaves:]
        return QSACompiledIndexerResult(
            selection,
            raw_next,
            pooled_next,
            pooled_len_next,
            offset_next,
        )

    def select_qk_rows(
        self,
        qk_rows: mx.array,
        raw_keys: mx.array,
        pooled: mx.array,
        *,
        pos_start: int | mx.array,
        total_tokens: int | mx.array,
        logical_blocks: int | mx.array,
        pooled_len: int | mx.array,
        mode: QSACompiledMode,
        dense_output_tokens: int | None = None,
    ) -> QSACompiledIndexerResult:
        """Run the compiled core from a shared/precomputed Q/K projection."""

        return self._dispatch(
            "qk_rows",
            qk_rows,
            raw_keys,
            pooled,
            pos_start=pos_start,
            total_tokens=total_tokens,
            logical_blocks=logical_blocks,
            pooled_len=pooled_len,
            mode=mode,
            dense_output_tokens=dense_output_tokens,
        )

    def select_hidden(
        self,
        hidden: mx.array,
        raw_keys: mx.array,
        pooled: mx.array,
        *,
        pos_start: int | mx.array,
        total_tokens: int | mx.array,
        logical_blocks: int | mx.array,
        pooled_len: int | mx.array,
        mode: QSACompiledMode,
        dense_output_tokens: int | None = None,
    ) -> QSACompiledIndexerResult:
        """Run projection and the complete indexer core in one compiled graph."""

        return self._dispatch(
            "hidden",
            hidden,
            raw_keys,
            pooled,
            pos_start=pos_start,
            total_tokens=total_tokens,
            logical_blocks=logical_blocks,
            pooled_len=pooled_len,
            mode=mode,
            dense_output_tokens=dense_output_tokens,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serializable engagement and structural-trace receipt."""

        report = {
            key: (dict(value) if isinstance(value, dict) else value)
            for key, value in self.stats.items()
        }
        report["entry_count"] = len(self._compiled)
        report["compiled_keys"] = [
            {**asdict(key), "label": key.label}
            for key in sorted(
                self._compiled,
                key=lambda item: item.label,
            )
        ]
        report["require_bucketed_backing"] = self._require_bucketed_backing
        report["minimum_raw_capacity"] = self._minimum_raw_capacity
        report["minimum_pooled_capacity"] = self._minimum_pooled_capacity
        report["selector_scratch_bytes"] = self._selector_scratch_bytes
        report["prefill_score_workspace_bytes"] = self._prefill_score_workspace_bytes
        report["rope_attention_scaling"] = self._rope_attention_scaling
        return report
