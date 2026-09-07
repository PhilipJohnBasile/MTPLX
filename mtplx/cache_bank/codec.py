"""No-pickle tensor/tree codec for persistent SessionBank snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import numpy as np

from mtplx.cache_state import CacheSnapshot


_MLX_TO_NUMPY_DTYPE: dict[str, Any] = {
    "bool": np.bool_,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
}

_NUMPY_TO_MLX_DTYPE: dict[str, Any] = {
    "bool": mx.bool_,
    "uint8": mx.uint8,
    "uint16": mx.uint16,
    "uint32": mx.uint32,
    "uint64": mx.uint64,
    "int8": mx.int8,
    "int16": mx.int16,
    "int32": mx.int32,
    "int64": mx.int64,
    "float16": mx.float16,
    "float32": mx.float32,
    "float64": mx.float64,
}


@dataclass(frozen=True)
class EncodedPayload:
    spec: dict[str, Any]
    tensors: dict[str, bytes]
    nbytes: int


@dataclass(frozen=True)
class DecodedPayload:
    cache_snapshot: CacheSnapshot
    logits: Any
    hidden: Any | None
    mtp_history_snapshot: CacheSnapshot | None
    # v3: (token_count, recurrent-only CacheSnapshot, hidden_last|None)
    gdn_boundaries: tuple = ()
    has_recurrent: bool = False
    # When a cold prefix restore decodes only the attention-KV blocks that
    # precede a safe boundary, these fields describe the materialized prefix
    # lengths.  The SessionBank keeps the entry's original token identity;
    # these are strictly the lengths represented by the decoded snapshots.
    cache_snapshot_prefix_len: int | None = None
    mtp_history_snapshot_prefix_len: int | None = None


class ColdEncodeInterrupted(RuntimeError):
    """Raised between tensor evals when the encode's should_abort fires.

    The SSD cold-tier encode runs on the single model-owner thread and a
    16k-context entry is ~2.5 GB of eval+copy — long enough that an arriving
    foreground request would queue behind it (surfacing as unattributed
    prompt-state wall, 0.66-3.6 s in the 2026-08-06/07 receipts). Aborting at
    a tensor boundary bounds that collision to one tensor's eval; the caller
    re-dispatches the job for the next quiet window.
    """


class TreeCodec:
    """Flatten JSON-safe trees plus MLX arrays into raw tensor blobs."""

    def __init__(
        self,
        *,
        block_size: int = 256,
        should_abort: Callable[[], bool] | None = None,
    ) -> None:
        self._next_tensor_id = 0
        self.tensors: dict[str, bytes] = {}
        self.block_size = max(1, int(block_size))
        self.should_abort = should_abort

    def _check_abort(self) -> None:
        check = self.should_abort
        if check is None:
            return
        try:
            interrupted = bool(check())
        except Exception:
            return
        if interrupted:
            raise ColdEncodeInterrupted()

    def encode(self, value: Any) -> Any:
        if value is None:
            return {"kind": "none"}
        if isinstance(value, bool):
            return {"kind": "bool", "value": bool(value)}
        if isinstance(value, int):
            return {"kind": "int", "value": int(value)}
        if isinstance(value, float):
            return {"kind": "float", "value": float(value)}
        if isinstance(value, str):
            return {"kind": "str", "value": value}
        if isinstance(value, np.generic):
            return self.encode(value.item())
        if isinstance(value, Path):
            return {"kind": "str", "value": str(value)}
        if isinstance(value, mx.array):
            return self._encode_tensor(value)
        if isinstance(value, tuple):
            return {"kind": "tuple", "items": [self.encode(item) for item in value]}
        if isinstance(value, list):
            return {"kind": "list", "items": [self.encode(item) for item in value]}
        if isinstance(value, dict):
            items = []
            for key, item in value.items():
                if not isinstance(key, (str, int, float, bool)):
                    raise TypeError(f"unsupported dict key in SessionBank snapshot: {type(key)!r}")
                items.append([self.encode(key), self.encode(item)])
            return {"kind": "dict", "items": items}
        raise TypeError(f"unsupported SessionBank snapshot leaf: {type(value)!r}")

    def _encode_tensor(self, value: Any) -> dict[str, Any]:
        self._check_abort()
        mx.eval(value)
        dtype = _dtype_name(value.dtype)
        shape = [int(dim) for dim in value.shape]
        if len(shape) >= 3 and shape[2] >= self.block_size * 2:
            return self._encode_tensor_blocks(value, dtype=dtype, shape=shape)
        name = f"tensor_{self._next_tensor_id:08d}"
        self._next_tensor_id += 1
        raw = bytes(memoryview(value))
        self.tensors[name] = raw
        return {
            "kind": "tensor",
            "name": name,
            "dtype": dtype,
            "shape": [int(dim) for dim in value.shape],
            "nbytes": len(raw),
        }

    def _encode_tensor_blocks(
        self,
        value: Any,
        *,
        dtype: str,
        shape: list[int],
    ) -> dict[str, Any]:
        axis = 2
        blocks: list[dict[str, Any]] = []
        total = 0
        for start in range(0, shape[axis], self.block_size):
            self._check_abort()
            end = min(shape[axis], start + self.block_size)
            slices = [slice(None)] * len(shape)
            slices[axis] = slice(start, end)
            chunk = value[tuple(slices)]
            mx.eval(chunk)
            raw = bytes(memoryview(chunk))
            name = f"tensor_{self._next_tensor_id:08d}"
            self._next_tensor_id += 1
            self.tensors[name] = raw
            total += len(raw)
            blocks.append(
                {
                    "name": name,
                    "start": int(start),
                    "end": int(end),
                    "shape": [int(dim) for dim in chunk.shape],
                    "nbytes": len(raw),
                }
            )
        return {
            "kind": "tensor_blocks",
            "dtype": dtype,
            "shape": shape,
            "axis": axis,
            "block_size": self.block_size,
            "blocks": blocks,
            "nbytes": total,
        }


def decode_tree(spec: Any, read_tensor: Callable[[str], bytes]) -> Any:
    kind = spec.get("kind") if isinstance(spec, dict) else None
    if kind == "none":
        return None
    if kind == "bool":
        return bool(spec["value"])
    if kind == "int":
        return int(spec["value"])
    if kind == "float":
        return float(spec["value"])
    if kind == "str":
        return str(spec["value"])
    if kind == "tuple":
        return tuple(decode_tree(item, read_tensor) for item in spec.get("items", []))
    if kind == "list":
        return [decode_tree(item, read_tensor) for item in spec.get("items", [])]
    if kind == "dict":
        return {
            decode_tree(key, read_tensor): decode_tree(value, read_tensor)
            for key, value in spec.get("items", [])
        }
    if kind == "tensor":
        return _decode_tensor(spec, read_tensor)
    if kind == "tensor_blocks":
        return _decode_tensor_blocks(spec, read_tensor)
    raise ValueError(f"unsupported SessionBank payload spec kind: {kind!r}")


def build_payload_spec(
    codec: TreeCodec,
    *,
    cache_snapshot: CacheSnapshot,
    logits: Any,
    hidden: Any | None,
    mtp_history_snapshot: CacheSnapshot | None,
    gdn_boundaries: tuple | list | None = None,
    has_recurrent: bool | None = None,
) -> dict[str, Any]:
    """The payload spec structure, shared by the staged and streaming
    encoders (cold_tier.put_entry vs cold_tier.spill_entry) so the two can
    never drift into incompatible on-disk formats."""
    return {
        "cache_snapshot": {
            "states": codec.encode(cache_snapshot.states),
            "meta_states": codec.encode(cache_snapshot.meta_states),
        },
        "logits": codec.encode(logits),
        "hidden": codec.encode(hidden),
        "mtp_history_snapshot": (
            None
            if mtp_history_snapshot is None
            else {
                "states": codec.encode(mtp_history_snapshot.states),
                "meta_states": codec.encode(mtp_history_snapshot.meta_states),
            }
        ),
        # v3: interior recurrent boundaries make persisted entries usable for
        # sub-prefix (partial) restores on hybrid models after a restart.
        "gdn_boundaries": [
            {
                "tokens": int(record[0]),
                "states": codec.encode(record[1].states),
                "meta_states": codec.encode(record[1].meta_states),
                "hidden_last": codec.encode(
                    record[2] if len(record) > 2 else None
                ),
            }
            for record in (gdn_boundaries or [])
        ],
        "has_recurrent": bool(
            has_recurrent
            if has_recurrent is not None
            else bool(gdn_boundaries)
        ),
    }


def encode_payload(
    *,
    cache_snapshot: CacheSnapshot,
    logits: Any,
    hidden: Any | None,
    mtp_history_snapshot: CacheSnapshot | None,
    gdn_boundaries: tuple | list | None = None,
    has_recurrent: bool | None = None,
    block_size: int = 256,
    should_abort: Callable[[], bool] | None = None,
) -> EncodedPayload:
    codec = TreeCodec(block_size=block_size, should_abort=should_abort)
    spec = build_payload_spec(
        codec,
        cache_snapshot=cache_snapshot,
        logits=logits,
        hidden=hidden,
        mtp_history_snapshot=mtp_history_snapshot,
        gdn_boundaries=gdn_boundaries,
        has_recurrent=has_recurrent,
    )
    return EncodedPayload(
        spec=spec,
        tensors=dict(codec.tensors),
        nbytes=sum(len(raw) for raw in codec.tensors.values()),
    )


def decode_payload(
    spec: dict[str, Any],
    read_tensor: Callable[[str], bytes],
    *,
    include_gdn_boundaries: bool = True,
) -> DecodedPayload:
    cache_spec = spec["cache_snapshot"]
    cache_snapshot = CacheSnapshot(
        states=tuple(decode_tree(cache_spec["states"], read_tensor)),
        meta_states=tuple(decode_tree(cache_spec["meta_states"], read_tensor)),
    )
    mtp_spec = spec.get("mtp_history_snapshot")
    mtp_history_snapshot = None
    if mtp_spec is not None:
        mtp_history_snapshot = CacheSnapshot(
            states=tuple(decode_tree(mtp_spec["states"], read_tensor)),
            meta_states=tuple(decode_tree(mtp_spec["meta_states"], read_tensor)),
        )
    gdn_boundaries = (
        decode_gdn_boundaries(spec, read_tensor)
        if include_gdn_boundaries
        else ()
    )
    decoded = DecodedPayload(
        cache_snapshot=cache_snapshot,
        logits=decode_tree(spec["logits"], read_tensor),
        hidden=decode_tree(spec["hidden"], read_tensor),
        mtp_history_snapshot=mtp_history_snapshot,
        gdn_boundaries=gdn_boundaries,
        has_recurrent=bool(spec.get("has_recurrent", False)),
    )
    _eval_decoded_arrays(decoded)
    return decoded


def decode_payload_prefix(
    spec: dict[str, Any],
    read_tensor: Callable[[str], bytes],
    *,
    cache_prefix_len: int,
    mtp_history_prefix_len: int | None = None,
    boundary_prefix_len: int | None = None,
) -> DecodedPayload:
    """Decode only the SSD payload needed for a sub-prefix restore.

    The cold tier stores attention tensors in fixed token blocks.  A normal
    ``decode_payload`` reconstructs every block in a long snapshot, even when
    a divergent follow-up can restore only a short recurrent-safe boundary.
    This variant reads only the attention blocks through that boundary and,
    when needed, the single matching recurrent boundary.  Non-trimmable top
    level states are deliberately skipped: the boundary snapshot owns those
    states and will overwrite them during restore.
    """

    cache_prefix_len = max(0, int(cache_prefix_len))
    mtp_prefix_len = (
        None
        if mtp_history_prefix_len is None
        else max(0, int(mtp_history_prefix_len))
    )
    selected_boundary = _gdn_boundary_spec_at_or_below(
        spec, boundary_prefix_len
    )
    boundary_states = (
        selected_boundary.get("states") if selected_boundary is not None else None
    )
    cache_spec = spec["cache_snapshot"]
    cache_snapshot = CacheSnapshot(
        states=_decode_cache_states_prefix(
            cache_spec["states"],
            boundary_states,
            read_tensor,
            prefix_len=cache_prefix_len,
        ),
        # Restoring a partial tensor state must not reinstate the original
        # full-prefix offset.  Real attention cache state setters derive the
        # offset from the truncated tensors; recurrent meta is restored from
        # the selected boundary below.
        meta_states=_none_tree_like(cache_spec["meta_states"]),
    )
    mtp_spec = spec.get("mtp_history_snapshot")
    mtp_history_snapshot = None
    if mtp_spec is not None:
        if mtp_prefix_len is None:
            mtp_history_snapshot = CacheSnapshot(
                states=tuple(decode_tree(mtp_spec["states"], read_tensor)),
                meta_states=tuple(
                    decode_tree(mtp_spec["meta_states"], read_tensor)
                ),
            )
        else:
            mtp_history_snapshot = CacheSnapshot(
                states=_decode_tree_prefix(
                    mtp_spec["states"],
                    read_tensor,
                    prefix_len=mtp_prefix_len,
                ),
                meta_states=_none_tree_like(mtp_spec["meta_states"]),
            )
    boundaries = ()
    if selected_boundary is not None:
        boundaries = (
            (
                int(selected_boundary["tokens"]),
                CacheSnapshot(
                    states=tuple(
                        decode_tree(item, read_tensor)
                        for item in _spec_items(selected_boundary["states"])
                    ),
                    meta_states=tuple(
                        decode_tree(item, read_tensor)
                        for item in _spec_items(selected_boundary["meta_states"])
                    ),
                ),
                decode_tree(
                    selected_boundary.get("hidden_last") or {"kind": "none"},
                    read_tensor,
                ),
            ),
        )
    decoded = DecodedPayload(
        cache_snapshot=cache_snapshot,
        # These are small final-token tensors and remain useful to callers
        # that reject the partial candidate before it reaches the boundary
        # restore.  They are not the source of the multi-GB hydration.
        logits=decode_tree(spec["logits"], read_tensor),
        hidden=decode_tree(spec["hidden"], read_tensor),
        mtp_history_snapshot=mtp_history_snapshot,
        gdn_boundaries=boundaries,
        has_recurrent=bool(spec.get("has_recurrent", False)),
        cache_snapshot_prefix_len=cache_prefix_len,
        mtp_history_snapshot_prefix_len=mtp_prefix_len,
    )
    _eval_decoded_arrays(decoded)
    return decoded


def _eval_decoded_arrays(decoded: DecodedPayload) -> None:
    """Single batched evaluation of every decoded array (vs per-tensor eval)."""
    arrays: list[Any] = []

    def collect(value: Any) -> None:
        if isinstance(value, mx.array):
            arrays.append(value)
        elif isinstance(value, CacheSnapshot):
            collect(value.states)
            collect(value.meta_states)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)

    collect(decoded.cache_snapshot)
    collect(decoded.logits)
    collect(decoded.hidden)
    collect(decoded.mtp_history_snapshot)
    collect(decoded.gdn_boundaries)
    if arrays:
        mx.eval(*arrays)


def decode_gdn_boundaries(
    spec: dict[str, Any], read_tensor: Callable[[str], bytes]
) -> tuple:
    """Decode only the interior recurrent boundaries from a payload spec.

    Used lazily by SSD-restored entries: exact restores skip the MB-scale
    boundary payloads entirely, and partial restores load them on demand
    through this helper (batched single eval)."""
    boundaries = tuple(
        (
            int(record["tokens"]),
            CacheSnapshot(
                states=tuple(decode_tree(record["states"], read_tensor)),
                meta_states=tuple(decode_tree(record["meta_states"], read_tensor)),
            ),
            decode_tree(record.get("hidden_last") or {"kind": "none"}, read_tensor),
        )
        for record in (spec.get("gdn_boundaries") or [])
    )
    arrays: list[Any] = []

    def collect(value: Any) -> None:
        if isinstance(value, mx.array):
            arrays.append(value)
        elif isinstance(value, CacheSnapshot):
            collect(value.states)
            collect(value.meta_states)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)

    collect(boundaries)
    if arrays:
        mx.eval(*arrays)
    return boundaries


def _gdn_boundary_spec_at_or_below(
    spec: dict[str, Any], prefix_len: int | None
) -> dict[str, Any] | None:
    if prefix_len is None:
        return None
    limit = int(prefix_len)
    best: dict[str, Any] | None = None
    for record in spec.get("gdn_boundaries") or []:
        try:
            tokens = int(record.get("tokens", 0))
        except (AttributeError, TypeError, ValueError):
            continue
        if 0 < tokens <= limit and (best is None or tokens > int(best["tokens"])):
            best = record
    return best


def payload_supports_prefix_decode(spec: dict[str, Any]) -> bool:
    return snapshot_supports_prefix_decode(spec.get("cache_snapshot"))


def snapshot_supports_prefix_decode(snapshot_spec: Any) -> bool:
    """Whether the target-cache representation can be safely block-sliced.

    Most attention caches derive their offset from the restored K/V tensors.
    A few cache layouts instead require coupled, model-specific metadata (or
    physically rotate their backing rows).  Reconstructing only their first
    tensor blocks is not equivalent to a trim, so those entries retain the
    established full-decode restore path.
    """

    try:
        meta_items = _spec_items(snapshot_spec["meta_states"])
    except (KeyError, ValueError, TypeError):
        return False
    for meta_spec in meta_items:
        values = _flat_string_sequence(meta_spec)
        if values is None:
            continue
        # DeepSeek-V4's five fields include rolling-window and compressor
        # counters which must stay coupled to its compressed tensor state.
        if values and values[0] == "mtplx-deepseek-v4-cache-v1":
            return False
        # Gemma's rotating cache carries (keep, max_size, offset, write_idx).
        # The persisted tensor order is ring-buffer order once full, so a
        # token-prefix slice cannot faithfully reconstruct it.
        if len(values) == 4 and all(_is_decimal_string(value) for value in values):
            return False
    return True


def _flat_string_sequence(spec: Any) -> tuple[str, ...] | None:
    kind = spec.get("kind") if isinstance(spec, dict) else None
    if kind == "none":
        return ()
    if kind not in {"tuple", "list"}:
        return None
    values: list[str] = []
    for item in spec.get("items") or []:
        if not isinstance(item, dict) or item.get("kind") != "str":
            return None
        values.append(str(item.get("value", "")))
    return tuple(values)


def _is_decimal_string(value: str) -> bool:
    return value.lstrip("-").isdigit()


def _spec_items(spec: Any) -> list[Any]:
    if not isinstance(spec, dict) or spec.get("kind") not in {"tuple", "list"}:
        raise ValueError("expected tuple/list cache-state payload spec")
    return list(spec.get("items") or [])


def _none_tree_like(spec: Any) -> Any:
    """Return a shape-compatible tree whose leaves are ``None``.

    CacheSnapshot's outer state/meta tuples are positional.  Keeping that
    shape lets ``restore_cache`` skip every original full-prefix meta state.
    """

    kind = spec.get("kind") if isinstance(spec, dict) else None
    if kind == "tuple":
        return tuple(_none_tree_like(item) for item in spec.get("items") or [])
    if kind == "list":
        return [_none_tree_like(item) for item in spec.get("items") or []]
    return None


def _decode_cache_states_prefix(
    states_spec: Any,
    boundary_states_spec: Any,
    read_tensor: Callable[[str], bytes],
    *,
    prefix_len: int,
) -> Any:
    states = _spec_items(states_spec)
    boundary_states = (
        _spec_items(boundary_states_spec)
        if boundary_states_spec is not None
        else [None] * len(states)
    )
    if len(states) != len(boundary_states):
        raise ValueError("SSD boundary cache-state shape mismatch")
    decoded = []
    for state_spec, boundary_spec in zip(states, boundary_states):
        # A non-None boundary state identifies a recurrent/non-trimmable
        # cache entry.  Its state comes from the selected boundary; loading
        # the full final snapshot here is both unnecessary and expensive.
        if isinstance(boundary_spec, dict) and boundary_spec.get("kind") != "none":
            decoded.append(None)
        else:
            decoded.append(
                _decode_tree_prefix(state_spec, read_tensor, prefix_len=prefix_len)
            )
    kind = states_spec.get("kind") if isinstance(states_spec, dict) else None
    return tuple(decoded) if kind == "tuple" else decoded


def _decode_tree_prefix(
    spec: Any,
    read_tensor: Callable[[str], bytes],
    *,
    prefix_len: int | None,
) -> Any:
    kind = spec.get("kind") if isinstance(spec, dict) else None
    if kind == "tuple":
        return tuple(
            _decode_tree_prefix(item, read_tensor, prefix_len=prefix_len)
            for item in spec.get("items") or []
        )
    if kind == "list":
        return [
            _decode_tree_prefix(item, read_tensor, prefix_len=prefix_len)
            for item in spec.get("items") or []
        ]
    if kind == "dict":
        return {
            _decode_tree_prefix(key, read_tensor, prefix_len=prefix_len):
            _decode_tree_prefix(value, read_tensor, prefix_len=prefix_len)
            for key, value in spec.get("items") or []
        }
    if kind == "tensor_blocks":
        return _decode_tensor_blocks(
            spec, read_tensor, prefix_len=prefix_len
        )
    return decode_tree(spec, read_tensor)


def _decode_tensor(spec: dict[str, Any], read_tensor: Callable[[str], bytes]) -> Any:
    dtype = str(spec["dtype"])
    shape = tuple(int(dim) for dim in spec.get("shape") or [])
    raw = read_tensor(str(spec["name"]))
    if dtype == "bfloat16":
        arr = mx.array(np.frombuffer(raw, dtype=np.uint16)).view(mx.bfloat16)
    else:
        np_dtype = _MLX_TO_NUMPY_DTYPE.get(dtype)
        mlx_dtype = _NUMPY_TO_MLX_DTYPE.get(dtype)
        if np_dtype is None or mlx_dtype is None:
            raise ValueError(f"unsupported persisted tensor dtype: {dtype!r}")
        arr = mx.array(np.frombuffer(raw, dtype=np_dtype), dtype=mlx_dtype)
    if shape:
        arr = arr.reshape(shape)
    return arr


def _decode_tensor_blocks(
    spec: dict[str, Any],
    read_tensor: Callable[[str], bytes],
    *,
    prefix_len: int | None = None,
) -> Any:
    axis = int(spec.get("axis", 2))
    limit = None if prefix_len is None else max(0, int(prefix_len))
    blocks = list(spec.get("blocks", []))
    if limit is not None:
        blocks = [block for block in blocks if int(block.get("start", 0)) < limit]
    chunks = [
        _decode_tensor({**block, "dtype": spec["dtype"]}, read_tensor)
        for block in blocks
    ]
    if not chunks:
        shape = tuple(int(dim) for dim in spec.get("shape") or [])
        if limit is not None and axis < len(shape):
            shape = (*shape[:axis], min(limit, shape[axis]), *shape[axis + 1 :])
        return mx.zeros(shape)
    arr = mx.concatenate(chunks, axis=axis)
    shape = tuple(int(dim) for dim in spec.get("shape") or [])
    if limit is not None and axis < len(shape):
        size = min(limit, shape[axis])
        if int(arr.shape[axis]) > size:
            slices = [slice(None)] * int(arr.ndim)
            slices[axis] = slice(0, size)
            arr = arr[tuple(slices)]
        shape = (*shape[:axis], size, *shape[axis + 1 :])
    if shape:
        arr = arr.reshape(shape)
    return arr


def _dtype_name(dtype: Any) -> str:
    raw = str(dtype)
    if raw.startswith("mlx.core."):
        raw = raw.removeprefix("mlx.core.")
    if raw == "bfloat16":
        return raw
    if raw in _MLX_TO_NUMPY_DTYPE:
        return raw
    raise TypeError(f"unsupported MLX tensor dtype for SessionBank SSD cache: {dtype!r}")
