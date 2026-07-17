"""Vision embedding splice for chunked prefill.

A vision request carries one embedding row per expanded image pad token,
in prompt order. Prefill consumes chunks strictly left to right on the
solo lane, so the splice is a sequential queue: each chunk replaces its
pad-token rows with the next rows from the queue. Deepstack features, if
any, ride alongside with the same ordering and are applied by the layer
injection when enabled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx


@dataclass
class VisionSplice:
    """Per-request vision state consumed during prefill."""

    image_pad_token_id: int
    embeddings: Any  # mx.array [total_pad_tokens, text_hidden_size]
    deepstack: dict[int, Any] = field(default_factory=dict)
    cursor: int = 0
    # Content identity for token-keyed caches: one 64-bit digest of the raw
    # image bytes per image, plus that image's expanded pad-token count, in
    # prompt order. None on legacy constructions; cache keying then stays
    # disabled for the request (bypass semantics).
    image_digests: tuple[int, ...] | None = None
    pad_counts: tuple[int, ...] | None = None

    @property
    def total_rows(self) -> int:
        return int(self.embeddings.shape[0])

    def remaining(self) -> int:
        return self.total_rows - self.cursor

    def reset(self) -> None:
        self.cursor = 0


# Surrogate ids live far above any real vocabulary id (vocab ~248k << 2^40)
# so a keyed sequence can never collide with a plain text prompt.
_BANK_KEY_FLAG = 1 << 62
_BANK_KEY_MIX = 0x9E3779B97F4A7C15  # golden-ratio odd constant, stable mix
_BANK_KEY_MASK = (1 << 62) - 1


def vision_bank_key_ids(
    prompt_ids: list[int], splice: VisionSplice
) -> list[int] | None:
    """Content-true cache-key view of a vision prompt.

    Every image pad token shares one vocab id, so a token-keyed cache cannot
    tell two different images apart — the reason vision requests historically
    bypassed the session bank outright. For cache keying only, each pad
    position is remapped to a surrogate derived from its image's content
    digest and row index: the key sequence becomes a pure function of
    (text tokens, pixel content, positions). Same pixels restore exactly;
    different pixels can never match. The model input is untouched.

    Returns None when the splice carries no content identity (legacy
    construction) or the pad layout does not match the supplied images;
    callers must then keep the conservative bypass behavior.
    """

    digests = splice.image_digests
    pad_counts = splice.pad_counts
    if not digests or not pad_counts or len(digests) != len(pad_counts):
        return None
    pad_id = splice.image_pad_token_id
    total_pads = sum(1 for token in prompt_ids if token == pad_id)
    if total_pads != sum(int(count) for count in pad_counts):
        return None
    keyed = list(prompt_ids)
    image_idx = 0
    row_in_image = 0
    for pos, token in enumerate(keyed):
        if token != pad_id:
            continue
        while row_in_image >= int(pad_counts[image_idx]):
            image_idx += 1
            row_in_image = 0
        mixed = (
            (int(digests[image_idx]) ^ (row_in_image * _BANK_KEY_MIX))
            & _BANK_KEY_MASK
        )
        keyed[pos] = _BANK_KEY_FLAG | mixed
        row_in_image += 1
    return keyed


def _splice_rows_into_embedded(
    embedded: Any,
    mask: Any,
    rows: Any,
) -> Any:
    flat_mask = mask.reshape(-1)
    positions = mx.array(
        [i for i, hit in enumerate(flat_mask.tolist()) if hit], dtype=mx.int32
    )
    batch, seq, hidden = embedded.shape
    flat = embedded.reshape(batch * seq, hidden)
    flat[positions] = rows.astype(embedded.dtype)
    return flat.reshape(batch, seq, hidden)


def spliced_chunk_embeddings(
    embed_tokens: Any,
    chunk_array: Any,
    splice: VisionSplice,
) -> Any | None:
    """Embed one prefill chunk, replacing pad rows with vision rows.

    Returns None when the chunk holds no image pad tokens, so callers can
    keep the plain token-id fast path. Advances the splice cursor by the
    number of pads consumed; raises if the prompt contains more pads than
    the request supplied vision rows for, which would silently misalign
    every later image.
    """

    ids = chunk_array
    mask = ids == splice.image_pad_token_id
    pad_count = int(mask.sum().item())
    if pad_count == 0:
        return None
    if splice.remaining() < pad_count:
        raise ValueError(
            "vision splice underflow: prompt has more image pad tokens "
            f"({splice.cursor + pad_count}) than vision rows ({splice.total_rows})"
        )
    embedded = embed_tokens(ids)
    rows = splice.embeddings[splice.cursor : splice.cursor + pad_count]
    splice.cursor += pad_count
    return _splice_rows_into_embedded(embedded, mask, rows)


def spliced_embeddings_for_window(
    embed_tokens: Any,
    window_array: Any,
    splice: VisionSplice,
    *,
    rows_before: int,
) -> Any | None:
    """Cursor-free splice for an arbitrary prompt window.

    The MTP committed-history stream pairs hidden state t with token t+1,
    so its embedding window is shifted one token right of the trunk prefill
    chunk that produced the hidden states. This variant reads vision rows
    at an explicit offset (``rows_before`` = pad tokens before the window
    start) without touching the sequential cursor the trunk consumes.

    Returns None when the window holds no image pad tokens.
    """

    mask = window_array == splice.image_pad_token_id
    pad_count = int(mask.sum().item())
    if pad_count == 0:
        return None
    if rows_before + pad_count > splice.total_rows:
        raise ValueError(
            "vision splice window overflow: window needs rows "
            f"[{rows_before}, {rows_before + pad_count}) but only "
            f"{splice.total_rows} vision rows exist"
        )
    embedded = embed_tokens(window_array)
    rows = splice.embeddings[rows_before : rows_before + pad_count]
    return _splice_rows_into_embedded(embedded, mask, rows)
