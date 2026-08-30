"""Multi-axis (M-RoPE) position tables for vision requests.

Qwen-VL-family models rope image tokens at (t, h, w) grid positions while
text tokens advance all three axes together; text after an image resumes at
max+1, so the image occupies max(frames, rows, cols) positions rather than
one per token ("position contraction"). Serving image tokens with plain
sequential positions diverges from training.

The table is a pure function of the expanded prompt (ids + per-image grids):
it is recomputed per request and never persisted in cache state, so warm
restores stay format-stable. Decode continues with equal-axes positions at
sequence_index + delta, which is exactly plain rope shifted by delta — the
scalar decode path stays untouched.

Algorithm adapted from mlx-vlm's get_rope_index (MIT, Prince Canuma and
contributors), simplified to batch-1 text+image serving (no attention-mask
padding, no video).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def build_mrope_positions(
    input_ids: Sequence[int],
    *,
    image_token_id: int,
    image_grids: Sequence[tuple[int, int, int]],
    spatial_merge_size: int,
    video_token_id: int | None = None,
) -> tuple[np.ndarray, int] | None:
    """Return (positions [3, len] int32, rope_delta) or None when unbuildable.

    ``image_grids`` are raw (t, h, w) patch grids in prompt order; the LLM
    sees h//merge x w//merge tokens per frame. Returns None (caller falls
    back to plain sequential rope) when a video pad is present or the pad
    layout disagrees with the grids — never a wrong table.
    """

    ids = list(int(t) for t in input_ids)
    n = len(ids)
    if video_token_id is not None and any(t == int(video_token_id) for t in ids):
        return None

    merge = max(1, int(spatial_merge_size))
    pad_id = int(image_token_id)
    chunks: list[np.ndarray] = []
    st = 0
    next_pos = 0
    for t, h, w in image_grids:
        try:
            ed = ids.index(pad_id, st)
        except ValueError:
            return None
        llm_t = int(t)
        llm_h = int(h) // merge
        llm_w = int(w) // merge
        block = llm_t * llm_h * llm_w
        if ed + block > n or any(x != pad_id for x in ids[ed : ed + block]):
            return None

        text_len = ed - st
        if text_len:
            text = np.arange(next_pos, next_pos + text_len, dtype=np.int32)
            chunks.append(np.broadcast_to(text, (3, text_len)).copy())
            next_pos += text_len

        t_idx = np.repeat(np.arange(llm_t, dtype=np.int32), llm_h * llm_w)
        h_idx = np.tile(
            np.repeat(np.arange(llm_h, dtype=np.int32), llm_w), llm_t
        )
        w_idx = np.tile(np.arange(llm_w, dtype=np.int32), llm_t * llm_h)
        chunks.append(np.stack([t_idx, h_idx, w_idx]) + next_pos)
        next_pos += max(llm_t, llm_h, llm_w)
        st = ed + block

    if any(x == pad_id for x in ids[st:]):
        # More pads than grids: layout mismatch, refuse rather than mis-rope.
        return None
    tail = n - st
    if tail:
        text = np.arange(next_pos, next_pos + tail, dtype=np.int32)
        chunks.append(np.broadcast_to(text, (3, tail)).copy())
        next_pos += tail

    if not chunks:
        return None
    table = np.concatenate(chunks, axis=1).astype(np.int32)
    if table.shape != (3, n):
        return None
    delta = int(next_pos - n)
    return table, delta
