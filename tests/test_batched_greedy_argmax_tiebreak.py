"""Argmax tie-break identity gate for the batched greedy accept (#315c1 port).

The batched accept reduces verify_logits[0, :R, :] with one 2-D argmax where
stock reduces each row 1-D. Same values, no accumulation — the entire
exactness claim rests on MLX dispatching the SAME tie-break (lowest index
wins) for both shapes. At half precision over a ~151k vocab, top-2 ties are
common, so an MMLU-style end-to-end pass cannot certify this; only direct
tie construction can. If this test fails on any device, the
MTPLX_BATCHED_GREEDY_ACCEPT knob must never default on there.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

VOCABS = [151_936, 4_096]
DTYPES = [mx.float16, mx.bfloat16]


def _tie_blocks(vocab: int, rows: int, dtype) -> list[mx.array]:
    base = (mx.random.normal((rows, vocab)) * 0.1).astype(dtype)
    top = mx.array(8.0, dtype=dtype)
    blocks = []

    def with_ties(positions_per_row: list[list[int]]) -> mx.array:
        block = mx.array(base)
        for r, positions in enumerate(positions_per_row):
            for p in positions:
                block[r, p] = top
        return block

    # Tie at index 0 vs a later index (first-wins is the observable).
    blocks.append(with_ties([[0, vocab // 2]] * rows))
    # Tie including the final index.
    blocks.append(with_ties([[vocab // 3, vocab - 1]] * rows))
    # Tie straddling a 1024-lane threadgroup boundary.
    blocks.append(with_ties([[1023, 1024]] * rows))
    # Three-way tie.
    blocks.append(with_ties([[7, 4096, vocab - 2]] * rows))
    # Different tie pair in every row simultaneously.
    blocks.append(
        with_ties([[r * 17 % vocab, (r * 17 + vocab // 2) % vocab] for r in range(rows)])
    )
    # Degenerate all-equal row (every index ties).
    blocks.append(mx.zeros((rows, vocab), dtype=dtype))
    return blocks


@pytest.mark.parametrize("vocab", VOCABS)
@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
@pytest.mark.parametrize("rows", [1, 2, 3, 4])
def test_batched_greedy_argmax_tiebreak_identity(vocab, dtype, rows):
    mx.random.seed(315)
    for block in _tie_blocks(vocab, rows, dtype):
        batched = mx.argmax(block, axis=-1)
        mx.eval(batched)
        batched_list = [int(v) for v in batched.tolist()]
        rowwise = []
        for r in range(rows):
            one_d = mx.argmax(block[r], axis=-1)
            mx.eval(one_d)
            rowwise.append(int(one_d.item()))
        assert batched_list == rowwise, (
            f"tie-break divergence: 2-D {batched_list} vs 1-D {rowwise} "
            f"(vocab={vocab} rows={rows} dtype={dtype})"
        )


@pytest.mark.skipif(not mx.metal.is_available(), reason="cpu-stream cross-check")
@pytest.mark.parametrize("dtype", DTYPES, ids=["fp16", "bf16"])
def test_batched_greedy_argmax_tiebreak_identity_cpu_stream(dtype):
    mx.random.seed(316)
    vocab, rows = 151_936, 3
    for block in _tie_blocks(vocab, rows, dtype):
        with mx.stream(mx.cpu):
            batched_cpu = mx.argmax(block, axis=-1)
            row_cpu = [mx.argmax(block[r], axis=-1) for r in range(rows)]
            mx.eval(batched_cpu, *row_cpu)
        assert [int(v) for v in batched_cpu.tolist()] == [
            int(r.item()) for r in row_cpu
        ]
