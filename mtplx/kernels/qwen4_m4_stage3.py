"""Physical-M4 stock-QMM combine tail for Qwen4 Flash-Next."""

from __future__ import annotations

from typing import Any, Callable

import mlx.core as mx


ROWS = 4
HIDDEN = 2560
TOP_K = 10
THREADS = 256

_KERNEL: Any | None = None


def source() -> str:
    return f"""
        using namespace metal;
        constexpr uint ROWS = {ROWS};
        constexpr uint HIDDEN = {HIDDEN};
        constexpr uint TOP_K = {TOP_K};

        uint index = thread_position_in_grid.x;
        if (index >= ROWS * HIDDEN) {{
            return;
        }}
        uint row = index / HIDDEN;
        uint column = index - row * HIDDEN;

        bfloat routed_products[TOP_K];
        for (uint slot = 0; slot < TOP_K; ++slot) {{
            bfloat down_value = routed_down[
                (row * TOP_K + slot) * HIDDEN + column];
            routed_products[slot] = bfloat(
                float(down_value) * float(route_scores[row * TOP_K + slot]));
        }}

        // Match MLX col_reduce_small for a ten-row BF16 strided reduction:
        // lanes 0 and 1 first fold rows 8 and 9, then lane 0 folds 1..7.
        bfloat routed_value = bfloat(
            float(routed_products[0]) + float(routed_products[8]));
        bfloat second = bfloat(
            float(routed_products[1]) + float(routed_products[9]));
        routed_value = bfloat(float(second) + float(routed_value));
        for (uint slot = 2; slot < 8; ++slot) {{
            routed_value = bfloat(
                float(routed_products[slot]) + float(routed_value));
        }}

        bfloat gated_shared = bfloat(
            float(shared_factor[row])
            * float(shared_down[row * HIDDEN + column]));
        output[index] = bfloat(float(routed_value) + float(gated_shared));
    """


def launch_geometry() -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    return ((ROWS * HIDDEN, 1, 1), (THREADS, 1, 1))


def bind() -> Callable[[Any, Any, Any, Any], mx.array]:
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="mtplx_qwen4_m4_combine_tail",
            input_names=[
                "routed_down",
                "shared_down",
                "route_scores",
                "shared_factor",
            ],
            output_names=["output"],
            source=source(),
            ensure_row_contiguous=True,
        )
    kernel = _KERNEL
    grid, threadgroup = launch_geometry()

    def combine(routed_down, shared_down, route_scores, shared_factor):
        (output,) = kernel(
            inputs=[routed_down, shared_down, route_scores, shared_factor],
            grid=grid,
            threadgroup=threadgroup,
            output_shapes=[(ROWS, HIDDEN)],
            output_dtypes=[mx.bfloat16],
        )
        return output

    return combine


__all__ = ["bind", "launch_geometry", "source"]
