from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest


class _ArraySpec:
    def __init__(self, shape, dtype) -> None:
        self.shape = tuple(shape)
        self.dtype = dtype


class _RoutedOwner:
    pass


class _SharedOwner:
    pass


def _projection(bits, group_size, weight_shape, metadata_shape):
    return SimpleNamespace(
        bits=bits,
        group_size=group_size,
        mode="affine",
        weight=_ArraySpec(weight_shape, mx.uint32),
        scales=_ArraySpec(metadata_shape, mx.bfloat16),
        biases=_ArraySpec(metadata_shape, mx.bfloat16),
    )


def _valid_block(monkeypatch):
    from mtplx import qwen4_m4_stage3 as stage3_module

    monkeypatch.setattr(stage3_module, "SparseMoeBlock", SimpleNamespace)
    monkeypatch.setattr(stage3_module, "_FusedGateUpSwitchGLU", _RoutedOwner)
    monkeypatch.setattr(stage3_module, "_FusedGateUpMLP", _SharedOwner)

    routed = _RoutedOwner()
    routed.bits = 4
    routed.group_size = 32
    routed.mode = "affine"
    routed.gu_weight = _ArraySpec((512, 1280, 320), mx.uint32)
    routed.gu_scales = _ArraySpec((512, 1280, 80), mx.bfloat16)
    routed.gu_biases = _ArraySpec((512, 1280, 80), mx.bfloat16)
    routed.down_proj = _projection(
        4,
        32,
        (512, 2560, 80),
        (512, 2560, 20),
    )

    shared = _SharedOwner()
    shared.bits = 8
    shared.group_size = 64
    shared.mode = "affine"
    shared.gu_weight = _ArraySpec((1280, 640), mx.uint32)
    shared.gu_scales = _ArraySpec((1280, 40), mx.bfloat16)
    shared.gu_biases = _ArraySpec((1280, 40), mx.bfloat16)
    shared.down_proj = _projection(8, 64, (2560, 160), (2560, 10))

    return SimpleNamespace(
        num_experts=512,
        top_k=10,
        norm_topk_prob=True,
        sharding_group=None,
        gate=_projection(8, 64, (512, 640), (512, 40)),
        shared_expert_gate=_projection(8, 64, (1, 640), (1, 40)),
        switch_mlp=routed,
        shared_expert=shared,
    )


def test_m4_stage3_keeps_quantized_matmuls_on_the_stock_path() -> None:
    from mtplx.kernels.qwen4_m4_stage3 import launch_geometry, source

    kernel_source = source()
    assert "constexpr uint ROWS = 4" in kernel_source
    assert "routed_down_weight" not in kernel_source
    assert "shared_down_weight" not in kernel_source
    assert launch_geometry() == ((10240, 1, 1), (256, 1, 1))


def test_m4_stage3_source_matches_mlx_bf16_column_reduction_order() -> None:
    from mtplx.kernels.qwen4_m4_stage3 import source

    kernel_source = source()
    assert "bfloat routed_products[TOP_K]" in kernel_source
    assert "routed_products[0]" in kernel_source
    assert "routed_products[8]" in kernel_source
    assert "routed_products[1]" in kernel_source
    assert "routed_products[9]" in kernel_source
    assert "for (uint slot = 2; slot < 8; ++slot)" in kernel_source


def test_m4_stage3_installer_keeps_both_down_projections_stock() -> None:
    import inspect

    from mtplx import qwen4_m4_stage3

    forward_source = inspect.getsource(qwen4_m4_stage3._m4_forward)
    install_source = inspect.getsource(qwen4_m4_stage3.install_qwen4_m4_stage3)
    assert "routed.down_proj(" in forward_source
    assert "shared.down_proj(" in forward_source
    assert "bind()" in install_source


def test_m4_stage3_flag_is_construction_bound(monkeypatch) -> None:
    from mtplx.qwen4_m4_stage3 import qwen4_m4_stage3_enabled

    monkeypatch.delenv("MTPLX_QWEN4_M4_STAGE3", raising=False)
    assert not qwen4_m4_stage3_enabled()
    monkeypatch.setenv("MTPLX_QWEN4_M4_STAGE3", "enabled")
    assert qwen4_m4_stage3_enabled()
    monkeypatch.setenv("MTPLX_QWEN4_M4_STAGE3", "disabled")
    assert not qwen4_m4_stage3_enabled()
    monkeypatch.setenv("MTPLX_QWEN4_M4_STAGE3", "maybe")
    with pytest.raises(ValueError, match="is not a boolean"):
        qwen4_m4_stage3_enabled()


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda block: setattr(block, "top_k", 9), "top-k"),
        (lambda block: setattr(block, "sharding_group", object()), "sharding"),
        (
            lambda block: setattr(block.switch_mlp, "group_size", 64),
            "routed fused GU",
        ),
    ],
)
def test_m4_stage3_rejects_wrong_construction_contract(
    monkeypatch, mutate, match
) -> None:
    from mtplx import qwen4_m4_stage3 as stage3_module

    block = _valid_block(monkeypatch)
    mutate(block)
    with pytest.raises(ValueError, match=match):
        stage3_module._validate_block_contract(block, index=7)


def test_m4_stage3_accepts_exact_construction_contract(monkeypatch) -> None:
    from mtplx import qwen4_m4_stage3 as stage3_module

    stage3_module._validate_block_contract(_valid_block(monkeypatch), index=7)


def test_m4_stage3_requires_bf16_hidden_input_ownership() -> None:
    from mtplx import qwen4_m4_stage3 as stage3_module

    layer = SimpleNamespace(
        mlp_hyper_connection=SimpleNamespace(
            hc_norm=SimpleNamespace(
                weight=_ArraySpec((4 * 2560,), mx.bfloat16),
            )
        )
    )
    stage3_module._validate_input_contract(layer, index=7)
    layer.mlp_hyper_connection.hc_norm.weight.dtype = mx.float16
    with pytest.raises(ValueError, match="BF16 hidden input"):
        stage3_module._validate_input_contract(layer, index=7)


def test_m4_stage3_routes_only_physical_m4_directly(monkeypatch) -> None:
    from mtplx import qwen4_m4_stage3 as stage3_module

    candidate = object()
    stock = object()
    block = stage3_module._M4Stage3SparseMoeBlock.__new__(
        stage3_module._M4Stage3SparseMoeBlock
    )
    object.__setattr__(block, "_mtplx_m4_stage3", object())
    monkeypatch.setattr(stage3_module, "_m4_forward", lambda *_args: candidate)
    monkeypatch.setattr(
        stage3_module.SparseMoeBlock,
        "__call__",
        lambda *_args: stock,
    )

    assert block(SimpleNamespace(size=4 * 2560, shape=(1, 4, 2560))) is candidate
    for rows in (1, 2, 3):
        assert (
            block(SimpleNamespace(size=rows * 2560, shape=(1, rows, 2560)))
            is stock
        )
