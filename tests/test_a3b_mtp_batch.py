from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import inspect
import json
from types import SimpleNamespace

import pytest


def _config() -> dict:
    layer_types = tuple(
        "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
        for index in range(40)
    )
    return {
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "dtype": "bfloat16",
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "layer_types": list(layer_types),
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512,
            "vocab_size": 248320,
            "mtp_num_hidden_layers": 1,
        },
        "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
        "mtplx_mtp_quantization": {
            "bits": 4,
            "group_size": 32,
            "mode": "affine",
            "policy": "prequantized-int4",
            "prequantized": True,
        },
    }


class _Runtime:
    def __init__(self, model_path):
        self.model_path = model_path
        self.mtp_enabled = True
        self.a3b_whole_moe_installed = False
        self.qwen_row_owned_router_report = {
            "installed": True,
            "target_routers": 40,
            "mtp_routers": 1,
            "validated_contract": {
                "routes": {"decode_verify": list(range(1, 17))},
                "combine_tail": {
                    "decode_verify": [1, 2, 8, 16],
                    "other_rows": "stock_weighted_reduction",
                },
            },
        }
        self.contract = SimpleNamespace(
            hidden_variant="post_norm",
            concat_order="embedding_hidden",
            mtp_quant_bits=4,
            mtp_quant_group_size=32,
            mtp_quant_mode="affine",
        )

        class Model(SimpleNamespace):
            def __call__(self, *args, **kwargs):
                return args, kwargs

        class FakeAttention:
            def __call__(self, *_args, **_kwargs):
                return "solo-attention"

        layers = []
        for index in range(40):
            is_linear = (index + 1) % 4 != 0
            layers.append(
                SimpleNamespace(
                    is_linear=is_linear,
                    self_attn=None if is_linear else FakeAttention(),
                )
            )
        self.model = Model(
            language_model=SimpleNamespace(
                model=SimpleNamespace(layers=layers),
                make_cache=self.make_cache,
            ),
            mtp=SimpleNamespace(layers=[SimpleNamespace(self_attn=FakeAttention())]),
            mtp_forward=self.draft_mtp,
            mtp_update_cache=self.update_mtp_cache,
            make_mtp_cache=self.make_mtp_cache,
        )
        self.a3b_compiled_target_prefix_factory = SimpleNamespace(
            layer_types=tuple(
                "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
                for index in range(40)
            ),
            gdn_layers=30,
            full_attention_layers=10,
            hidden_size=2048,
            quantization="affine_q4_group64",
            gdn_postconv=SimpleNamespace(
                m2_implementations=tuple((lambda *args: args) for _ in range(30)),
                b8_t2_implementations=tuple((lambda *args: args) for _ in range(30)),
            ),
        )

    def forward_ar(self, *args, **kwargs):
        return args, kwargs

    def draft_mtp(self, *args, **kwargs):
        return args, kwargs

    def make_cache(self):
        return []

    def make_mtp_cache(self):
        return []

    def update_mtp_cache(self, *args, **kwargs):
        return args, kwargs

    def _forward_ar_capture_a3b_postconv(self, *args, **kwargs):
        return args, kwargs


def _runtime(tmp_path, config=None):
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        json.dumps(config or _config()), encoding="utf-8"
    )
    return _Runtime(model_path)


def _passing_selfcheck(lane):
    return {
        "ok": True,
        "target_shape": [lane.geometry.cohort_slots, lane.geometry.verify_tokens],
        "projection_rows": lane.geometry.projection_rows,
        "solo_parity": True,
        "heterogeneous_row_parity": True,
        "heterogeneous_numerical_parity": True,
        "heterogeneous_argmax_parity": True,
        "b8_t2_gdn_numerical_parity": True,
        "compiled_eager_numerical_parity": True,
        "compiled_eager_argmax_parity": True,
        "compiled_eager_offset_parity": True,
        "same_geometry_numerical_parity": True,
        "same_geometry_argmax_parity": True,
        "same_geometry_attention_parity": True,
        "stock_b8_unchanged_moe_reference": True,
        "captured_gdn_layers": 30,
        "row_commit": True,
        "fixed_row_commit": True,
        "mixed_commit_parity": True,
        "prefill_contract": True,
        "prefill_numerical_parity": True,
        "empty_mtp_draft_parity": True,
        "empty_mtp_draft_numerical_parity": True,
        "empty_mtp_draft_argmax_parity": True,
        "empty_mtp_row_isolation_parity": True,
        "row_isolation_parity": True,
        "balanced_l0_qkv_z_b_b1_bitwise": True,
    }


def _fake_profile_factories():
    from mtplx.mtp_batch_numerics import MTPBatchNumerics

    return {
        MTPBatchNumerics.BALANCED: lambda base: replace(
            base,
            numerics=MTPBatchNumerics.BALANCED,
            route_id="qwen35b_a3b_mtp_batch_b8_t2_balanced",
        ),
        MTPBatchNumerics.B1_EXACT: lambda base: replace(
            base,
            numerics=MTPBatchNumerics.B1_EXACT,
            route_id="qwen35b_a3b_mtp_batch_b8_t2_b1_exact",
        ),
    }


@pytest.mark.parametrize(
    ("profile", "suffix"),
    [
        ("throughput", "m16_throughput"),
        ("balanced", "balanced"),
        ("b1-exact", "b1_exact_serial"),
    ],
)
def test_installer_route_identity_includes_numerics_profile(tmp_path, profile, suffix):
    from mtplx.a3b_mtp_batch import install_a3b_mtp_batch_lane

    lane = install_a3b_mtp_batch_lane(
        _runtime(tmp_path),
        numerics=profile,
        selfcheck=_passing_selfcheck,
        profile_factories=_fake_profile_factories(),
    )

    assert lane.numerics_profile == profile
    assert lane.route_id.endswith(suffix)
    assert profile in lane.config_fingerprint


def test_balanced_profile_is_construction_bound_without_external_factory(
    tmp_path, monkeypatch
):
    import mtplx.a3b_mtp_batch as module

    sentinel = object()
    monkeypatch.setattr(
        module,
        "_bind_balanced_capture_forward",
        lambda _runtime: sentinel,
        raising=False,
    )

    lane = module.install_a3b_mtp_batch_lane(
        _runtime(tmp_path),
        numerics="balanced",
        selfcheck=_passing_selfcheck,
    )

    assert lane.route_id == ("qwen35b_a3b_mtp_batch_b8_t2_l0_b1_qkv_z_b_balanced")
    assert lane.capture_forward.keywords["call"] is sentinel


def test_b1_exact_profile_is_builtin_and_names_serial_b1_execution(tmp_path):
    from mtplx.a3b_mtp_batch import install_a3b_mtp_batch_lane

    lane = install_a3b_mtp_batch_lane(
        _runtime(tmp_path),
        numerics="b1-exact",
        selfcheck=_passing_selfcheck,
    )

    assert lane.numerics_profile == "b1-exact"
    assert lane.route_id == "qwen35b_mtp_batch_b1_exact_serial"
    assert lane.selfcheck["solo_parity"] is True
    assert lane.selfcheck["b1_exact_bitwise"] is True
    assert lane.selfcheck["b1_exact_failed_boundaries"] == []
    assert lane.selfcheck["b1_exact_execution"] == "unchanged_solo_runner"


def test_balanced_contract_uses_b1_and_own_eager_receipts_not_throughput_b8():
    from mtplx.a3b_mtp_batch import (
        _balanced_selfcheck_contract,
        _throughput_selfcheck_contract,
    )

    report = _passing_selfcheck(
        SimpleNamespace(
            geometry=SimpleNamespace(
                cohort_slots=8,
                verify_tokens=2,
                projection_rows=16,
            )
        )
    )
    report["same_geometry_numerical_parity"] = False

    assert _throughput_selfcheck_contract(report) is False
    assert _balanced_selfcheck_contract(report) is True

    report["balanced_l0_qkv_z_b_b1_bitwise"] = False
    assert _balanced_selfcheck_contract(report) is False


def test_balanced_full_graph_bound_is_profile_specific_and_construction_fixed():
    from mtplx.a3b_mtp_batch import _geometry_relative_limit

    assert _geometry_relative_limit("throughput") == 9.0 / 128.0
    assert _geometry_relative_limit("balanced") == 9.0 / 128.0
    assert _geometry_relative_limit("b1-exact") == 9.0 / 128.0


def test_balanced_binds_b1_qkv_z_b_only_at_the_first_divergent_gdn():
    import mlx.core as mx

    from mtplx.a3b_mtp_batch import _bind_balanced_projection_implementations

    calls = []
    gdns = []
    for layer in range(30):
        projections = {}
        for name in ("qkv", "z", "b", "a"):
            projections[f"in_proj_{name}"] = lambda value, layer=layer, name=name: (
                calls.append(("throughput", layer, name, tuple(value.shape))) or value
            )
        gdns.append(SimpleNamespace(**projections))
    gdns = tuple(gdns)

    def stock_qlinear(module, value):
        calls.append(("b1", module, tuple(value.shape)))
        return value

    qkv, z, b, a = _bind_balanced_projection_implementations(gdns, stock_qlinear)
    inputs = mx.zeros((8, 2, 1), dtype=mx.bfloat16)

    first = [implementations[0](inputs) for implementations in (qkv, z, b, a)]
    second = [implementations[1](inputs) for implementations in (qkv, z, b, a)]
    mx.eval(*first, *second)

    assert all(len(implementations) == 30 for implementations in (qkv, z, b, a))
    b1_calls = [item for item in calls if item[0] == "b1"]
    throughput_calls = [item for item in calls if item[0] == "throughput"]
    assert [item[2] for item in b1_calls] == [(1, 2, 1)] * 24
    assert [(item[1], item[2]) for item in throughput_calls] == [
        (0, "a"),
        (1, "qkv"),
        (1, "z"),
        (1, "b"),
        (1, "a"),
    ]
    assert [item[3] for item in throughput_calls] == [(8, 2, 1)] * 5


def test_installer_pins_qwen35b_width8_depth1_geometry(tmp_path):
    from mtplx.a3b_mtp_batch import (
        _prefill_qwen35b_batch_request,
        install_a3b_mtp_batch_lane,
    )

    runtime = _runtime(tmp_path)
    lane = install_a3b_mtp_batch_lane(runtime, selfcheck=_passing_selfcheck)

    assert lane.geometry.cohort_slots == 8
    assert lane.geometry.speculative_depth == 1
    assert lane.geometry.verify_tokens == 2
    assert lane.geometry.projection_rows == 16
    assert lane.geometry.hidden_size == 2048
    assert lane.geometry.vocab_size == 248320
    assert lane.route_id == "qwen35b_a3b_mtp_batch_b8_t2_m16_throughput"
    assert lane.numerics_profile == "throughput"
    assert lane.attention_route_id == "qwen35b_b8_t2_stock_fused_sdpa"
    assert lane.target_forward.keywords["call"] is runtime.model
    assert lane.draft_forward.keywords["call"].func.__self__ is runtime
    assert "mtp_depth" not in lane.draft_forward.keywords["call"].keywords
    assert callable(lane.update_mtp_cache)
    assert getattr(
        lane.capture_forward.keywords["call"],
        "_mtplx_compiled_qwen35b_b8_t2",
        False,
    )
    assert lane.prefill_request.func is _prefill_qwen35b_batch_request
    assert lane.prefill_request.keywords["target_forward"] is runtime.model
    assert lane.prefill_request.keywords["target_cache_factory"] == runtime.make_cache
    assert lane.prefill_request.keywords["mtp_cache_factory"] == runtime.make_mtp_cache
    assert lane.make_cache == runtime.model.language_model.make_cache
    assert lane.make_mtp_cache == runtime.model.make_mtp_cache
    full_attention = [
        layer.self_attn
        for layer in runtime.model.language_model.model.layers
        if not layer.is_linear
    ]
    assert len(full_attention) == 10
    assert all(
        type(attention).__call__.__name__ == "_qwen35b_b8_stock_attention"
        for attention in full_attention
    )
    assert all(attention(None) == "solo-attention" for attention in full_attention)
    assert lane.selfcheck["solo_parity"] is True
    with pytest.raises(FrozenInstanceError):
        lane.route_id = "changed"


def test_installer_rejects_mlx_lm_without_arrays_cache_fix(tmp_path, monkeypatch):
    import mtplx.a3b_mtp_batch as module

    class ReleasedArraysCache:
        def __init__(self, _size):
            pass

    monkeypatch.setattr(module, "ArraysCache", ReleasedArraysCache)

    with pytest.raises(module.A3BMTPBatchInstallError) as exc_info:
        module.install_a3b_mtp_batch_lane(
            _runtime(tmp_path),
            selfcheck=_passing_selfcheck,
        )

    message = str(exc_info.value)
    assert "mlx-lm PR #1642" in message
    assert "985af30df768a6f4dd2d0c7969d1868ca5dc3e1a" in message
    assert "uv pip install --python" in message
    assert " -m pip install --no-deps" in message


def test_installer_selfcheck_exercises_decode_verify_kernel_phase():
    from mtplx import a3b_mtp_batch

    source = inspect.getsource(a3b_mtp_batch._default_selfcheck)

    assert 'with attention_phase("decode_verify")' in source
    assert 'with attention_phase("ar_decode")' in source


def test_batch_driver_executes_draft_and_verify_in_installed_kernel_phases():
    from mtplx import a3b_mtp_batch

    source = inspect.getsource(a3b_mtp_batch.generate_a3b_mtp_batch)

    assert 'with attention_phase("ar_decode")' in source
    assert 'with attention_phase("decode_verify")' in source
    assert "solo prefill did not preserve" not in source


def test_batch_driver_does_not_call_numerics_attribution():
    from mtplx import a3b_mtp_batch

    source = inspect.getsource(a3b_mtp_batch.generate_a3b_mtp_batch)

    assert "attribution" not in source
    assert "first_material_divergence" not in source


@pytest.mark.parametrize(
    ("path", "value", "reason"),
    [
        (("model_type",), "deepseek_v4", "model_type"),
        (("text_config", "num_hidden_layers"), 39, "num_hidden_layers"),
        (("text_config", "mtp_num_hidden_layers"), 2, "mtp_num_hidden_layers"),
        (("text_config", "hidden_size"), 4096, "hidden_size"),
        (("text_config", "num_experts"), 128, "num_experts"),
        (("quantization", "group_size"), 128, "body group_size"),
        (("mtplx_mtp_quantization", "group_size"), 64, "MTP group_size"),
    ],
)
def test_installer_rejects_wrong_runtime_contract(tmp_path, path, value, reason):
    from mtplx.a3b_mtp_batch import (
        A3BMTPBatchInstallError,
        install_a3b_mtp_batch_lane,
    )

    config = _config()
    node = config
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    runtime = _runtime(tmp_path, config)

    with pytest.raises(A3BMTPBatchInstallError, match=reason):
        install_a3b_mtp_batch_lane(runtime, selfcheck=_passing_selfcheck)


def test_installer_rejects_missing_prebound_callable(tmp_path):
    from mtplx.a3b_mtp_batch import (
        A3BMTPBatchInstallError,
        install_a3b_mtp_batch_lane,
    )

    runtime = _runtime(tmp_path)
    runtime.model.mtp_forward = None

    with pytest.raises(A3BMTPBatchInstallError, match="mtp_forward"):
        install_a3b_mtp_batch_lane(runtime, selfcheck=_passing_selfcheck)


def test_installer_rejects_mtp_adapter_that_needs_hot_depth_routing(tmp_path):
    from mtplx.a3b_mtp_batch import (
        A3BMTPBatchInstallError,
        install_a3b_mtp_batch_lane,
    )

    runtime = _runtime(tmp_path)
    runtime.mtp_adapter_path = tmp_path / "adapter"

    with pytest.raises(A3BMTPBatchInstallError, match="MTP adapter"):
        install_a3b_mtp_batch_lane(runtime, selfcheck=_passing_selfcheck)


def test_installer_rejects_missing_row_owned_m1_m16_router(tmp_path):
    from mtplx.a3b_mtp_batch import (
        A3BMTPBatchInstallError,
        install_a3b_mtp_batch_lane,
    )

    runtime = _runtime(tmp_path)
    runtime.qwen_row_owned_router_report["installed"] = False

    with pytest.raises(A3BMTPBatchInstallError, match="row-owned"):
        install_a3b_mtp_batch_lane(runtime, selfcheck=_passing_selfcheck)


def test_installer_rejects_missing_compiled_capture_factory(tmp_path):
    from mtplx.a3b_mtp_batch import (
        A3BMTPBatchInstallError,
        install_a3b_mtp_batch_lane,
    )

    runtime = _runtime(tmp_path)
    runtime.a3b_compiled_target_prefix_factory = None

    with pytest.raises(A3BMTPBatchInstallError, match="compiled target-prefix"):
        install_a3b_mtp_batch_lane(runtime, selfcheck=_passing_selfcheck)


def test_installer_rejects_incomplete_postconv_capture_factory(tmp_path):
    from mtplx.a3b_mtp_batch import (
        A3BMTPBatchInstallError,
        install_a3b_mtp_batch_lane,
    )

    runtime = _runtime(tmp_path)
    runtime.a3b_compiled_target_prefix_factory.gdn_postconv.b8_t2_implementations = (
        object(),
    )

    with pytest.raises(A3BMTPBatchInstallError, match="30 B8/T2 post-conv"):
        install_a3b_mtp_batch_lane(runtime, selfcheck=_passing_selfcheck)


def test_installer_rejects_missing_b8_t2_postconv_capture_factory(tmp_path):
    from mtplx.a3b_mtp_batch import (
        A3BMTPBatchInstallError,
        install_a3b_mtp_batch_lane,
    )

    runtime = _runtime(tmp_path)
    runtime.a3b_compiled_target_prefix_factory.gdn_postconv.b8_t2_implementations = ()

    with pytest.raises(A3BMTPBatchInstallError, match="30 B8/T2 post-conv"):
        install_a3b_mtp_batch_lane(runtime, selfcheck=_passing_selfcheck)


def test_installer_rejects_failed_numerical_selfcheck(tmp_path):
    from mtplx.a3b_mtp_batch import (
        A3BMTPBatchInstallError,
        install_a3b_mtp_batch_lane,
    )

    runtime = _runtime(tmp_path)

    with pytest.raises(A3BMTPBatchInstallError, match="self-check"):
        install_a3b_mtp_batch_lane(
            runtime,
            selfcheck=lambda lane: {"ok": False, "solo_parity": False},
        )


def test_installed_lane_keeps_bound_routes_when_runtime_attributes_change(tmp_path):
    from mtplx.a3b_mtp_batch import install_a3b_mtp_batch_lane

    runtime = _runtime(tmp_path)
    lane = install_a3b_mtp_batch_lane(runtime, selfcheck=_passing_selfcheck)
    target = lane.target_forward
    draft = lane.draft_forward
    runtime.forward_ar = None
    runtime.draft_mtp = None

    assert lane.target_forward is target
    assert lane.draft_forward is draft


def test_installed_decode_bypasses_runtime_counter_wrappers(tmp_path):
    from mtplx.a3b_mtp_batch import install_a3b_mtp_batch_lane

    runtime = _runtime(tmp_path)
    lane = install_a3b_mtp_batch_lane(runtime, selfcheck=_passing_selfcheck)

    assert lane.target_forward.keywords["call"] is runtime.model
    assert lane.make_cache is runtime.model.language_model.make_cache
    assert lane.make_mtp_cache is runtime.model.make_mtp_cache


def test_fixed_b8_commit_selects_real_conv_and_gdn_state_ranks():
    import mlx.core as mx
    import numpy as np

    from mtplx.a3b_mtp_batch import (
        _LAYER_TYPES,
        _commit_qwen35b_b8_t2_rows,
    )

    cache = []
    captures = {}
    base_recurrent = {}
    for layer_idx, layer_type in enumerate(_LAYER_TYPES):
        if layer_type == "full_attention":
            cache.append(SimpleNamespace(offsets=mx.full((8,), 2, mx.int32)))
            continue
        conv_states = mx.arange(8 * 2 * 2 * 3).reshape(8, 2, 2, 3)
        states = mx.arange(8 * 2 * 2 * 3 * 4).reshape(8, 2, 2, 3, 4)
        base_conv = mx.full((8, 2, 3), -1)
        base_state = mx.full((8, 2, 3, 4), -2)
        cache.append([conv_states[:, -1], states[:, -1]])
        captures[layer_idx] = {
            "conv_states": conv_states,
            "states": states,
        }
        base_recurrent[layer_idx] = (base_conv, base_state)

    _commit_qwen35b_b8_t2_rows(
        cache,
        captures,
        [0, 1, 2, 0, 1, 2, 0, 1],
        base_recurrent,
    )
    first_linear = next(
        index
        for index, layer_type in enumerate(_LAYER_TYPES)
        if layer_type == "linear_attention"
    )
    np.testing.assert_array_equal(cache[first_linear][0][0], -1)
    np.testing.assert_array_equal(cache[first_linear][1][0], -2)
    np.testing.assert_array_equal(
        cache[first_linear][0][1], captures[first_linear]["conv_states"][1, 0]
    )
    np.testing.assert_array_equal(
        cache[first_linear][1][2], captures[first_linear]["states"][2, 1]
    )


def test_installer_rejects_uncancellable_mtp_batch_prefill_chunk(monkeypatch, tmp_path):
    from mtplx.a3b_mtp_batch import (
        A3BMTPBatchInstallError,
        install_a3b_mtp_batch_lane,
    )

    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_SIZE", "32768")

    with pytest.raises(A3BMTPBatchInstallError, match="prefill chunk"):
        install_a3b_mtp_batch_lane(_runtime(tmp_path), selfcheck=_passing_selfcheck)


@pytest.mark.parametrize(
    "receipt",
    [
        "heterogeneous_numerical_parity",
        "heterogeneous_argmax_parity",
        "b8_t2_gdn_numerical_parity",
        "compiled_eager_numerical_parity",
        "compiled_eager_argmax_parity",
        "compiled_eager_offset_parity",
        "same_geometry_numerical_parity",
        "same_geometry_argmax_parity",
        "same_geometry_attention_parity",
        "stock_b8_unchanged_moe_reference",
        "empty_mtp_draft_numerical_parity",
        "empty_mtp_draft_argmax_parity",
        "empty_mtp_row_isolation_parity",
        "row_isolation_parity",
    ],
)
def test_installer_rejects_missing_exact_batch_numerical_receipt(tmp_path, receipt):
    from mtplx.a3b_mtp_batch import (
        A3BMTPBatchInstallError,
        install_a3b_mtp_batch_lane,
    )

    def failed_selfcheck(lane):
        report = _passing_selfcheck(lane)
        report[receipt] = False
        return report

    with pytest.raises(A3BMTPBatchInstallError, match="numerical self-check"):
        install_a3b_mtp_batch_lane(_runtime(tmp_path), selfcheck=failed_selfcheck)


def test_batch_prefill_uses_only_prebound_routes_without_runtime_counters(
    monkeypatch,
):
    import mlx.core as mx

    from mtplx import generation
    from mtplx.a3b_mtp_batch import _prefill_qwen35b_batch_request

    target_cache = []
    mtp_cache = []
    target_lengths = []
    history_tokens = []

    def target_forward(input_ids, *, cache, return_hidden, hidden_variant):
        assert cache is target_cache
        assert return_hidden is True
        assert hidden_variant == "post_norm"
        length = int(input_ids.shape[1])
        target_lengths.append(length)
        return (
            mx.zeros((1, length, 7), dtype=mx.float32),
            mx.zeros((1, length, 3), dtype=mx.float32),
        )

    def update_mtp_cache(hidden, token_ids, *, mtp_cache, position_offset):
        assert mtp_cache is globals_mtp_cache
        assert position_offset is None
        history_tokens.append(token_ids.tolist())
        return hidden

    globals_mtp_cache = mtp_cache
    monkeypatch.setattr(
        generation,
        "_runtime_count",
        lambda *_args, **_kwargs: pytest.fail("batch prefill used runtime counters"),
    )

    result = _prefill_qwen35b_batch_request(
        [10, 11, 12, 13, 14],
        target_forward=target_forward,
        target_cache_factory=lambda: target_cache,
        mtp_cache_factory=lambda: mtp_cache,
        update_mtp_cache=update_mtp_cache,
        chunk_size=2,
        cleanup_every=0,
    )

    assert target_lengths == [2, 2, 1]
    assert history_tokens == [[[11, 12]], [[13, 14]]]
    assert result[0] is target_cache
    assert result[3] is mtp_cache
    assert tuple(result[1].shape) == (1, 7)
    assert tuple(result[2].shape) == (1, 1, 3)
    source = inspect.getsource(_prefill_qwen35b_batch_request)
    assert "os.environ" not in source
    assert "_runtime_count" not in source


def test_batch_prefill_freezes_dense_cleanup_cadence_at_construction(
    monkeypatch, tmp_path
):
    from mtplx.a3b_mtp_batch import install_a3b_mtp_batch_lane

    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_CACHE_CLEANUP", "1")
    monkeypatch.setenv("MTPLX_PREFILL_CHUNK_CACHE_CLEANUP_EVERY", "auto")
    monkeypatch.setenv("MTPLX_SUSTAINED_PREFILL_LAYOUT", "auto")
    monkeypatch.delenv("MTPLX_CURRENT_PREFILL_CONTEXT_TOKENS", raising=False)

    lane = install_a3b_mtp_batch_lane(_runtime(tmp_path), selfcheck=_passing_selfcheck)

    assert lane.prefill_request.keywords["cleanup_every"] == 4
