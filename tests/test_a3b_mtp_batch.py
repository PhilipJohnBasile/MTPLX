from __future__ import annotations

from dataclasses import FrozenInstanceError
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
                    "decode_verify": [1, 2],
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
        self.model = SimpleNamespace(
            language_model=SimpleNamespace(
                model=SimpleNamespace(layers=[object() for _ in range(40)])
            ),
            mtp=SimpleNamespace(layers=[object()]),
            mtp_forward=self.draft_mtp,
            mtp_update_cache=self.update_mtp_cache,
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
                m2_implementations=tuple((lambda *args: args) for _ in range(30))
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
        "captured_gdn_layers": 30,
        "row_commit": True,
        "fixed_row_commit": True,
    }


def test_installer_pins_qwen35b_width8_depth1_geometry(tmp_path):
    from mtplx.a3b_mtp_batch import install_a3b_mtp_batch_lane

    runtime = _runtime(tmp_path)
    lane = install_a3b_mtp_batch_lane(runtime, selfcheck=_passing_selfcheck)

    assert lane.geometry.cohort_slots == 8
    assert lane.geometry.speculative_depth == 1
    assert lane.geometry.verify_tokens == 2
    assert lane.geometry.projection_rows == 16
    assert lane.geometry.hidden_size == 2048
    assert lane.geometry.vocab_size == 248320
    assert lane.route_id == "qwen35b_a3b_mtp_batch_b8_t2_m16"
    assert lane.target_forward.__self__ is runtime
    assert lane.draft_forward.func.__self__ is runtime
    assert callable(lane.update_mtp_cache)
    assert lane.capture_forward.func.__self__ is runtime
    assert lane.prefill_request.func is not None
    assert lane.selfcheck["solo_parity"] is True
    with pytest.raises(FrozenInstanceError):
        lane.route_id = "changed"


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
    runtime.a3b_compiled_target_prefix_factory.gdn_postconv.m2_implementations = (
        object(),
    )

    with pytest.raises(A3BMTPBatchInstallError, match="30 M2 post-conv"):
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
