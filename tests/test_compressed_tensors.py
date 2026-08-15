from __future__ import annotations

import json
import struct
from pathlib import Path

import mlx.core as mx
import numpy as np

from mtplx.compressed_tensors import (
    convert_compressed_tensors_awq_to_mlx,
    convert_compressed_tensors_nvfp4_to_mlx,
)
from mtplx.expert_layout import NumberedExpertAccumulator


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_raw_safetensors(path: Path, tensors: dict[str, tuple[str, tuple[int, ...], bytes]]) -> None:
    offset = 0
    header: dict[str, dict] = {}
    payload = bytearray()
    for name, (dtype, shape, raw) in tensors.items():
        start = offset
        payload.extend(raw)
        offset += len(raw)
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [start, offset],
        }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + payload)


def test_numbered_expert_accumulator_waits_until_final_when_expert_count_unknown():
    accumulator = NumberedExpertAccumulator()

    assert accumulator.add("layers.0.mlp.experts.0.gate_proj.weight", mx.array([1]))
    assert accumulator.flush_complete() == {}
    assert accumulator.add("layers.0.mlp.experts.1.gate_proj.weight", mx.array([2]))

    stacked = accumulator.flush_remaining(strict=True)

    assert tuple(stacked["layers.0.mlp.switch_mlp.gate_proj.weight"].shape) == (2, 1)


def test_compressed_tensors_converter_reads_split_scale_shards(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    weight_prefix = "model.language_model.layers.0.self_attn.q_proj"
    mtp_prefix = "mtp.layers.0.self_attn.q_proj"
    _write_json(
        source / "config.json",
        {
            "model_type": "qwen3_5",
            "quantization_config": {
                "format": "pack-quantized",
                "quant_method": "compressed-tensors",
                "config_groups": {
                    "group_0": {
                        "weights": {
                            "num_bits": 4,
                            "group_size": 8,
                        }
                    }
                },
            },
        },
    )
    _write_json(
        source / "model.safetensors.index.json",
        {
            "metadata": {},
            "weight_map": {
                f"{weight_prefix}.weight_packed": "model-00001-of-00002.safetensors",
                f"{weight_prefix}.weight_zero_point": "model-00001-of-00002.safetensors",
                f"{weight_prefix}.weight_shape": "model-00001-of-00002.safetensors",
                f"{weight_prefix}.weight_scale": "model-00002-of-00002.safetensors",
                f"{mtp_prefix}.weight_packed": "model-00001-of-00002.safetensors",
                f"{mtp_prefix}.weight_zero_point": "model-00001-of-00002.safetensors",
                f"{mtp_prefix}.weight_shape": "model-00001-of-00002.safetensors",
                f"{mtp_prefix}.weight_scale": "model-00002-of-00002.safetensors",
                "lm_head.weight": "model-00001-of-00002.safetensors",
            },
        },
    )
    mx.save_safetensors(
        str(source / "model-00001-of-00002.safetensors"),
        {
            f"{weight_prefix}.weight_packed": mx.array([[1], [2], [3], [4], [5], [6], [7], [8]], dtype=mx.int32),
            f"{weight_prefix}.weight_zero_point": mx.array([[0]], dtype=mx.int32),
            f"{weight_prefix}.weight_shape": mx.array([8, 8], dtype=mx.int64),
            f"{mtp_prefix}.weight_packed": mx.array([[9], [10], [11], [12], [13], [14], [15], [16]], dtype=mx.int32),
            f"{mtp_prefix}.weight_zero_point": mx.array([[0]], dtype=mx.int32),
            f"{mtp_prefix}.weight_shape": mx.array([8, 8], dtype=mx.int64),
            "lm_head.weight": mx.ones((2, 2), dtype=mx.bfloat16),
        },
    )
    mx.save_safetensors(
        str(source / "model-00002-of-00002.safetensors"),
        {
            f"{weight_prefix}.weight_scale": mx.ones((8, 1), dtype=mx.float16),
            f"{mtp_prefix}.weight_scale": mx.ones((8, 1), dtype=mx.float16),
        },
    )

    report = convert_compressed_tensors_awq_to_mlx(
        source,
        output,
        source_repo="owner/model",
        source_sha="abc123",
    )

    main = mx.load(str(output / "model-00001-of-00002.safetensors"))
    mtp = mx.load(str(output / "mtp.safetensors"))
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    index = json.loads((output / "model.safetensors.index.json").read_text(encoding="utf-8"))

    assert report["audit"]["passed"] is True
    assert "language_model.model.layers.0.self_attn.q_proj.weight" in main
    assert "language_model.model.layers.0.self_attn.q_proj.scales" in main
    assert "language_model.model.layers.0.self_attn.q_proj.biases" in main
    assert "language_model.lm_head.weight" in main
    assert main["language_model.lm_head.weight"].dtype == mx.bfloat16
    assert "mtp.layers.0.self_attn.q_proj.weight" in mtp
    assert config["mlx_lm_extra_tensors"]["mtp_file"] == "mtp.safetensors"
    assert config["mtplx_mtp_quantization"]["policy"] == "cyankiwi"
    assert config["mtplx_mtp_quantization"]["source_format"] == "compressed-tensors-awq"
    assert config["quantization"]["group_size"] == 8
    assert config["mtplx_policy"]["source"] == "owner/model"
    assert index["metadata"]["source_sha"] == "abc123"


def test_compressed_tensors_converter_stacks_numbered_moe_experts(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    main_prefix = "model.language_model.layers.0.mlp.experts"
    mtp_prefix = "mtp.layers.0.mlp.experts"
    weight_map = {"lm_head.weight": "model-00001-of-00002.safetensors"}
    for root in (main_prefix, mtp_prefix):
        for expert in range(2):
            prefix = f"{root}.{expert}.gate_proj"
            filename = "model-00001-of-00002.safetensors" if expert == 0 else "model-00002-of-00002.safetensors"
            weight_map[f"{prefix}.weight_packed"] = filename
            weight_map[f"{prefix}.weight_zero_point"] = filename
            weight_map[f"{prefix}.weight_shape"] = filename
            weight_map[f"{prefix}.weight_scale"] = "model-00002-of-00002.safetensors"

    _write_json(
        source / "config.json",
        {
            "model_type": "qwen3_5_moe",
            "text_config": {
                "model_type": "qwen3_5_moe_text",
                "n_routed_experts": 2,
                "mtp_num_hidden_layers": 1,
            },
            "quantization_config": {
                "format": "pack-quantized",
                "quant_method": "compressed-tensors",
                "config_groups": {
                    "group_0": {
                        "weights": {
                            "num_bits": 4,
                            "group_size": 8,
                        }
                    }
                },
            },
        },
    )
    _write_json(source / "model.safetensors.index.json", {"metadata": {}, "weight_map": weight_map})

    file1 = {
        "lm_head.weight": mx.ones((2, 2), dtype=mx.bfloat16),
    }
    file2 = {}
    for root in (main_prefix, mtp_prefix):
        for expert in range(2):
            prefix = f"{root}.{expert}.gate_proj"
            target = file1 if expert == 0 else file2
            target[f"{prefix}.weight_packed"] = mx.array(
                [[expert + 1] for _ in range(8)],
                dtype=mx.int32,
            )
            target[f"{prefix}.weight_zero_point"] = mx.array([[0]], dtype=mx.int32)
            target[f"{prefix}.weight_shape"] = mx.array([8, 8], dtype=mx.int64)
            file2[f"{prefix}.weight_scale"] = mx.ones((8, 1), dtype=mx.float16)
    mx.save_safetensors(str(source / "model-00001-of-00002.safetensors"), file1)
    mx.save_safetensors(str(source / "model-00002-of-00002.safetensors"), file2)

    report = convert_compressed_tensors_awq_to_mlx(
        source,
        output,
        source_repo="owner/moe",
        source_sha="rev",
    )

    index = json.loads((output / "model.safetensors.index.json").read_text(encoding="utf-8"))
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    main_file = index["weight_map"]["language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"]
    main = mx.load(str(output / main_file))
    mtp = mx.load(str(output / "mtp.safetensors"))

    assert report["audit"]["passed"] is True
    assert not any(".mlp.experts." in key for key in index["weight_map"])
    assert "language_model.model.layers.0.mlp.switch_mlp.gate_proj" in config["quantization"]
    assert not any(".mlp.experts." in key for key in config["quantization"])
    assert tuple(main["language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"].shape)[0] == 2
    assert "mtp.layers.0.mlp.switch_mlp.gate_proj.weight" in mtp
    assert not any(".mlp.experts." in key for key in mtp)
    assert config["mlx_lm_extra_tensors"]["mtp_tensor_count"] == len(mtp)


def test_autoawq_converter_preserves_glm_key_layout(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    prefix = "model.layers.0.mlp.down_proj"
    _write_json(
        source / "config.json",
        {
            "architectures": ["Glm4MoeLiteForCausalLM"],
            "model_type": "glm4_moe_lite",
            "num_hidden_layers": 47,
            "num_nextn_predict_layers": 1,
            "quantization_config": {
                "quant_method": "awq",
                "bits": 4,
                "group_size": 8,
                "zero_point": True,
            },
        },
    )
    _write_json(
        source / "model.safetensors.index.json",
        {
            "metadata": {},
            "weight_map": {
                f"{prefix}.qweight": "model.safetensors",
                f"{prefix}.qzeros": "model.safetensors",
                f"{prefix}.scales": "model.safetensors",
                "model.layers.47.enorm.weight": "model.safetensors",
                "model.layers.47.hnorm.weight": "model.safetensors",
                "model.layers.47.eh_proj.weight": "model.safetensors",
                "lm_head.weight": "model.safetensors",
            },
        },
    )
    mx.save_safetensors(
        str(source / "model.safetensors"),
        {
            f"{prefix}.qweight": mx.array([[0x76543210] for _ in range(8)], dtype=mx.int32),
            f"{prefix}.qzeros": mx.array([[0x76543210]], dtype=mx.int32),
            f"{prefix}.scales": mx.ones((1, 8), dtype=mx.float16),
            "model.layers.47.enorm.weight": mx.ones((8,), dtype=mx.float16),
            "model.layers.47.hnorm.weight": mx.ones((8,), dtype=mx.float16),
            "model.layers.47.eh_proj.weight": mx.ones((8, 16), dtype=mx.float16),
            "lm_head.weight": mx.ones((8, 8), dtype=mx.float16),
        },
    )

    report = convert_compressed_tensors_awq_to_mlx(
        source,
        output,
        source_repo="owner/glm-awq",
        source_sha="rev",
    )

    index = json.loads((output / "model.safetensors.index.json").read_text(encoding="utf-8"))
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    converted = mx.load(str(output / index["weight_map"][f"{prefix}.weight"]))

    assert report["audit"]["passed"] is True
    assert f"{prefix}.weight" in index["weight_map"]
    assert "language_model.model.layers.0.mlp.down_proj.weight" not in index["weight_map"]
    assert "model.layers.47.eh_proj.weight" in index["weight_map"]
    assert "lm_head.weight" in index["weight_map"]
    assert tuple(converted[f"{prefix}.weight"].shape) == (8, 1)
    assert tuple(converted[f"{prefix}.scales"].shape) == (8, 1)
    assert tuple(converted[f"{prefix}.biases"].shape) == (8, 1)
    assert config["quantization"]["group_size"] == 8
    assert f"{prefix}" in config["quantization"]


def test_autoawq_converter_routes_mtp_qweight_to_sidecar(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    main_prefix = "model.language_model.layers.0.self_attn.q_proj"
    mtp_prefix = "mtp.layers.0.self_attn.q_proj"
    _write_json(
        source / "config.json",
        {
            "model_type": "qwen3_5_moe",
            "text_config": {
                "model_type": "qwen3_5_moe_text",
                "mtp_num_hidden_layers": 1,
            },
            "quantization_config": {
                "quant_method": "awq",
                "bits": 4,
                "group_size": 8,
                "zero_point": True,
            },
        },
    )
    _write_json(
        source / "model.safetensors.index.json",
        {
            "metadata": {},
            "weight_map": {
                f"{main_prefix}.qweight": "model.safetensors",
                f"{main_prefix}.qzeros": "model.safetensors",
                f"{main_prefix}.scales": "model.safetensors",
                f"{mtp_prefix}.qweight": "model.safetensors",
                f"{mtp_prefix}.qzeros": "model.safetensors",
                f"{mtp_prefix}.scales": "model.safetensors",
                "lm_head.weight": "model.safetensors",
            },
        },
    )
    mx.save_safetensors(
        str(source / "model.safetensors"),
        {
            f"{main_prefix}.qweight": mx.array([[0x76543210] for _ in range(8)], dtype=mx.int32),
            f"{main_prefix}.qzeros": mx.array([[0x76543210]], dtype=mx.int32),
            f"{main_prefix}.scales": mx.ones((1, 8), dtype=mx.float16),
            f"{mtp_prefix}.qweight": mx.array([[0x76543210] for _ in range(8)], dtype=mx.int32),
            f"{mtp_prefix}.qzeros": mx.array([[0x76543210]], dtype=mx.int32),
            f"{mtp_prefix}.scales": mx.ones((1, 8), dtype=mx.float16),
            "lm_head.weight": mx.ones((8, 8), dtype=mx.float16),
        },
    )

    report = convert_compressed_tensors_awq_to_mlx(
        source,
        output,
        source_repo="owner/qwen-awq",
        source_sha="rev",
    )

    index = json.loads((output / "model.safetensors.index.json").read_text(encoding="utf-8"))
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    mtp = mx.load(str(output / "mtp.safetensors"))

    assert report["audit"]["passed"] is True
    assert report["stats"]["mtp_quantized_modules"] == 1
    assert f"{mtp_prefix}.weight" in mtp
    assert f"{mtp_prefix}.scales" in mtp
    assert f"{mtp_prefix}.biases" in mtp
    assert f"{mtp_prefix}.weight" not in index["weight_map"]
    assert f"{main_prefix}.weight" not in index["weight_map"]
    assert "language_model.model.layers.0.self_attn.q_proj.weight" in index["weight_map"]
    assert config["mlx_lm_extra_tensors"]["mtp_file"] == "mtp.safetensors"
    assert config["mtplx_mtp_quantization"]["quantized_modules"] == 1


def test_nvfp4_converter_requantizes_to_mlx_affine(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    prefix = "model.language_model.layers.0.self_attn.q_proj"
    _write_json(
        source / "config.json",
        {
            "model_type": "qwen3_5",
            "quantization_config": {
                "quant_method": "compressed-tensors",
                "format": "nvfp4-pack-quantized",
                "config_groups": {
                    "group_0": {
                        "weights": {
                            "num_bits": 4,
                            "type": "float",
                            "strategy": "tensor_group",
                            "group_size": 16,
                        }
                    }
                },
            },
        },
    )
    _write_json(
        source / "model.safetensors.index.json",
        {
            "metadata": {},
            "weight_map": {
                f"{prefix}.weight_packed": "model.safetensors",
                f"{prefix}.weight_scale": "model.safetensors",
                f"{prefix}.weight_global_scale": "model.safetensors",
                f"{prefix}.input_global_scale": "model.safetensors",
                "lm_head.weight": "model.safetensors",
            },
        },
    )
    _write_raw_safetensors(
        source / "model.safetensors",
        {
            f"{prefix}.weight_packed": (
                "U8",
                (1, 16),
                np.array([0x21, 0x43, 0x65, 0x87] * 4, dtype=np.uint8).tobytes(),
            ),
            f"{prefix}.weight_scale": (
                "F8_E4M3",
                (1, 2),
                np.array([0x38, 0x38], dtype=np.uint8).tobytes(),
            ),
            f"{prefix}.weight_global_scale": (
                "F32",
                (1,),
                np.array([1.0], dtype="<f4").tobytes(),
            ),
            f"{prefix}.input_global_scale": (
                "F32",
                (1,),
                np.array([1.0], dtype="<f4").tobytes(),
            ),
            "lm_head.weight": (
                "F32",
                (2, 2),
                np.ones((2, 2), dtype="<f4").tobytes(),
            ),
        },
    )

    report = convert_compressed_tensors_nvfp4_to_mlx(
        source,
        output,
        source_repo="owner/nvfp4",
        source_sha="rev",
        target_bits=4,
        target_group_size=32,
    )

    index = json.loads((output / "model.safetensors.index.json").read_text(encoding="utf-8"))
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    converted = mx.load(str(output / index["weight_map"]["language_model.model.layers.0.self_attn.q_proj.weight"]))

    assert report["audit"]["passed"] is True
    assert "language_model.model.layers.0.self_attn.q_proj.weight" in index["weight_map"]
    assert config["quantization"]["group_size"] == 32
    assert config["mtplx_policy"]["source_format"] == "compressed-tensors-nvfp4-w4a16"
    assert config["mtplx_policy"]["awq_calibrated"] is False
    dequantized = mx.dequantize(
        converted["language_model.model.layers.0.self_attn.q_proj.weight"],
        converted["language_model.model.layers.0.self_attn.q_proj.scales"],
        converted["language_model.model.layers.0.self_attn.q_proj.biases"],
        group_size=32,
        bits=4,
    )
    assert tuple(dequantized.shape) == (1, 32)
    assert float(mx.max(mx.abs(dequantized)).item()) > 0.0


def test_nvfp4_converter_preserves_glm_key_layout(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    prefix = "model.layers.0.mlp.down_proj"
    _write_json(
        source / "config.json",
        {
            "architectures": ["Glm4MoeLiteForCausalLM"],
            "model_type": "glm4_moe_lite",
            "num_hidden_layers": 47,
            "num_nextn_predict_layers": 1,
            "quantization_config": {
                "quant_method": "compressed-tensors",
                "format": "nvfp4-pack-quantized",
                "config_groups": {
                    "group_0": {
                        "weights": {
                            "num_bits": 4,
                            "type": "float",
                            "strategy": "tensor_group",
                            "group_size": 16,
                        }
                    }
                },
            },
        },
    )
    _write_json(
        source / "model.safetensors.index.json",
        {
            "metadata": {},
            "weight_map": {
                f"{prefix}.weight_packed": "model.safetensors",
                f"{prefix}.weight_scale": "model.safetensors",
                f"{prefix}.weight_global_scale": "model.safetensors",
                f"{prefix}.input_global_scale": "model.safetensors",
                "lm_head.weight": "model.safetensors",
            },
        },
    )
    _write_raw_safetensors(
        source / "model.safetensors",
        {
            f"{prefix}.weight_packed": (
                "U8",
                (1, 16),
                np.array([0x21, 0x43, 0x65, 0x87] * 4, dtype=np.uint8).tobytes(),
            ),
            f"{prefix}.weight_scale": (
                "F8_E4M3",
                (1, 2),
                np.array([0x38, 0x38], dtype=np.uint8).tobytes(),
            ),
            f"{prefix}.weight_global_scale": (
                "F32",
                (1,),
                np.array([1.0], dtype="<f4").tobytes(),
            ),
            f"{prefix}.input_global_scale": (
                "F32",
                (1,),
                np.array([1.0], dtype="<f4").tobytes(),
            ),
            "lm_head.weight": (
                "F32",
                (2, 2),
                np.ones((2, 2), dtype="<f4").tobytes(),
            ),
        },
    )

    report = convert_compressed_tensors_nvfp4_to_mlx(
        source,
        output,
        source_repo="owner/glm-nvfp4",
        source_sha="rev",
        target_bits=4,
        target_group_size=32,
    )

    index = json.loads((output / "model.safetensors.index.json").read_text(encoding="utf-8"))

    assert report["audit"]["passed"] is True
    assert f"{prefix}.weight" in index["weight_map"]
    assert f"language_model.{prefix}.weight" not in index["weight_map"]


def test_mlx_key_preserves_vision_tower_prefixes():
    # The model.visual.* -> vision_tower.* remap is what keeps multimodal
    # checkpoints sighted through the compressed-tensors lane (issue #263).
    from mtplx.compressed_tensors import _mlx_key

    assert (
        _mlx_key("model.visual.blocks.0.attn.qkv.weight")
        == "vision_tower.blocks.0.attn.qkv.weight"
    )
    assert (
        _mlx_key("model.visual.patch_embed.proj.weight")
        == "vision_tower.patch_embed.proj.weight"
    )
    assert (
        _mlx_key("vision_tower.merger.linear_fc1.weight")
        == "vision_tower.merger.linear_fc1.weight"
    )
    assert (
        _mlx_key("model.language_model.layers.0.mlp.gate_proj.weight")
        == "language_model.model.layers.0.mlp.gate_proj.weight"
    )
    assert _mlx_key("lm_head.weight") == "language_model.lm_head.weight"
