from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from mtplx.backends.deepseek_v4_mlxserve import (
    BACKEND_ID,
    DeepSeekV4MlxServeError,
    build_command,
    child_environment,
    resolve_binary,
)
from mtplx.backends.descriptors import descriptor_for_backend_id
from mtplx.backends.registry import compatibility_for_inspection
from mtplx.cli import build_parser
from mtplx.commands import public
from mtplx.models.deepseek_v4_target_only_config import (
    DEEPSEEK_V4_TARGET_ONLY_REPO_BYTES,
    DEEPSEEK_V4_TARGET_ONLY_REPO_ID,
    DEEPSEEK_V4_TARGET_ONLY_REVISION,
    DEEPSEEK_V4_TARGET_ONLY_SHARD_SHA256,
    DEEPSEEK_V4_TARGET_ONLY_SHARD_SIZES,
    is_deepseek_v4_target_only_config,
)


def _quantization() -> dict[str, object]:
    quantization: dict[str, object] = {
        "bits": 8,
        "group_size": 64,
        "mode": "affine",
        "embed": {"bits": 8, "group_size": 64, "mode": "affine"},
        "head": {"bits": 8, "group_size": 64, "mode": "affine"},
    }
    for layer in range(43):
        bits = (2, 3, 2) if layer < 39 else (4, 4, 4)
        group_size = 128 if layer < 39 else 64
        for projection, projection_bits in zip(("w1", "w2", "w3"), bits, strict=True):
            quantization[f"layers.{layer}.ffn.experts.{projection}"] = {
                "bits": projection_bits,
                "group_size": group_size,
                "mode": "affine",
            }
    return quantization


def _inspection(*, mtp_layers: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        model_dir="/tmp/deepseek-v4-target-only",
        architecture="DeepseekV4ForCausalLM",
        model_type="deepseek_v4",
        mtp_num_hidden_layers=mtp_layers,
        hidden_size=4096,
        num_hidden_layers=43,
        vocab_size=129_280,
        num_experts_per_tok=6,
        deepseek_v4_target_only_match=True,
        deepseek_v4_target_only_artifacts_complete=True,
        model_files=tuple(
            [f"model-layer-{idx}.safetensors" for idx in range(43)]
            + ["model-top.safetensors"]
        ),
        quantization=_quantization(),
        weight_keys=(),
        mtp=None,
        runtime_contract_data=None,
        runtime_contract_error=None,
        runtime_contract_path=None,
    )


def _config() -> dict[str, object]:
    return {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "num_hidden_layers": 43,
        "hidden_size": 4096,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "vocab_size": 129_280,
        "num_nextn_predict_layers": 0,
        "dspark_block_size": 0,
        "num_experts_per_tok": 6,
        "n_routed_experts": 256,
        "quantization": _quantization(),
    }


def test_exact_target_only_artifact_is_runnable_through_external_backend() -> None:
    verdict = compatibility_for_inspection(_inspection())

    assert verdict.tier == "AR-only"
    assert verdict.arch_id == "deepseek-v4-mlxserve-ar"
    assert verdict.can_run is True
    assert verdict.supported is True
    assert verdict.recommended_backend == BACKEND_ID
    assert verdict.mtp_supported == "no"


def test_public_artifact_contract_is_exact_and_self_consistent() -> None:
    assert DEEPSEEK_V4_TARGET_ONLY_REPO_ID == (
        "philipjohnbasile/DeepSeek-V4-Flash-0731-MLX-M5Max-TargetOnly"
    )
    assert DEEPSEEK_V4_TARGET_ONLY_REVISION == (
        "ac33e4f3ca3546e6cec104558d42161e15814e33"
    )
    assert len(DEEPSEEK_V4_TARGET_ONLY_SHARD_SIZES) == 44
    assert set(DEEPSEEK_V4_TARGET_ONLY_SHARD_SHA256) == set(
        DEEPSEEK_V4_TARGET_ONLY_SHARD_SIZES
    )
    assert sum(DEEPSEEK_V4_TARGET_ONLY_SHARD_SIZES.values()) == 103_849_215_724
    assert DEEPSEEK_V4_TARGET_ONLY_REPO_BYTES == 103_855_774_263
    assert is_deepseek_v4_target_only_config(_config()) is True


def test_public_artifact_contract_rejects_recipe_drift() -> None:
    config = _config()
    quantization = config["quantization"]
    assert isinstance(quantization, dict)
    # The published target-only Gold view is a UNIFORM affine 8-bit g64
    # artifact. Drift to an unpublished global geometry must be rejected.
    quantization["bits"] = 4
    quantization["group_size"] = 32

    assert is_deepseek_v4_target_only_config(config) is False


def test_public_artifact_contract_rejects_same_size_weight_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from mtplx.models import deepseek_v4_target_only_config as contract

    shard = tmp_path / "model-layer-0.safetensors"
    shard.write_bytes(b"good")
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(
        json.dumps({"weight_map": {"weight": shard.name}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        contract, "DEEPSEEK_V4_TARGET_ONLY_SHARD_SIZES", {shard.name: 4}
    )
    monkeypatch.setattr(
        contract,
        "DEEPSEEK_V4_TARGET_ONLY_SHARD_SHA256",
        {shard.name: hashlib.sha256(b"good").hexdigest()},
    )
    monkeypatch.setattr(
        contract,
        "DEEPSEEK_V4_TARGET_ONLY_SIDECAR_SHA256",
        {index.name: hashlib.sha256(index.read_bytes()).hexdigest()},
    )
    monkeypatch.setattr(
        contract,
        "DEEPSEEK_V4_TARGET_ONLY_WEIGHT_SHARDS",
        (shard.name,),
    )

    assert contract.deepseek_v4_target_only_artifact_integrity_errors(tmp_path) == ()
    shard.write_bytes(b"evil")
    assert contract.deepseek_v4_target_only_artifact_integrity_errors(tmp_path) == (
        shard.name,
    )


def test_target_only_gate_rejects_unbound_artifact() -> None:
    inspection = _inspection()
    inspection.deepseek_v4_target_only_artifacts_complete = False

    verdict = compatibility_for_inspection(inspection)

    assert verdict.can_run is False
    assert verdict.tier == "no-MTP"


def test_target_only_gate_rejects_quantization_drift() -> None:
    inspection = _inspection()
    # The published target-only Gold view is uniform affine 8-bit g64.
    # Drift to an unpublished global geometry (e.g. 4-bit) must be rejected.
    inspection.quantization["bits"] = 4
    inspection.quantization["group_size"] = 32

    verdict = compatibility_for_inspection(inspection)

    assert verdict.can_run is False
    assert verdict.tier == "no-MTP"


def test_dspark_declaring_artifact_stays_on_pending_dspark_identity() -> None:
    verdict = compatibility_for_inspection(_inspection(mtp_layers=1))

    assert verdict.arch_id == "deepseek-v4-mtp"
    assert verdict.can_run is False
    assert verdict.recommended_backend == "deepseek_v4_dspark"


def test_descriptor_is_honest_about_external_target_only_runtime() -> None:
    descriptor = descriptor_for_backend_id(BACKEND_ID)

    assert descriptor.backend_id == BACKEND_ID
    assert descriptor.uses_draft_lm_head is False
    assert descriptor.tune_policy.supported is False
    assert descriptor.supports("external_mlx_serve")
    assert descriptor.context_window_policy.default == 8_192


def test_external_command_and_environment_are_closed(tmp_path: Path) -> None:
    binary = tmp_path / "mlx-serve"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)
    admitted = resolve_binary({"MTPLX_MLX_SERVE_BIN": str(binary)})

    env = child_environment(
        {
            "PATH": "/usr/bin",
            "MLX_SERVE_DSV4_SINK_SOFTMAX": "1",
            "MLXSERVE_DEVICE": "foreign",
        }
    )
    command = build_command(
        binary=admitted,
        model="/models/dsv4",
        host="127.0.0.1",
        port=8123,
        context_window=None,
        api_key="test-only-key",
    )

    assert env == {
        "PATH": "/usr/bin",
        "MLX_SERVE_WIRED": "fit",
        "MLX_SERVE_CACHE_LIMIT": "268435456",
    }
    assert command[0] == str(binary.resolve())
    assert command[command.index("--ctx-size") + 1] == "8192"
    assert command[command.index("--api-key") + 1] == "test-only-key"
    for required in (
        "--serve",
        "--no-pld",
        "--no-decode-attn-quant",
        "--no-vision",
        "--skip-mem-preflight",
    ):
        assert required in command


def test_child_environment_wired_override(tmp_path: Path) -> None:
    from mtplx.backends.deepseek_v4_mlxserve import child_environment

    env = child_environment({"MTPLX_DSV4_WIRED": "max", "PATH": "/usr/bin"})
    assert env["MLX_SERVE_WIRED"] == "max"
    # the non-filtered override variable must not leak into the child
    assert "MTPLX_DSV4_WIRED" not in env
    # supplied MLX_SERVE_* vars stay filtered even when the override is set
    env2 = child_environment(
        {"MTPLX_DSV4_WIRED": "fit", "MLX_SERVE_WIRED": "off"})
    assert env2["MLX_SERVE_WIRED"] == "fit"


def test_external_command_rejects_invalid_context(tmp_path: Path) -> None:
    binary = tmp_path / "mlx-serve"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)

    with pytest.raises(DeepSeekV4MlxServeError, match="context window"):
        build_command(
            binary=binary,
            model="/models/dsv4",
            host="127.0.0.1",
            port=8123,
            context_window=1_048_577,
            api_key=None,
        )


def test_public_serve_dry_run_routes_to_external_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    binary = tmp_path / "mlx-serve"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)
    inspection = {
        "model_dir": str(tmp_path),
        "architecture": "DeepseekV4ForCausalLM",
        "model_type": "deepseek_v4",
        "recommended_backend": BACKEND_ID,
        "compatibility": {
            "tier": "AR-only",
            "arch_id": "deepseek-v4-mlxserve-ar",
            "can_run": True,
            "exit_code": 0,
            "runtime_compatibility": "native-ar-only",
            "recommended_backend": BACKEND_ID,
        },
    }
    monkeypatch.setattr(public, "_serve_should_onboard", lambda _args: False)
    monkeypatch.setattr(
        public,
        "_resolve_runtime_model_path",
        lambda model, cache_dir=None: (str(tmp_path), None),
    )
    monkeypatch.setattr(
        public,
        "_model_gate",
        lambda *_args, **_kwargs: (inspection, None),
    )
    monkeypatch.setattr(
        public, "resolve_deepseek_v4_mlxserve_binary", lambda: binary.resolve()
    )
    monkeypatch.setattr(
        public,
        "resolve_deepseek_v4_mlxserve_working_directory",
        lambda _binary: tmp_path,
    )
    monkeypatch.setattr(os, "environ", {"PATH": "/usr/bin"})
    args = build_parser().parse_args(
        [
            "serve",
            "--model",
            str(tmp_path),
            "--yes",
            "--no-mtp",
        ]
    )
    args.dry_run = True
    args.json = True

    assert public.cmd_serve_public(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["backend_id"] == BACKEND_ID
    assert payload["generation_mode"] == "ar"
    assert payload["mtp_available"] is False
    assert payload["dspark_available"] is False
    assert payload["env"]["MLX_SERVE_WIRED"] == "fit"
    assert payload["env"]["MLX_SERVE_CACHE_LIMIT"] == "268435456"
    assert "--no-pld" in payload["argv"]
    assert "--no-load-mtp" not in payload["argv"]


def _patch_external_dry_run_route(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    inspection = {
        "model_dir": str(tmp_path),
        "architecture": "DeepseekV4ForCausalLM",
        "model_type": "deepseek_v4",
        "recommended_backend": BACKEND_ID,
        "compatibility": {
            "tier": "AR-only",
            "arch_id": "deepseek-v4-mlxserve-ar",
            "can_run": True,
            "exit_code": 0,
            "runtime_compatibility": "native-ar-only",
            "recommended_backend": BACKEND_ID,
        },
    }
    monkeypatch.setattr(public, "_serve_should_onboard", lambda _args: False)
    monkeypatch.setattr(
        public,
        "_resolve_runtime_model_path",
        lambda model, cache_dir=None: (str(tmp_path), None),
    )
    monkeypatch.setattr(
        public,
        "_model_gate",
        lambda *_args, **_kwargs: (inspection, None),
    )
    monkeypatch.setattr(os, "environ", {"PATH": "/usr/bin"})


@pytest.mark.parametrize(
    ("failure", "expected_detail"),
    [
        ("binary", "synthetic missing binary"),
        ("cwd", "synthetic missing cwd"),
        ("command", "synthetic invalid command"),
    ],
)
def test_external_dry_run_json_admission_errors_are_parseable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    failure: str,
    expected_detail: str,
) -> None:
    _patch_external_dry_run_route(monkeypatch, tmp_path)
    binary = tmp_path / "mlx-serve"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)

    if failure == "binary":
        monkeypatch.setattr(
            public,
            "resolve_deepseek_v4_mlxserve_binary",
            lambda: (_ for _ in ()).throw(
                DeepSeekV4MlxServeError("synthetic missing binary")
            ),
        )
    else:
        monkeypatch.setattr(
            public,
            "resolve_deepseek_v4_mlxserve_binary",
            lambda: binary.resolve(),
        )
        if failure == "cwd":
            monkeypatch.setattr(
                public,
                "resolve_deepseek_v4_mlxserve_working_directory",
                lambda _binary: (_ for _ in ()).throw(
                    DeepSeekV4MlxServeError("synthetic missing cwd")
                ),
            )
        else:
            monkeypatch.setattr(
                public,
                "resolve_deepseek_v4_mlxserve_working_directory",
                lambda _binary: tmp_path,
            )
            monkeypatch.setattr(
                public,
                "build_deepseek_v4_mlxserve_command",
                lambda **_kwargs: (_ for _ in ()).throw(
                    DeepSeekV4MlxServeError("synthetic invalid command")
                ),
            )

    args = build_parser().parse_args(
        ["serve", "--model", str(tmp_path), "--yes", "--no-mtp"]
    )
    args.dry_run = True
    args.json = True

    assert public.cmd_serve_public(args) == 2
    stdout = capsys.readouterr().out
    payload = json.loads(stdout)
    assert stdout.lstrip().startswith("{")
    assert payload == {
        "ok": False,
        "dry_run": True,
        "target": "server",
        "error": "external_runtime_admission_failed",
        "backend_id": BACKEND_ID,
        "external_runtime": "mlx-serve",
        "detail": expected_detail,
    }


def test_external_dry_run_json_rejects_fan_mode_without_human_prefix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_external_dry_run_route(monkeypatch, tmp_path)
    monkeypatch.setattr(
        public,
        "resolve_deepseek_v4_mlxserve_binary",
        lambda: pytest.fail("fan-mode admission must run before binary lookup"),
    )
    args = build_parser().parse_args(
        [
            "serve",
            "--model",
            str(tmp_path),
            "--yes",
            "--no-mtp",
            "--fan-mode",
            "max",
        ]
    )
    args.dry_run = True
    args.json = True

    assert public.cmd_serve_public(args) == 2
    stdout = capsys.readouterr().out
    payload = json.loads(stdout)
    assert stdout.lstrip().startswith("{")
    assert payload["error"] == "external_runtime_admission_failed"
    assert payload["backend_id"] == BACKEND_ID
    assert payload["detail"].endswith("--fan-mode default")
