"""Integrity gates for the DeepSeek-V4 MoE-tail K3 E2E bracket."""

import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest


_ROOT = Path(__file__).parents[1]
_VALIDATOR = _ROOT / "scripts" / "deepseek_v4_validate_moe_tail_k3_bracket.py"
_GUARD = _ROOT / "scripts" / "deepseek_v4_guard_window.py"
_BENCHMARK = _ROOT / "scripts" / "deepseek_v4_mtpk_bench.py"
_ARMS = _ROOT / "scripts" / "deepseek_v4_moe_tail_arms.sh"

_spec = importlib.util.spec_from_file_location("dsv4_moe_tail_bracket", _VALIDATOR)
V = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V)

if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))
_bench_spec = importlib.util.spec_from_file_location("dsv4_moe_tail_bench", _BENCHMARK)
H = importlib.util.module_from_spec(_bench_spec)
_bench_spec.loader.exec_module(H)


def _stage4_env(enabled: bool) -> dict[str, str]:
    return {
        "MTPLX_COMPILED_VERIFY": "off",
        "MTPLX_DSV4_ATTN": "fused",
        "MTPLX_DSV4_FP32_ACTIVATIONS": "0",
        "MTPLX_DSV4_HC_COMPILE": "1",
        "MTPLX_DSV4_MOE_TAIL": "1" if enabled else "0",
        "MTPLX_DSV4_O_LORA": "cached",
        "MTPLX_DSV4_SINKHORN_KERNEL": "1",
    }


def _guard_window(child_pid: int = 200) -> dict:
    attestation = {
        "schema_version": 1,
        "guard_pid": 100,
        "child_pid": child_pid,
        "issued_monotonic_ns": 1_000_000,
        "expires_monotonic_ns": 61_000_000,
        "lock_path": "/tmp/mtplx-gpu-exclusive.lock",
        "lock_device": 1,
        "lock_inode": 2,
        "nonce_sha256": "c" * 64,
    }
    encoded_attestation = json.dumps(
        attestation, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    document = {
        "schema_version": 1,
        "kind": "mtplx_verified_guard_window",
        "verified": True,
        "verified_monotonic_ns": 2_000_000,
        "window_id": hashlib.sha256(encoded_attestation).hexdigest(),
        "attestation": attestation,
    }
    encoded_document = json.dumps(
        document, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return {
        **document,
        "receipt_path": "/tmp/mtplx-dsv4-guard-window-test/window.json",
        "receipt_sha256": hashlib.sha256(encoded_document).hexdigest(),
    }


def _install_report() -> dict:
    return {
        "route": "decode_verify_m4",
        "body_layers_installed": 43,
        "mtp_layers_stock": 1,
        "verify_rows": 4,
        "repair_rows": 1,
        "topk": 6,
        "hidden_size": 4096,
        "kernel_selfcheck_exact": True,
    }


def _receipt(
    tps: float,
    *,
    candidate: bool = False,
    role: str = "measurement",
    tokens: list[int] | None = None,
    guard_window: dict | None = None,
) -> dict:
    tokens = list(range(256)) if tokens is None else tokens
    stats = {
        "accepted_by_depth": [60, 40, 20],
        "drafted_by_depth": [80, 60, 40],
        "accepted_drafts": 120,
        "rejected_drafts": 40,
        "drafted_tokens": 180,
        "skipped_drafts": 0,
        "bonus_tokens": 20,
        "correction_tokens": 0,
        "verify_calls": 80,
        "mtp_forward_calls": 180,
        "make_mtp_cache_calls": 80,
        "update_mtp_cache_calls": 80,
        "mtp_history_append_calls": 80,
        "forward_ar_hidden_calls": 161,
        "forward_ar_plain_calls": 0,
    }
    return {
        "status": 0,
        "source_commit": "8" * 40,
        "model_path": "/Users/davidtai/models/DeepSeek-V4-Flash-2bit-DQ-mtp",
        "model_type": "deepseek_v4",
        "num_hidden_layers": 43,
        "num_nextn_predict_layers": 1,
        "host": {"mlx_version": "0.31.2"},
        "mlx_identity": {
            "version": "0.31.2",
            "core_sha256": "d7bd29fc20b4a08318d21161c3dfb340889cc9454c5e554ad749eb0127cfa2d6",
            "lib_sha256": "2ee6fbd32ff22e22e1301ebe3c3bece95584104ff9cbc900513d41a095211bbd",
        },
        "artifact_identity": {
            "config_sha256": "c8ff87fd5ee5c9587d0c937e9bfd3193e1a1621141aa367848a9610b3291fa6f",
            "index_sha256": "c84d2b369f5d5023d0f2d183fc36a935a3981751414996243b65f069983e43d8",
            "model_type": "deepseek_v4",
            "num_hidden_layers": 43,
            "num_nextn_predict_layers": 1,
            "body_q2_routed_projections": 129,
            "body_q2_manifest_tensors": 387,
            "mtp_manifest_tensors": 35,
            "index_weight_count": 2645,
        },
        "loaded_runtime_identity": {
            "runtime_mtp_enabled": True,
            "body_layers_loaded": 43,
            "mtp_blocks_bound": 1,
            "body_q2_routed_projections": 129,
            "body_q2_weight_dtype": "uint32",
            "mtp_mxfp4_routed_projections": 3,
            "mtp_routed_weight_dtype": "uint32",
        },
        "prompt_file": (
            "/Users/davidtai/projects/OpenSourceWTF/bench/deepseek-v4/"
            "smoke-2bitdq-20260731-prompt2.txt"
        ),
        "prompt": {
            "path": (
                "/Users/davidtai/projects/OpenSourceWTF/bench/deepseek-v4/"
                "smoke-2bitdq-20260731-prompt2.txt"
            ),
            "sha256": "ee94397faa812c91d5f1a0ee17c5bb6ca6032883653591dd33d4cfddb737ac33",
            "tokens": 328,
        },
        "prompt_tokens": 328,
        "max_tokens": 256,
        "depths": [3],
        "verify_strategy": "capture_commit",
        "verify_core": "stock",
        "mtp_history_policy": "committed",
        "receipt_role": role,
        "performance_eligible": role == "measurement",
        "launch_mtplx_env": _stage4_env(candidate),
        "guard_window": _guard_window() if guard_window is None else guard_window,
        "deepseek_v4_moe_tail": _install_report() if candidate else None,
        "arms": [
            {
                "speculative_depth": 3,
                "generated_tokens": 256,
                "tokens": tokens,
                "peak_gib": 97.0,
                "decode_tokens_per_second": tps,
                "stats": stats,
            }
        ],
    }


def test_shell_is_hermetic_and_orders_primer_c0_candidate_c1_validator():
    source = _ARMS.read_text()
    issue = source.index('deepseek_v4_guard_window.py" issue')
    primer = source.index('run_arm "DISCARDED full K3 control primer" 0 primer')
    c0 = source.index('run_arm "C0 Stage-4 control" 0 before')
    candidate = source.index('run_arm "MoE-tail M4 candidate" 1 candidate')
    c1 = source.index('run_arm "C1 Stage-4 control" 0 after')
    validator = source.index('if "$VENV" -u "$VALIDATOR"')
    assert issue < source.index("shasum -a 256") < primer < c0 < candidate < c1 < validator
    assert '"$name" == MTPLX_*' in source
    assert "HF_HUB_OFFLINE=1" in source and "PYTHONNOUSERSITE=1" in source
    assert "--max-tokens 256 --depths 3" in source
    assert "--verify-strategy capture_commit --verify-core stock" in source
    assert "--mtp-history-policy committed" in source
    assert "discarded_control_primer" in source
    assert "if \"$VENV\" -u \"$VALIDATOR\"" in source
    assert "receipts preserved" in source


def test_benchmark_verifies_guard_before_mlx_and_records_tail_installation():
    source = _BENCHMARK.read_text()
    assert source.index("load_verified_guard_window()") < source.index(
        "import mlx.core as mx"
    )
    assert "_deepseek_v4_moe_tail_install_report" in source
    assert "loaded_runtime_identity" in source
    assert "mlx_identity" in source
    assert "receipt_role" in source


def test_benchmark_install_report_proves_43_body_routes_and_stock_mtp():
    class Route:
        pass

    def stock(*_args):
        return None

    backend = SimpleNamespace(
        _InstalledMoETailRoute=Route,
        _stock_moe_tail_combine=stock,
        _MOE_TAIL=True,
        _MOE_TAIL_SELF_CHECKED=True,
        _MOE_TAIL_KERNEL=object(),
    )
    model = SimpleNamespace(
        layers=[SimpleNamespace(ffn=SimpleNamespace(_tail_combine=Route()))] * 43,
        mtp_blocks=[SimpleNamespace(ffn=SimpleNamespace(_tail_combine=stock))],
    )
    assert H._deepseek_v4_moe_tail_install_report(
        SimpleNamespace(model=model), backend
    ) == _install_report()

    backend._MOE_TAIL = False
    for layer in model.layers:
        layer.ffn._tail_combine = stock
    assert H._deepseek_v4_moe_tail_install_report(
        SimpleNamespace(model=model), backend
    ) is None


def test_direct_benchmark_refuses_before_importing_mlx(tmp_path: Path):
    fake_package = tmp_path / "mlx"
    fake_package.mkdir()
    marker = tmp_path / "mlx-imported"
    (fake_package / "__init__.py").write_text("")
    (fake_package / "core.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('imported')\n"
    )
    environment = {**os.environ, "PYTHONPATH": str(tmp_path)}
    for key in tuple(environment):
        if key.startswith("MTPLX_GUARD_ATTEST_") or key.startswith(
            "MTPLX_DSV4_GUARD_WINDOW_"
        ):
            del environment[key]
    completed = subprocess.run(
        [sys.executable, str(_BENCHMARK), "--tiny"],
        capture_output=True,
        text=True,
        timeout=10,
        env=environment,
        check=False,
    )
    assert completed.returncode != 0
    assert "verified guard window environment is absent or malformed" in completed.stderr
    assert not marker.exists()


def test_guard_attestation_survives_real_zsh_four_grandchild_hops(tmp_path: Path):
    lock_path = tmp_path / "mlx.lock"
    lock_path.write_bytes(b"")
    lock_descriptor = os.open(lock_path, os.O_RDONLY)
    fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock_stat = os.fstat(lock_descriptor)
    read_fd, write_fd = os.pipe()
    nonce = "a" * 64
    output = tmp_path / "windows.jsonl"
    command = (
        'issued=$("$1" -u "$2" issue --expected-lock "$3") || exit $?; '
        "export MTPLX_DSV4_GUARD_WINDOW_PATH=${issued%%$'\\t'*}; "
        "export MTPLX_DSV4_GUARD_WINDOW_SHA256=${issued#*$'\\t'}; "
        '"$1" -u "$2" verify >> "$4" || exit $?; '
        '"$1" -u "$2" verify >> "$4" || exit $?; '
        '"$1" -u "$2" verify >> "$4" || exit $?; '
        '"$1" -u "$2" verify >> "$4"'
    )
    environment = {
        **os.environ,
        "MTPLX_GUARD_ATTEST_FD": str(read_fd),
        "MTPLX_GUARD_ATTEST_NONCE": nonce,
    }
    process = subprocess.Popen(
        (
            "/bin/zsh",
            "-c",
            command,
            "zsh",
            sys.executable,
            str(_GUARD),
            str(lock_path),
            str(output),
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
        pass_fds=(read_fd,),
    )
    issued = time.monotonic_ns()
    payload = {
        "schema_version": 1,
        "nonce": nonce,
        "guard_pid": os.getpid(),
        "child_pid": process.pid,
        "issued_monotonic_ns": issued,
        "expires_monotonic_ns": issued + 60_000_000_000,
        "lock_path": str(lock_path.resolve()),
        "lock_device": lock_stat.st_dev,
        "lock_inode": lock_stat.st_ino,
    }
    os.close(read_fd)
    os.write(write_fd, json.dumps(payload).encode())
    os.close(write_fd)
    _stdout, stderr = process.communicate(timeout=15)
    fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
    os.close(lock_descriptor)
    assert process.returncode == 0, stderr
    windows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(windows) == 4 and all(window == windows[0] for window in windows)
    receipt_path = Path(windows[0]["receipt_path"])
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o400
    assert hashlib.sha256(receipt_path.read_bytes()).hexdigest() == windows[0][
        "receipt_sha256"
    ]
    receipt_path.unlink()
    receipt_path.parent.rmdir()


def test_validator_passes_only_clear_gain_beyond_post_primer_control_drift():
    primer = _receipt(1_000_000.0, role="discarded_control_primer")
    before = _receipt(30.0)
    candidate = _receipt(32.0, candidate=True)
    after = _receipt(30.4)
    result = V.validate_moe_tail_k3_bracket(
        primer, before, candidate, after, peak_ceiling_gib=108.0
    )
    assert result["status"] == "PASS"
    assert result["integrity_pass"] is True
    assert result["tokens"]["all_equal"] is True
    assert result["counters"]["all_equal"] is True
    assert result["primer"]["performance_data_used"] is False
    assert result["control"]["candidate_delta_fraction"] > result["control"][
        "drift_fraction"
    ]


def test_validator_preserves_a_correct_but_slower_candidate_as_loss():
    result = V.validate_moe_tail_k3_bracket(
        _receipt(1000.0, role="discarded_control_primer"),
        _receipt(30.0),
        _receipt(29.0, candidate=True),
        _receipt(30.2),
        peak_ceiling_gib=108.0,
    )
    assert result["status"] == "LOSS"
    assert result["integrity_pass"] is True
    assert result["performance_pass"] is False


def test_validator_cli_writes_loss_receipt_before_returning_nonzero(tmp_path: Path):
    inputs = {
        "primer": _receipt(1000.0, role="discarded_control_primer"),
        "before": _receipt(30.0),
        "candidate": _receipt(29.0, candidate=True),
        "after": _receipt(30.2),
    }
    paths = {}
    for name, receipt in inputs.items():
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(receipt))
    verdict = tmp_path / "validation.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(_VALIDATOR),
            "--primer",
            str(paths["primer"]),
            "--before",
            str(paths["before"]),
            "--candidate",
            str(paths["candidate"]),
            "--after",
            str(paths["after"]),
            "--out",
            str(verdict),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    assert verdict.is_file()
    assert json.loads(verdict.read_text())["status"] == "LOSS"


@pytest.mark.parametrize("mutation", ("tokens", "counters", "peak", "guard"))
def test_validator_rejects_integrity_mismatch(mutation: str):
    primer = _receipt(1000.0, role="discarded_control_primer")
    before = _receipt(30.0)
    candidate = _receipt(32.0, candidate=True)
    after = _receipt(30.2)
    if mutation == "tokens":
        candidate["arms"][0]["tokens"][4] = 999
    elif mutation == "counters":
        candidate["arms"][0]["stats"]["verify_calls"] += 1
    elif mutation == "peak":
        candidate["arms"][0]["peak_gib"] = 109.0
    else:
        candidate["guard_window"] = _guard_window(child_pid=201)
    result = V.validate_moe_tail_k3_bracket(
        primer, before, candidate, after, peak_ceiling_gib=108.0
    )
    assert result["status"] == "INVALID_BRACKET"
    assert result["integrity_pass"] is False


@pytest.mark.parametrize(
    "mutation",
    ("model", "config", "index", "prompt", "mlx", "topology", "quant", "env"),
)
def test_validator_rejects_noncanonical_identity(mutation: str):
    primer = _receipt(1000.0, role="discarded_control_primer")
    before = _receipt(30.0)
    candidate = _receipt(32.0, candidate=True)
    after = _receipt(30.2)
    target = candidate
    if mutation == "model":
        target["model_path"] += "-wrong"
    elif mutation == "config":
        target["artifact_identity"]["config_sha256"] = "0" * 64
    elif mutation == "index":
        target["artifact_identity"]["index_sha256"] = "0" * 64
    elif mutation == "prompt":
        target["prompt"]["sha256"] = "0" * 64
    elif mutation == "mlx":
        target["mlx_identity"]["version"] = "0.32.0"
    elif mutation == "topology":
        target["artifact_identity"]["num_nextn_predict_layers"] = 0
    elif mutation == "quant":
        target["loaded_runtime_identity"]["body_q2_routed_projections"] = 0
    else:
        target["launch_mtplx_env"]["SURPRISE"] = "1"
    result = V.validate_moe_tail_k3_bracket(
        primer, before, candidate, after, peak_ceiling_gib=108.0
    )
    assert result["status"] == "INVALID_BRACKET"


def test_validator_requires_candidate_report_and_stock_controls():
    primer = _receipt(1000.0, role="discarded_control_primer")
    before = _receipt(30.0)
    candidate = _receipt(32.0, candidate=True)
    after = _receipt(30.2)
    candidate["deepseek_v4_moe_tail"] = None
    before["deepseek_v4_moe_tail"] = _install_report()
    result = V.validate_moe_tail_k3_bracket(
        primer, before, candidate, after, peak_ceiling_gib=108.0
    )
    assert result["status"] == "INVALID_BRACKET"
    assert any("installation" in error or "stock" in error for error in result["errors"])
