"""Compare a tiny Torch Qwen4-Exp reference with an arbitrary MLX port.

The two backends intentionally run in separate interpreters.  MLX and Torch are
installed in different Python environments on the target machine, and mixing
their binary wheels through PYTHONPATH is unsafe.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TORCH_PYTHON = Path("/Users/mac/.venvs/pm-transformers/bin/python")
DEFAULT_MLX_PYTHON = Path(sys.executable)
DEFAULT_MLX_IMPL = REPO_ROOT / "mtplx/models/qwen4_exp.py"
SEED = 20260826


def _command_output(argv: list[str]) -> str:
    result = subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _machine_safety_gate() -> bool:
    """Refuse model workers while another model process is live."""

    processes = _command_output(
        [
            "pgrep",
            "-fl",
            "mtplx(\\.cli)? (serve|bench prefill-ladder)|mtplx.server.openai|mlx_lm",
        ]
    )
    pressure = _command_output(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"])
    print(
        f"SAFETY pressure={pressure or 'unknown'} "
        f"concurrent_model_process={bool(processes)}",
        flush=True,
    )
    if processes:
        print("SAFETY_REFUSE another model process is live:", flush=True)
        print(processes, flush=True)
        return False
    return True


def tiny_config() -> dict[str, object]:
    """Return a geometry that exercises every Qwen4-Exp text subsystem."""
    yarn_factor_raw = (os.environ.get("QWEN4EXP_NUMERIC_YARN_FACTOR") or "").strip()
    yarn_factor = float(yarn_factor_raw) if yarn_factor_raw else None
    rope_parameters: dict[str, object] = {
        "rope_type": "default",
        "rope_theta": 10_000.0,
        "partial_rotary_factor": 0.5,
        "mrope_section": [1, 1, 0],
    }
    max_position_embeddings = 64
    if yarn_factor is not None:
        if yarn_factor < 1.0:
            raise ValueError("QWEN4EXP_NUMERIC_YARN_FACTOR must be >= 1")
        rope_parameters.update(
            {
                "rope_type": "yarn",
                "factor": yarn_factor,
                "original_max_position_embeddings": 64,
            }
        )
        max_position_embeddings = int(64 * yarn_factor)
    return {
        "model_type": "qwen4_exp_text",
        "vocab_size": 32,
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "max_position_embeddings": max_position_embeddings,
        "rope_parameters": rope_parameters,
        "layer_types": ["linear_attention", "full_attention"],
        "linear_conv_kernel_dim": 2,
        "linear_key_head_dim": 4,
        "linear_value_head_dim": 4,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        # Keep 2*intermediate != hidden: v2.10.0's raw-HF sanitizer uses
        # those axes to distinguish the two packed expert layouts.  A square
        # test tensor is shape-ambiguous and can validate the wrong transpose.
        "moe_intermediate_size": 7,
        "shared_expert_intermediate_size": 8,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "hc_count": 2,
        "hc_lowrank": 4,
        "ple_layer_ids": [1],
        "ple_embed_dim": 8,
        "ple_conv_kernel_size": 2,
        "ngram_size": 3,
        "heads_per_ngram": 2,
        "ngram_vocab_size_base": 17,
        "make_ngram_vocab_size_divisible_by": 8,
        "split_ngram_parts": 1,
        "indexer_n_heads": 2,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 8,
        "indexer_budget": 4,
        "indexer_compress_ratio": 2,
        "eos_token_id": 1,
        "bos_token_id": 1,
        "pad_token_id": 0,
        "output_gate_type": "sigmoid",
        "norm_topk_prob": True,
        "tie_word_embeddings": False,
        "seed": 1234,
    }


def write_inputs(path: Path) -> None:
    import numpy as np

    rng = np.random.default_rng(SEED)
    seq_len = 7
    hidden_size = 16
    hc_count = 2
    ids = np.array([[1, 5, 7, 1, 9, 10, 11]], dtype=np.int64)
    hidden = rng.normal(0.0, 0.2, (1, seq_len, hidden_size)).astype(np.float32)
    hyper = rng.normal(0.0, 0.2, (1, seq_len, hidden_size * hc_count)).astype(
        np.float32
    )
    causal = np.full(
        (1, 1, seq_len, seq_len), np.finfo(np.float32).min, dtype=np.float32
    )
    causal = np.triu(causal, k=1)
    np.savez(path, ids=ids, hidden=hidden, hyper=hyper, causal=causal)


def torch_worker(workspace: Path) -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["USE_HUB_KERNELS"] = "NO"

    import numpy as np
    import torch
    from transformers import Qwen4ExpForCausalLM, Qwen4ExpTextConfig

    config = Qwen4ExpTextConfig(**tiny_config())
    config._attn_implementation = "eager"
    # The deliberately odd tiny expert width violates Torch CPU grouped_mm's
    # 16-byte stride contract; use Qwen's architecture-native eager reference.
    config._experts_implementation = "eager"
    torch.manual_seed(SEED)
    model = Qwen4ExpForCausalLM(config).eval()

    state = {
        name: tensor.detach().cpu().numpy()
        for name, tensor in model.state_dict().items()
    }
    np.savez(workspace / "torch-state.npz", **state)

    fixture = np.load(workspace / "inputs.npz")
    ids = torch.from_numpy(fixture["ids"])
    hidden = torch.from_numpy(fixture["hidden"])
    hyper = torch.from_numpy(fixture["hyper"])
    causal = torch.from_numpy(fixture["causal"])
    positions = torch.arange(ids.shape[1], dtype=torch.long).unsqueeze(0)
    position_embeddings = model.model.rotary_emb(hidden, positions)
    linear_layer = model.model.layers[0]
    sparse_layer = model.model.layers[1]

    with torch.inference_mode():
        qsa_mask = sparse_layer.self_attn.indexer(
            hidden,
            position_embeddings,
            causal,
            None,
        )
        outputs = {
            "gated_delta_net": linear_layer.linear_attn(hidden),
            "qsa_mask": ((qsa_mask == 0) & (causal == 0)).float(),
            "sparse_attention": sparse_layer.self_attn(
                hidden,
                position_embeddings,
                attention_mask=causal,
                past_key_values=None,
            )[0],
            "moe": linear_layer.mlp(hidden),
            "ngram": linear_layer.ple.ple_embedding(ids, None),
            "ple": linear_layer.ple(hyper, ids, None),
            "linear_decoder_layer": linear_layer(
                hyper,
                position_embeddings,
                attention_mask=causal,
                conv_mask=None,
                past_key_values=None,
                ple_input_ids=ids,
            ),
            "sparse_decoder_layer": sparse_layer(
                hyper,
                position_embeddings,
                attention_mask=causal,
                conv_mask=None,
                past_key_values=None,
                ple_input_ids=ids,
            ),
            "end_to_end_logits": model(input_ids=ids, use_cache=False).logits,
        }

    np.savez(
        workspace / "torch-outputs.npz",
        **{
            name: value.detach().float().cpu().numpy()
            for name, value in outputs.items()
        },
    )


def load_external_mlx_module(source: Path) -> object:
    package_parts = [source.stem]
    package_root = source.parent
    while (package_root / "__init__.py").is_file():
        package_parts.insert(0, package_root.name)
        package_root = package_root.parent
    module_name = (
        ".".join(package_parts) if len(package_parts) > 1 else "mlx_lm.models.qwen4_exp"
    )
    sys.path.insert(0, str(package_root))
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import MLX implementation: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _install_indexer_route_probe(module: object) -> dict[str, object]:
    """Record eager-fused and compiled-core entries without changing work."""

    fused_calls: list[str] = []
    compiled_calls: list[dict[str, str]] = []
    compiled_cores: list[object] = []
    original = module.QSAIndexer._select_fused

    def recorded(
        self,
        q,
        pos_start,
        pooled_backing,
        logical_blocks,
        total,
        mode,
    ):
        fused_calls.append(str(mode))
        return original(
            self,
            q,
            pos_start,
            pooled_backing,
            logical_blocks,
            total,
            mode,
        )

    module.QSAIndexer._select_fused = recorded
    from mtplx.kernels.qsa_indexer_compile import QSACompiledIndexerCore

    original_hidden = QSACompiledIndexerCore.select_hidden
    original_qk_rows = QSACompiledIndexerCore.select_qk_rows

    def remember_core(core: object) -> None:
        if not any(existing is core for existing in compiled_cores):
            compiled_cores.append(core)

    def recorded_hidden(self, *args, **kwargs):
        result = original_hidden(self, *args, **kwargs)
        remember_core(self)
        compiled_calls.append({"source": "hidden", "mode": str(kwargs.get("mode"))})
        return result

    def recorded_qk_rows(self, *args, **kwargs):
        result = original_qk_rows(self, *args, **kwargs)
        remember_core(self)
        compiled_calls.append({"source": "qk_rows", "mode": str(kwargs.get("mode"))})
        return result

    QSACompiledIndexerCore.select_hidden = recorded_hidden
    QSACompiledIndexerCore.select_qk_rows = recorded_qk_rows
    return {
        "fused_calls": fused_calls,
        "compiled_calls": compiled_calls,
        "compiled_cores": compiled_cores,
    }


def _indexer_route_receipt(probe: dict[str, object]) -> dict[str, object]:
    fused_calls = probe["fused_calls"]
    compiled_calls = probe["compiled_calls"]
    compiled_cores = probe["compiled_cores"]
    reports = [core.to_dict() for core in compiled_cores]
    return {
        "fused_calls": len(fused_calls),
        "fused_modes": sorted(set(fused_calls)),
        "compiled_calls": len(compiled_calls),
        "compiled_sources": sorted({call["source"] for call in compiled_calls}),
        "compiled_modes": sorted({call["mode"] for call in compiled_calls}),
        "compiled_cores": len(reports),
        "compiled_core_calls": sum(int(report["compiled_calls"]) for report in reports),
        "compiled_core_traces": sum(int(report["traces"]) for report in reports),
        "compiled_core_entries": sum(int(report["entry_count"]) for report in reports),
    }


def _selector_from_environment() -> str:
    truthy = {"1", "true", "yes", "on"}
    compiled = (os.environ.get("MTPLX_COMPILED_QSA_INDEXER") or "0").strip().lower()
    fused = (os.environ.get("MTPLX_FUSED_QSA_INDEXER") or "0").strip().lower()
    if compiled in truthy and fused in truthy:
        return "compiled"
    if fused in truthy:
        return "fused"
    return "eager"


def mlx_worker(workspace: Path, mlx_impl: Path, selector: str) -> None:
    for key in tuple(os.environ):
        if key.startswith("MTPLX_"):
            os.environ.pop(key)
    os.environ["MTPLX_FUSED_QSA_INDEXER"] = (
        "1" if selector in {"fused", "compiled"} else "0"
    )
    os.environ["MTPLX_COMPILED_QSA_INDEXER"] = "1" if selector == "compiled" else "0"
    # Numeric parity does not exercise generation's speculative replay policy.
    os.environ["MTPLX_QSA_MTP_PRECOMPUTE"] = "0"

    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten

    module = load_external_mlx_module(mlx_impl)
    route_probe = _install_indexer_route_probe(module)
    args = module.ModelArgs(
        model_type="qwen4_exp",
        text_config=tiny_config(),
    )
    model = module.Model(args)
    # The serving kernel requires production-scale head geometry.  Training mode
    # selects the same deterministic reference recurrence for this tiny fixture.
    model.train()

    with np.load(workspace / "torch-state.npz") as archive:
        state = {name: archive[name] for name in archive.files}
    # v2.10.0 owns the raw-HF conversion on Model.sanitize().  Feeding that
    # boundary directly is important: it remaps the new language_model tree,
    # stacks the packed experts, transposes both convolution layouts, shifts
    # zero-centered RMSNorm weights, and materializes the tiny PLE table.
    remapped = model.sanitize({name: mx.array(value) for name, value in state.items()})
    expected = {name: value for name, value in tree_flatten(model.parameters())}
    missing = sorted(set(expected) - set(remapped))
    unexpected = sorted(set(remapped) - set(expected))
    shape_errors = [
        f"{name}: expected {tuple(expected[name].shape)}, got {tuple(remapped[name].shape)}"
        for name in sorted(set(expected) & set(remapped))
        if tuple(expected[name].shape) != tuple(remapped[name].shape)
    ]
    if missing or unexpected or shape_errors:
        details = {
            "missing": missing,
            "unexpected": unexpected,
            "shape_errors": shape_errors,
        }
        raise RuntimeError(
            f"Torch-to-MLX state mapping is not bijective: {json.dumps(details)}"
        )

    model.load_weights(list(remapped.items()), strict=True)
    mx.eval(model.parameters())

    fixture = np.load(workspace / "inputs.npz")
    ids = mx.array(fixture["ids"])
    hidden = mx.array(fixture["hidden"])
    hyper = mx.array(fixture["hyper"])
    causal = mx.array(fixture["causal"])
    text_model = model.language_model
    linear_layer = text_model.model.layers[0]
    sparse_layer = text_model.model.layers[1]
    ratio = text_model.args.indexer_compress_ratio
    qsa_cache = module.QSACache(ratio)
    qsa_sel = sparse_layer.self_attn.indexer(hidden, 0, qsa_cache)
    if qsa_sel is None:
        raise RuntimeError("tiny fixture did not activate QSA sparsification")
    if isinstance(qsa_sel, tuple) or qsa_sel.ndim != 4:
        raise RuntimeError(
            "numeric fixture requires the v2.10.0 dense QSA reference lane; "
            f"got {type(qsa_sel).__name__}"
        )
    qsa_keep = qsa_sel

    outputs = {
        "gated_delta_net": linear_layer.linear_attn(hidden, None, None),
        "qsa_mask": (qsa_keep & (causal == 0)).astype(mx.float32),
        "sparse_attention": sparse_layer.self_attn(hidden, module.QSACache(ratio)),
        "moe": linear_layer.mlp(hidden),
        "ngram": linear_layer.ple.ple_embedding(ids, None, linear_layer.ple.NGRAM_IDX),
        "ple": linear_layer.ple(hyper, ids, None),
        "linear_decoder_layer": linear_layer(
            hyper, input_ids=ids, ssm_mask=None, cache=None
        ),
        "sparse_decoder_layer": sparse_layer(
            hyper,
            input_ids=ids,
            ssm_mask=None,
            cache=module.QSACache(ratio),
        ),
        "end_to_end_logits": model(ids, cache=model.make_cache()),
    }
    mx.eval(*outputs.values())
    route = _indexer_route_receipt(route_probe)
    if selector == "compiled":
        route_ok = route["compiled_calls"] > 0 and route["fused_calls"] == 0
    elif selector == "fused":
        route_ok = route["fused_calls"] > 0 and route["compiled_calls"] == 0
    else:
        route_ok = route["fused_calls"] == 0 and route["compiled_calls"] == 0
    if not route_ok:
        raise RuntimeError(
            "QSA indexer route mismatch: "
            f"requested={selector} receipt={json.dumps(route, sort_keys=True)}"
        )
    print(
        f"INDEXER_ROUTE PASS requested={selector} "
        f"fused_calls={route['fused_calls']} fused_modes={route['fused_modes']} "
        f"compiled_calls={route['compiled_calls']} "
        f"compiled_sources={route['compiled_sources']} "
        f"compiled_modes={route['compiled_modes']} "
        f"compiled_traces={route['compiled_core_traces']} "
        f"compiled_entries={route['compiled_core_entries']}",
        flush=True,
    )
    np.savez(
        workspace / "mlx-outputs.npz",
        **{
            name: np.asarray(value, dtype=np.float32) for name, value in outputs.items()
        },
    )
    (workspace / "mapping.json").write_text(
        json.dumps(
            {
                "mapped_parameters": len(remapped),
                "indexer_requested": selector,
                "indexer_route": route,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def run_worker(command: Sequence[str], name: str) -> None:
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    if result.stdout:
        print(result.stdout, end="", file=sys.stderr)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"{name} worker exited {result.returncode}")


def compare_outputs(
    torch_path: Path,
    mlx_path: Path,
    threshold: float,
) -> tuple[list[dict[str, object]], bool]:
    import numpy as np

    rows: list[dict[str, object]] = []
    passed = True
    with np.load(torch_path) as torch_outputs, np.load(mlx_path) as mlx_outputs:
        if set(torch_outputs.files) != set(mlx_outputs.files):
            raise RuntimeError(
                "backend output sets differ: "
                f"torch={sorted(torch_outputs.files)}, mlx={sorted(mlx_outputs.files)}"
            )
        for name in torch_outputs.files:
            reference = torch_outputs[name].astype(np.float64)
            candidate = mlx_outputs[name].astype(np.float64)
            if reference.shape != candidate.shape:
                raise RuntimeError(
                    f"{name} shape mismatch: torch={reference.shape}, mlx={candidate.shape}"
                )
            difference = np.abs(reference - candidate)
            finite = bool(np.isfinite(reference).all() and np.isfinite(candidate).all())
            max_abs = float(difference.max(initial=0.0))
            mean_abs = float(difference.mean())
            row_passed = finite and max_abs <= threshold
            passed = passed and row_passed
            rows.append(
                {
                    "module": name,
                    "shape": list(reference.shape),
                    "max_abs": max_abs,
                    "mean_abs": mean_abs,
                    "finite": finite,
                    "passed": row_passed,
                }
            )
    return rows, passed


def parent(args: argparse.Namespace) -> int:
    script = Path(__file__).resolve()
    selector = args.selector or _selector_from_environment()
    if args.yarn_factor is not None:
        if args.yarn_factor < 1.0:
            raise ValueError("--yarn-factor must be >= 1")
        os.environ["QWEN4EXP_NUMERIC_YARN_FACTOR"] = str(args.yarn_factor)
    for label, path in (
        ("Torch interpreter", args.torch_python),
        ("MLX interpreter", args.mlx_python),
        ("MLX implementation", args.mlx_impl),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    with tempfile.TemporaryDirectory(prefix="qwen4exp-numerics-") as raw_workspace:
        workspace = Path(raw_workspace)
        write_inputs(workspace / "inputs.npz")
        run_worker(
            [
                str(args.torch_python),
                str(script),
                "--worker",
                "torch",
                "--workspace",
                str(workspace),
            ],
            "Torch",
        )
        run_worker(
            [
                str(args.mlx_python),
                str(script),
                "--worker",
                "mlx",
                "--workspace",
                str(workspace),
                "--mlx-impl",
                str(args.mlx_impl),
                "--selector",
                selector,
            ],
            "MLX",
        )
        rows, passed = compare_outputs(
            workspace / "torch-outputs.npz",
            workspace / "mlx-outputs.npz",
            args.threshold,
        )
        mapping = json.loads((workspace / "mapping.json").read_text(encoding="utf-8"))

    rope_label = (
        "default" if args.yarn_factor is None else f"yarn-{args.yarn_factor:g}x"
    )
    print(
        "Qwen4-Exp tiny numeric check "
        f"(threshold={args.threshold:.3e}, rope={rope_label})"
    )
    print(f"mapped parameters: {mapping['mapped_parameters']}")
    print(
        f"indexer route: requested={mapping['indexer_requested']} "
        f"receipt={json.dumps(mapping['indexer_route'], sort_keys=True)}"
    )
    print(f"{'module':<25} {'max_abs':>12} {'mean_abs':>12} {'verdict':>8}")
    for row in rows:
        verdict = "PASS" if row["passed"] else "FAIL"
        print(
            f"{row['module']:<25} {row['max_abs']:>12.5e} "
            f"{row['mean_abs']:>12.5e} {verdict:>8}"
        )
    print(f"overall: {'PASS' if passed else 'FAIL'}")

    receipt = {
        "schema": "qwen4exp.numeric.v1",
        "seed": SEED,
        "threshold": args.threshold,
        "rope": rope_label,
        "torch_python": str(args.torch_python),
        "mlx_python": str(args.mlx_python),
        "mlx_impl": str(args.mlx_impl),
        "mapped_parameters": mapping["mapped_parameters"],
        "indexer_requested": mapping["indexer_requested"],
        "indexer_route": mapping["indexer_route"],
        "modules": rows,
        "passed": passed,
    }
    if args.json_report is not None:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )
    return 0 if passed or args.report_only else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-python", type=Path, default=DEFAULT_TORCH_PYTHON)
    parser.add_argument("--mlx-python", type=Path, default=DEFAULT_MLX_PYTHON)
    parser.add_argument("--mlx-impl", type=Path, default=DEFAULT_MLX_IMPL)
    parser.add_argument(
        "--selector",
        choices=("eager", "fused", "compiled"),
        default=None,
        help=(
            "QSA indexer lane to gate; when omitted, resolve the existing "
            "MTPLX_COMPILED_QSA_INDEXER/MTPLX_FUSED_QSA_INDEXER environment"
        ),
    )
    parser.add_argument("--threshold", type=float, default=1e-4)
    parser.add_argument(
        "--yarn-factor",
        type=float,
        help=(
            "Gate static YaRN against Transformers using a tiny native window; "
            "for example --yarn-factor 4"
        ),
    )
    parser.add_argument("--json-report", type=Path)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="emit the baseline but do not fail the process when the threshold is exceeded",
    )
    parser.add_argument("--worker", choices=("torch", "mlx"), help=argparse.SUPPRESS)
    parser.add_argument("--workspace", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not _machine_safety_gate():
        return 2
    if args.worker is not None:
        if args.workspace is None:
            raise ValueError("--workspace is required in worker mode")
        if args.worker == "torch":
            torch_worker(args.workspace)
        else:
            mlx_worker(
                args.workspace,
                args.mlx_impl,
                args.selector or _selector_from_environment(),
            )
        return 0
    return parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
