"""Tiny-random qwen4_exp MTP exactness and rollback gate.

This is deliberately a no-artifact gate: it constructs the v2.10.0 in-tree
qwen4_exp trunk and MTP head from small random parameters.  It exercises the
same public generation and cache-repair paths used by serving while avoiding
all real model weights.

The default lane enables the fused QSA selector so the command in the QSA
kernel acceptance gate tests the new path.  Pass ``--selector compiled`` for
the whole-indexer graph or ``--selector eager`` for the retained oracle.
Phase-3 replay staging remains independently default-off and is exercised
only with ``--mtp-precompute on``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, ClassVar

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _command_output(argv: list[str]) -> str:
    result = subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _machine_safety_gate() -> bool:
    """Refuse even a tiny model while another model process is live."""

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


def _tiny_text_args():
    from mtplx.models.qwen4_exp import TextArgs

    return TextArgs(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=128,
        layer_types=["linear_attention", "full_attention"],
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        # mlx-lm's GPU gated-delta kernel assigns Dk/32 values per lane, so
        # Dk=16 instantiates an invalid zero-length Metal array. Keep the
        # no-artifact fixture tiny while exercising the supported Dk floor.
        linear_key_head_dim=32,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        hc_count=2,
        hc_lowrank=16,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=16,
        indexer_budget=8,
        indexer_compress_ratio=2,
        ple_layer_ids=[],
        ple_embed_dim=64,
        ngram_vocab_size_base=128,
        heads_per_ngram=2,
        partial_rotary_factor=0.25,
        rope_theta=10_000.0,
        eos_token_id=0,
    )


def _build_tiny_model(seed: int):
    import mlx.core as mx

    from mtplx.models.qwen4_exp import Model, ModelArgs, Qwen4ExpMTP
    from mtplx.mtp_patch import validate_mtp_support

    mx.random.seed(seed)
    args = _tiny_text_args()
    model = Model(
        ModelArgs(
            model_type="qwen4_exp",
            text_config=asdict(args),
        )
    )
    model.language_model.mtp = Qwen4ExpMTP(model.language_model.args)
    model.eval()
    mx.eval(model.parameters())
    if not validate_mtp_support(model):
        raise RuntimeError("tiny qwen4_exp MTP surface failed validation")
    return model


class _TinyTokenizer:
    eos_token_id = None
    eos_token_ids: ClassVar[set[int]] = set()

    def decode(self, tokens, **_kwargs):
        return " ".join(str(int(token)) for token in tokens)


def _tiny_runtime(model):
    from mtplx.mtp_patch import MTPContract
    from mtplx.runtime import MTPLXRuntime

    return MTPLXRuntime(
        model=model,
        tokenizer=_TinyTokenizer(),
        model_path=Path("."),
        mtp_enabled=True,
        contract=MTPContract(),
    )


def _array_leaves(value: Any) -> list[Any]:
    import mlx.core as mx

    if isinstance(value, mx.array):
        return [value]
    if isinstance(value, dict):
        out = []
        for key in sorted(value, key=str):
            out.extend(_array_leaves(value[key]))
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(_array_leaves(item))
        return out
    return []


def _trees_equal(left: Any, right: Any) -> bool:
    import mlx.core as mx

    if isinstance(left, mx.array) or isinstance(right, mx.array):
        if not isinstance(left, mx.array) or not isinstance(right, mx.array):
            return False
        if left.shape != right.shape or left.dtype != right.dtype:
            return False
        return bool(mx.all(left == right).item())
    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict) or not isinstance(right, dict):
            return False
        if left.keys() != right.keys():
            return False
        return all(_trees_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
            return False
        return len(left) == len(right) and all(
            _trees_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _cache_state(cache) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    return (
        tuple(getattr(entry, "state", None) for entry in cache),
        tuple(getattr(entry, "meta_state", None) for entry in cache),
    )


def _cache_offsets(cache) -> tuple[int | None, ...]:
    return tuple(
        int(entry.offset) if getattr(entry, "offset", None) is not None else None
        for entry in cache
    )


def _install_indexer_route_probe() -> dict[str, object]:
    """Record eager-fused and compiled-core entries without changing work."""

    from mtplx.models.qwen4_exp import QSAIndexer

    fused_calls: list[str] = []
    compiled_calls: list[dict[str, str]] = []
    compiled_cores: list[object] = []
    original = QSAIndexer._select_fused

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

    QSAIndexer._select_fused = recorded
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
    reports = [core.to_dict() for core in probe["compiled_cores"]]
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


def _install_mtp_precompute_route_probe() -> dict[str, list[dict[str, object]]]:
    """Record generation-level Phase-3 staging and replay reconciliation."""

    from mtplx import generation

    stage_calls: list[dict[str, object]] = []
    replay_calls: list[dict[str, object]] = []
    original_stage = generation.precompute_and_stage_qsa_replay_caches
    original_replay = generation.precompute_mtp_indexer_replay

    def recorded_stage(caches, *, window_tokens):
        plans = original_stage(caches, window_tokens=window_tokens)
        stage_calls.append(
            {
                "window_tokens": int(window_tokens),
                "plans": len(plans),
            }
        )
        return plans

    def recorded_replay(*, cycle_offset, observed_offset):
        plan = original_replay(
            cycle_offset=cycle_offset,
            observed_offset=observed_offset,
        )
        replay_calls.append(
            {
                "speculative_rows": int(plan.speculative_rows),
                "primary_staged": bool(plan.primary_staged),
                "rollback_offset": int(plan.rollback_offset),
                "reappend_start": int(plan.reappend_start),
            }
        )
        return plan

    generation.precompute_and_stage_qsa_replay_caches = recorded_stage
    generation.precompute_mtp_indexer_replay = recorded_replay
    return {"stage_calls": stage_calls, "replay_calls": replay_calls}


def _mtp_precompute_route_receipt(
    probe: dict[str, list[dict[str, object]]],
) -> dict[str, object]:
    stage_calls = probe["stage_calls"]
    replay_calls = probe["replay_calls"]
    return {
        "stage_calls": len(stage_calls),
        "staged_plans": sum(int(call["plans"]) for call in stage_calls),
        "stage_windows": sorted({int(call["window_tokens"]) for call in stage_calls}),
        "replay_calls": len(replay_calls),
        "primary_staged_calls": sum(
            bool(call["primary_staged"]) for call in replay_calls
        ),
        "replay_speculative_rows": sorted(
            {int(call["speculative_rows"]) for call in replay_calls}
        ),
    }


def _run_generation_exactness(model, prompt_ids: list[int], tokens: int) -> bool:
    from mtplx.generation import generate_ar, generate_mtpk
    from mtplx.sampling import SamplerConfig

    greedy = SamplerConfig(temperature=0.0, top_p=1.0, top_k=0)
    ar = generate_ar(
        _tiny_runtime(model),
        prompt_ids,
        max_tokens=tokens,
        sampler=greedy,
        stop_token_ids=set(),
    )
    mtp = generate_mtpk(
        _tiny_runtime(model),
        prompt_ids,
        max_tokens=tokens,
        sampler=greedy,
        draft_sampler=greedy,
        speculative_depth=3,
        mtp_cache_policy="persistent",
        mtp_history_policy="committed",
        verify_strategy="batched",
        stop_token_ids=set(),
    )
    ar_tokens = list(ar.tokens)
    mtp_tokens = list(mtp.tokens)
    first_mismatch = next(
        (
            index
            for index, (ar_token, mtp_token) in enumerate(zip(ar_tokens, mtp_tokens))
            if ar_token != mtp_token
        ),
        None,
    )
    identical = ar_tokens == mtp_tokens
    print(f"AR_GREEDY_OK n={len(ar_tokens)} ids={ar_tokens[:8]}...", flush=True)
    print(
        f"MTP_GREEDY_OK n={len(mtp_tokens)} ids={mtp_tokens[:8]}... "
        f"accepted={mtp.stats.accepted_drafts} "
        f"rejected={mtp.stats.rejected_drafts}",
        flush=True,
    )
    print(
        f"EXACTNESS {'PASS' if identical else 'FAIL'} "
        f"ar==mtp {identical} first_mismatch={first_mismatch}",
        flush=True,
    )
    if not identical:
        print(f"AR  {ar_tokens}", flush=True)
        print(f"MTP {mtp_tokens}", flush=True)
    return identical


def _run_full_reject_rollback(model, prompt, verify_ids) -> bool:
    import mlx.core as mx

    from mtplx.cache_state import (
        rollback_after_verify,
        snapshot_cache,
        snapshot_untrimmable_cache,
    )

    cache = model.make_cache()
    logits, _hidden = model(prompt, cache=cache, return_hidden=True)
    mx.eval(logits)

    full_before = snapshot_cache(cache)
    rollback_snapshot = snapshot_untrimmable_cache(cache)
    before_offsets = _cache_offsets(cache)
    mx.eval(
        *_array_leaves(full_before.states),
        *_array_leaves(full_before.meta_states),
        *_array_leaves(rollback_snapshot.states),
        *_array_leaves(rollback_snapshot.meta_states),
    )

    verify_logits, _hidden = model(verify_ids, cache=cache, return_hidden=True)
    current_states, current_meta = _cache_state(cache)
    mx.eval(
        verify_logits,
        *_array_leaves(current_states),
        *_array_leaves(current_meta),
    )
    after_verify_offsets = _cache_offsets(cache)
    mutated = (
        after_verify_offsets != before_offsets
        or not _trees_equal(full_before.states, current_states)
        or not _trees_equal(full_before.meta_states, current_meta)
    )

    rollback_after_verify(cache, rollback_snapshot, int(verify_ids.shape[1]))
    restored_states, restored_meta = _cache_state(cache)
    mx.eval(*_array_leaves(restored_states), *_array_leaves(restored_meta))
    restored_offsets = _cache_offsets(cache)
    restored_equal = (
        restored_offsets == before_offsets
        and _trees_equal(full_before.states, restored_states)
        and _trees_equal(full_before.meta_states, restored_meta)
    )
    ok = mutated and restored_equal
    print(
        f"ROLLBACK {'PASS' if ok else 'FAIL'} "
        f"mutated={mutated} restored_equal={restored_equal} "
        f"before_offsets={before_offsets} "
        f"verify_offsets={after_verify_offsets} "
        f"restored_offsets={restored_offsets}",
        flush=True,
    )
    return ok


def _run_partial_accept_commit(model, prompt, verify_ids, next_ids) -> bool:
    import mlx.core as mx

    from mtplx.cache_state import snapshot_untrimmable_cache_lazy

    verified_tokens = int(verify_ids.shape[1])
    keep_tokens = 2

    cache = model.make_cache()
    model(prompt, cache=cache, return_hidden=True)
    snapshot = snapshot_untrimmable_cache_lazy(cache)
    with model.verify_capture_scope():
        model(verify_ids, cache=cache, return_hidden=True)
    committed = model.commit_verified_window(
        cache,
        snapshot.states,
        keep_tokens=keep_tokens,
        verified_tokens=verified_tokens,
    )
    out = model(next_ids, cache=cache)

    golden_cache = model.make_cache()
    model(prompt, cache=golden_cache, return_hidden=True)
    model(verify_ids[:, :keep_tokens], cache=golden_cache, return_hidden=True)
    golden = model(next_ids, cache=golden_cache)
    mx.eval(out, golden)

    candidate_states, candidate_meta = _cache_state(cache)
    golden_states, golden_meta = _cache_state(golden_cache)
    mx.eval(
        *_array_leaves(candidate_states),
        *_array_leaves(candidate_meta),
        *_array_leaves(golden_states),
        *_array_leaves(golden_meta),
    )

    max_abs = float(
        mx.max(mx.abs(out.astype(mx.float32) - golden.astype(mx.float32))).item()
    )
    logits_close = bool(mx.allclose(out, golden, atol=2e-5, rtol=2e-5).item())
    argmax_equal = bool(
        mx.all(mx.argmax(out, axis=-1) == mx.argmax(golden, axis=-1)).item()
    )
    offsets_equal = _cache_offsets(cache) == _cache_offsets(golden_cache)
    cache_equal = _trees_equal(candidate_states, golden_states) and _trees_equal(
        candidate_meta, golden_meta
    )
    ok = bool(
        committed and logits_close and argmax_equal and offsets_equal and cache_equal
    )
    print(
        f"PARTIAL_ACCEPT {'PASS' if ok else 'FAIL'} "
        f"keep={keep_tokens}/{verified_tokens} committed={committed} "
        f"argmax_equal={argmax_equal} logits_close={logits_close} "
        f"max_abs={max_abs:.9g} offsets_equal={offsets_equal} "
        f"cache_equal={cache_equal}",
        flush=True,
    )
    return ok


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selector",
        choices=("fused", "compiled", "eager"),
        default="fused",
        help="QSA indexer lane to gate (default: fused Metal selector)",
    )
    parser.add_argument(
        "--mtp-precompute",
        choices=("off", "on"),
        default="off",
        help="Phase-3 QSA/MTP replay staging lane (default: off)",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=32,
        help="number of greedy AR/MTP completion tokens (default: 32)",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.tokens <= 0:
        raise SystemExit("--tokens must be positive")
    if not _machine_safety_gate():
        return 2

    # Keep this gate focused on native MTP + the selected QSA implementation.
    # A serving shell can carry dozens of unrelated MTPLX optimization flags;
    # remove all of them before installing this fixture's explicit contract.
    for key in tuple(os.environ):
        if key.startswith("MTPLX_"):
            os.environ.pop(key)
    os.environ["MTPLX_FUSED_QSA_INDEXER"] = (
        "1" if args.selector in {"fused", "compiled"} else "0"
    )
    os.environ["MTPLX_COMPILED_QSA_INDEXER"] = (
        "1" if args.selector == "compiled" else "0"
    )
    os.environ["MTPLX_QSA_MTP_PRECOMPUTE"] = "1" if args.mtp_precompute == "on" else "0"
    os.environ["MTPLX_CONTEXT_COPY"] = "0"
    os.environ["MTPLX_GREEDY_DRAFT_CHAIN"] = "0"
    os.environ["MTPLX_FAMILY_CAPTURE_COMMIT"] = "0"
    os.environ["MTPLX_QSA_FLASH"] = "0"
    os.environ["MTPLX_QSA_GATHER"] = "0"
    os.environ["MTPLX_QSA_GATHER_DECODE"] = "0"

    import mlx.core as mx

    indexer_probe = _install_indexer_route_probe()
    mtp_precompute_probe = _install_mtp_precompute_route_probe()
    model = _build_tiny_model(args.seed)
    prompt = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]], mx.int32)
    verify_ids = mx.array([[17, 23, 31, 47]], mx.int32)
    next_ids = mx.array([[53, 59, 61]], mx.int32)
    print(
        f"CONFIG_OK base=v2.10.0 random_weights=True selector={args.selector} "
        f"mtp_precompute={args.mtp_precompute} "
        f"layers={len(model.layers)} mtp_layers={len(model.mtp.layers)}",
        flush=True,
    )

    exact = _run_generation_exactness(
        model,
        [int(token) for token in prompt[0].tolist()],
        args.tokens,
    )
    rollback = _run_full_reject_rollback(model, prompt, verify_ids)
    partial = _run_partial_accept_commit(model, prompt, verify_ids, next_ids)
    route = _indexer_route_receipt(indexer_probe)
    if args.selector == "compiled":
        route_ok = route["compiled_calls"] > 0 and route["fused_calls"] == 0
    elif args.selector == "fused":
        route_ok = route["fused_calls"] > 0 and route["compiled_calls"] == 0
    else:
        route_ok = route["fused_calls"] == 0 and route["compiled_calls"] == 0
    print(
        f"INDEXER_ROUTE {'PASS' if route_ok else 'FAIL'} "
        f"requested={args.selector} "
        f"fused_calls={route['fused_calls']} fused_modes={route['fused_modes']} "
        f"compiled_calls={route['compiled_calls']} "
        f"compiled_sources={route['compiled_sources']} "
        f"compiled_modes={route['compiled_modes']} "
        f"compiled_traces={route['compiled_core_traces']} "
        f"compiled_entries={route['compiled_core_entries']}",
        flush=True,
    )
    precompute_route = _mtp_precompute_route_receipt(mtp_precompute_probe)
    if args.mtp_precompute == "on":
        precompute_ok = (
            precompute_route["stage_calls"] > 0
            and precompute_route["staged_plans"] > 0
            and precompute_route["replay_calls"] > 0
        )
    else:
        precompute_ok = (
            precompute_route["stage_calls"] == 0
            and precompute_route["replay_calls"] == 0
        )
    print(
        f"MTP_PRECOMPUTE_ROUTE {'PASS' if precompute_ok else 'FAIL'} "
        f"requested={args.mtp_precompute} "
        f"stage_calls={precompute_route['stage_calls']} "
        f"staged_plans={precompute_route['staged_plans']} "
        f"stage_windows={precompute_route['stage_windows']} "
        f"replay_calls={precompute_route['replay_calls']} "
        f"primary_staged_calls={precompute_route['primary_staged_calls']} "
        f"speculative_rows={precompute_route['replay_speculative_rows']}",
        flush=True,
    )
    ok = exact and rollback and partial and route_ok and precompute_ok
    print(
        f"QWEN4EXP_MTP_TINY_SMOKE_{'OK' if ok else 'FAIL'}",
        flush=True,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
