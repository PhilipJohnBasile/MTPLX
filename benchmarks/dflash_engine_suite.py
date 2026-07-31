#!/usr/bin/env python3
"""Protocol-matched AR/MTP-D1/DFlash-B8 suite in one MTPLX runtime."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
from typing import Any

from mtplx.benchmarks.protocol import (
    assert_protocol_match,
    build_effective_run_record,
    weighted_tok_s,
)
from mtplx.benchmarks.schema import encode_prompt_case, load_prompt_suite
from mtplx.dflash_source import DFlashDraftSource
from mtplx.generation import generate_ar, generate_mtpk
from mtplx.runtime import load
from mtplx.sampling import SamplerConfig


AR = "ar"
MTP_D1 = "mtp_d1"
DFLASH_B8 = "dflash_b8"
ARMS = (AR, MTP_D1, DFLASH_B8)
RUNTIME_SWITCH_KEYS = (
    "MTPLX_COMPILED_VERIFY",
    "MTPLX_COMPILED_TARGET_PREFIX",
    "MTPLX_FUSE_GDN_POST_CONV",
    "MTPLX_QWEN_ROW_OWNED_ROUTER",
    "MTPLX_QWEN_MOE_PACK_GATE_UP",
    "MTPLX_A3B_WHOLE_MOE_FUSION",
    "MTPLX_SUSTAINED_PREFILL_LAYOUT",
    "MTPLX_CONTEXT_COPY",
    "MTPLX_SKIP_VERIFY_SNAPSHOT",
    "MTPLX_NAX_VERIFY",
)


def _env_on(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--draft-revision", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=112)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument(
        "--dflash-lane",
        choices=("wide", "staged-k1"),
        default="wide",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--include-traces",
        action="store_true",
        help="Include token IDs and per-cycle diagnostics in the JSON receipt.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _artifact_layout_revision(model: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(model.glob("*")):
        if not path.is_file():
            continue
        if path.name.endswith(".safetensors"):
            digest.update(f"{path.name}:{path.stat().st_size}\n".encode())
        elif path.name in {
            "config.json",
            "model.safetensors.index.json",
            "mtplx_runtime.json",
        }:
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return f"layout-sha256:{digest.hexdigest()}"


def _run_arm(
    arm: str,
    *,
    runtime: Any,
    prompt_ids: list[int],
    max_tokens: int,
    sampler: SamplerConfig,
    seed: int,
    draft_model: Any,
    draft_ref: str,
    block_size: int,
    dflash_lane: str,
):
    if arm == AR:
        return generate_ar(
            runtime,
            prompt_ids,
            max_tokens=max_tokens,
            sampler=sampler,
            seed=seed,
        )
    if arm == MTP_D1:
        whole_moe = _env_on("MTPLX_A3B_WHOLE_MOE_FUSION")
        return generate_mtpk(
            runtime,
            prompt_ids,
            max_tokens=max_tokens,
            sampler=sampler,
            speculative_depth=1,
            seed=seed,
            verify_strategy="target_prefix" if whole_moe else "capture_commit",
        )
    if arm == DFLASH_B8:
        source = DFlashDraftSource(
            draft_ref,
            block_size=block_size,
            staged_k1=dflash_lane == "staged-k1",
            draft_model=draft_model,
        )
        try:
            return generate_mtpk(
                runtime,
                prompt_ids,
                max_tokens=max_tokens,
                sampler=sampler,
                speculative_depth=(
                    1 if dflash_lane == "staged-k1" else block_size - 1
                ),
                seed=seed,
                verify_strategy=(
                    "target_prefix"
                    if dflash_lane == "staged-k1"
                    else "capture_commit"
                ),
                block_draft_source=source,
            )
        finally:
            source.close()
    raise ValueError(f"unknown arm {arm!r}")


def _effective_records(args: argparse.Namespace, model_revision: str):
    compiled_verify = os.environ.get(
        "MTPLX_COMPILED_VERIFY",
        "off",
    ).strip().lower()
    nax_enabled = _env_on("MTPLX_NAX_VERIFY")
    whole_moe = _env_on("MTPLX_A3B_WHOLE_MOE_FUSION")
    runtime_switches = {
        key: os.environ.get(key, "<unset>")
        for key in RUNTIME_SWITCH_KEYS
    }
    common = {
        "model_ref": args.model,
        "model_revision": model_revision,
        "prompt_suite": args.prompts,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "seed_policy": "base_plus_prompt_index",
        "enable_thinking": False,
        "nax_enabled": nax_enabled,
        "runtime_switches": runtime_switches,
    }
    records = {
        AR: build_effective_run_record(
            backend="mtplx",
            draft_ref=None,
            draft_revision=None,
            block_size=None,
            generation_mode="ar",
            verify_strategy="ar",
            compiled_verify="not_applicable",
            **common,
        ),
        MTP_D1: build_effective_run_record(
            backend="mtplx",
            draft_ref=None,
            draft_revision=None,
            block_size=1,
            generation_mode="mtp_d1",
            verify_strategy="target_prefix" if whole_moe else "capture_commit",
            compiled_verify=compiled_verify,
            **common,
        ),
        DFLASH_B8: build_effective_run_record(
            backend="mtplx",
            draft_ref=args.draft_model,
            draft_revision=args.draft_revision,
            block_size=args.block_size,
            generation_mode=(
                "dflash_staged_k1"
                if args.dflash_lane == "staged-k1"
                else "dflash_one_hot"
            ),
            verify_strategy=(
                "target_prefix"
                if args.dflash_lane == "staged-k1"
                else "capture_commit"
            ),
            compiled_verify=compiled_verify,
            **common,
        ),
    }
    assert_protocol_match(records[AR], records[MTP_D1])
    assert_protocol_match(records[AR], records[DFLASH_B8])
    return records


def main() -> int:
    args = _args()
    if os.environ.get("MTPLX_DFLASH_DRAFT") != "1":
        raise SystemExit("set MTPLX_DFLASH_DRAFT=1")
    compiled_verify = os.environ.get(
        "MTPLX_COMPILED_VERIFY",
        "off",
    ).strip().lower()
    if args.dflash_lane == "wide" and compiled_verify != "off":
        raise SystemExit("set MTPLX_COMPILED_VERIFY=off for the wide lane")
    if args.dflash_lane == "staged-k1" and compiled_verify not in {
        "off",
        "on",
    }:
        raise SystemExit("set MTPLX_COMPILED_VERIFY=off or on")
    if args.dflash_lane == "staged-k1":
        required_on = (
            "MTPLX_COMPILED_TARGET_PREFIX",
            "MTPLX_FUSE_GDN_POST_CONV",
            "MTPLX_QWEN_ROW_OWNED_ROUTER",
            "MTPLX_QWEN_MOE_PACK_GATE_UP",
            "MTPLX_A3B_WHOLE_MOE_FUSION",
        )
        missing = [name for name in required_on if not _env_on(name)]
        if compiled_verify != "on":
            missing.append("MTPLX_COMPILED_VERIFY=on")
        if os.environ.get("MTPLX_SUSTAINED_PREFILL_LAYOUT") != (
            "contiguous_dense_decode"
        ):
            missing.append(
                "MTPLX_SUSTAINED_PREFILL_LAYOUT=contiguous_dense_decode"
            )
        if missing:
            raise SystemExit(
                "staged-K1 requires the fail-closed compiled A3B stack: "
                + ", ".join(missing)
            )

    cases = load_prompt_suite(args.prompts)
    if args.limit is not None:
        cases = cases[: args.limit]
    runtime = load(args.model, mtp=True)
    draft_module = importlib.import_module("dflash.model_mlx")
    draft_model = draft_module.load_draft(args.draft_model)
    sampler = SamplerConfig(temperature=0.0, top_p=1.0, top_k=0)
    model_revision = _artifact_layout_revision(args.model)
    effective_runs = _effective_records(args, model_revision)

    rows: list[dict[str, Any]] = []
    exact_cases = 0
    dflash_exact_cases = 0
    mtp_exact_cases = 0
    for case_index, case in enumerate(cases):
        prompt_ids = encode_prompt_case(
            runtime.tokenizer,
            case,
            chat_template=True,
            enable_thinking=False,
        )
        case_max_tokens = min(args.max_tokens, case.max_tokens)
        rotation = case_index % len(ARMS)
        order = ARMS[rotation:] + ARMS[:rotation]
        outputs = {}
        for arm in order:
            output = _run_arm(
                arm,
                runtime=runtime,
                prompt_ids=prompt_ids,
                max_tokens=case_max_tokens,
                sampler=sampler,
                seed=args.seed + case_index,
                draft_model=draft_model,
                draft_ref=args.draft_model,
                block_size=args.block_size,
                dflash_lane=args.dflash_lane,
            )
            outputs[arm] = output
            row = {
                "prompt_id": case.id,
                "category": case.category,
                "prompt_sha256": case.prompt_sha256,
                "arm": arm,
                "order": list(order),
                "generated_tokens": len(output.tokens),
                "decode_seconds": output.stats.decode_elapsed_s,
                "tok_s": output.stats.decode_tok_s,
                "finish_reason": output.finish_reason,
                "accepted_drafts": output.stats.accepted_drafts,
                "rejected_drafts": output.stats.rejected_drafts,
                "drafted_tokens": output.stats.drafted_tokens,
                "verify_calls": output.stats.verify_calls,
                "peak_memory_bytes": output.stats.peak_memory_bytes,
                "token_sha256": hashlib.sha256(
                    json.dumps(list(output.tokens)).encode()
                ).hexdigest(),
                "text_sha256": hashlib.sha256(
                    output.text.encode("utf-8")
                ).hexdigest(),
            }
            if args.include_traces:
                row["token_ids"] = [int(token) for token in output.tokens]
                row["cycles"] = [
                        {
                            "step": event.get("step"),
                            "primary": event.get("primary"),
                            "accepted_depths": event.get("accepted_depths"),
                            "rejected_at_depth": event.get("rejected_at_depth"),
                            "block_draft_source": event.get("block_draft_source"),
                            "drafts": [
                                {
                                    "token": draft.get("token"),
                                    "accepted": draft.get("accepted"),
                                    "correction": draft.get("correction"),
                                    "target_top2_margin": draft.get(
                                        "target_top2_margin"
                                    ),
                                }
                                for draft in event.get("drafts", [])
                            ],
                        }
                        for event in output.stats.events
                    ]
            rows.append(row)
        hashes = {
            hashlib.sha256(json.dumps(list(output.tokens)).encode()).hexdigest()
            for output in outputs.values()
        }
        if len(hashes) == 1:
            exact_cases += 1
        if outputs[AR].tokens == outputs[DFLASH_B8].tokens:
            dflash_exact_cases += 1
        if outputs[AR].tokens == outputs[MTP_D1].tokens:
            mtp_exact_cases += 1

    summaries = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        summaries[arm] = {
            "prompts": len(arm_rows),
            "generated_tokens": sum(row["generated_tokens"] for row in arm_rows),
            "decode_seconds": sum(row["decode_seconds"] for row in arm_rows),
            "weighted_tok_s": weighted_tok_s(arm_rows),
            "accepted_drafts": sum(row["accepted_drafts"] for row in arm_rows),
            "rejected_drafts": sum(row["rejected_drafts"] for row in arm_rows),
            "drafted_tokens": sum(row["drafted_tokens"] for row in arm_rows),
            "verify_calls": sum(row["verify_calls"] for row in arm_rows),
            "peak_memory_bytes": max(
                (row["peak_memory_bytes"] for row in arm_rows),
                default=0,
            ),
        }
    ar_tok_s = summaries[AR]["weighted_tok_s"]
    for arm in (MTP_D1, DFLASH_B8):
        summaries[arm]["vs_ar"] = (
            summaries[arm]["weighted_tok_s"] / ar_tok_s
            if ar_tok_s > 0
            else 0.0
        )

    payload = {
        "schema": 1,
        "includes_traces": bool(args.include_traces),
        "effective_runs": effective_runs,
        "exact_cases": exact_cases,
        "dflash_exact_cases": dflash_exact_cases,
        "mtp_exact_cases": mtp_exact_cases,
        "total_cases": len(cases),
        "all_exact": exact_cases == len(cases),
        "dflash_all_exact": dflash_exact_cases == len(cases),
        "mtp_all_exact": mtp_exact_cases == len(cases),
        "summary": summaries,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in payload.items() if k != "rows"}, indent=2))
    return 0 if payload["dflash_all_exact"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
