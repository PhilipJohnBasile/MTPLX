#!/usr/bin/env python3
"""Minimal stochastic-contract smoke for the env-gated DFlash source."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from mtplx.benchmarks.protocol import summarize_external_draft_contract
from mtplx.dflash_source import DFlashDraftSource, load_dflash_draft
from mtplx.generation import generate_ar, generate_mtpk
from mtplx.runtime import load
from mtplx.sampling import SamplerConfig


EVIDENCE_SAMPLER = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--draft-revision", required=True)
    parser.add_argument(
        "--prompt", default="Write a Python function that adds two integers."
    )
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _encode_prompt(tokenizer, prompt: str) -> list[int]:
    messages = [{"role": "user", "content": prompt}]
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    return [int(token) for token in encoded]


def main() -> int:
    args = _args()
    if os.environ.get("MTPLX_DFLASH_DRAFT") != "1":
        raise SystemExit("set MTPLX_DFLASH_DRAFT=1")
    if os.environ.get("MTPLX_COMPILED_VERIFY", "off").strip().lower() != "off":
        raise SystemExit("set MTPLX_COMPILED_VERIFY=off for the tap-capture smoke")

    runtime = load(args.model, mtp=True)
    draft_model, draft_artifact = load_dflash_draft(
        args.draft_model,
        revision=args.draft_revision,
    )
    if not draft_artifact.get("immutable_revision"):
        raise SystemExit("smoke requires a Hub draft resolved to an immutable SHA")
    prompt_ids = _encode_prompt(runtime.tokenizer, args.prompt)
    draft_temperature_scale = os.environ.get(
        "MTPLX_DRAFT_TEMPERATURE_SCALE", ""
    ).strip()
    if draft_temperature_scale not in {"", "1", "1.0"}:
        raise SystemExit(
            "unset MTPLX_DRAFT_TEMPERATURE_SCALE for protocol-matched target/q sampling"
        )

    sampler = SamplerConfig(**EVIDENCE_SAMPLER)

    ar = generate_ar(
        runtime,
        prompt_ids,
        max_tokens=args.max_tokens,
        sampler=sampler,
        seed=args.seed,
    )
    mtp_d1 = generate_mtpk(
        runtime,
        prompt_ids,
        max_tokens=args.max_tokens,
        sampler=sampler,
        draft_sampler=sampler,
        speculative_depth=1,
        seed=args.seed,
        verify_strategy="capture_commit",
    )
    source = DFlashDraftSource(
        args.draft_model,
        block_size=args.block_size,
        draft_model=draft_model,
        draft_artifact=draft_artifact,
    )
    try:
        dflash = generate_mtpk(
            runtime,
            prompt_ids,
            max_tokens=args.max_tokens,
            sampler=sampler,
            draft_sampler=sampler,
            speculative_depth=args.block_size - 1,
            seed=args.seed,
            verify_strategy="sequential",
            block_draft_source=source,
        )
    finally:
        source.close()
    seeded_stream_equality = list(dflash.tokens) == list(ar.tokens) and list(
        mtp_d1.tokens
    ) == list(ar.tokens)
    observed_contract = summarize_external_draft_contract(
        dflash.stats.events,
        expected_sampler=EVIDENCE_SAMPLER,
        expected_draft_revision=str(draft_artifact["resolved_revision"]),
    )
    contract_smoke_pass = bool(
        observed_contract["contract_matches"]
        and observed_contract["soft_q_proposals"] > 0
    )
    payload = {
        "schema": 2,
        "model": str(args.model.resolve()),
        "model_config_sha256": hashlib.sha256(
            (args.model / "config.json").read_bytes()
        ).hexdigest(),
        "draft_model": args.draft_model,
        "draft_revision": args.draft_revision,
        "draft_resolved_revision": draft_artifact["resolved_revision"],
        "draft_artifact": draft_artifact,
        "prompt_sha256": hashlib.sha256(args.prompt.encode("utf-8")).hexdigest(),
        "enable_thinking": False,
        "target_sampler": dict(EVIDENCE_SAMPLER),
        "draft_sampler": dict(EVIDENCE_SAMPLER),
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "block_size": args.block_size,
        "generation_mode": "dflash-sampled-q-sequential",
        "verify_strategy": "sequential",
        "nax_enabled": False,
        "observed_dflash_contract": observed_contract,
        "contract_smoke_pass": contract_smoke_pass,
        "seeded_stream_equality_diagnostic": {
            "equal": seeded_stream_equality,
            "release_gate": False,
            "note": (
                "Same-seed equality, including greedy equality, is diagnostic "
                "only and cannot establish stochastic exactness."
            ),
        },
        "release_gate": {
            "pass": False,
            "status": "deferred_single_sample_no_distribution_receipt",
            "required_harness": "benchmarks/dflash_engine_suite.py",
        },
        "ar": {
            "tokens": len(ar.tokens),
            "decode_tok_s": ar.stats.decode_tok_s,
            "text_sha256": hashlib.sha256(ar.text.encode("utf-8")).hexdigest(),
        },
        "mtp_d1": {
            "tokens": len(mtp_d1.tokens),
            "decode_tok_s": mtp_d1.stats.decode_tok_s,
            "accepted_drafts": mtp_d1.stats.accepted_drafts,
            "rejected_drafts": mtp_d1.stats.rejected_drafts,
            "drafted_tokens": mtp_d1.stats.drafted_tokens,
            "verify_calls": mtp_d1.stats.verify_calls,
            "text_sha256": hashlib.sha256(mtp_d1.text.encode("utf-8")).hexdigest(),
        },
        "dflash": {
            "tokens": len(dflash.tokens),
            "decode_tok_s": dflash.stats.decode_tok_s,
            "accepted_drafts": dflash.stats.accepted_drafts,
            "rejected_drafts": dflash.stats.rejected_drafts,
            "drafted_tokens": dflash.stats.drafted_tokens,
            "verify_calls": dflash.stats.verify_calls,
            "verify_windows": dflash.stats.verify_calls,
            "verify_forward_calls": dflash.stats.verify_forward_calls,
            "repair_forward_calls": dflash.stats.repair_forward_calls,
            "text_sha256": hashlib.sha256(dflash.text.encode("utf-8")).hexdigest(),
        },
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if contract_smoke_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
