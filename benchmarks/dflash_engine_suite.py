#!/usr/bin/env python3
"""Protocol-matched AR/MTP-D1/DFlash-B8 suite in one MTPLX runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from mtplx.benchmarks.protocol import (
    assert_protocol_match,
    build_effective_run_record,
    compare_token_position_distributions,
    summarize_external_draft_contract,
    weighted_tok_s,
)
from mtplx.benchmarks.schema import encode_prompt_case, load_prompt_suite
from mtplx.dflash_source import DFlashDraftSource, load_dflash_draft
from mtplx.generation import generate_ar, generate_mtpk
from mtplx.runtime import load
from mtplx.sampling import SamplerConfig


AR = "ar"
MTP_D1 = "mtp_d1"
DFLASH_B8 = "dflash_b8"
ARMS = (AR, MTP_D1, DFLASH_B8)
EVIDENCE_SAMPLER = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
}
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
        "--distribution-repeats",
        type=int,
        default=0,
        help=(
            "Independent AR/DFlash samples for the stochastic distribution "
            "gate. Zero records a fail-closed deferred gate."
        ),
    )
    parser.add_argument("--distribution-max-tokens", type=int, default=16)
    parser.add_argument("--distribution-max-positions", type=int, default=16)
    parser.add_argument("--distribution-min-samples", type=int, default=128)
    parser.add_argument("--distribution-permutations", type=int, default=2000)
    parser.add_argument("--distribution-alpha", type=float, default=0.01)
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


def _engine_source_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]

    def git(*args: str) -> bytes:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout

    head = git("rev-parse", "HEAD").decode().strip()
    status = git("status", "--porcelain=v1", "-z")
    digest = hashlib.sha256()
    digest.update(git("diff", "--binary", "HEAD", "--"))
    entries = [entry for entry in status.decode(errors="replace").split("\0") if entry]
    for entry in entries:
        if not entry.startswith("?? "):
            continue
        path = root / entry[3:]
        digest.update(entry.encode())
        if path.is_file():
            digest.update(path.read_bytes())
    return {
        "git_head": head,
        "dirty": bool(entries),
        "dirty_diff_sha256": digest.hexdigest(),
        "status": entries,
    }


def _run_arm(
    arm: str,
    *,
    runtime: Any,
    prompt_ids: list[int],
    max_tokens: int,
    sampler: SamplerConfig,
    seed: int,
    draft_model: Any,
    draft_artifact: dict[str, Any],
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
            draft_sampler=sampler,
            speculative_depth=1,
            seed=seed,
            verify_strategy="target_prefix" if whole_moe else "capture_commit",
            base_hidden_variant=runtime.contract.base_hidden_variant,
            mtp_hidden_variant=runtime.contract.hidden_variant,
        )
    if arm == DFLASH_B8:
        source = DFlashDraftSource(
            draft_ref,
            block_size=block_size,
            staged_k1=dflash_lane == "staged-k1",
            draft_model=draft_model,
            draft_artifact=draft_artifact,
        )
        try:
            return generate_mtpk(
                runtime,
                prompt_ids,
                max_tokens=max_tokens,
                sampler=sampler,
                draft_sampler=sampler,
                speculative_depth=(1 if dflash_lane == "staged-k1" else block_size - 1),
                seed=seed,
                verify_strategy="sequential",
                base_hidden_variant=runtime.contract.base_hidden_variant,
                mtp_hidden_variant=runtime.contract.hidden_variant,
                block_draft_source=source,
            )
        finally:
            source.close()
    raise ValueError(f"unknown arm {arm!r}")


def _effective_records(
    args: argparse.Namespace,
    model_revision: str,
    runtime: Any,
    draft_artifact: dict[str, Any],
):
    compiled_verify = (
        os.environ.get(
            "MTPLX_COMPILED_VERIFY",
            "off",
        )
        .strip()
        .lower()
    )
    nax_enabled = _env_on("MTPLX_NAX_VERIFY")
    whole_moe = _env_on("MTPLX_A3B_WHOLE_MOE_FUSION")
    runtime_switches = {
        key: os.environ.get(key, "<unset>") for key in RUNTIME_SWITCH_KEYS
    }
    common = {
        "model_ref": args.model,
        "model_revision": model_revision,
        "prompt_suite": args.prompts,
        **EVIDENCE_SAMPLER,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "seed_policy": "base_plus_prompt_index",
        "enable_thinking": False,
        "nax_enabled": nax_enabled,
        "runtime_switches": runtime_switches,
        "runtime_contract": runtime.contract.to_dict(),
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
            draft_revision=str(draft_artifact["resolved_revision"]),
            block_size=args.block_size,
            generation_mode=(
                "dflash_staged_k1"
                if args.dflash_lane == "staged-k1"
                else "dflash_sampled_q_sequential"
            ),
            verify_strategy="sequential",
            compiled_verify=compiled_verify,
            **common,
        ),
    }
    assert_protocol_match(records[AR], records[MTP_D1])
    assert_protocol_match(records[AR], records[DFLASH_B8])
    return records


def _aggregate_contracts(contracts: list[dict[str, Any]]) -> dict[str, Any]:
    declarations: dict[str, int] = {}
    acceptance_lanes: dict[str, int] = {}
    draft_q_kinds: dict[str, int] = {}
    sampler_mismatches: list[dict[str, Any]] = []
    draft_revision_mismatches: list[dict[str, Any]] = []
    engine_q_mismatches: list[dict[str, Any]] = []
    for contract in contracts:
        for target, key in (
            (declarations, "declarations"),
            (acceptance_lanes, "acceptance_lanes"),
            (draft_q_kinds, "draft_q_kinds"),
        ):
            for name, count in contract[key].items():
                target[name] = target.get(name, 0) + int(count)
        sampler_mismatches.extend(contract["sampler_mismatches"])
        draft_revision_mismatches.extend(contract["draft_revision_mismatches"])
        engine_q_mismatches.extend(contract["engine_q_mismatches"])
    proposal_count = sum(int(row["proposal_count"]) for row in contracts)
    soft_q_proposals = sum(int(row["soft_q_proposals"]) for row in contracts)
    residual_corrections = sum(
        int(row["rejected_with_residual_correction"]) for row in contracts
    )
    contract_matches = bool(
        contracts
        and all(bool(row["contract_matches"]) for row in contracts)
        and proposal_count > 0
        and soft_q_proposals == proposal_count
        and draft_q_kinds == {"soft": proposal_count}
    )
    return {
        "proposal_count": proposal_count,
        "declarations": dict(sorted(declarations.items())),
        "acceptance_lanes": dict(sorted(acceptance_lanes.items())),
        "draft_q_kinds": dict(sorted(draft_q_kinds.items())),
        "soft_q_proposals": soft_q_proposals,
        "rejected_with_residual_correction": residual_corrections,
        "sampler_mismatches": sampler_mismatches,
        "draft_revision_mismatches": draft_revision_mismatches,
        "engine_q_mismatches": engine_q_mismatches,
        "contract_matches": contract_matches,
        "residual_path_exercised": residual_corrections > 0,
    }


def main() -> int:
    args = _args()
    if os.environ.get("MTPLX_DFLASH_DRAFT") != "1":
        raise SystemExit("set MTPLX_DFLASH_DRAFT=1")
    if args.distribution_repeats < 0:
        raise SystemExit("--distribution-repeats must be >= 0")
    if args.distribution_max_tokens < 2:
        raise SystemExit("--distribution-max-tokens must be >= 2")
    if args.distribution_min_samples < 2:
        raise SystemExit("--distribution-min-samples must be >= 2")
    draft_temperature_scale = os.environ.get(
        "MTPLX_DRAFT_TEMPERATURE_SCALE", ""
    ).strip()
    if draft_temperature_scale not in {"", "1", "1.0"}:
        raise SystemExit(
            "unset MTPLX_DRAFT_TEMPERATURE_SCALE for protocol-matched target/q sampling"
        )
    compiled_verify = (
        os.environ.get(
            "MTPLX_COMPILED_VERIFY",
            "off",
        )
        .strip()
        .lower()
    )
    if args.dflash_lane == "wide" and compiled_verify != "off":
        raise SystemExit("set MTPLX_COMPILED_VERIFY=off for the wide lane")
    if args.dflash_lane == "staged-k1":
        raise SystemExit(
            "staged-K1 is not an evidence lane for stochastic sampled-q yet; "
            "use --dflash-lane wide until compiled target-prefix supports "
            "probability-ratio acceptance and residual correction"
        )
    cases = load_prompt_suite(args.prompts)
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("prompt suite is empty after --limit")
    runtime = load(args.model, mtp=True)
    draft_model, draft_artifact = load_dflash_draft(
        args.draft_model,
        revision=args.draft_revision,
    )
    if not draft_artifact.get("immutable_revision"):
        raise SystemExit(
            "evidence harness requires a Hub draft resolved to an immutable SHA"
        )
    sampler = SamplerConfig(**EVIDENCE_SAMPLER)
    model_revision = _artifact_layout_revision(args.model)
    engine_source = _engine_source_identity()
    effective_runs = _effective_records(
        args,
        model_revision,
        runtime,
        draft_artifact,
    )

    rows: list[dict[str, Any]] = []
    seeded_equal_cases = 0
    dflash_seeded_equal_cases = 0
    mtp_seeded_equal_cases = 0
    observed_dflash_contracts: list[dict[str, Any]] = []
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
                draft_artifact=draft_artifact,
                draft_ref=args.draft_model,
                block_size=args.block_size,
                dflash_lane=args.dflash_lane,
            )
            expected_mode = "ar" if arm == AR else "mtpk"
            if output.stats.mode != expected_mode:
                raise RuntimeError(
                    f"{arm} requested {expected_mode} but observed {output.stats.mode}"
                )
            compiled_report = output.stats.graphbank.get("compiled_verify")
            if args.dflash_lane == "staged-k1" and arm != AR:
                if (
                    not isinstance(compiled_report, dict)
                    or compiled_report.get("mode") != "a3b_k1_target_prefix"
                    or int(compiled_report.get("fallback_calls", -1)) != 0
                ):
                    raise RuntimeError(
                        f"{arm} did not execute the compiled A3B K1 route"
                    )
            observed_contract = None
            if arm == DFLASH_B8:
                observed_contract = summarize_external_draft_contract(
                    output.stats.events,
                    expected_sampler=EVIDENCE_SAMPLER,
                    expected_draft_revision=str(
                        draft_artifact["resolved_revision"]
                    ),
                )
                observed_dflash_contracts.append(observed_contract)
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
                "verify_windows": output.stats.verify_calls,
                "verify_forward_calls": output.stats.verify_forward_calls,
                "repair_forward_calls": output.stats.repair_forward_calls,
                "observed_mode": output.stats.mode,
                "observed_speculative_depth": (output.stats.speculative_depth),
                "observed_requested_speculative_depth": (
                    output.stats.requested_speculative_depth
                ),
                "observed_runtime_mtp_enabled": (output.stats.runtime_mtp_enabled),
                "observed_mtp_forward_calls": (output.stats.mtp_forward_calls),
                "observed_forward_ar_plain_calls": (
                    output.stats.forward_ar_plain_calls
                ),
                "observed_forward_ar_hidden_calls": (
                    output.stats.forward_ar_hidden_calls
                ),
                "observed_compiled_verify": compiled_report,
                "observed_dflash_contract": observed_contract,
                "peak_memory_bytes": output.stats.peak_memory_bytes,
                "token_sha256": hashlib.sha256(
                    json.dumps(list(output.tokens)).encode()
                ).hexdigest(),
                "text_sha256": hashlib.sha256(output.text.encode("utf-8")).hexdigest(),
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
                                "accept_probability": draft.get("accept_probability"),
                                "correction": draft.get("correction"),
                                "correction_origin": draft.get("correction_origin"),
                                "target_top2_margin": draft.get("target_top2_margin"),
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
            seeded_equal_cases += 1
        if outputs[AR].tokens == outputs[DFLASH_B8].tokens:
            dflash_seeded_equal_cases += 1
        if outputs[AR].tokens == outputs[MTP_D1].tokens:
            mtp_seeded_equal_cases += 1

    distribution_receipts: list[dict[str, Any]] = []
    distribution_ar_tokens: list[list[int]] = []
    distribution_dflash_tokens: list[list[int]] = []
    distribution_max_tokens = min(
        args.distribution_max_tokens,
        cases[0].max_tokens,
    )
    if args.distribution_repeats > 0:
        distribution_case = cases[0]
        distribution_prompt_ids = encode_prompt_case(
            runtime.tokenizer,
            distribution_case,
            chat_template=True,
            enable_thinking=False,
        )
        for sample_index in range(args.distribution_repeats):
            # Disjoint seeds make these independent samples, not a same-seed
            # byte-equality diagnostic dressed up as a distribution test.
            ar_seed = args.seed + 1_000_000 + 2 * sample_index
            dflash_seed = ar_seed + 1
            ar_output = _run_arm(
                AR,
                runtime=runtime,
                prompt_ids=distribution_prompt_ids,
                max_tokens=distribution_max_tokens,
                sampler=sampler,
                seed=ar_seed,
                draft_model=draft_model,
                draft_artifact=draft_artifact,
                draft_ref=args.draft_model,
                block_size=args.block_size,
                dflash_lane=args.dflash_lane,
            )
            dflash_output = _run_arm(
                DFLASH_B8,
                runtime=runtime,
                prompt_ids=distribution_prompt_ids,
                max_tokens=distribution_max_tokens,
                sampler=sampler,
                seed=dflash_seed,
                draft_model=draft_model,
                draft_artifact=draft_artifact,
                draft_ref=args.draft_model,
                block_size=args.block_size,
                dflash_lane=args.dflash_lane,
            )
            ar_tokens = [int(token) for token in ar_output.tokens]
            dflash_tokens = [int(token) for token in dflash_output.tokens]
            distribution_ar_tokens.append(ar_tokens)
            distribution_dflash_tokens.append(dflash_tokens)
            contract = summarize_external_draft_contract(
                dflash_output.stats.events,
                expected_sampler=EVIDENCE_SAMPLER,
                expected_draft_revision=str(
                    draft_artifact["resolved_revision"]
                ),
            )
            observed_dflash_contracts.append(contract)
            receipt = {
                "sample_index": sample_index,
                "ar_seed": ar_seed,
                "dflash_seed": dflash_seed,
                "ar_token_sha256": hashlib.sha256(
                    json.dumps(ar_tokens).encode()
                ).hexdigest(),
                "dflash_token_sha256": hashlib.sha256(
                    json.dumps(dflash_tokens).encode()
                ).hexdigest(),
                "ar_token_ids": ar_tokens,
                "dflash_token_ids": dflash_tokens,
                "observed_dflash_contract": contract,
            }
            distribution_receipts.append(receipt)

    distribution_gate = compare_token_position_distributions(
        distribution_ar_tokens,
        distribution_dflash_tokens,
        max_positions=args.distribution_max_positions,
        permutations=args.distribution_permutations,
        alpha=args.distribution_alpha,
        min_samples_per_arm=args.distribution_min_samples,
        seed=args.seed + 2_000_000,
    )
    distribution_gate.update(
        {
            "prompt_id": cases[0].id,
            "prompt_sha256": cases[0].prompt_sha256,
            "max_tokens": distribution_max_tokens,
            "independent_seed_streams": True,
        }
    )
    observed_dflash_contract = _aggregate_contracts(observed_dflash_contracts)

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
            "verify_windows": sum(row["verify_windows"] for row in arm_rows),
            "verify_forward_calls": sum(
                row["verify_forward_calls"] for row in arm_rows
            ),
            "repair_forward_calls": sum(
                row["repair_forward_calls"] for row in arm_rows
            ),
            "peak_memory_bytes": max(
                (row["peak_memory_bytes"] for row in arm_rows),
                default=0,
            ),
        }
    ar_tok_s = summaries[AR]["weighted_tok_s"]
    for arm in (MTP_D1, DFLASH_B8):
        summaries[arm]["vs_ar"] = (
            summaries[arm]["weighted_tok_s"] / ar_tok_s if ar_tok_s > 0 else 0.0
        )

    contract_gate_pass = bool(
        observed_dflash_contract["contract_matches"]
        and observed_dflash_contract["residual_path_exercised"]
    )
    distribution_gate_pass = bool(distribution_gate["release_gate_pass"])
    if not observed_dflash_contract["contract_matches"]:
        gate_status = "failed_observed_proposal_contract"
        exit_code = 2
    elif not observed_dflash_contract["residual_path_exercised"]:
        gate_status = "failed_residual_path_not_exercised"
        exit_code = 2
    elif distribution_gate["status"].startswith("deferred_"):
        gate_status = "deferred_no_evidence_grade_distribution_receipt"
        exit_code = 3
    elif distribution_gate["status"] == "divergence_detected":
        gate_status = "failed_distribution_divergence_detected"
        exit_code = 2
    elif not distribution_gate_pass:
        gate_status = "deferred_statistical_equivalence_not_established"
        exit_code = 3
    else:
        gate_status = "passed"
        exit_code = 0

    payload = {
        "schema": 2,
        "includes_traces": bool(args.include_traces),
        "evidence_sampler": {
            "target": dict(EVIDENCE_SAMPLER),
            "draft": dict(EVIDENCE_SAMPLER),
        },
        "runtime_observed": {
            "mtp_enabled": bool(runtime.mtp_enabled),
            "contract": runtime.contract.to_dict(),
            "a3b_whole_moe_installed": bool(runtime.a3b_whole_moe_installed),
        },
        "engine_source": engine_source,
        "draft_artifact": draft_artifact,
        "effective_runs": effective_runs,
        "total_cases": len(cases),
        "seeded_stream_equality_diagnostic": {
            "all_arms_equal_cases": seeded_equal_cases,
            "dflash_vs_ar_equal_cases": dflash_seeded_equal_cases,
            "mtp_vs_ar_equal_cases": mtp_seeded_equal_cases,
            "total_cases": len(cases),
            # Gate semantics are governed by docs/dflash-gate-preregistration.md (§3 R3),
            # pre-registered before the deciding measurement.
            "release_gate": False,
            "note": (
                "Same-seed byte equality, including greedy equality, is only "
                "a diagnostic and cannot establish stochastic exactness."
            ),
        },
        "observed_dflash_contract": observed_dflash_contract,
        "distribution_gate": distribution_gate,
        "distribution_receipts": distribution_receipts,
        "release_gate": {
            "status": gate_status,
            "pass": exit_code == 0,
            "contract_gate_pass": contract_gate_pass,
            "distribution_gate_pass": distribution_gate_pass,
            "requires_real_stochastic_receipts": True,
        },
        "summary": summaries,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in payload.items()
                if key not in {"rows", "distribution_receipts"}
            },
            indent=2,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
