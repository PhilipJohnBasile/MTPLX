#!/usr/bin/env python3
"""Quality-FP16 vs bf16 Quality parent: same-route logit-diff gate (Phase 3.2).

Loads the parent and the fp16 sibling sequentially (one GPU job at a time),
runs the identical prompt through the same forward route, and compares the
logits at the last N prompt positions plus a greedy 30-token continuation.

bf16 -> fp16 weight casting is value-exact (fp16 has the wider mantissa; all
bf16 magnitudes here are far below the fp16 max), so every difference comes
from activation dtype rounding along the forward. The gate is therefore a
distribution-safety band, not bit-equality:
  - argmax agreement >= 0.90 across checked positions,
  - mean top-20 overlap >= 0.90,
  - raw dmax recorded for the ledger.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_PROMPT = (
    "Write a complete snake game in Python with pygame. Include scoring, "
    "increasing speed, wall and self collision, a start screen and a game "
    "over screen. Write clean, commented code."
)


def _collect(model_path: Path, prompt: str, tail_positions: int, greedy_tokens: int):
    import mlx.core as mx

    from mtplx.generation import generate_mtpk
    from mtplx.runtime import load
    from mtplx.sampling import SamplerConfig

    rt = load(model_path, mtp=True)
    ids = list(rt.tokenizer.encode(prompt))
    inputs = mx.array([ids])
    text_model = getattr(rt.model, "language_model", rt.model)
    logits = text_model(inputs)
    logits = logits[0, -tail_positions:, :].astype(mx.float32)
    mx.eval(logits)
    tail = np.array(logits)

    out = generate_mtpk(
        rt,
        ids,
        max_tokens=greedy_tokens,
        sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=0),
        speculative_depth=3,
        verify_strategy="capture_commit",
        stop_token_ids=set(),
        seed=0,
    )
    tokens = list(out.tokens)
    del rt
    gc.collect()
    mx.clear_cache()
    return tail, tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--sibling", type=Path, required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--tail-positions", type=int, default=16)
    parser.add_argument("--greedy-tokens", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    started = time.perf_counter()
    parent_tail, parent_tokens = _collect(
        args.parent.expanduser(), args.prompt, args.tail_positions, args.greedy_tokens
    )
    sibling_tail, sibling_tokens = _collect(
        args.sibling.expanduser(), args.prompt, args.tail_positions, args.greedy_tokens
    )

    diffs = np.abs(parent_tail - sibling_tail)
    dmax = float(diffs.max())
    dmean = float(diffs.mean())
    argmax_agree = float(
        np.mean(parent_tail.argmax(axis=-1) == sibling_tail.argmax(axis=-1))
    )
    k = int(args.top_k)
    overlaps = []
    for row_p, row_s in zip(parent_tail, sibling_tail):
        top_p = set(np.argpartition(row_p, -k)[-k:].tolist())
        top_s = set(np.argpartition(row_s, -k)[-k:].tolist())
        overlaps.append(len(top_p & top_s) / k)
    topk_overlap = float(np.mean(overlaps))

    first_div = None
    for i, (a, b) in enumerate(zip(parent_tokens, sibling_tokens)):
        if a != b:
            first_div = i
            break

    passed = argmax_agree >= 0.90 and topk_overlap >= 0.90
    result = {
        "run_id": f"quality-fp16-parent-logitdiff-{time.strftime('%Y%m%d-%H%M%S')}",
        "parent": str(args.parent),
        "sibling": str(args.sibling),
        "tail_positions": int(args.tail_positions),
        "logit_dmax": dmax,
        "logit_dmean": dmean,
        "argmax_agreement": argmax_agree,
        "top20_overlap_mean": topk_overlap,
        "greedy_tokens": int(args.greedy_tokens),
        "greedy_first_divergence": first_div,
        "greedy_matched_prefix": first_div if first_div is not None else len(parent_tokens),
        "elapsed_s": time.perf_counter() - started,
        "passed": passed,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
