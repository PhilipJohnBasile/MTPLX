# Why agent loops kept breaking one harness at a time, and what vLLM does instead

Written 2026-09-03 after the Flash-Next / OpenCode "21 tok/s" session. Source
for the vLLM side: a sparse checkout of `vllm-project/vllm` at `8a72866`
(2026-09-03), `vllm/v1/core/kv_cache_utils.py`,
`vllm/v1/core/single_type_kv_cache_manager.py`, `vllm/config/cache.py`.

## What the session actually lost

17 turns, ~14 minutes, 43k context. Decode rounds ran at 65-70 tok/s the whole
time. The user saw 21 tok/s because of dead time, not slow decode:

| Dead time | Seconds | Mechanism |
|---|---|---|
| before first token, uncharged | 41 | early n-gram gather started at request arrival for token 0-2048, owner thread blocked on it at scope exit; page-warm of the whole prompt's rows behind it |
| hidden postcommit waits | 49 | previous turn's snapshot re-prefilling the assistant turn on the GPU while the next request waited (30 s bound, then abort) |
| whole-turn re-prefills | 56 | 25,647 tokens after a reasoning turn whose snapshot was aborted; 9,082 after the RAM entry was superseded and SSD served a 30k prefix |
| SSD twin hydrations | ~10 | every warm turn decoded its own cold-tier copy for a candidate the sort discarded |

Every one of these was server-side and harness-independent in mechanism, but
each *surfaced* through a harness-specific quirk (OpenCode re-sending tool
calls, Pi's convergence contract, a forced `tool_choice`), which is why the
fix history reads as "fix OpenCode, then Hermes breaks, then Pi breaks".

## The structural difference

**vLLM keys the cache on content, at block granularity, statelessly.**
`hash_block_tokens(parent_hash, block_token_ids, extra_keys)` chains a hash per
16-token block over the token ids alone (plus LoRA / multimodal / cache-salt
keys). Any request whose prefix hashes match reuses those blocks: no session
id, no client label, no policy fingerprint, no transcript canonicalization. A
divergence anywhere costs only the blocks after it. For hybrid (Mamba / GDN)
models, `mamba_cache_mode="align"` stores the recurrent state at every
`block_size`-aligned position **during prefill and decode alike**, so a
generated turn has recurrent checkpoints every block and a divergence inside
it still costs one block.

**MTPLX keys the cache on session entries with sparse recurrent boundaries.**
A bank entry is exact for its token ids, restorable at its end or at a GDN
boundary; boundaries are captured at prefill chunk edges (2048 + a 256-token
tail grid) and never during decode, because a Flash-Next boundary is a ~200 MB
recurrent snapshot (entries read ~1 GB at 3.6k tokens, 2.4 GB at 43k). So a
one-token divergence anywhere inside a 2,300-token generated turn — a
trailing newline stripped from a tool argument, a `<think>` seam re-tokenized
differently — rewinds the restore to the prompt end and re-prefills the whole
turn. That is why MTPLX needs byte-exact re-rendering of its own turns
(committed think / committed body substitution) where vLLM needs nothing, and
why every harness quirk that touches a byte of history has been a cache
killer here and a non-event there.

Two other places where per-request state leaked into cache identity, both
things vLLM structurally cannot have:

- Prompt-text policy flags in the bank fingerprint (`tool_choice`, the
  post-tool / force-answer / Pi-convergence contracts). The bytes of those
  contracts already sit after the stable prefix; keying the entry on them
  made every transition round miss the whole session. Removed 2026-09-03.
- Request-arrival optimizations that assumed a cold prompt (the early
  first-chunk gather). A warm turn's prefill starts where the bank leaves
  off; anything that predicts from the prompt alone must ask the bank first.

## What closes the gap, in order

1. **Done 2026-09-03:** committed-body substitution (tool-call turns
   re-render byte-exactly from the generated stream), `tool_calls` finishes
   advance the committed stream, tail-only contracts out of the fingerprint,
   forced `tool_choice` as a trailing sentinel, bank-aware early gather,
   progress-gated postcommit wait, no SSD twin hydration, lane verdicts as
   receipts. Effect: the agent-session gate passes at 40k with every turn
   warm, including the auto and forced tool rounds.
2. **Next (architectural):** decode-time recurrent boundaries. Capture a GDN
   boundary every K generated tokens (K ~ 512), spilled to the SSD tier, so
   a divergence inside a generated turn costs ≤ K tokens instead of the
   turn — vLLM's `align` semantics. With that, re-rendering fidelity stops
   being load-bearing for cache survival and the canonicalization machinery
   becomes an optimization rather than a requirement.
3. **Then:** content-hash lookup across sessions at boundary granularity
   (the bank already matches cross-session prefixes; a hash index makes it
   O(1) instead of a token-compare scan over entries).

## How to keep it from regressing

`scripts/agent_session_gate.py` runs the loop every harness sends against a
serving daemon and fails on the receipts above. `release_macos_v1.sh` runs
it after the pillar gate. Add a harness-specific probe to it only when the
harness sends a request *shape* the gate does not already cover; never a
harness-specific fix to the engine.
