# Benchmarking MTPLX honestly

MTPLX ships the measurement surface benchmark harnesses need. This page
documents the contracts that keep cross-engine numbers honest.

## Server-side stats are the source of truth

Every `/v1/chat/completions` response carries an `mtplx_stats` block with
authoritative server-side timings: `prefill_tok_s`, `decode_tok_s`,
`ttft_s`, `prompt_eval_time_s`, `decode_elapsed_s`, `cached_tokens`,
`new_prefill_tokens`, `session_cache_hit`, `peak_memory_bytes`, and (for
speculative decode) `accepted_by_depth` / `drafted_by_depth`. Prefer these
over client-side wall clocks: they separate prefill from decode, which
client timing cannot.

For cold-prefill rows, POST `/admin/cache/clear` between rows (it also
resets the peak-memory watermark) and salt each prompt with a unique
prefix so the session bank cannot serve a warm prefix.

## Reasoning models and small output caps

MTPLX serves its reasoning models with thinking ON by default. A capped
row (for example `max_tokens: 128`) will spend its entire budget inside
the think channel: `message.reasoning_content` fills, `message.content`
stays empty, and `finish_reason` is `length`. That is faithful model
behavior, not a serving bug. When a benchmark needs visible content in
short rows, send `"enable_thinking": false` in the request body. The
token split is always reported in
`usage.completion_tokens_details.reasoning_tokens`.

## Accuracy runs (AIME and similar) must be uncapped

Never cap accuracy arms. A reasoning model that hits `max_tokens`
mid-think abstains on every problem and the run scores near zero —
that is a truncated run, not a model score. The contract for extracting
answers is: the model's answer is the content AFTER the closing
`</think>`; reasoning text is never parsed for answers. `mtplx bench
aime` (against a running daemon) applies both rules; if you build your
own harness, apply them too, and treat any row with
`finish_reason != "stop"` as void rather than wrong.

## Quality lane: prompt scoring for KL divergence

`/v1/completions` supports teacher-forced prompt scoring:

```json
{"prompt": "...", "echo": true, "logprobs": 64, "max_tokens": 0, "temperature": 0}
```

The response's `choices[0].logprobs.top_logprobs[i]` is a token-string to
logprob dict for the model's distribution after prefix token `i` (it
predicts token `i+1`) — the llama.cpp-compatible shape KL-divergence
harnesses consume. Bounds: `logprobs <= 128`
(`MTPLX_PROMPT_LOGPROBS_MAX`) and prompt length `<= 8192` tokens
(`MTPLX_PROMPT_SCORE_MAX_TOKENS`). One forward pass, chunk-bounded
memory, zero effect on decode paths.

## Thermal discipline

Apple Silicon throttles quietly. Numbers taken with uncontrolled fans and
hot dies are noise: pin fans to maximum and verify the actual RPM before
loading the model, equalize die temperature between A/B arms (fans at max
is not the same as an equalized die), interleave arm order (ABBA), and
repeat at least three times. MTPLX's `--fan-mode max` requests the ramp;
verify it happened rather than trusting the request.

## Comparing engines fairly

- Same quantization class and same tokenizer family per lane.
- Separate prefill from decode in every reported number; a "generation"
  rate whose denominator includes prefill is a different metric.
- Apply cache-busting symmetrically across engines, or not at all.
- Report the sampler. Greedy (`temperature 0`) and sampled runs are
  different regimes for speculative engines; MTPLX's speculative
  acceptance is mathematically exact at any temperature, and greedy-only
  receipts are not product evidence.
