# Benchmarking MTPLX honestly

MTPLX ships the measurement surface benchmark harnesses need. This page
documents the contracts that keep cross-engine numbers honest.

## Server-side stats are the source of truth

Every `/v1/chat/completions` response carries an `mtplx_stats` block with
authoritative server-side timings: `prefill_tok_s`, `decode_tok_s`,
`ttft_s`, `prompt_eval_time_s`, `decode_elapsed_s`, `server_elapsed_s`,
`cached_tokens`, `new_prefill_tokens`, `session_cache_hit`,
`peak_memory_bytes`, and (for speculative decode) `accepted_by_depth` /
`drafted_by_depth`. Streamed responses carry the same block on the final
chunk, next to `usage` and `timings`. Prefer these over client-side wall
clocks: they separate prefill from decode, which client timing cannot,
and `server_elapsed_s` separates server time from client overhead.

For cold-prefill rows, POST `/admin/cache/clear` between rows (it also
resets the peak-memory watermark) and salt each prompt with a unique
prefix so the session bank cannot serve a warm prefix.

## Response caps: request, server, context

`mtplx serve --max-tokens N` sets a server-side response-token ceiling.
The cap applied to a request is the smallest of: the request's
`max_tokens` (omitting it requests the remaining context), the server's
`--max-tokens`, and the context remaining after the prompt. Every clamp
is visible in `mtplx_stats`: `request_max_tokens`,
`server_max_response_tokens`, `effective_max_tokens`, and the
`server_cap_applied` / `context_cap_applied` booleans. Before charting a
capped row, assert `effective_max_tokens` equals the cap the harness
meant to request.

A prompt that fills or overflows the context window is rejected with
HTTP 400, error code `context_length_exceeded`, and both numbers in the
message — never silently degraded into a one-token generation.

## Reasoning models and small output caps

A capped-token harness that expects visible content must disable the
think channel per request on `/v1/chat/completions`:

```json
{"model": "...", "messages": [{"role": "user", "content": "..."}],
 "max_tokens": 128, "enable_thinking": false}
```

MTPLX serves its reasoning models with thinking ON by default, so a
capped row (for example `max_tokens: 128`) spends its entire budget
inside the think channel before any visible answer exists:
`message.reasoning_content` fills, `message.content` stays empty, and
`finish_reason` is `length`. That is faithful model behavior under the
cap, not a serving bug. The token split is reported in
`usage.completion_tokens_details.reasoning_tokens` whenever the server
routed a think channel; when no reasoning was routed the field is
absent — read absent as zero, not as an error.

## Accuracy runs (AIME and similar) must be uncapped

Never cap accuracy arms. A reasoning model that hits `max_tokens`
mid-think abstains on every problem and the run scores near zero —
that is a truncated run, not a model score. The contract for extracting
answers is: the model's answer is the content AFTER the closing
`</think>`; reasoning text is never parsed for answers. `mtplx bench
aime` (against a running daemon) applies both rules and sends uncapped
requests by default; its one capped default is Gemma-4 with thinking
disabled (`max_tokens` 2048, plus cap-recovery and answer-verification
rescue passes that are off for every other model — the run summary's
rescue-policy payload discloses them, and a score is rescue-free exactly
when its `active` flag is false).

Void rules for your own harness: treat any row with
`finish_reason != "stop"` as void rather than wrong, and likewise any
row whose `mtplx_stats` carries `repetition_stop_triggered: true`. The
repetition guard (armed by default on uncapped requests) ends a
degenerate row early yet still reports `finish_reason: "stop"`, so that
stat — stamped only when the guard fired, with `repetition_stop_reason`
beside it — is the only signal that the guard, not the model, ended the
row.

## Quality lane: prompt scoring for KL divergence

`/v1/completions` supports teacher-forced prompt scoring:

```json
{"prompt": "...", "echo": true, "logprobs": 64, "max_tokens": 0, "temperature": 0}
```

The response follows the OpenAI echo+logprobs alignment (the shape
llama.cpp emits and KL harnesses consume). In `choices[0].logprobs`, the
arrays `tokens`, `token_logprobs`, `top_logprobs`, `text_offset`, and
`token_ids` all have length n (the prompt token count), with index `i`
describing prompt token `i`. `token_logprobs[0]` and `top_logprobs[0]`
are null — the first token has no conditional. For `i >= 1`,
`top_logprobs[i]` is the token-string to logprob dict of the
distribution that predicted token `i`, and it always contains token `i`
itself (up to k+1 entries), so string-keyed KL never loses the scored
token. `token_ids` is the stable identity lane: distinct byte-level
pieces can decode to identical strings, ids never collapse. The arrays
are correct under both harness zip conventions (`zip(tokens,
token_logprobs)` and the skip-nulls variant).

Request rules: `logprobs` on `/v1/completions` requires `echo: true`
with `max_tokens: 0`; any other combination is a 400 whose message says
why (decode-time logprobs are not supported yet). `logprobs: 0` is a
valid request and returns each scored token's own logprob. Bounds:
`logprobs <= 128` (`MTPLX_PROMPT_LOGPROBS_MAX`) and prompt length
`<= 8192` tokens (`MTPLX_PROMPT_SCORE_MAX_TOKENS`). One prefill-shaped
pass over the prompt, chunk-bounded memory, zero effect on decode
paths.

## Streaming measurement

- Progress and heartbeat frames are spec-valid `chat.completion.chunk`s
  with an empty `delta: {}` (plus an `mtplx_progress` extension). A
  harness that counts chunks counts them; count content deltas instead.
- The role chunk (`delta: {"role": "assistant"}`) is emitted before
  prefill starts. Measure TTFT at the first content delta, not the first
  frame — or read the authoritative server-side `ttft_s` from
  `mtplx_stats`.
- The visible TPS stats footer is off for API clients by design: it
  renders only on MTPLX's own UI surfaces, so `content` stays clean and
  temperature-0 byte-equality holds (`MTPLX_STATS_FOOTER_SCOPE=all`
  opts back in).
- The final chunk carries `usage`, `mtplx_stats`, and `timings`;
  `server_elapsed_s` separates server time from client overhead.

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
