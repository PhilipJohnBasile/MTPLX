# Phase 6.1 — RequestPolicy extraction: /v1/completions behavior deltas

The three per-request prologues (`chat_completions`, `completions`,
`anthropic_count_tokens`) now share one resolver
(`mtplx/server/request_policy.py`). `/v1/completions` was a diverged partial
copy; unifying it onto the shared path intentionally changes the following.
Chat and count_tokens are observationally unchanged (see the fidelity notes
at the bottom). The golden matrix
(`tests/test_request_observability_golden.py`, 13 arms) passes byte-identical
with **zero** golden-file regeneration — including the `plain_completions`
arm, because every public-envelope key the shared path emits for completions
is either value-identical to what the generation layer already emitted or
filtered out by `PUBLIC_MTPLX_STATS_KEYS`.

## Deltas on /v1/completions

- **Server sampler fallback resolved at the prologue.** Server-owned sampler
  fields (hints mode, or omitted fields) now resolve to the launch sampler
  (`state.args.temperature/top_p/top_k`, hard fallback
  `DEFAULT_TEMPERATURE`/`DEFAULT_TOP_P`/`DEFAULT_TOP_K` = 0.6/0.95/20 when an
  arg is unset) *before* generation, exactly as chat always did, instead of
  passing `None` down. Why it's correct: `_generation_params` already applied
  the same `state.args` defaults downstream, so the sampled distribution is
  unchanged whenever the args are set; the request is additionally hardened
  for the args-unset edge, and the resolved target now reaches
  `_resolve_draft_sampler_for_request(target_temperature=...)` — see next
  bullet.
- **Draft-sampler resolution sees the true effective target temperature.**
  Previously a completions request without an explicit temperature handed
  `target_temperature=None` to the per-family draft-temperature curve, which
  then used the launch draft default; now the curve maps the actual effective
  target (e.g. 0.6), matching chat. On launches without a curve (or with a
  pinned draft sampler) nothing changes.
- **`effective_*` sampler telemetry.** `request_observability` now carries
  `effective_temperature`/`effective_top_p`/`effective_top_k` (public values
  identical to the copies `_generation_params` already wrote — golden
  unchanged) plus `effective_presence_penalty`/`effective_frequency_penalty`
  (internal metrics only; not in `PUBLIC_MTPLX_STATS_KEYS`).
- **`request_presence_penalty` / `request_frequency_penalty` recorded** when
  the client sends them (internal metrics; chat parity).
- **`client_sampler_fields_ignored` computed the chat way.** Written only
  when at least one of the five sampler fields was actually ignored.
  Previously completions wrote an empty list whenever any non-sampler control
  (e.g. `generation_mode`, `depth`) was ignored. `client_control_fields_ignored`
  is unchanged.
- **OpenCode sampler/draft override path now runs on completions.** It
  cannot trigger for a raw-prompt request (it requires a chat transcript with
  tools or simple chitchat), so behavior is unchanged today; the two
  generation calls now pass `draft_sampler=` through (always `None` until an
  override can fire), so the completions and chat lanes are structurally
  identical.
- **Client-controls ownership evaluated after prompt encoding** (was before).
  `_client_controls_allowed` is a pure function of headers/metadata/env, so
  this is unobservable; error precedence (empty-prompt 400 → mode 400 →
  depth 400 → non-finite-sampler 400) is unchanged.
- **Prompt-scoring lane (`echo`+`logprobs`+`max_tokens=0`) inherits the
  richer observability dict** (additive sampler telemetry keys). Teacher-
  forced scoring never samples, so this is telemetry-only; the contract test
  asserts shape keys, not an exact key set, and still passes.

## Test updated for an intended delta

- `tests/test_server_openai.py::test_completion_request_controls_are_server_owned_without_override`
  pinned the old mechanism (`temperature/top_p/top_k is None` reaching
  generation in hints mode). The invariant it guards — client sampler values
  must not be applied when the server owns controls — still holds and is now
  pinned more strongly: the resolved values must equal the server defaults
  (0.6/0.95/20), never the client's, and the new
  `client_sampler_fields_ignored` + `effective_*` telemetry is asserted.

## Fidelity notes (chat / count_tokens)

- **Busy-background 503 order is preserved exactly**: the resolver raises
  `BackgroundBusyBypass` at the same pipeline position the inline check
  occupied (after transcript canonicalization, before thinking/mode/depth
  resolution), so a busy background request never reaches the later
  400-raising steps.
- **Single-fault requests are byte-identical on chat.** The one observable
  reordering: mode/depth/non-finite-sampler validation now runs as one unit
  ahead of the response_format/strict-tool constraint block and the vision
  block. A request with *two or more* invalid elements straddling that seam
  (e.g. invalid `response_format` *and* out-of-range `depth`) now gets the
  policy 400 instead of the constraint/vision 400. Status codes are
  unchanged; only which 400 detail wins on multi-fault requests.
- **count_tokens keeps its historical shape** (deliberately, per the
  observational-identity constraint): the REQUESTED toolset is encoded
  unfiltered, and no prompt contracts / OpenCode system-prompt replacement
  are applied. The resolver's `count_tokens` profile encodes exactly the
  prompt that endpoint always counted and introduces no new raise paths.
