# every-desktop-chat-froze-8-to-11s-because-a-stream-holdback-shipped-tested-only-with-capped-requests-qa-must-exercise-the-uncapped-client-path-and-measure-delivered-cadence-not-server-tps

**Symptom** → Field reports within hours of 2.8.0-2.8.2: "reasoning freezes,
speed drops to ~20-30, then it vomits the output in a burst"; founder measured
27-47 tok/s on prompts that used to show 55-80. Server-side decode numbers and
all release gates were green.

**Cause** → Three independent 2.8.0 changes, none caught because QA never
measured what a desktop user actually receives:
1. F35 armed-stream holdback held a fixed ~448-token tail off the wire on
   every UNCAPPED request → wire blackout from token ~320 to ~768 (6-11 s at
   chat rates) + an end-of-response burst. Every benchmark and pillar-gate
   request passed `max_tokens` → repetition stop disarmed → holdback never
   engaged in QA. Every real chat is uncapped.
2. F6 put the deep warm ladder (pow2 rungs to 32768) in the TURBO product
   profile for benchmark-row cosmetics → 30-60+ s of max GPU after every boot
   (field "idle GPU pin" reports), rungs re-queued between chat turns, and a
   request landing mid-rung had its prefill contended to ~166 tok/s (healthy
   ~800 → founder saw 6.6 s TTFT).
3. A first client-side probe "measured" periodic 8 s stalls that were the
   probe's own `HTTPResponse.read(65536)` looping to FILL 64 KiB across
   chunked frames. `read1()` is the only honest cadence read.

**Fix / rule** →
- Fix: candidate-gated holdback (`_RepetitionStreamGate`, env
  `MTPLX_REPETITION_STREAM_HOLDBACK=candidate|strict|off`) — wire lag only
  while a loop is actually forming; turbo ladder back to `512,2560` (deep
  rungs are benchmark-harness env); background warm steps wait 90 s of
  foreground quiet (`MTPLX_WARMUP_IDLE_GRACE_S`).
- Rule 1: any feature that touches the stream path ships only after a run on
  the UNCAPPED client path (no max_tokens — what the app/web/agents send).
  `scripts/pillar_gate_qa.py::gate_uncapped_stream_cadence` now fails the
  release on any >2 s delivered-content gap; do not cap it, do not skip it.
- Rule 2: delivered cadence is a pillar signal distinct from decode TPS —
  a healthy `decode_tok_s` proves nothing about what the user's screen does.
- Rule 3: cadence probes must use `read1()`/line-iteration, never `read(n)`.
- Rule 4: warm/benchmark-cosmetic GPU work does not belong in product
  profiles; it belongs in the harness that wants it.
