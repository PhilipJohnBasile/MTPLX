# Mean TPS and sliding averages hid thirty sub-second emit silences per turn — felt smoothness is the p99 inter-emit gap, so census gaps in every request record

**Symptom (2026-08-18):** founder: chip says 55 tok/s, "feels like 30
— freezes half a second then vomits two lines." An earlier wire probe
had already measured 0.51 s read gaps and was WRONGLY read as
exoneration because mean throughput held. The app-side stall census
then proved the app innocent (apply p95 0.17 ms, 3 stalls in 4 min)
while 22-30 gaps ≥200 ms per turn arrived with only 3-11 BYTES waiting
behind them: the GENERATOR itself went silent, repeatedly.

**Causes found (server/engine, 2.8.3):**
- `_IncrementalTokenDecoder` held visible text to whitespace
  boundaries — a table separator row / URL / minified run froze the
  visible stream for its full length, then landed as one paste.
  Invisible to byte-gap probes (progress frames keep flowing).
- `emit_new_tokens()` ran trunk-cache materialize + `mx.clear_cache`
  BEFORE the token callback — the ≥16k-context housekeeping barrier
  blocked production and delivery together.
- Auth middleware was Starlette `BaseHTTPMiddleware` — every SSE frame
  relayed through a zero-buffer anyio channel (several extra loop
  turns per frame, clumping under load).
- Residual: KV-growth reallocation stalls at step boundaries (follow-up
  lane), sustained-decay tail.

**Fixes/rules:**
1. Felt speed = inter-emit gap distribution, not the mean. Every
   request record now carries `producer_gap_ms_p95/max` and
   `producer_gaps_over_200ms` — read them BEFORE calling a stream
   smooth.
2. The QA pillar gate previously failed only on gaps >2 s — the whole
   0.2-0.8 s freeze regime passed green. Gate ceilings must sit just
   above the measured-good distribution, not an order of magnitude up.
3. Any deliberate wire hold (repetition holdback etc.) must log its
   engage/release transitions — a silence you can't attribute in logs
   will be blamed on the wrong subsystem for a week.
4. When a probe's mean is green but the user feels stutter, histogram
   the gaps before exonerating anything.
