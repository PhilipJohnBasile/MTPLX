# A request-arrival optimisation that predicted from the prompt alone fired on every warm bank turn and blocked the owner thread on work no prefill consumed — ask the bank first, and never await unconsumed work

**Symptom:** OpenCode on Flash-Next at 43k context: 5-7.5 s of dead time before the first
token of every tool turn, charged to nothing in the receipt (`prompt_state_unattributed_time_s`),
divided into `decode_tok_s` on 200-token turns → "21 tok/s". Memory-pressure notices. Decode
rounds themselves ran at 65-70 tok/s.

**Cause:** `MTPLX_QWEN4_PLE_FIRST_GATHER_EARLY` (default-on since the #391 port) predicted the
first prefill chunk as tokens 0-2048 *from the prompt alone*, gathered their n-gram rows on a
worker at request arrival, chained a page-warm of the rest of the prompt's rows (650k at 43k),
and at scope exit blocked on the gather (`close()` → `future.result()`). On a warm bank turn
the prefill starts 40k tokens later; the gather was pure waste and its wait was pure dead time,
as long as the SSD preads took once the table had been evicted.

**Fix / rule:** anything that runs at request arrival must ask the session bank whether the
prefill will start at token 0 (`SessionBank.shares_ram_prefix`), and an owner thread must never
wait for work it will not consume (cancel, orphan, move on). Measure warm turns with the bank
populated, not cold single requests: `scripts/agent_session_gate.py`.
