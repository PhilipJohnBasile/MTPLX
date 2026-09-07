# A benchmark-arm assertion shipped in a default-on lane and returned finish_reason "error" to a user after a 30 s prefill — verdicts are receipts in serving and raises only behind a strict flag

**Symptom:** a cold 31k-token OpenCode-shaped prompt paid its full 30 s prefill on the live
daemon, then the stream ended with `finish_reason: "error"`: "MTPLX_QWEN4_PLE_PREFILL_LOOKAHEAD=1
did not engage on every prefill chunk ... refusing to report a measurement that did not run the
candidate."

**Cause:** the PLE lookahead's engagement verdict (right for an A/B arm, where an inert lane must
never be reported as the candidate) raised inside `prefill_lookahead_scope`'s clean exit, which
is the serving path. The sidecar had declined a low-entropy first chunk, so a "required" span
was "ineligible" and the verdict fired on a user request.

**Fix / rule:** a lane's inertness verdict is a counter + `last_scope_status()` + one warning in
serving; it raises only under `MTPLX_QWEN4_PLE_PREFILL_LOOKAHEAD_STRICT=1`, which measurement
arms export. Before default-arming any lane from a port, grep it for `raise` on paths a request
can reach.
