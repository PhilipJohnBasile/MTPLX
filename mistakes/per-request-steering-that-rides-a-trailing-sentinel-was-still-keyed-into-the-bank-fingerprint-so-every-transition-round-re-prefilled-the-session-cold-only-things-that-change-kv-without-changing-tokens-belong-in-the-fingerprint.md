# Per-request steering that rides a trailing sentinel was still keyed into the bank fingerprint, so every transition round re-prefilled the session cold — only things that change KV without changing tokens belong in the fingerprint

**Symptom:** agent-session gate at 41k: a forced `tool_choice` round re-prefilled the whole
session (41 s), and the following auto round did it again (44 s). Same class as the earlier
Pi-convergence and read-only force-answer "transition round re-prefills cold" reports.

**Cause:** two layers. (1) The forced-tool clause was appended to the system contract in msg0,
so the prompt's first bytes changed with `tool_choice`. (2) Even after moving it to a trailing
transient sentinel (the pattern every other per-request contract already used), the bank's
`policy_fingerprint` carried `tool_choice=` and the post-tool / force-answer / Pi-convergence
flags, so identical token prefixes were treated as different cache identities.

**Fix / rule:** a bank entry is exact for its token ids. The fingerprint carries only what can
change the KV or the MTP history *without* changing the tokens (model, draft head, depth and
history policy). Anything a harness can flip per request goes in a trailing sentinel registered
in `_transient_trailing_user_sentinel_texts()` and stays out of the fingerprint. The gate's
forced round is the regression check.
