# Wizard rebuilt the tune-state key by hand and drifted from the save side, so every start re-tuned — derive shared keys from one constructor with a parity test

**Symptom →** Hours after 2.8.0/2.8.1 shipped, users reported "load the model,
machine heats up with zero requests, API unreachable" (#280 + X). `mtplx
start` re-offered tuning on every launch; accepting meant minutes of maxed GPU
before the port opened.

**Cause →** The tune record is keyed by sha256 over the exact tune settings.
2.8.0's per-model profile work updated the settings the tuner *saves* under;
the start wizard looked records up with its own hand-built copy of that dict
(stale profile literal, different depths encoding, missing keys). Two hashes,
zero matches, forever. Two sibling leaks found underneath: wizard model picks
carried no explicitness markers (`_cli_flags`/`_model_explicit`), so tune
resolved the hardware default (picked 4B → tuned 27B) and default-named local
folders were swapped for the repo id (#279).

**Fix / rule →** Any hashed/derived contract value gets ONE constructor that
every reader and writer calls (`_tune_state_context_for_args`), plus a
regression test that fails when the sides drift
(tests/test_tune_record_wizard_parity.py). Never duplicate a settings dict
across save/lookup sites — the duplicate WILL rot on the next defaults change.
When a wizard/programmatic namespace sets a value a CLI flag would set, it
must also set the flag-provenance markers, or provenance-gated code re-resolves
defaults over the user's choice.
