# pytest rewrote the founder's real OpenCode config mid-run with a fixture model id — every external config path the code can write gets an autouse scratch override in conftest

**Symptom:** during a full `pytest tests/` run, `~/.config/opencode/opencode.json` was rewritten
(mtime 22:28:15, the run ended 22:29:30) with `mtplx-qwen38-27b-optimized-speed` as the only
mtplx model; every `opencode run -m mtplx/mtplx-flash-next-optimized-speed` afterwards failed
with "Model not found ... Did you mean: mtplx-qwen38-27b-optimized-speed?". This class of
breakage looks exactly like "the OpenCode update broke the integration again".

**Cause:** `mtplx.opencode.write_opencode_config` resolves its path from `MTPLX_OPENCODE_CONFIG`
or the real home directory; most tests that exercise it set the env var, at least one path did
not, and `tests/conftest.py` isolated settings/models but not this file.

**Fix / rule:** `tests/conftest.py`'s autouse fixture now points `MTPLX_OPENCODE_CONFIG` at a
scratch file. Any code path that writes outside the repo (configs, plugins, launchers) gets the
same treatment the day it is added. Repair: `write_opencode_config(...)` with the served model
restores the entry; the writer backs the old file up as `opencode.json.before-mtplx-*.bak`.
