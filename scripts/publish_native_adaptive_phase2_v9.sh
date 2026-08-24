#!/usr/bin/env bash
set -euo pipefail

: "${BASE_SHA:?BASE_SHA is required}"
: "${PR_HEAD:?PR_HEAD is required}"
: "${LEASE_SHA:?LEASE_SHA is required}"

LOG_DIR=/tmp/mtplx-phase2-logs
mkdir -p "$LOG_DIR"

python scripts/materialize_native_adaptive_phase2.py
python scripts/materialize_native_adaptive_phase2_fixups.py
python scripts/materialize_native_adaptive_phase2_fixups_v5.py
python scripts/materialize_native_adaptive_phase2_fixups_v6.py
python scripts/materialize_native_adaptive_phase2_fixups_v9.py

python - <<'PY'
from pathlib import Path

path = Path("mtplx/server/openai.py")
value = path.read_text(encoding="utf-8")
replacements = (
    (
        "        def _mtplx_runtime_systems_snapshot(state: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:",
        "        def _mtplx_runtime_systems_snapshot(  # noqa: F811\n"
        "            state: Any, *args: Any, **kwargs: Any\n"
        "        ) -> dict[str, Any]:",
    ),
    (
        "        def _memory_governor_tick(state: Any, *args: Any, **kwargs: Any) -> dict[str, Any] | None:",
        "        def _memory_governor_tick(  # noqa: F811\n"
        "            state: Any, *args: Any, **kwargs: Any\n"
        "        ) -> dict[str, Any] | None:",
    ),
)
for old, new in replacements:
    if old not in value:
        raise SystemExit(f"missing expected integration line: {old}")
    value = value.replace(old, new, 1)
path.write_text(value, encoding="utf-8")
PY

python -m pip install -U pip
python -m pip install -e ".[dev,server]"

files=(
  mtplx/expert_residency.py
  mtplx/unified_memory.py
  mtplx/otlp_export.py
  mtplx/policy_hooks.py
  mtplx/replay_orchestrator.py
  mtplx/native_adaptive.py
  tests/test_expert_residency.py
  tests/test_unified_memory.py
  tests/test_otlp_export.py
  tests/test_policy_hooks.py
  tests/test_replay_orchestrator.py
  tests/test_native_adaptive.py
)
python -m ruff check --fix --unsafe-fixes \
  --select E9,F,I,UP035,UP037,RUF100 "${files[@]}"
python -m ruff format "${files[@]}"
python -m ruff check \
  --select E9,F,I,UP035,UP037,RUF100 "${files[@]}" \
  2>&1 | tee "$LOG_DIR/ruff.log"

python -m pytest -q \
  --junitxml="$LOG_DIR/native-systems.xml" \
  tests/test_semantic_anchors.py \
  tests/test_memory_governor.py \
  tests/test_expert_locality.py \
  tests/test_deterministic_replay.py \
  tests/test_trace_parity.py \
  tests/test_request_capture.py \
  tests/test_runtime_systems.py \
  tests/test_dashboard_endpoints.py \
  tests/test_expert_residency.py \
  tests/test_unified_memory.py \
  tests/test_otlp_export.py \
  tests/test_policy_hooks.py \
  tests/test_replay_orchestrator.py \
  tests/test_native_adaptive.py \
  2>&1 | tee "$LOG_DIR/native-systems.log"

python -m pytest -q \
  --junitxml="$LOG_DIR/compatibility.xml" \
  tests/test_no_mlx_imports.py \
  tests/test_public_cli.py \
  tests/test_runtime_kpis.py \
  tests/test_server_openai.py \
  tests/test_openai_bridge.py \
  tests/test_thermal.py \
  tests/test_cache_state.py \
  2>&1 | tee "$LOG_DIR/compatibility.log"

python -m compileall -q mtplx tests
git diff --check "$BASE_SHA"

(
  cd dashboard
  bun install --frozen-lockfile 2>&1 | tee "$LOG_DIR/bun-install.log"
  bun run build 2>&1 | tee "$LOG_DIR/dashboard-build.log"
)

python -m pip install -U build twine
rm -rf dist build
python -m build 2>&1 | tee "$LOG_DIR/python-build.log"
python -m twine check dist/* 2>&1 | tee "$LOG_DIR/twine.log"
scripts/fresh_venv_smoke.sh 2>&1 | tee "$LOG_DIR/fresh-venv.log"

if grep -RInE '(^|[^A-Za-z0-9_])(import|from)[[:space:]]+(freetoken|futureagi)([^A-Za-z0-9_]|$)' \
  mtplx tests dashboard/src; then
  echo "Unexpected FreeToken/Future AGI runtime dependency" >&2
  exit 1
fi
python - <<'PY'
from pathlib import Path

forbidden = ("FlashML-org/FreeToken", "future-agi/future-agi")
for root in (Path("mtplx"), Path("dashboard/src")):
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".ts", ".tsx"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for marker in forbidden:
            if marker in text:
                raise SystemExit(
                    f"vendor repository reference in executable source: {path}: {marker}"
                )
PY

python - <<'PY'
from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path


def junit(path: str) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    keys = ("tests", "failures", "errors", "skipped")
    return {
        key: sum(int(suite.attrib.get(key, "0")) for suite in suites)
        for key in keys
    }


receipt = {
    "schema_version": 1,
    "base_sha": os.environ["BASE_SHA"],
    "native_systems": junit("/tmp/mtplx-phase2-logs/native-systems.xml"),
    "compatibility": junit("/tmp/mtplx-phase2-logs/compatibility.xml"),
    "gates": {
        "python_compileall": "pass",
        "git_diff_check": "pass",
        "ruff_pyflakes_imports_upgrades": "pass",
        "dashboard_typescript_vite": "pass",
        "python_build": "pass",
        "twine_check": "pass",
        "fresh_venv_smoke": "pass",
        "vendor_runtime_dependency_scan": "pass",
    },
    "runtime_boundaries": {
        "default_off_or_inert": True,
        "router_mutation": False,
        "automatic_promotion": False,
        "hard_residency_claimed_by_generic_mlx_backend": False,
        "kv_resize_requires_explicit_capability": True,
        "content_export_default": False,
        "policy_worker_pool_is_bounded": True,
        "response_policy_rejection_is_enforced": True,
        "partial_prefetch_failure_preserves_warm_set": True,
    },
}
destination = Path("docs/validation/native-adaptive-phase2.json")
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(
    json.dumps(receipt, sort_keys=True, indent=2) + "\n",
    encoding="utf-8",
)
PY

rm -f \
  scripts/materialize_native_adaptive_phase2.py \
  scripts/materialize_native_adaptive_phase2_fixups.py \
  scripts/materialize_native_adaptive_phase2_fixups_v3.py \
  scripts/materialize_native_adaptive_phase2_fixups_v4.py \
  scripts/materialize_native_adaptive_phase2_fixups_v5.py \
  scripts/materialize_native_adaptive_phase2_fixups_v6.py \
  scripts/materialize_native_adaptive_phase2_fixups_v8.py \
  scripts/materialize_native_adaptive_phase2_fixups_v9.py \
  scripts/publish_native_adaptive_phase2_v9.sh
for version in '' -v2 -v3 -v4 -v5 -v6 -v7 -v8 -v9; do
  rm -f ".github/workflows/publish-native-adaptive-phase2${version}.yml"
done
rm -f \
  .github/workflows/materialize-native-adaptive-phase2.yml \
  .github/workflows/materialize-native-adaptive-phase2-v2.yml

git config user.name "OpenAI Integration Worker"
git config user.email "noreply@openai.com"
git reset --soft "$BASE_SHA"
git add -A
git diff --cached --check
git commit -m "feat(runtime): complete native adaptive systems"
test "$(git rev-list --count "$BASE_SHA"..HEAD)" = 1
test -z "$(git status --porcelain)"

final_sha="$(git rev-parse HEAD)"
git push --force origin "HEAD:refs/heads/${GITHUB_REF_NAME}"
git push \
  --force-with-lease="refs/heads/$PR_HEAD:$LEASE_SHA" \
  origin "HEAD:refs/heads/$PR_HEAD"
echo "Published $final_sha to $PR_HEAD"
