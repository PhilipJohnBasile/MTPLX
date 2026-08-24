# Native runtime systems

MTPLX carries two vendor-neutral system families behind one truthful dashboard surface.

## Semantic runtime intelligence

- **Exact semantic anchors** convert only byte-exact complete-message prefixes into mandatory recurrent-cache edges. Enable with `MTPLX_SEMANTIC_ANCHORS=1`.
- **Expert locality** samples MoE router reuse without changing router choices or expert placement. Enable with `MTPLX_EXPERT_LOCALITY=1`.
- **Safe-point memory governor** adjusts SessionBank budgets only while generation, restore, commit, and MTP work are idle. Enable with `MTPLX_MEMORY_GOVERNOR=1`.

## Deterministic replay and evaluation

- **Request capture** writes a bounded, atomic reproduction envelope before generation. It is disabled unless `MTPLX_REQUEST_CAPTURE_DIR` is set.
- Prompt tokens, messages, prompt text, response text, and exception text are **not persisted by default**. Counts and SHA-256 digests remain available for correlation. Content-bearing fields require separate explicit opt-ins.
- **Counterfactual replay** is an offline callable core. It preserves input order, isolates candidate and evaluator mutations, separates private credential-aware deduplication from public redacted fingerprints, and returns an explicit regression-policy decision.
- **Trace parity** compares ordered tensor boundaries with explicit numeric tolerances.
- Replay never changes the serving runtime and never promotes a candidate automatically.

## Dashboard contract

`GET /v1/mtplx/systems` reports whether each subsystem is available, enabled, wired, and sampled. The dashboard does not infer activity from source files or configuration alone.

## Operator verification

After starting the server, inspect the JSON contract directly before relying on the dashboard:

```bash
curl -fsS http://127.0.0.1:8000/v1/mtplx/systems | python -m json.tool
```

Treat `available`, `enabled`, `wired`, and `sampled` as separate states. A subsystem that is installed but disabled or has not yet observed a qualifying request must remain visibly inactive rather than being reported as healthy. Request-capture directories should be treated as private diagnostic data even when content capture remains disabled.

## Phase-two adaptive systems

MTPLX also provides active expert warm-set planning, shared unified-memory
budgets, privacy-first OTLP/HTTP export, bounded lifecycle policy hooks, and
capture-to-replay planning with stale-source checks and non-automatic
promotion receipts. See [Native adaptive systems](native-adaptive-systems.md).
