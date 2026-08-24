<div align="center">

<img src="docs/assets/readme/hero.svg" alt="MTPLX" width="100%" />

# Run local LLMs on Apple Silicon, around twice as fast.

[![PyPI](https://img.shields.io/pypi/v/mtplx?label=PyPI)](https://pypi.org/project/mtplx/)
[![CI](https://github.com/youssofal/MTPLX/actions/workflows/ci.yml/badge.svg)](https://github.com/youssofal/MTPLX/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![macOS Apple Silicon](https://img.shields.io/badge/macOS-Apple%20Silicon-black?logo=apple)](https://developer.apple.com/metal/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

</div>

MTPLX is a native Mac app and a command line for running local language models with multi-token prediction. Modern models like Qwen 3.5/3.6/3.8 ship with built-in MTP heads. Almost no runtime uses them. MTPLX does: the model drafts several tokens ahead of itself, verifies each drafted block in a single batched forward pass, and commits tokens through exact rejection sampling with residual correction. Same model, same output distribution, measured 1.6x faster on a 16 GB M4 Mac mini and 2.24x on an M5 Max.

There is no second draft model eating your RAM, and no greedy shortcut that quietly changes what the model would have said at real sampling settings. The acceptance math is the Leviathan and Chen rejection sampling theorem with residual correction, so `temperature=0.6, top_p=0.95` behaves exactly like normal decoding, just faster.

## Native runtime intelligence and adaptive systems

MTPLX now includes a vendor-neutral runtime intelligence layer built directly on MLX, SessionBank, and the existing OpenAI-compatible server. It does **not** import, embed, launch, or require FreeToken, Future AGI, or another external runtime. Every serving-path feature is disabled or inert by default; enabling diagnostics or offline replay does not silently change decoding behavior.

### Before and after

| Area | Before | After |
|---|---|---|
| **Session reuse** | SessionBank reused exact token prefixes, but complete-message boundaries were not first-class recurrent-cache edges. | Exact semantic anchors admit only byte-exact, complete-message prefixes and can mark instructions, user/assistant turns, reasoning/tool boundaries, and compaction summaries as reusable edges. |
| **MoE working set** | Expert routing ran without persistent locality evidence or an MTPLX-owned warm-set plan. | Opt-in expert-locality telemetry feeds a byte-bounded, hysteretic expert warm-set controller without changing router choices. |
| **Memory governance** | SessionBank, expert state, and KV headroom were managed independently. | Safe-point governance and an atomic unified-memory coordinator plan SessionBank, the expert warm set, and protected KV headroom together, with rollback on failed mutations. |
| **Production diagnosis** | Metrics and the flight recorder exposed runtime events, but there was no bounded capture-to-replay contract. | Privacy-default request capture stores counts and SHA-256 digests, then supports deterministic replay and ordered trace-parity diagnosis. |
| **Candidate evaluation** | Candidate changes required ad hoc comparison and promotion decisions. | Offline counterfactual replay isolates candidates and evaluators, applies explicit regression gates, and writes auditable receipts; promotion is never automatic. |
| **Observability and policy** | No native OTLP exporter or lifecycle policy bus covered the complete serving path. | Dependency-free OTLP/HTTP export and trusted, bounded request, stream-event, response, and error hooks are available without an external policy service. |
| **Operations** | The dashboard focused on throughput, cache, memory, requests, and thermals. | `GET /v1/mtplx/systems`, a truthful Systems view, a read-only Native command surface, and a permanent cross-version/configuration test matrix expose the new systems. |

### What the new systems do

**Semantic and memory intelligence.** `MTPLX_SEMANTIC_ANCHORS=1` enables exact complete-message cache anchors. `MTPLX_EXPERT_LOCALITY=1` records bounded MoE reuse telemetry without changing routing. `MTPLX_MEMORY_GOVERNOR=1` allows SessionBank budget changes only at a verified safe point while the model lock is held. `MTPLX_EXPERT_RESIDENCY=1` turns locality evidence into a bounded warm-set plan; the generic MLX backend reports `materialize_only` because it can evaluate lazy expert arrays but cannot promise physical page pinning or per-expert unloading. `MTPLX_UNIFIED_MEMORY=1` coordinates SessionBank, expert, and protected KV budgets atomically.

**Capture, replay, telemetry, and policy.** Setting `MTPLX_REQUEST_CAPTURE_DIR` enables bounded atomic request envelopes. Prompt tokens, messages, prompt text, response text, and exception text remain absent by default; counts and SHA-256 digests are retained for correlation. `MTPLX_OTLP_ENDPOINT` enables a bounded, fail-open OTLP/HTTP exporter with the same privacy-default treatment. Trusted application code can register deterministic request, stream-event, response, and error hooks through `mtplx.policy_hooks.PolicyBus`. Replay remains offline, rejects stale capture plans, and cannot alter live serving or promote a candidate automatically.

The features are independent. A representative opt-in configuration is:

```bash
export MTPLX_SEMANTIC_ANCHORS=1
export MTPLX_EXPERT_LOCALITY=1
export MTPLX_MEMORY_GOVERNOR=1

export MTPLX_EXPERT_RESIDENCY=1
export MTPLX_EXPERT_RESIDENCY_BYTES=$((8 * 1024 * 1024 * 1024))

export MTPLX_UNIFIED_MEMORY=1
export MTPLX_UNIFIED_MEMORY_TARGET=0.88
export MTPLX_UNIFIED_MEMORY_RESERVE_BYTES=$((4 * 1024 * 1024 * 1024))

export MTPLX_REQUEST_CAPTURE_DIR="$HOME/.mtplx/captures"
export MTPLX_OTLP_ENDPOINT=http://127.0.0.1:4318/v1/traces
```

Inspect the runtime's actual state instead of assuming that configuration equals successful work:

```bash
curl -fsS http://127.0.0.1:8000/v1/mtplx/systems | python -m json.tool
```

The Systems surface distinguishes availability, enablement, wiring, observation, activity, and blocked work. The safety contract is explicit: no router mutation from telemetry, no memory mutation outside a proven safe point, no live KV resize without a backend capability, no content persistence/export without a separate opt-in, fail-open observability, and no automatic replay promotion.

Full operator details are in [Native runtime systems](docs/native-systems.md) and [Native adaptive systems](docs/native-adaptive-systems.md).

<details>
<summary><strong>Maintainer review map and verification</strong></summary>

Suggested review order:

1. `mtplx/semantic_anchors.py`, `memory_governor.py`, and `expert_locality.py`
2. `mtplx/deterministic_replay.py`, `request_capture.py`, and `replay_orchestrator.py`
3. `mtplx/expert_residency.py`, `unified_memory.py`, and `native_adaptive.py`
4. `mtplx/otlp_export.py` and `policy_hooks.py`
5. `mtplx/runtime_systems.py` and `mtplx/server/openai.py`
6. Dashboard source, focused tests, and `docs/validation/native-adaptive-phase2.json`

The permanent `native-adaptive` workflow covers pure-system tests on Python 3.11, 3.12, 3.13, and 3.14; isolated default-off, expert-only, unified-memory-only, and combined-memory process profiles; and macOS 14 ARM64 runtime integration. The validation receipt records the broader publication gate, including compatibility tests, TypeScript/Vite production build, wheel/sdist creation, Twine validation, fresh-environment installation, and external-runtime dependency scans.

</details>

## Get it

**The Mac app** is the easiest way in. Download the DMG at [mtplx.com](https://mtplx.com/download), drag it to Applications, and the app takes care of everything else: it checks your hardware, recommends a model that actually fits your memory, downloads it, sets up its own Python engine (no Homebrew needed), installs fan control, puts `mtplx` on your PATH, and then measures your machine to pick the fastest decoding depth.

**Recommended for coding:** Qwen 3.8 27B Optimized Speed is a 4-bit dynamic
quant with great coding speeds and good quality. Its two siblings sit right
under it in the app and CLI: Bare Speed (quickest burst chat speeds, lower
quality and slower on long coding tasks) and Optimized Quality (8-bit dynamic
quant, good coding speeds and perfect quality). Qwen 3.6 Optimized Speed V2
remains available directly below them.

**The CLI** on its own:

```bash
brew install youssofal/mtplx/mtplx
mtplx start
```

or `python3 -m pip install mtplx` if you prefer pip. All releases are listed at [mtplx.com/releases](https://mtplx.com/releases/).

Requirements: Apple Silicon (M1 or newer), macOS 14+. 16 GB of memory runs the
4B and 9B models comfortably. Qwen 3.8 Optimized Speed is recommended on Macs
with 32 GB or more; on M1 and M2 the app and CLI pick its FP16 build (same
weights, native precision for those chips) automatically. Both check your Mac
before recommending anything.

## The app

<img src="docs/assets/readme/app-dashboard.jpg" alt="MTPLX dashboard with live decode gauge" width="100%" />

The dashboard shows what your model is doing while it does it: live tokens per second, acceptance rate by draft depth, the verify waterfall, cache state, and system pressure. The Systems view separately reports each native runtime subsystem as unavailable, inactive, enabled, observed, active, or blocked; source-code presence alone is never presented as healthy operation. The Native view provides copyable, read-only commands for startup, tuning, diagnostics, inspection, and benchmarks without giving the browser a shell-execution endpoint. When you start a chat, code an agent against the local server, or run a benchmark, the numbers are right there.

<img src="docs/assets/readme/app-chat.jpg" alt="Chat streaming with live speed badge" width="100%" />

Chat is native, streams with thinking cards, takes file attachments, and can search the web. One click launches OpenCode, Pi, Hermes, Open WebUI, or anything else that speaks the OpenAI or Anthropic API against your local server. There is also a built-in AIME benchmark runner with fully disclosed, coaching-free prompts, so you can score a model yourself instead of trusting a chart.

## Auto-tune

The right draft depth depends on your specific Mac: chip, memory bandwidth, thermals. During onboarding (and any time after), MTPLX runs the real model on your machine at each depth, with fans pinned for clean timing, and keeps autoregressive decoding as the baseline. If an MTP depth beats it, that depth is saved. If nothing beats the baseline, nothing is saved and the app says so. From the terminal it is one command:

```bash
mtplx tune --model <model-or-path> --retune
```

On a 16 GB M4 Mac mini, tuning the 9B model lands on depth 1: 14.4 tok/s baseline becomes 23.0 tok/s.

## Forge: make your own MTP models

<img src="docs/assets/readme/app-forge.jpg" alt="Forge verifying a freshly built MTP model" width="100%" />

Forge takes a Hugging Face repo and turns it into an MTPLX-ready MTP model: convert to MLX, train the MTP adapter, verify that the result is actually faster and still exact, and publish back to the Hub if you want to share it. The honest part matters: Forge measures before and after on your hardware and shows you the verdict ("Depth 1 is fastest: 227.1 to 296.1, 1.30x") rather than assuming the adapter helped. Available in the app and as `mtplx forge` subcommands.

MTPLX does not support attaching a separately supplied MTP sidecar to an arbitrary MLX trunk. Matching architecture fields, tensor shapes, or provenance labels cannot prove that the head was trained against those exact trunk weights. Use a complete model that already includes its matching MTP weights, or use Forge to build and verify an artifact from its original source checkpoint.

The official catalog lives on Hugging Face under [Youssofal](https://huggingface.co/Youssofal): Qwen 3.8 27B (Bare Speed, Optimized Speed, Optimized Quality, each with an FP16 build for M1 and M2), Qwen 3.6 (27B, 35B MoE) in speed and quality builds (the 35B MoE adds a balance build), Qwen 3.5 (4B, 9B), plus Gemma 4. The app and the CLI recommend from these based on your hardware.

## The server

`mtplx start` (or the app's play button) serves an OpenAI-compatible API on `127.0.0.1:8000`: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, the optional `/v1/embeddings` and `/v1/rerank` (see below), plus an Anthropic-compatible `/v1/messages` with streaming, tool calls in both styles, `/health`, `/metrics`, and the native-systems contract at `/v1/mtplx/systems`. Claude Code, Cline, Continue, Open WebUI, curl, the openai and anthropic Python clients: if it speaks the API, it works. The app and CLI share one server, so `mtplx start` attaches to the app's running model instead of loading a second copy.

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mtplx","messages":[{"role":"user","content":"hi"}],"stream":true}'
```

Sessions survive: a warm-prefix session bank keeps multi-turn chats fast, and a default-on SSD session cache restores sessions near-instantly across restarts (disable with `--ssd-session-cache off`).

### Embeddings and reranking

The same daemon can serve retrieval models, so a RAG or agent-memory setup does not need a second inference server beside MTPLX. Point it at any MLX embedding or reranker model — Hugging Face id or local path, optionally with a `REF=served-id` alias:

```bash
mtplx serve \
  --embedding-model mlx-community/Qwen3-Embedding-8B-4bit-DWQ \
  --reranker-model vserifsaglam/Qwen3-Reranker-4B-4bit-MLX
```

```bash
curl http://127.0.0.1:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3-Embedding-8B-4bit-DWQ","input":["hello","world"]}'

curl http://127.0.0.1:8000/v1/rerank \
  -H 'Content-Type: application/json' \
  -d '{"query":"where is the cache?","documents":["the cache lives in ~/.mtplx","unrelated text"]}'
```

Both flags repeat, so several models can be served at once and picked per request via `"model"`. Listing the same reference as both an embedder and a reranker loads **one** copy of the weights and serves both roles from it. Retrieval models load on first request and are capped by `--retrieval-max-resident` (default 2), which unloads the least recently used one beyond the cap — an unused endpoint costs nothing. `/v1/models` stays chat-only by default so chat clients that enumerate models never offer an embedder as a conversation target; list retrieval models with `?capability=embedding` or `?capability=rerank` (every entry carries its `capability`), and a chat completion that requests a retrieval id gets a clear 400 rather than a silent answer from the chat model.

These models do not go through the MTP path, and that is deliberate: multi-token prediction makes *next-token* decoding cheaper, which means nothing for a model that returns a vector instead of a token stream. Configure them in the app under Settings → Retrieval endpoints, or persist them in `~/.mtplx/config.toml` as `embedding_models` and `reranker_models`. With nothing configured the endpoints answer 404 and chat behaves exactly as before. One safety gate: checkpoints that bundle their own Python inference code (the jina embedding/reranker MLX releases do) are refused with a 403 until you opt in with `--retrieval-trust-remote-code` (or `retrieval_trust_remote_code = true` in the config file) — a model download never gains code execution just by being pointed at.

Sampler controls cover `temperature`, `top_p`, `top_k`, and the OpenAI penalty pair `presence_penalty` / `frequency_penalty` — per request, as server defaults (`--default-presence-penalty` / `--default-frequency-penalty` on `start`/`serve`/`quickstart`), or live via `mtplx settings set` and the app's Presence Penalty dial. Penalties default to 0, which is an exact no-op that preserves MTP exactness. Qwen's guidance: leave them at 0 for coding and agent work; ~0.5–1.5 presence penalty helps creative writing or when a model loops on itself.

Concurrent scheduler modes, ownership guarantees, and backend-specific
implementations are documented in [Concurrency modes](docs/concurrency.md).

## CLI quick reference

```bash
mtplx start                # interactive: pick model, mode, surface, then chat
mtplx serve --port 8000    # API server only
mtplx stop                 # stop the running server cleanly
mtplx pull <hf-repo>       # download a model safely
mtplx models               # what is cached, sizes, validation
mtplx inspect <model>      # compatibility report before anything runs
mtplx inspect <model> --require-mtp --json  # fail closed unless native MTP is valid
mtplx tune --retune        # measure AR vs D1/D2/D3 on your Mac
mtplx forge --help         # build, verify, and publish MTP models (probe/build/publish/verify subcommands)
mtplx bench aime --quick   # run the AIME benchmark from the terminal
mtplx bench run --profile sustained --generation-mode mtp --strict --json
mtplx doctor --deep --json # deep install, runtime, and integration health
mtplx max --install        # fan control (one sudo prompt, crash-safe)
mtplx settings get/set     # read or change live server settings
```

Every command takes `--help`, and most inspection/diagnostic commands take `--json`. The CLI works without MLX installed for everything that does not need a model, so `doctor` and `inspect` run on any machine.

## Modes

| Mode | What it does | When |
|---|---|---|
| **Turbo** | NAX verify kernels + compiled verify; the default for the quantized 27B and 9B flagship models | Picked automatically for those models |
| **Sustained** | Default for all other models. Long-context MTP path with chunked prefill and request-sized KV | Everyday use, big files, 16K-200K prompts |
| **Sustained Max** | Sustained with fans pinned at 100% | Long work where you want maximum cooling |
| **Burst** | Legacy short-context benchmark lane, loud | Short prompts and benchmarks only |

Fan-backed modes restore your fans to automatic if MTPLX dies for any reason, including `kill -9` and closing the terminal. A detached watchdog handles it; this is verified on hardware, not assumed.

## Compatibility, honestly

`mtplx inspect` classifies models before anything runs: verified, family-compatible but unverified, architecture-compatible but unverified, AR-only, incompatible architecture, or no MTP heads at all. Unverified models load with an explicit unverified label. There are no silent fallbacks: if MTPLX cannot run a model correctly, it tells you instead of running it badly.

[Laguna-S-2.1 oQ4e](https://huggingface.co/mlx-community/Laguna-S-2.1-oQ4e) is supported through its exact MLX architecture in target-only AR mode:

```bash
mtplx start cli \
  --model mlx-community/Laguna-S-2.1-oQ4e \
  --download \
  --no-mtp
```

MTPLX pins that model to revision
`8e3f5cad513746264940c1c4195de48d7ea345a5` and verifies the 13-shard layout,
tokenizer, generation config, special tokens map, and Poolside chat template
before admitting it. The checkpoint has no native MTP head, so an MTP launch is
rejected before weights load instead of falling back during execution. The
weights occupy 59.72 GiB, a 64.13 GB snapshot on disk. The launch preflight
requires about 85 GiB of unified memory (weights, runtime headroom, and a
16 GiB system reserve) — in practice a 96 GB Mac; 128 GB is
comfortable. MTPLX defaults Laguna to a 32,768-token context
and response cap, and checks larger explicit server contexts against the active
Metal memory cap.

## What MTPLX is not

- Not an external-drafter system. The drafter is the target model's own MTP heads.
- Not a greedy-argmax trick. Acceptance is exact rejection sampling, correct at any temperature.
- Not a wrapper around FreeToken, Future AGI, or another runtime. The new systems are independent MTPLX implementations.
- Not an automatic optimizer or promotion service. Replay decisions are advisory and never alter serving by themselves.
- Not claiming hard per-expert page pinning from generic MLX. The built-in backend reports `materialize_only` unless a stronger capability exists.
- Not a CUDA project. MTPLX is MLX-native and Apple Silicon first. For Linux, use vLLM.

## License and credit

Apache-2.0: use it, modify it, ship it commercially. Keep the license and the [NOTICE](NOTICE) file if you redistribute.

**Attribution is required.** If you ship a product, app, or service that includes or is built on MTPLX, it has to say so inside the product itself, somewhere a user can see it (About screen, credits, settings, shipped docs, or a CLI startup banner):

> Powered by MTPLX
> https://github.com/youssofal/MTPLX

A mention in your repo or on your website does not cover it. The full terms are in [NOTICE](NOTICE), which Apache-2.0 section 4(d) carries with every copy.

MTPLX builds on [MLX](https://github.com/ml-explore/mlx) and the Qwen and Gemma model families; the speculative sampling math follows Leviathan and Chen (2023). Fan control via [ThermalForge](https://github.com/ProducerGuy/ThermalForge). Model weights remain governed by their upstream licenses.

Built by [Youssof Altoukhi](https://github.com/youssofal). Bug reports and benchmark replications welcome via [Issues](https://github.com/youssofal/MTPLX/issues).
