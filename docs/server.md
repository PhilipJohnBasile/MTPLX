# Server

The server target is OpenAI-compatible local serving, with Anthropic
Messages compatibility available for coding harness smoke tests.

```bash
mtplx serve --host 127.0.0.1 --port 8000 --no-stats-footer
```

See [Concurrency modes](concurrency.md) for scheduler selection, ownership
rules, and model/backend-specific implementations.

Endpoints:

- `GET /health`
- `GET /metrics`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/messages`
- `GET /admin/sessions`
- `POST /admin/cache/clear`

## n-gram table pre-read (`--ngram-prewarm`, on by default)

Models with a streamed n-gram sidecar (the Qwen 3.8 Flash-Next family) keep a
~30 GiB PLE table on SSD and gather rows from it through a memmap. Those rows
are hash-scattered, so a cold table means demand faults during decode —
serial, and flat at ~1.4 GiB/s however many threads touch them — while reading
the file sequentially runs at ~12 GiB/s. So the server reads the whole table
once at model load:

```
[mtplx] n-gram table pre-read 29.8 GiB in 2.5 s (12.1 GiB/s)
```

That ~2.5 s is bought back immediately: cold sidecar rows measured 56 tok/s
against 68.8 tok/s warm, and the first prefill chunk stops being bimodal
(1.9 s vs 4.4 s on the same build, tracking nothing but table residency).

Opt out when you want the old as-found behaviour — a machine where the load
time matters more than the first replies, or an A/B of this exact effect:

```bash
mtplx serve --no-ngram-prewarm
mtplx start --no-ngram-prewarm
```

The equivalent environment variable is `MTPLX_NGRAM_PREWARM` (`1`/`0`), and
the CLI flag wins over it. `GET /health` reports what actually happened:

```json
"ngram_prewarm": {"enabled": true, "bytes": 32000000000, "seconds": 2.5, "gib_per_s": 12.1}
```

**Caveat.** This warms the page cache; it does not pin it. On a 128 GB Mac
running ~85 GB of wired weights next to a 32 GB table, memory pressure can
evict those pages again, and the pre-read has no way to notice. The hot-row
LRU (`MTPLX_NGRAM_HOT_MB`, default 1024) holds the popular rows in RAM and is
unaffected. A periodic re-warm, or growing the hot-row cache to cover the
working set, is the follow-up — deliberately not part of this change.

## Sharing on your network (other devices, Parallels/VM guests)

The default bind is `127.0.0.1`: only this Mac can connect. To reach MTPLX
from other devices — or from a Windows VM in Parallels/VMware/UTM on the same
Mac, which arrives over the virtual network rather than loopback — bind all
interfaces. Non-localhost binds require an API key; if the key file doesn't
exist yet it is created with a fresh key and printed once:

```bash
mtplx serve --host 0.0.0.0 --port 8000 --api-key-file ~/.mtplx/api-key
```

Startup prints a `Network OpenAI API Base URL` (your Mac's LAN address, e.g.
`http://192.168.1.20:8000/v1`). On the other machine, point any
OpenAI-compatible client at that base URL with the printed key as the API
key (sent as a Bearer token). Parallels shared networking reaches the Mac's
LAN address directly; macOS may ask once to allow incoming connections —
click Allow. To pass the key inline instead of a file:

```bash
mtplx serve --host 0.0.0.0 --port 8000 --api-key "$MTPLX_API_KEY"
```

For Open WebUI, set the OpenAI-compatible base URL to:

```text
http://127.0.0.1:8000/v1
```

For Dockerized Open WebUI, the container must use the host gateway URL, not the host's loopback URL:

```bash
mtplx openwebui docker-command
```

That helper disables Open WebUI's Ollama probe and background task generations
so MTPLX only serves visible chat turns by default.

For Anthropic Messages-compatible clients, point the client base URL at the
bare server root — no `/v1` suffix:

```text
http://127.0.0.1:8000
```

The Anthropic SDK appends `/v1/messages` itself; a `/v1` base would request
`/v1/v1/messages`, which is not a registered route.

## Android Studio

Android Studio's external model provider should use the OpenAI-compatible URL
schema and the MTPLX `/v1` base URL:

```text
URL: http://127.0.0.1:8008/v1
URL schema: OpenAI-compatible
API key: leave blank for localhost unless MTPLX was started with --api-key
```

Refresh the model list after the server starts. MTPLX supports the OpenAI chat,
streaming, and tool-call request shape used by local coding clients; Gemini-only
proprietary behavior is outside that compatibility contract. To verify a local
setup, run:

```bash
mtplx doctor android-studio --port 8008
```

Since 2.5.3 the stats footer only appears on MTPLX-owned surfaces (the app
and the built-in browser chat); API clients such as Open WebUI, Claude Code,
and OpenCode never receive it, so no flag is needed for them.
`--no-stats-footer` still turns it off everywhere, and
`MTPLX_STATS_FOOTER_SCOPE=all` restores the pre-2.5.3 behavior. Metrics
remain available at `/metrics`.
