# Quickstart

```bash
brew install youssofal/mtplx/mtplx

mtplx help
mtplx doctor --summary
mtplx pull Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed
mtplx inspect Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed --json
```

Homebrew is the recommended macOS path. Python-only installs can use PyPI:

```bash
python3 -m pip install -U mtplx
```

The GitHub release wheel remains available for reproducible installs:

```bash
gh release download --repo youssofal/mtplx --pattern '*.whl'   # latest tagged release
python3 -m pip install ./mtplx-*-py3-none-any.whl
```

The commands above are no-MLX-safe except generation and serving. A missing MLX runtime should appear in `doctor` as an actionable dependency issue, not a traceback.

After the verified model is available:

```bash
mtplx start
mtplx start cli
mtplx start cli --no-mtp
mtplx quickstart --port 8000 --no-stats-footer
```

`--no-mtp` switches generation to target-only AR. For MTP-equipped models the
MTP runtime stays loaded, so terminal chat can use `/mtp off`, `/mtp on`, and
`/mtp status` without reloading. Native AR-only models such as
`mlx-community/Laguna-S-2.1-oQ4e` instead install an unloaded AR route at
construction because there is no MTP head to retain.

## Choose the Qwen 35B concurrent-MTP numerics profile

Qwen 35B's `mtp_batch` scheduler accepts
`--mtp-batch-numerics throughput|balanced|b1-exact`. The fastest profile is
named `throughput` (there is no `performance` value):

```bash
# Fast B8 MTP; default.
mtplx serve --model <qwen-35b-model-or-path> \
  --scheduler-mode mtp_batch \
  --mtp-batch-numerics throughput

# B8 MTP with selected B1 arithmetic; closer, but not token-exact with B1.
mtplx serve --model <qwen-35b-model-or-path> \
  --scheduler-mode mtp_batch \
  --mtp-batch-numerics balanced

# Exact unchanged B1 behavior; serialized, so this is not B8 throughput.
mtplx serve --model <qwen-35b-model-or-path> \
  --scheduler-mode mtp_batch \
  --mtp-batch-numerics b1-exact
```

To make the choice persistent:

```bash
mtplx config set scheduler_mode mtp_batch
mtplx config set mtp_batch_numerics throughput  # or balanced / b1-exact
mtplx config show --json
```

This is a construction-time setting, not a per-request or live setting. Stop
and restart the server after changing the saved profile. A CLI value overrides
the saved value for that launch. The option applies only to the Qwen 35B
`mtp_batch` route; incompatible scheduler/model combinations fail during
startup instead of falling back silently.

The Laguna download is pinned automatically. It needs about 64.13 GB of disk
space, and the runtime's admission gate requires ≈85.3 GiB of unified memory
(weights plus runtime headroom and a 16 GiB system reserve) — in practice a
96 GB Mac, with 128 GB comfortable. Its default
context and maximum response are 32,768 tokens. A larger explicit server
context is accepted only when it fits the active Metal resident-memory cap.

Use `mtplx doctor --deep --json` for exhaustive diagnostics and `mtplx doctor --bundle` to create a redacted support bundle.
