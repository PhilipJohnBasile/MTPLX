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
gh release download v0.3.0 --repo youssofal/mtplx --pattern 'mtplx-0.3.0-py3-none-any.whl'
python3 -m pip install ./mtplx-0.3.0-py3-none-any.whl
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

The Laguna download is pinned automatically. It needs about 64.13 GB of disk
space and at least 96 GiB unified memory; 128 GiB is recommended. Its default
context and maximum response are 32,768 tokens. A larger explicit server
context is accepted only when it fits the active Metal resident-memory cap.

For the 128 GB DeepSeek-V4 target-only artifact, install or build `mlx-serve`
and select AR explicitly:

```bash
mtplx pull philipjohnbasile/DeepSeek-V4-Flash-0731-MLX-M5Max-TargetOnly

MTPLX_MLX_SERVE_BIN=/path/to/mlx-serve \
mtplx serve \
  --model philipjohnbasile/DeepSeek-V4-Flash-0731-MLX-M5Max-TargetOnly \
  --no-mtp --yes
```

MTPLX strips ambient `MLX_SERVE_*` controls before launch, then installs the
reviewed `MLX_SERVE_WIRED=off` and 256 MB cache policy. This backend is an
experimental external native target runtime, not a speculative MTPLX backend.
Two short non-streaming smokes were around 23.5 tok/s, but representative
streaming attempts hit severe slowdown with no Metal headroom. Do not make a
throughput claim from the smoke results; representative streaming remains
unapproved.

Use `mtplx doctor --deep --json` for exhaustive diagnostics and `mtplx doctor --bundle` to create a redacted support bundle.
