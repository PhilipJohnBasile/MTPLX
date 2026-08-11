# Model Compatibility

MTPLX separates detection from support.

| Tier | Meaning | Default behavior |
|---|---|---|
| Verified | `mtplx_runtime.json` exists and matches the expected contract | Run |
| Architecture-compatible, unverified | Qwen3-Next MTP markers exist, but no MTPLX contract | Refuse unless explicitly forced |
| AR-only | An exact architecture-specific AR loader is installed, but the checkpoint has no MTP head | Run only with target-only AR selected |
| Incompatible architecture | MTP markers exist for an unsupported architecture | Exit with roadmap pointer |
| No MTP | No MTP head detected | Exit with a clear message |

The AR-only tier is narrow by design. It currently recognizes the exact
mixed-precision geometry and storage map of `mlx-community/Laguna-S-2.1-oQ4e` at
revision `8e3f5cad513746264940c1c4195de48d7ea345a5`. Local cache admission also
requires the pinned source marker, all 13 shards at their reviewed sizes, the
index, tokenizer, generation config, special tokens map, and Poolside chat
template. Other Laguna variants — including the earlier uniform-4bit build —
remain blocked until they have their own construction-time validation and
runtime evidence.

The second AR-only route recognizes the exact 43-layer target-only derivative
`philipjohnbasile/DeepSeek-V4-Flash-0731-MLX-M5Max-TargetOnly`. Admission binds
the immutable public revision, DeepSeek V4 architecture and core geometry,
all 44 shard sizes, the small-file hashes and index closure, and every layer's
reviewed expert quantization recipe. Execution is
delegated to a separately installed native `mlx-serve` binary. The artifact's
`num_nextn_predict_layers=0` contract is mandatory; a V4 artifact that declares
or contains DSpark/MTP remains on the separate pending DSpark backend and is
not silently routed through target-only AR.

The artifact identity gate permits this external AR route, but its product
support level is experimental. Two short non-streaming smokes completed around
23.5 tok/s; two representative streaming attempts instead suffered severe
slowdown with no Metal headroom. Representative streaming and throughput are
therefore unapproved, and the smoke receipts do not support a speed claim.
