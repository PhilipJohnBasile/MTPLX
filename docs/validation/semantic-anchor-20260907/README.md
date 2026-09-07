# Semantic-anchor real-model evidence — September 7, 2026

**The acceptance gate is not cleared.** One isolated OFF/ON pair completed for each workload. Neither recovered any additional cached tokens with semantic anchors enabled. The long pair failed exact output parity on the plain-append turn. This is preliminary negative evidence, not a completed three-pair campaign or a statistically established speed result.

A subsequent long ON arm was rejected by the host isolation guard when another MLX campaign controller appeared. It is retained under `excluded/` and is not counted as a valid pair. The other campaign subsequently started its model worker. Remaining repetitions require an uninterrupted GPU window; no other task was interrupted to obtain one.

## Pinned configuration

- Feature source: `dfe28bf94fb62d35ba305d3b764c044c880f0905`, including upstream correctness baseline `21be78b3f51820eecef020e5e4855c0715eaf9a5`. The source was unchanged between measured arms.
- Model: `Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed`, revision `766cec2bb474544381139cc036ae1c2267759b33`. Installed file sizes, available published SHA-256 values and measured hashes of every file were verified; see `provenance.json`.
- Runtime: MLX 0.32.2, mlx-lm 0.31.3, Transformers 5.8.0, tokenizers 0.22.2. Apple M5 Max, 128 GiB, macOS 27.0 build 26A5425a.
- Turbo profile, target-only AR, MTP sidecar not loaded, 131,072-token context, greedy sampling, request thinking disabled, native tokenizer template, preserved prior reasoning, no agent rewrites, one-token streaming and no stats footer.
- Each arm used a new server process, initially empty SessionBank and fresh private SSD directory. Bank cap 32 GiB, 24 entries, three per session; SSD cap 20 GiB; postcommit wait timeout 30 seconds. These are controlled settings, not a claim about automatic defaults.
- Initial 32-token warmup plus the identical blocking extended warmup finished before readiness. N-gram prewarming was disabled. AC power and Apple's default fan policy were used. Before/after process checks found no competing model job for the two included pairs; there was no continuous system-wide GPU trace.
- OFF and ON used the same complete settings, differing only in `MTPLX_SEMANTIC_ANCHORS`. Launch records, effective health settings and per-request flag telemetry accompany the operator-attested manifests. The safe-path server child explicitly imported the pinned checkout through `PYTHONPATH`.

## Frozen real-model workloads

Both conversations use public synthetic reference records and actual model-generated `lookup_record` tool calls, followed by deterministic tool replies. The five turns are cold tool call A, tool result A, plain append, repeated tool call B and tool result B. Complete requests were captured once and frozen. Replay outputs were never appended to later requests.

The long sequence contains **126,982–127,166 actual replay prompt tokens**, and the short sequence 4,097–4,281. Capture requests counted one token fewer; both replay arms use identical canonical request bytes and identical measured prompt counts. The source of that capture/replay difference has not been separately established, and capture timing is not used in the comparison.

The included long pair ran OFF then ON; the short pair ran ON then OFF. The protocol requires at least three fresh pairs per workload. One pair each is insufficient to establish repeatability or distinguish flag-correlated differences from baseline variation.

## Observations

The long plain-append output changed from `Marker: amber-17` with OFF to `amber-17` with ON. The output hashes differ and completion length changes from seven to five tokens; both finish normally. The other four long turns and all five short turns have matching output hashes, finish reasons and completion counts. The excluded subsequent ON arm repeated the shorter answer, but it is not an isolated replication and is not used to establish causation.

Cached-token recovery is identical in both included arms on every turn:

| Workload | Cold | Tool result | Plain append | Repeated tool call | Repeated tool result |
|---|---:|---:|---:|---:|---:|
| Short, OFF and ON | 0 | 4,125 | 4,125 | 4,125 | 4,125 |
| Long, OFF and ON | 0 | 127,010 | 127,010 | 126,976 | 126,976 |

Client time to first generated delta, in seconds; positive delta means ON was slower. These single-pair timings are diagnostic observations. The failed-parity long sequence is not eligible for a speed claim.

| Workload / warm turn | OFF | ON | ON − OFF |
|---|---:|---:|---:|
| Short / tool result | 0.595 | 0.742 | +0.147 |
| Short / plain append | 0.633 | 0.780 | +0.147 |
| Short / repeated tool call | 1.713 | 1.867 | +0.154 |
| Short / repeated tool result | 0.713 | 0.887 | +0.174 |
| Long / tool result | 11.036 | 10.409 | −0.627 |
| Long / plain append | 1.309 | 1.682 | +0.373 |
| Long / repeated tool call | 2.970 | 3.360 | +0.390 |
| Long / repeated tool result | 1.652 | 2.247 | +0.595 |

Cold generated TTFT is separate: short OFF 5.996 s / ON 6.117 s; long OFF 249.613 s / ON 248.625 s. Content TTFT equals generated TTFT for these text-only answers and is null for the tool-only answers; both fields are retained in `per-turn.csv` and the raw replay receipts.

Both arms use existing RAM clone restoration on the first two warm long turns and the existing 126,976-token block boundary for the last two. The long first tool-result turn waits 10.507 s OFF / 9.536 s ON for an already-running retokenized-history postcommit job. That waiting time must not be presented as a new semantic-anchor restoration benefit.

ON admits only the initial user boundary: token 4,090 for short and 126,975 for long. Later candidates are rejected by strict `not_exact_prefix` validation; rejection counts are 0, 2, 4, 6 and 8 across the sequence. The long extra boundary is one token before the existing block boundary used later. A prefill-boundary numerical effect is a possible lead for the output mismatch, but this receipt does not establish its cause. No prefix validation or parity check was weakened.

## Reproduction and evidence

Use the protocol in `../semantic-anchor-ab.md` and `scripts/bench_semantic_anchor_replay.py` at the pinned source commit. Its 15 contract tests passed locally before measurement; these tests validate the harness, not model behavior.

1. Decompress `short-transcript.json.gz` and `long-transcript.json.gz`; the decompressed bytes exactly match the frozen local inputs. Original and exported file hashes are recorded in `inventory.json`.
2. Replace `<SOURCE_DIR>`, `<MODEL_DIR>`, `<PYTHON>` and `<CAMPAIGN_DIR>` in launch records with local paths. `<SOURCE_DIR>` must be the pinned source checkout. Use a new SSD directory for every arm. The explicit `PYTHONPATH` is needed for the CLI's safe-path child to load that checkout.
3. Complete the same warmup and verify health settings before invoking the harness. Replay each frozen conversation into fresh OFF and ON server processes; compare with the unchanged harness.
4. Run at least three balanced pairs per workload with no other GPU campaign. Retain parity failures, zero gains and all excluded attempts separately.

`replay.json`, pair receipts, `per-turn.csv`, and `summary.json` preserve every measured value used above. `restore-commit.jsonl` contains the five measured requests' selected cache, anchor and timing fields. `generation-events.jsonl` retains generation events from the public synthetic fixture. `capture-*-responses.json` proves the model actually issued the captured tool calls. Full original local logs and the excluded setup attempts remain in the project workspace.

Export changes are limited to documented local-path substitutions, selected log fields and lossless transcript compression. This evidence does not certify model quality, other models, MTP mode, other context sizes, physical-device packaging or release readiness. The next gate is repeated isolated measurement and diagnosis of the observed parity failure; the measured baseline cache benefit cannot be credited to this feature.
