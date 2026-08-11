# SessionBank checkpoint replay

Checkpoint replay is a CPU-only falsification gate for changing MTPLX's sparse
recurrent-checkpoint placement. It records bounded, content-free overlap facts
at the request boundary and compares equal-budget placement policies offline.

Tracing does not change cache capture, restore, admission, or eviction. This
change does separately harden `put_snapshot`: when recurrence provenance is
unknown, the entry is treated as recurrent instead of receiving unsafe
attention-only sub-prefix restore rights. A favorable counterfactual only
identifies an experiment worth implementing; it is not a deployable result,
runtime evidence, or performance evidence.

## Collect a trace

Tracing is off by default. Set a bounded capacity before starting the server:

```bash
MTPLX_SESSION_OVERLAP_TRACE_MAX_EVENTS=4096 mtplx serve ...
```

After representative traffic, export the in-memory snapshot:

```bash
curl -s http://127.0.0.1:8000/admin/session-overlap-trace \
  -o session-overlap-trace.json
```

If the server requires an API key, use the same authorization header as the
other admin endpoints. Clear observations without clearing reusable cache
state:

```bash
curl -s -X POST http://127.0.0.1:8000/admin/session-overlap-trace/clear
```

The trace contains no prompt text, token IDs, token hashes, request or session
IDs, model paths, identity hashes, or wall-clock timestamps. It records prompt
lengths; process-local bank epochs and entry ordinals; compatible resident RAM
candidates; retained and actually capturable checkpoint positions; coarse
terminal/cache outcomes; and bounded duration counters. The producer records at
most 16 RAM candidates per event, even though the default SessionBank capacity
is 24 entries. A report with `compatible_entry_count` above its recorded
candidate count is truncated and cannot pass the gate. Trace clearing does not
clear SessionBank. Cache clearing increments the bank epoch so ordinals cannot
be compared across lifecycles.

Candidate scans use the operational idle-TTL and consumable-state rules without
evicting entries or hydrating SSD data. Placement-relevant metadata changes,
including recurrent-boundary hydration, receive a new content-free ordinal.
Shared-prefix snapshots preserve whether their cache contains recurrent state;
unknown legacy provenance fails closed as recurrent rather than being treated
as arbitrarily sliceable attention state.

One event is finalized per request, after completion, cancellation, or error.
Its sequence number is reserved during the server's SessionBank probe/admission
phase, before finalization, so replay recovers that process-local chronology
when concurrent requests finish out of order. It is not a client-observed
wall-clock HTTP-request-start timestamp.
Blank-generation retries do not create additional events. Warmup and background
maintenance requests are excluded. The hot path performs no trace I/O and does
not query the SSD tier.

For a streaming HTTP request that performs internal inspection, tool, or repair
dispatches, the first generation dispatch owns the recorded prompt, candidate,
and token-accounting fields. Later dispatches may only add evidence that the
bank was consulted; the outer HTTP lifecycle supplies the single terminal
status. This keeps one coherent replay row instead of mixing later prompts with
the first probe's candidate set.

The replay consumer validates the complete snapshot root: tracing was enabled;
`max_events`, `events_collected`, `events_dropped`, `pending_probe_count`,
`sequence_high_watermark`, and `bank_epoch` are mutually consistent; retained
events fit the bounded capacity; and finalized sequences are exactly contiguous.
Replay requires a quiescent snapshot with no pending probes or dropped events.
Candidate truncation also blocks favorable evidence.

## Replay the trace

```bash
mtplx checkpoint-replay \
  --trace session-overlap-trace.json \
  --output checkpoint-replay-report.json
```

The output must not resolve to the trace file itself, including through a
hardlink.

For each completed RAM SessionBank hit, replay evaluates every eligible resident
candidate and lets each policy choose its best entry. RAM is the only source
eligible for the checkpoint-placement comparison. SSD hits, cache misses,
bypasses, cancellations, and errors remain in the workload totals as
incumbent-equal zero-benefit rows; none can satisfy the placement or incumbent
provenance minimums. A `committed` or `last_window` MTP lane also excludes an
entry unless it has a durable committed-history snapshot or a live
committed-history cache reference; replay never sees a restore that generation
would reject for missing committed MTP history. The full-prefix endpoint is
mandatory and free; policies receive the same observed interior-checkpoint
budget.

The report separates two comparisons:

- `observed_fitted` is a capture-candidate counterfactual fitted only on the
  earlier request-start partition. The runtime observed that these positions
  were capturable, but geometric retention may have discarded their snapshot
  payloads. This arm is therefore not deployable in this PR.
- `future_known_upper` is a non-deployable oracle that knows each evaluation
  request's common-prefix length. With the same candidate entry and interior
  checkpoint budget, it chooses the best recorded capture boundary at or below
  that prefix, even when the incumbent retention made the entry unrestorable.
  It never invents an unobserved capture position and is used only as an
  optimistic no-go falsifier.

Replay validates unique sequence numbers, sorts by request start, fits placement
on the earlier chronological partition, and evaluates on the frozen later
partition. Completed events must account for the full prompt exactly:
`cached_tokens + new_prefill_tokens == prompt_tokens`. Cancelled and error
events may be partial, but cannot exceed the prompt and are forced to
zero-benefit rows. For pure-attention entries, verified incumbent restore is
the exact common prefix, not a retained recurrent checkpoint.

The canonical CPU counterfactual requires all of the following:

- zero pending probes and a contiguous finalized sequence through the recorded
  high-watermark;
- zero dropped trace events;
- zero candidate truncation events;
- every eligible RAM-hit row to identify a selected incumbent and have a
  verifiable restored-token count, with at least 100 verified incumbent
  observations;
- at least 100 eligible evaluation events;
- at least two non-append divergence buckets with 20 events each;
- a paired moving-block bootstrap with 10,000 replicates;
- the worse lower 95% bound from block lengths `ceil(n ** (1/3))` and twice
  that value to show at least 10% workload-wide token-work improvement.

Only the exact preregistered defaults can return `counterfactual_only`. Any
override to block size, decay, event thresholds, bootstrap iterations or block
length, or seed is labeled `exploratory_configuration`. Both statuses exit
nonzero. Every report includes `canonical_gate` and a complete `gate_config` so
a weakened exploratory run cannot be mistaken for the canonical analysis.

The divergence buckets are at most 256, 257-1,024, 1,025-8,192, and greater
than 8,192 tokens.

`no_go` means the fully evidenced comparison ran, but either the fitted
counterfactual's workload-wide point estimate does not reach 10% or the
optimistic same-candidate, same-budget oracle cannot justify the experiment.
`amber` means the point estimate reaches 10% but its lower confidence bound does
not.
`insufficient_data` means required provenance, zero-loss trace completeness, or
coverage is missing; it is not a favorable result.
`counterfactual_only` means the conservative CPU lower bound reaches 10%, but
some selected capture-candidate payloads may no longer exist. This PR never
emits a deployable GO status; `deployable_go_available` and `runtime_go` remain
false, and the CLI exits nonzero.

## Runtime promotion boundary

Do not wire replay output into SessionBank from this counterfactual. A later PR
must first implement policy-before-capture, bounded retention of the selected
payloads, or measured rematerialization. Only then can an M5 gate prove exact
continuation, at least 10% paired measured cost reduction, positive results in
two non-append buckets, no important bucket regression over 5%, and bounded
physical memory.

Before runtime promotion, also close the complete-state prerequisites from the
architecture audit: fail-closed snapshot validation, complete MTP positional
state, strong runtime/model/cache identity, compatibility-before-selection,
serialized clear/publication lifecycle, and SSD extra-state integrity.
