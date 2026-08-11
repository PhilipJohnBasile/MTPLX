"""Pure-CPU replay for SessionBank schema-v2 overlap traces.

The trace is deliberately a producer/consumer boundary.  The producer emits a
request-level ``bank_consulted`` flag and an ``outcome`` mapping, plus the
compatible ``candidates`` that were available at that request.  It does *not*
emit a per-candidate cache hit or identity flag: compatibility was already
filtered before candidates were recorded.  This module consumes that contract
directly and rejects the retired winner-only/synthetic candidate formats.

The chronological first 70% of request-start sequence numbers forms an
empirical overlap histogram for each ``entry_ordinal``.  The final 30% is
evaluated without contributing to placement.  A fitted placement uses the
candidate's observed interior checkpoint budget.  The endpoint ``N`` is free
in every compared policy.

For an observed candidate, ``L`` is its last successful interior capture.
Before its exact planner runs, all non-exact tail observations ``L <= t < N``
are folded into depth ``L``.  For every interior placement, their original
loss is ``t - c = (L - c) + (t - L)``.  The second term is
placement-independent, so the transformation preserves the optimum.  Exact
``t == N`` observations are removed first because the free endpoint already
gives them zero loss.

``future_known_upper`` is a non-deployable no-go oracle.  It retains the same
candidate and incumbent interior-checkpoint budget, but may reposition one
checkpoint only to a successful, recorded capture boundary at or below that
event's known common-prefix position.  It can therefore assess the best
capturable future placement without inventing a snapshot that the trace did
not observe.

``observed_fitted`` is also non-deployable.  It is a capture-candidate
counterfactual: the producer records positions where recurrent snapshots were
successfully captured, while geometric retention may later discard their
payloads.  Because the placement is fit from later requests, this replay
cannot apply the choice before the source entry was captured.  The actual
retained/incumbent arm remains separately reported as the recorded RAM restore
baseline.  A future change must implement policy-before-capture, bounded
retention, or measured rematerialization before a GO gate can exist.

This is an optimistic same-trace *token-work* replay.  It is not M5 timing,
quality, memory, admission, eviction/LRU, or runtime-correctness evidence.
``runtime_go`` is therefore always false.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Integral, Real
import random
from typing import Any

from mtplx.checkpoint_placement import (
    balanced_checkpoint_positions,
    geometric_tail_checkpoint_positions,
    optimal_checkpoint_positions_from_candidates,
)


__all__ = ["replay_checkpoint_trace"]


_SCHEMA_VERSION = 2
_MAX_PRODUCER_EVENTS = 4096
_MAX_PRODUCER_CANDIDATES = 16
_PRODUCER_LANES = frozenset({"solo_mtp", "solo_ar", "ar_batch", "unknown"})
_PRODUCER_TERMINAL_STATUSES = frozenset({"completed", "cancelled", "error"})
_PRODUCER_CACHE_SOURCES = frozenset({"none", "ram", "ssd", "other"})
_PRODUCER_RESTORE_KINDS = frozenset(
    {
        "cold",
        "clone",
        "exact",
        "near",
        "block",
        "boundary",
        "reference",
        "ssd_clone",
        "other",
    }
)
_CANONICAL_BLOCK_SIZE = 256
_CANONICAL_DECAY = 1.0
_CANONICAL_MIN_EVENTS = 100
_CANONICAL_MIN_BUCKET_EVENTS = 20
_CANONICAL_BOOTSTRAP_ITERATIONS = 10_000
_CANONICAL_SEED = 0
_CANONICAL_HISTORY_NUMERATOR = 70
_CANONICAL_HISTORY_DENOMINATOR = 100
_CANONICAL_HISTORY_FRACTION = (
    _CANONICAL_HISTORY_NUMERATOR / _CANONICAL_HISTORY_DENOMINATOR
)
_CANONICAL_DECISION_THRESHOLD_PERCENT = 10.0
_CPU_FALSIFIER_THRESHOLD_PERCENT = _CANONICAL_DECISION_THRESHOLD_PERCENT
_HISTORY_FRACTION = _CANONICAL_HISTORY_FRACTION
_MIN_HISTORY_OBSERVATIONS = 2
_MIN_BOOTSTRAP_ITERATIONS = _CANONICAL_BOOTSTRAP_ITERATIONS
_MIN_POPULATED_NONAPPEND_BUCKETS = 2
_BOOTSTRAP_LOWER_PERCENTILE = 0.05
_BOOTSTRAP_UPPER_PERCENTILE = 0.95
_POLICIES = (
    "prompt_end_only",
    "incumbent",
    "observed_fitted",
    "balanced",
    "geometric_tail",
    "future_known_upper",
)
_BUCKETS = (
    ("<=256", 1, 256),
    ("257-1024", 257, 1024),
    ("1025-8192", 1025, 8192),
    (">8192", 8193, None),
)


@dataclass(frozen=True, slots=True)
class _Candidate:
    """A content-free, producer-observed SessionBank candidate."""

    index: int
    entry_ordinal: int
    stored_prefix_tokens: int
    common_prefix_tokens: int
    retained_checkpoint_tokens: tuple[int, ...]
    capture_candidate_tokens: tuple[int, ...]
    incumbent_interior_budget: int
    has_recurrent: bool
    structurally_restorable: bool

    @property
    def incumbent_interior_positions(self) -> tuple[int, ...]:
        return tuple(
            position
            for position in self.retained_checkpoint_tokens
            if position < self.stored_prefix_tokens
        )


@dataclass(frozen=True, slots=True)
class _Event:
    """Validated trace values needed for token-work replay."""

    sequence: int
    bank_epoch: int
    prompt_tokens: int
    bank_consulted: bool
    cache_case: str
    cache_source: str
    lane: str
    terminal_status: str
    compatible_entry_count: int
    candidates: tuple[_Candidate, ...]
    selected_entry_ordinal: int | None
    outcome_cached_tokens: int


@dataclass(frozen=True, slots=True)
class _TraceContract:
    """Validated root-level producer provenance for one trace snapshot."""

    max_events: int
    events_collected: int
    events_dropped: int
    pending_probe_count: int
    sequence_high_watermark: int
    bank_epoch: int


@dataclass(frozen=True, slots=True)
class _Selection:
    """One policy's best candidate or its required no-cache fallback."""

    candidate: _Candidate | None
    checkpoint_tokens: tuple[int, ...]
    checkpoint_loss_tokens: int
    target_suffix_tokens: int
    work_tokens: int
    requested_budget: int
    effective_budget: int
    planned_budget: int
    rejection_counts: Mapping[str, int]

    @property
    def covered(self) -> bool:
        return self.candidate is not None


_CandidateIdentity = tuple[
    int,
    tuple[int, ...],
    tuple[int, ...],
    int,
    bool,
]


def _integer(value: object, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _real(value: object, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _sequence(value: object, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    return value


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _positions(value: object, *, limit: int, name: str) -> tuple[int, ...]:
    raw_positions = _sequence(value, name)
    result: list[int] = []
    previous = 0
    for index, raw_position in enumerate(raw_positions):
        position = _integer(raw_position, f"{name}[{index}]", minimum=1)
        if position > limit:
            raise ValueError(f"{name}[{index}] must not exceed stored_prefix_tokens")
        if position <= previous:
            raise ValueError(f"{name} must be sorted and unique")
        result.append(position)
        previous = position
    return tuple(result)


def _validated_candidate(
    raw: object, *, event_index: int, index: int, prompt: int
) -> _Candidate:
    name = f"events[{event_index}].candidates[{index}]"
    candidate = _mapping(raw, name)
    required = (
        "entry_ordinal",
        "stored_prefix_tokens",
        "common_prefix_tokens",
        "retained_checkpoint_tokens",
        "capture_candidate_tokens",
        "incumbent_interior_budget",
        "has_recurrent",
        "structurally_restorable",
    )
    missing = [field for field in required if field not in candidate]
    if missing:
        raise ValueError(f"{name} is missing required fields: {', '.join(missing)}")
    ordinal = _integer(candidate["entry_ordinal"], f"{name}.entry_ordinal", minimum=0)
    stored = _integer(
        candidate["stored_prefix_tokens"], f"{name}.stored_prefix_tokens", minimum=1
    )
    common = _integer(
        candidate["common_prefix_tokens"], f"{name}.common_prefix_tokens", minimum=0
    )
    if common > stored:
        raise ValueError(
            f"{name}.common_prefix_tokens must not exceed stored_prefix_tokens"
        )
    if common > prompt:
        raise ValueError(f"{name}.common_prefix_tokens must not exceed prompt_tokens")
    retained = _positions(
        candidate["retained_checkpoint_tokens"],
        limit=stored,
        name=f"{name}.retained_checkpoint_tokens",
    )
    if stored not in retained:
        raise ValueError(
            f"{name}.retained_checkpoint_tokens must include the stored endpoint"
        )
    capture = _positions(
        candidate["capture_candidate_tokens"],
        limit=stored,
        name=f"{name}.capture_candidate_tokens",
    )
    budget = _integer(
        candidate["incumbent_interior_budget"],
        f"{name}.incumbent_interior_budget",
        minimum=0,
    )
    actual_budget = sum(position < stored for position in retained)
    if budget != actual_budget:
        raise ValueError(
            f"{name}.incumbent_interior_budget must equal retained interior checkpoints"
        )
    return _Candidate(
        index=index,
        entry_ordinal=ordinal,
        stored_prefix_tokens=stored,
        common_prefix_tokens=common,
        retained_checkpoint_tokens=retained,
        capture_candidate_tokens=capture,
        incumbent_interior_budget=budget,
        has_recurrent=_boolean(candidate["has_recurrent"], f"{name}.has_recurrent"),
        structurally_restorable=_boolean(
            candidate["structurally_restorable"], f"{name}.structurally_restorable"
        ),
    )


def _parse_event(raw: object, *, index: int) -> _Event:
    name = f"events[{index}]"
    event = _mapping(raw, name)
    required = (
        "schema_version",
        "sequence",
        "bank_epoch",
        "prompt_tokens",
        "bank_consulted",
        "lane",
        "terminal_status",
        "compatible_entry_count",
        "candidates",
        "outcome",
        "timings",
    )
    missing = [field for field in required if field not in event]
    if missing:
        raise ValueError(f"{name} is missing required fields: {', '.join(missing)}")
    if _integer(event["schema_version"], f"{name}.schema_version") != _SCHEMA_VERSION:
        raise ValueError(f"{name}.schema_version must be {_SCHEMA_VERSION}")
    sequence = _integer(event["sequence"], f"{name}.sequence", minimum=0)
    bank_epoch = _integer(event["bank_epoch"], f"{name}.bank_epoch", minimum=0)
    prompt = _integer(event["prompt_tokens"], f"{name}.prompt_tokens", minimum=1)
    bank_consulted = _boolean(event["bank_consulted"], f"{name}.bank_consulted")
    lane = event["lane"]
    terminal_status = event["terminal_status"]
    if lane not in _PRODUCER_LANES:
        raise ValueError(f"{name}.lane is not producer-recognized")
    if terminal_status not in _PRODUCER_TERMINAL_STATUSES:
        raise ValueError(f"{name}.terminal_status is not producer-recognized")
    candidates = tuple(
        _validated_candidate(
            candidate, event_index=index, index=candidate_index, prompt=prompt
        )
        for candidate_index, candidate in enumerate(
            _sequence(event["candidates"], f"{name}.candidates")
        )
    )
    ordinals = [candidate.entry_ordinal for candidate in candidates]
    if len(set(ordinals)) != len(ordinals):
        raise ValueError(f"{name}.candidates entry_ordinal values must be unique")
    if len(candidates) > _MAX_PRODUCER_CANDIDATES:
        raise ValueError(
            f"{name}.candidates must not exceed producer cap {_MAX_PRODUCER_CANDIDATES}"
        )
    compatible_entry_count = _integer(
        event["compatible_entry_count"], f"{name}.compatible_entry_count", minimum=0
    )
    if compatible_entry_count < len(candidates):
        raise ValueError(
            f"{name}.compatible_entry_count must cover recorded candidates"
        )
    outcome = _mapping(event["outcome"], f"{name}.outcome")
    outcome_required = (
        "selected_entry_ordinal",
        "cache_hit",
        "cached_tokens",
        "new_prefill_tokens",
        "cache_source",
        "restore_kind",
    )
    missing_outcome = [field for field in outcome_required if field not in outcome]
    if missing_outcome:
        raise ValueError(
            f"{name}.outcome is missing required fields: {', '.join(missing_outcome)}"
        )
    selected_raw = outcome["selected_entry_ordinal"]
    if selected_raw is None:
        selected = None
    else:
        selected = _integer(
            selected_raw, f"{name}.outcome.selected_entry_ordinal", minimum=0
        )
        if selected not in set(ordinals):
            raise ValueError(
                f"{name}.outcome.selected_entry_ordinal is not a candidate"
            )
    cache_hit = _boolean(outcome["cache_hit"], f"{name}.outcome.cache_hit")
    cached_tokens = _integer(
        outcome["cached_tokens"], f"{name}.outcome.cached_tokens", minimum=0
    )
    new_prefill_tokens = _integer(
        outcome["new_prefill_tokens"], f"{name}.outcome.new_prefill_tokens", minimum=0
    )
    if cached_tokens > prompt or new_prefill_tokens > prompt:
        raise ValueError(f"{name}.outcome token counts must not exceed prompt_tokens")
    cache_source = outcome["cache_source"]
    restore_kind = outcome["restore_kind"]
    if not isinstance(cache_source, str) or not isinstance(restore_kind, str):
        raise TypeError(f"{name}.outcome cache_source and restore_kind must be strings")
    if cache_source not in _PRODUCER_CACHE_SOURCES:
        raise ValueError(f"{name}.outcome.cache_source is not producer-recognized")
    if restore_kind not in _PRODUCER_RESTORE_KINDS:
        raise ValueError(f"{name}.outcome.restore_kind is not producer-recognized")
    if terminal_status == "completed":
        if cached_tokens + new_prefill_tokens != prompt:
            raise ValueError(
                f"{name}.outcome cached_tokens + new_prefill_tokens must equal "
                "prompt_tokens for completed events"
            )
    elif cached_tokens + new_prefill_tokens > prompt:
        raise ValueError(
            f"{name}.outcome token counts must not exceed prompt_tokens for "
            "cancelled or error events"
        )
    if not bank_consulted:
        if cache_hit or selected is not None or cached_tokens != 0:
            raise ValueError(
                f"{name}.outcome bypass cannot report a cache hit, selected entry, "
                "or cached tokens"
            )
        if cache_source not in {"none", "other"}:
            raise ValueError(
                f"{name}.outcome bypass must use cache_source none or other"
            )
    elif not cache_hit:
        if selected is not None or cached_tokens != 0:
            raise ValueError(
                f"{name}.outcome cache miss cannot report a selected entry or "
                "cached tokens"
            )
        if cache_source not in {"none", "other"}:
            raise ValueError(
                f"{name}.outcome cache miss must use cache_source none or other"
            )
    else:
        if cache_source not in {"ram", "ssd"}:
            raise ValueError(
                f"{name}.outcome cache hit must use cache_source ram or ssd"
            )
        if cache_source == "ssd" and selected is not None:
            raise ValueError(
                f"{name}.outcome SSD cache hit cannot select a RAM probe candidate"
            )
    _mapping(event["timings"], f"{name}.timings")
    if selected is not None:
        selected_candidate = next(
            candidate for candidate in candidates if candidate.entry_ordinal == selected
        )
        if cached_tokens > selected_candidate.common_prefix_tokens:
            raise ValueError(
                f"{name}.outcome.cached_tokens must not exceed selected common_prefix_tokens"
            )
    cache_case = "bypass" if not bank_consulted else "hit" if cache_hit else "miss"
    return _Event(
        sequence=sequence,
        bank_epoch=bank_epoch,
        prompt_tokens=prompt,
        bank_consulted=bank_consulted,
        cache_case=cache_case,
        cache_source=cache_source,
        lane=lane,
        terminal_status=terminal_status,
        compatible_entry_count=compatible_entry_count,
        candidates=candidates,
        selected_entry_ordinal=selected,
        outcome_cached_tokens=cached_tokens,
    )


def _parse_payload(payload: object) -> tuple[list[_Event], _TraceContract]:
    root = _mapping(payload, "payload")
    required = (
        "schema_version",
        "enabled",
        "max_events",
        "events_collected",
        "events_dropped",
        "pending_probe_count",
        "sequence_high_watermark",
        "bank_epoch",
        "events",
    )
    missing = [field for field in required if field not in root]
    if missing:
        raise ValueError(f"payload is missing required fields: {', '.join(missing)}")
    if _integer(root["schema_version"], "payload.schema_version") != _SCHEMA_VERSION:
        raise ValueError(f"payload.schema_version must be {_SCHEMA_VERSION}")
    if not _boolean(root["enabled"], "payload.enabled"):
        raise ValueError("payload.enabled must be true for replay")
    max_events = _integer(root["max_events"], "payload.max_events", minimum=1)
    if max_events > _MAX_PRODUCER_EVENTS:
        raise ValueError(
            f"payload.max_events must not exceed producer cap {_MAX_PRODUCER_EVENTS}"
        )
    collected = _integer(
        root["events_collected"], "payload.events_collected", minimum=0
    )
    dropped = _integer(root["events_dropped"], "payload.events_dropped", minimum=0)
    pending = _integer(
        root["pending_probe_count"], "payload.pending_probe_count", minimum=0
    )
    sequence_high_watermark = _integer(
        root["sequence_high_watermark"],
        "payload.sequence_high_watermark",
        minimum=0,
    )
    bank_epoch = _integer(root["bank_epoch"], "payload.bank_epoch", minimum=0)
    if pending != 0:
        raise ValueError(
            "payload.pending_probe_count must equal zero for a quiescent replay snapshot"
        )
    if dropped != 0:
        raise ValueError(
            "payload.events_dropped must equal zero for complete replay evidence"
        )
    events = [
        _parse_event(event, index=index)
        for index, event in enumerate(_sequence(root["events"], "payload.events"))
    ]
    if collected != len(events):
        raise ValueError(
            "payload.events_collected must equal payload.events length for complete replay evidence"
        )
    if collected != sequence_high_watermark:
        raise ValueError(
            "payload.events_collected must equal payload.sequence_high_watermark"
        )
    if collected > max_events:
        raise ValueError("complete replay evidence must fit within payload.max_events")
    if any(event.bank_epoch > bank_epoch for event in events):
        raise ValueError(
            "payload.bank_epoch must cover every retained event bank_epoch"
        )
    expected_sequences = list(range(1, sequence_high_watermark + 1))
    if sorted(event.sequence for event in events) != expected_sequences:
        raise ValueError(
            "payload.events sequence values must exactly cover "
            "1..payload.sequence_high_watermark"
        )
    ordered_events = sorted(events, key=lambda event: event.sequence)
    if any(
        previous.bank_epoch > current.bank_epoch
        for previous, current in zip(ordered_events, ordered_events[1:])
    ):
        raise ValueError(
            "payload.events bank_epoch must be nondecreasing by request-start sequence"
        )
    return (
        ordered_events,
        _TraceContract(
            max_events=max_events,
            events_collected=collected,
            events_dropped=dropped,
            pending_probe_count=pending,
            sequence_high_watermark=sequence_high_watermark,
            bank_epoch=bank_epoch,
        ),
    )


def _last_legal_interior(prefix_length: int, block_size: int) -> int:
    return ((prefix_length - 1) // block_size) * block_size


def _with_free_endpoint(interior: Sequence[int], prefix_length: int) -> tuple[int, ...]:
    return tuple(sorted({*interior, prefix_length}))


def _restore_position(checkpoints: Sequence[int], overlap: int) -> int:
    restore = 0
    for position in checkpoints:
        if position > overlap:
            break
        restore = position
    return restore


def _candidate_view(candidate: _Candidate) -> dict[str, object]:
    return {
        "entry_ordinal": candidate.entry_ordinal,
        "stored_prefix_tokens": candidate.stored_prefix_tokens,
        "common_prefix_tokens": candidate.common_prefix_tokens,
        "has_recurrent": candidate.has_recurrent,
        "structurally_restorable": candidate.structurally_restorable,
        "incumbent_interior_budget": candidate.incumbent_interior_budget,
    }


def _fallback(event: _Event, reasons: Mapping[str, int]) -> _Selection:
    return _Selection(
        candidate=None,
        checkpoint_tokens=(),
        checkpoint_loss_tokens=0,
        target_suffix_tokens=event.prompt_tokens,
        work_tokens=event.prompt_tokens,
        requested_budget=0,
        effective_budget=0,
        planned_budget=0,
        rejection_counts=dict(reasons),
    )


def _selection_for_candidate(
    event: _Event,
    candidate: _Candidate,
    interior: Sequence[int],
    *,
    requested_budget: int,
    effective_budget: int,
    planned_budget: int,
) -> _Selection:
    if candidate.has_recurrent:
        checkpoints = _with_free_endpoint(interior, candidate.stored_prefix_tokens)
        restore = _restore_position(checkpoints, candidate.common_prefix_tokens)
    else:
        # Attention-only cache state is sliceable at the exact shared prefix;
        # retained recurrent checkpoints neither limit nor explain its restore.
        checkpoints = (
            (candidate.common_prefix_tokens,)
            if candidate.common_prefix_tokens > 0
            else ()
        )
        restore = candidate.common_prefix_tokens
    checkpoint_loss = candidate.common_prefix_tokens - restore
    suffix = event.prompt_tokens - candidate.common_prefix_tokens
    return _Selection(
        candidate=candidate,
        checkpoint_tokens=checkpoints,
        checkpoint_loss_tokens=checkpoint_loss,
        target_suffix_tokens=suffix,
        work_tokens=checkpoint_loss + suffix,
        requested_budget=requested_budget,
        effective_budget=effective_budget,
        planned_budget=planned_budget,
        rejection_counts={},
    )


def _is_completed_ram_bank_hit(event: _Event) -> bool:
    """Whether the producer says this completed request restored SessionBank RAM."""

    return (
        event.terminal_status == "completed"
        and event.bank_consulted
        and event.cache_case == "hit"
        and event.cache_source == "ram"
    )


def _selected_candidate(event: _Event) -> _Candidate | None:
    if event.selected_entry_ordinal is None:
        return None
    return next(
        (
            candidate
            for candidate in event.candidates
            if candidate.entry_ordinal == event.selected_entry_ordinal
        ),
        None,
    )


def _verified_incumbent_candidate(event: _Event) -> _Candidate | None:
    """Return the observed RAM restore only when its token accounting proves it.

    The candidate list is captured before runtime selection.  It is therefore
    useful for a proposed placement, but it cannot identify the incumbent by
    itself.  A completed RAM row joins the comparison only when the producer
    also names one candidate and its reported cached-token count equals the
    only restorable point for that candidate.
    """

    if not _is_completed_ram_bank_hit(event):
        return None
    candidate = _selected_candidate(event)
    if candidate is None or not candidate.structurally_restorable:
        return None
    expected = _expected_incumbent_restore(candidate)
    if expected <= 0 or event.outcome_cached_tokens != expected:
        return None
    return candidate


def _eligible_candidates(event: _Event) -> tuple[tuple[_Candidate, ...], Counter[str]]:
    rejected: Counter[str] = Counter()
    if not _is_completed_ram_bank_hit(event):
        if event.candidates:
            if event.terminal_status != "completed":
                reason = f"terminal_status_{event.terminal_status}"
            elif event.cache_source != "ram":
                reason = f"cache_source_{event.cache_source}"
            else:
                reason = f"cache_case_{event.cache_case}"
            rejected[reason] = len(event.candidates)
        return (), rejected
    if _verified_incumbent_candidate(event) is None:
        if event.candidates:
            rejected["unverified_incumbent_provenance"] = len(event.candidates)
        return (), rejected
    eligible: list[_Candidate] = []
    for candidate in event.candidates:
        if not candidate.structurally_restorable:
            rejected["not_structurally_restorable"] += 1
            continue
        eligible.append(candidate)
    return tuple(eligible), rejected


def _future_known_restore_position(candidate: _Candidate) -> int | None:
    """Return the best actual boundary a future recurrent placement can use.

    The endpoint is an already-retained state and is free when the request
    shares the entire stored prefix.  Every interior choice must be one of the
    producer's successful capture positions; the oracle may rearrange a
    checkpoint budget, but may not synthesize a snapshot at the observed
    prefix.
    """

    if not candidate.has_recurrent:
        return None
    if candidate.common_prefix_tokens == candidate.stored_prefix_tokens:
        return candidate.stored_prefix_tokens
    if candidate.incumbent_interior_budget <= 0:
        return None
    return max(
        (
            position
            for position in candidate.capture_candidate_tokens
            if position < candidate.stored_prefix_tokens
            and position <= candidate.common_prefix_tokens
        ),
        default=None,
    )


def _eligible_future_known_upper_candidates(
    event: _Event,
) -> tuple[tuple[_Candidate, ...], Counter[str]]:
    """Return RAM candidates available to the optimistic capture-bound oracle.

    Normal policies need the incumbent's currently retained recurrent state to
    be structurally restorable.  A future policy can instead retain a recorded
    capture boundary, so a currently non-restorable *recurrent* candidate is
    admissible only when its recorded capture list proves an exact usable
    boundary (or the free endpoint).  The actual selected RAM restore remains
    provenance-verified before any counterfactual candidate may participate.
    """

    rejected: Counter[str] = Counter()
    if not _is_completed_ram_bank_hit(event):
        if event.candidates:
            if event.terminal_status != "completed":
                reason = f"terminal_status_{event.terminal_status}"
            elif event.cache_source != "ram":
                reason = f"cache_source_{event.cache_source}"
            else:
                reason = f"cache_case_{event.cache_case}"
            rejected[reason] = len(event.candidates)
        return (), rejected
    if _verified_incumbent_candidate(event) is None:
        if event.candidates:
            rejected["unverified_incumbent_provenance"] = len(event.candidates)
        return (), rejected

    eligible: list[_Candidate] = []
    for candidate in event.candidates:
        if not candidate.has_recurrent:
            if not candidate.structurally_restorable:
                rejected["not_structurally_restorable"] += 1
                continue
        elif not candidate.structurally_restorable and (
            _future_known_restore_position(candidate) is None
        ):
            rejected["no_captured_restore_at_common_prefix"] += 1
            continue
        eligible.append(candidate)
    return tuple(eligible), rejected


def _select_from_candidates(
    event: _Event,
    candidates: Sequence[_Candidate],
    rejected: Counter[str],
    *,
    placement: Callable[[_Candidate], tuple[Sequence[int], int, int, int] | str | None],
) -> _Selection:
    best: _Selection | None = None
    for candidate in candidates:
        planned = placement(candidate)
        if isinstance(planned, str):
            rejected[planned] += 1
            continue
        if planned is None:
            rejected["unavailable"] += 1
            continue
        positions, requested, effective, planned_budget = planned
        selection = _selection_for_candidate(
            event,
            candidate,
            positions,
            requested_budget=requested,
            effective_budget=effective,
            planned_budget=planned_budget,
        )
        if best is None or (
            selection.work_tokens,
            selection.checkpoint_loss_tokens,
            -_restore_position(
                selection.checkpoint_tokens, candidate.common_prefix_tokens
            ),
            candidate.entry_ordinal,
            candidate.index,
        ) < (
            best.work_tokens,
            best.checkpoint_loss_tokens,
            -_restore_position(
                best.checkpoint_tokens,
                best.candidate.common_prefix_tokens
                if best.candidate is not None
                else 0,
            ),
            best.candidate.entry_ordinal if best.candidate is not None else math.inf,
            best.candidate.index if best.candidate is not None else math.inf,
        ):
            best = selection
    return best if best is not None else _fallback(event, rejected)


def _select(
    event: _Event,
    *,
    placement: Callable[[_Candidate], tuple[Sequence[int], int, int, int] | str | None],
) -> _Selection:
    candidates, rejected = _eligible_candidates(event)
    return _select_from_candidates(event, candidates, rejected, placement=placement)


def _select_future_known_upper(event: _Event) -> _Selection:
    candidates, rejected = _eligible_future_known_upper_candidates(event)
    return _select_from_candidates(
        event, candidates, rejected, placement=_placement_future_known_upper
    )


def _select_incumbent(event: _Event) -> _Selection:
    """Use the actual, provenance-verified RAM restore as the baseline.

    Candidate-list policies may choose among compatible entries.  The incumbent
    cannot: it is the entry that the producer records as having been restored.
    Otherwise an unobserved candidate could silently replace the baseline in a
    supposedly paired comparison.
    """

    candidate = _verified_incumbent_candidate(event)
    if candidate is None:
        return _fallback(event, {"unverified_or_non_ram_incumbent": 1})
    interior = candidate.incumbent_interior_positions if candidate.has_recurrent else ()
    return _selection_for_candidate(
        event,
        candidate,
        interior,
        requested_budget=candidate.incumbent_interior_budget
        if candidate.has_recurrent
        else 0,
        effective_budget=len(interior),
        planned_budget=len(interior),
    )


def _fitted_positions_on_capture_candidates(
    candidate: _Candidate,
    history: Sequence[int],
    *,
    decay: float,
) -> tuple[int, ...]:
    """Fit a capture-candidate counterfactual, preserving tail-folding.

    The legal universe is *only* successful ``capture_candidate_tokens`` below
    the stored endpoint.  Those positions are not a retained-payload contract:
    geometric retention can have discarded their snapshot payloads before this
    later-history fit exists.  The result is therefore diagnostic only, not a
    runtime placement.  Its last position replaces ``L`` in the module-level
    constant-offset proof: observations at or above that position carry the
    same placement-independent tail offset.
    """
    legal = tuple(
        position
        for position in candidate.capture_candidate_tokens
        if position < candidate.stored_prefix_tokens
    )
    if candidate.incumbent_interior_budget <= 0 or not legal:
        return ()
    last_legal = legal[-1]
    weights = [0.0] * last_legal
    total_observations = len(history)
    for index, observed_overlap in enumerate(history):
        overlap = min(observed_overlap, candidate.stored_prefix_tokens)
        if overlap <= 0 or overlap == candidate.stored_prefix_tokens:
            continue
        transformed_depth = min(overlap, last_legal)
        weights[transformed_depth - 1] += decay ** (total_observations - index - 1)
    if not any(weights):
        return ()
    return optimal_checkpoint_positions_from_candidates(
        weights, candidate.incumbent_interior_budget, legal
    )


def _candidate_identity(candidate: _Candidate) -> _CandidateIdentity:
    """Return the immutable placement/restore contract for one entry revision."""

    return (
        candidate.stored_prefix_tokens,
        candidate.retained_checkpoint_tokens,
        candidate.capture_candidate_tokens,
        candidate.incumbent_interior_budget,
        candidate.has_recurrent,
    )


def _validate_candidate_identities(
    events: Sequence[_Event],
) -> dict[tuple[int, int], _CandidateIdentity]:
    """Reject a trace that reuses an entry revision with changed placement state.

    ``structurally_restorable`` is intentionally not part of the identity.  The
    producer derives it from the fixed cached-payload state *and* the current
    request's common prefix, so one unchanged recurrent entry can correctly be
    restorable for one event and not for another.
    """

    identities: dict[tuple[int, int], _CandidateIdentity] = {}
    for event in events:
        for candidate in event.candidates:
            key = (event.bank_epoch, candidate.entry_ordinal)
            descriptor = _candidate_identity(candidate)
            previous = identities.setdefault(key, descriptor)
            if previous != descriptor:
                raise ValueError(
                    "(bank_epoch, entry_ordinal) must retain its complete "
                    "placement and restore contract across the trace"
                )
    return identities


def _fitted_history(events: Sequence[_Event]) -> dict[tuple[int, int], tuple[int, ...]]:
    values: dict[tuple[int, int], list[int]] = defaultdict(list)
    for event in events:
        if _verified_incumbent_candidate(event) is None:
            continue
        for candidate in event.candidates:
            if not candidate.structurally_restorable:
                continue
            key = (event.bank_epoch, candidate.entry_ordinal)
            values[key].append(candidate.common_prefix_tokens)
    return {key: tuple(overlaps) for key, overlaps in values.items()}


def _placement_incumbent(candidate: _Candidate) -> tuple[Sequence[int], int, int, int]:
    if not candidate.has_recurrent:
        return (), 0, 0, 0
    interior = candidate.incumbent_interior_positions
    return interior, candidate.incumbent_interior_budget, len(interior), len(interior)


def _placement_prompt_end(candidate: _Candidate) -> tuple[Sequence[int], int, int, int]:
    return (
        (),
        candidate.incumbent_interior_budget if candidate.has_recurrent else 0,
        0,
        0,
    )


def _placement_balanced(
    candidate: _Candidate, block_size: int
) -> tuple[Sequence[int], int, int, int]:
    if not candidate.has_recurrent:
        return (), 0, 0, 0
    last_interior = _last_legal_interior(candidate.stored_prefix_tokens, block_size)
    positions = (
        balanced_checkpoint_positions(
            last_interior, candidate.incumbent_interior_budget, block_size
        )
        if last_interior > 0
        else ()
    )
    return (
        positions,
        candidate.incumbent_interior_budget,
        len(positions),
        len(positions),
    )


def _placement_geometric(
    candidate: _Candidate, block_size: int
) -> tuple[Sequence[int], int, int, int]:
    if not candidate.has_recurrent:
        return (), 0, 0, 0
    last_interior = _last_legal_interior(candidate.stored_prefix_tokens, block_size)
    positions = (
        geometric_tail_checkpoint_positions(
            last_interior, candidate.incumbent_interior_budget, block_size
        )
        if last_interior > 0
        else ()
    )
    return (
        positions,
        candidate.incumbent_interior_budget,
        len(positions),
        len(positions),
    )


def _placement_future_known_upper(
    candidate: _Candidate,
) -> tuple[Sequence[int], int, int, int] | str:
    """Return the same-budget future-known token-work upper bound.

    The endpoint is free.  For recurrent caches, every interior checkpoint
    must be a successful capture boundary present in the trace; the best such
    boundary at or below the event's common prefix minimizes recompute work.
    This is at least as good as every observed capture-candidate placement for
    the same candidate, while never claiming a deployable policy.
    """
    requested = candidate.incumbent_interior_budget if candidate.has_recurrent else 0
    if not candidate.has_recurrent:
        return (), requested, 0, 0
    restore = _future_known_restore_position(candidate)
    if restore is None:
        return "no_captured_restore_at_common_prefix"
    if restore == candidate.stored_prefix_tokens:
        return (), requested, 0, 0
    return (restore,), requested, 1, 1


def _selection_view(selection: _Selection) -> dict[str, object]:
    return {
        "covered": selection.covered,
        "candidate": _candidate_view(selection.candidate)
        if selection.candidate
        else None,
        "checkpoint_tokens": list(selection.checkpoint_tokens),
        "checkpoint_loss_tokens": selection.checkpoint_loss_tokens,
        "target_suffix_tokens": selection.target_suffix_tokens,
        "work_tokens": selection.work_tokens,
        "requested_budget": selection.requested_budget,
        "effective_budget": selection.effective_budget,
        "planned_budget": selection.planned_budget,
        "rejection_counts": dict(sorted(selection.rejection_counts.items())),
    }


def _quantiles(values: Sequence[int]) -> dict[str, float | int]:
    if not values:
        return {"total": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        if len(ordered) == 1:
            return float(ordered[0])
        rank = fraction * (len(ordered) - 1)
        lower = math.floor(rank)
        upper = math.ceil(rank)
        if lower == upper:
            return float(ordered[lower])
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)

    return {
        "total": sum(values),
        "mean": math.fsum(values) / len(values),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
    }


def _gain_percent(baseline: int, comparison: int) -> float:
    return 0.0 if baseline <= 0 else 100.0 * (baseline - comparison) / baseline


def _summary(
    rows: Sequence[Mapping[str, _Selection]], policy: str
) -> dict[str, object]:
    selections = [row[policy] for row in rows]
    work = [selection.work_tokens for selection in selections]
    loss = [selection.checkpoint_loss_tokens for selection in selections]
    suffix = [selection.target_suffix_tokens for selection in selections]
    covered = [selection for selection in selections if selection.covered]
    requested = sum(selection.requested_budget for selection in covered)
    effective = sum(selection.effective_budget for selection in covered)
    planned = sum(selection.planned_budget for selection in covered)
    return {
        "recompute_work_tokens": _quantiles(work),
        "checkpoint_loss_tokens": _quantiles(loss),
        "target_suffix_tokens": _quantiles(suffix),
        "coverage": {
            "covered_events": len(covered),
            "total_events": len(selections),
            "rate": 0.0 if not selections else len(covered) / len(selections),
        },
        "budget_utilization": {
            "requested_interior_checkpoints": requested,
            "effective_interior_checkpoints": effective,
            "planned_interior_checkpoints": planned,
            "planned_over_effective": 0.0 if effective == 0 else planned / effective,
        },
    }


def _conditional_gain(
    rows: Sequence[Mapping[str, _Selection]], policy: str
) -> dict[str, object]:
    selected = [
        row
        for row in rows
        if row[policy].candidate is not None
        and row[policy].candidate.common_prefix_tokens
        < row[policy].candidate.stored_prefix_tokens
    ]
    incumbent_work = sum(row["incumbent"].work_tokens for row in selected)
    policy_work = sum(row[policy].work_tokens for row in selected)
    return {
        "event_count": len(selected),
        "gain_vs_incumbent_percent": _gain_percent(incumbent_work, policy_work),
    }


def _bucket_for(divergence: int) -> str | None:
    for name, lower, upper in _BUCKETS:
        if divergence >= lower and (upper is None or divergence <= upper):
            return name
    return None


def _bootstrap_once(
    pairs: Sequence[tuple[int, int]],
    *,
    iterations: int,
    block_length: int,
    seed: int,
) -> tuple[float, float]:
    """Return paired circular-moving-block 5th/95th percentile gains."""
    size = len(pairs)
    if size == 0:
        return 0.0, 0.0
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(iterations):
        remaining = size
        baseline = 0
        comparison = 0
        while remaining:
            start = rng.randrange(size)
            take = min(block_length, remaining)
            for offset in range(take):
                left, right = pairs[(start + offset) % size]
                baseline += left
                comparison += right
            remaining -= take
        samples.append(_gain_percent(baseline, comparison))
    samples.sort()
    return (
        _percentile(samples, _BOOTSTRAP_LOWER_PERCENTILE),
        _percentile(samples, _BOOTSTRAP_UPPER_PERCENTILE),
    )


def _percentile(values: Sequence[float], fraction: float) -> float:
    if len(values) == 1:
        return values[0]
    rank = fraction * (len(values) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (rank - lower)


def _bootstrap(
    rows: Sequence[Mapping[str, _Selection]],
    *,
    policy: str,
    iterations: int,
    block_length: int,
    seed: int,
) -> dict[str, object]:
    pairs = [(row["incumbent"].work_tokens, row[policy].work_tokens) for row in rows]
    point = _gain_percent(
        sum(left for left, _ in pairs), sum(right for _, right in pairs)
    )
    lengths = (block_length, block_length * 2)
    bounds = [
        _bootstrap_once(pairs, iterations=iterations, block_length=length, seed=seed)
        for length in lengths
    ]
    return {
        "point_gain_vs_incumbent_percent": point,
        "lower95_percent": min(lower for lower, _ in bounds),
        "upper95_percent": max(upper for _, upper in bounds),
        "dependence_runs": 2,
        "replicate_bounds_percent": [
            {
                "seed": seed,
                "block_length": length,
                "lower95_percent": lower,
                "upper95_percent": upper,
            }
            for length, (lower, upper) in zip(lengths, bounds, strict=True)
        ],
        "base_block_length": block_length,
        "block_lengths": list(lengths),
        "iterations_per_replicate": iterations,
        "comparison": "paired incumbent token-work minus policy token-work",
    }


def _expected_incumbent_restore(candidate: _Candidate) -> int:
    """Return the only cached-token count that proves the selected restore."""
    if not candidate.has_recurrent:
        return candidate.common_prefix_tokens
    return _restore_position(
        _with_free_endpoint(
            candidate.incumbent_interior_positions, candidate.stored_prefix_tokens
        ),
        candidate.common_prefix_tokens,
    )


def _incumbent_sanity(events: Sequence[_Event]) -> dict[str, int]:
    """Classify incumbent provenance for completed RAM hits only.

    The trace records RAM candidates before selection, so a RAM hit with no
    selected ordinal is an unverified row rather than evidence for a favorable
    incumbent baseline. SSD and terminal outcomes are deliberately excluded:
    they stay workload-wide zero-benefit rows, but cannot prove RAM placement.
    """
    result = {
        "eligible_ram_hits": 0,
        "unambiguous": 0,
        "verified": 0,
        "matches": 0,
        "mismatches": 0,
        "not_reported": 0,
        "impossible": 0,
    }
    for event in events:
        if not _is_completed_ram_bank_hit(event):
            continue
        result["eligible_ram_hits"] += 1
        candidate = _selected_candidate(event)
        if candidate is None:
            result["not_reported"] += 1
            continue
        if not candidate.structurally_restorable:
            result["impossible"] += 1
            continue
        expected = _expected_incumbent_restore(candidate)
        result["unambiguous"] += 1
        if expected <= 0:
            result["impossible"] += 1
        elif event.outcome_cached_tokens == expected:
            result["matches"] += 1
            result["verified"] += 1
        else:
            result["mismatches"] += 1
    return result


def _status(
    *,
    dropped: int,
    truncated_candidate_events: int,
    incumbent_sanity: Mapping[str, int],
    eligible: int,
    min_events: int,
    bucket_counts: Mapping[str, int],
    min_bucket_events: int,
    observed_bootstrap: Mapping[str, object] | None,
    optimistic_bootstrap: Mapping[str, object] | None,
) -> tuple[str, list[str]]:
    failures: list[str] = []
    if dropped != 0:
        failures.append("trace events_dropped must equal zero")
    if truncated_candidate_events != 0:
        failures.append(
            f"recorded candidates are truncated on {truncated_candidate_events} events"
        )
    if incumbent_sanity["mismatches"] != 0:
        failures.append(
            "incumbent provenance has "
            f"{incumbent_sanity['mismatches']} unambiguous cached-token mismatches"
        )
    if incumbent_sanity["not_reported"] != 0:
        failures.append(
            "incumbent provenance has "
            f"{incumbent_sanity['not_reported']} eligible RAM hits without a "
            "selected incumbent"
        )
    if incumbent_sanity["impossible"] != 0:
        failures.append(
            "incumbent provenance has "
            f"{incumbent_sanity['impossible']} impossible eligible RAM-hit selections"
        )
    if incumbent_sanity["verified"] != incumbent_sanity["eligible_ram_hits"]:
        failures.append(
            "every completed RAM-hit row in the full trace must have verified "
            "incumbent provenance"
        )
    if incumbent_sanity["verified"] < min_events:
        failures.append(
            "verified incumbent observations "
            f"{incumbent_sanity['verified']} < {min_events}"
        )
    if eligible < min_events:
        failures.append(f"eligible evaluation events {eligible} < {min_events}")
    populated = sum(count >= min_bucket_events for count in bucket_counts.values())
    if populated < _MIN_POPULATED_NONAPPEND_BUCKETS:
        failures.append(
            "only "
            f"{populated} nonappend buckets have at least {min_bucket_events} eligible events"
        )
    if failures:
        return "insufficient_data", failures
    assert observed_bootstrap is not None and optimistic_bootstrap is not None
    optimistic_point = float(optimistic_bootstrap["point_gain_vs_incumbent_percent"])
    optimistic_upper = float(optimistic_bootstrap["upper95_percent"])
    observed_point = float(observed_bootstrap["point_gain_vs_incumbent_percent"])
    observed_lower = float(observed_bootstrap["lower95_percent"])
    if (
        optimistic_point < _CPU_FALSIFIER_THRESHOLD_PERCENT
        or optimistic_upper < _CPU_FALSIFIER_THRESHOLD_PERCENT
    ):
        return "no_go", ["future-known upper bound does not clear 10%"]
    if observed_point < _CPU_FALSIFIER_THRESHOLD_PERCENT:
        return "no_go", [
            "capture-candidate counterfactual point gain does not clear 10%"
        ]
    if observed_lower < _CPU_FALSIFIER_THRESHOLD_PERCENT:
        return "amber", ["capture-candidate counterfactual lower95 gain is below 10%"]
    return "counterfactual_only", [
        "capture-candidate counterfactual lower95 gain clears 10%, but selected "
        "snapshot payloads may be unavailable; no deployable GO gate exists"
    ]


def _canonical_gate(
    *,
    block_size: int,
    decay: float,
    min_events: int,
    min_bucket_events: int,
    bootstrap_iterations: int,
    bootstrap_block_length: int | None,
    seed: int,
) -> bool:
    """Whether these are exactly the preregistered CPU-falsifier inputs.

    The split fraction and decision threshold are intentionally not public
    overrides. They are still compared here so a code change cannot quietly
    mint the canonical label with a different fixed value.
    """

    return (
        block_size == _CANONICAL_BLOCK_SIZE
        and decay == _CANONICAL_DECAY
        and min_events == _CANONICAL_MIN_EVENTS
        and min_bucket_events == _CANONICAL_MIN_BUCKET_EVENTS
        and bootstrap_iterations == _CANONICAL_BOOTSTRAP_ITERATIONS
        and bootstrap_block_length is None
        and seed == _CANONICAL_SEED
        and _HISTORY_FRACTION == _CANONICAL_HISTORY_FRACTION
        and _CPU_FALSIFIER_THRESHOLD_PERCENT == _CANONICAL_DECISION_THRESHOLD_PERCENT
    )


def _gate_config(
    *,
    block_size: int,
    decay: float,
    min_events: int,
    min_bucket_events: int,
    bootstrap_iterations: int,
    bootstrap_block_length: int | None,
    seed: int,
    fit_window_event_count: int,
    evaluation_window_event_count: int,
    completed_evaluation_event_count: int,
    bootstrap_event_count: int,
    derived_bootstrap_base_block_length: int,
) -> dict[str, object]:
    """Return the complete, content-free configuration provenance."""

    return {
        "effective_inputs": {
            "block_size": block_size,
            "decay": decay,
            "min_events": min_events,
            "min_bucket_events": min_bucket_events,
            "bootstrap_iterations": bootstrap_iterations,
            "bootstrap_block_length": bootstrap_block_length,
            "seed": seed,
            "chronological_fit_fraction": _HISTORY_FRACTION,
            "decision_threshold_percent": _CANONICAL_DECISION_THRESHOLD_PERCENT,
        },
        "derived": {
            "minimum_history_observations": _MIN_HISTORY_OBSERVATIONS,
            "minimum_populated_nonappend_buckets": _MIN_POPULATED_NONAPPEND_BUCKETS,
            "bootstrap_lower_percentile": _BOOTSTRAP_LOWER_PERCENTILE,
            "bootstrap_upper_percentile": _BOOTSTRAP_UPPER_PERCENTILE,
            "fit_window_event_count": fit_window_event_count,
            "evaluation_window_event_count": evaluation_window_event_count,
            "completed_evaluation_event_count": completed_evaluation_event_count,
            "workload_evaluation_event_count": bootstrap_event_count,
            "bootstrap_event_count": bootstrap_event_count,
            "bootstrap_base_block_length": derived_bootstrap_base_block_length,
            "bootstrap_block_lengths": [
                derived_bootstrap_base_block_length,
                derived_bootstrap_base_block_length * 2,
            ],
            "bootstrap_block_length_rule": (
                "ceil(workload_evaluation_event_count ** (1 / 3))"
                if bootstrap_block_length is None
                else "explicit bootstrap_block_length"
            ),
            "bootstrap_dependence_runs": 2,
        },
        "canonical_defaults": {
            "block_size": _CANONICAL_BLOCK_SIZE,
            "decay": _CANONICAL_DECAY,
            "min_events": _CANONICAL_MIN_EVENTS,
            "min_bucket_events": _CANONICAL_MIN_BUCKET_EVENTS,
            "bootstrap_iterations": _CANONICAL_BOOTSTRAP_ITERATIONS,
            "bootstrap_block_length": None,
            "seed": _CANONICAL_SEED,
            "chronological_fit_fraction": _CANONICAL_HISTORY_FRACTION,
            "decision_threshold_percent": _CPU_FALSIFIER_THRESHOLD_PERCENT,
        },
    }


def replay_checkpoint_trace(
    payload: Mapping[str, Any],
    *,
    block_size: int = _CANONICAL_BLOCK_SIZE,
    decay: float = _CANONICAL_DECAY,
    min_events: int = _CANONICAL_MIN_EVENTS,
    min_bucket_events: int = _CANONICAL_MIN_BUCKET_EVENTS,
    bootstrap_iterations: int = _CANONICAL_BOOTSTRAP_ITERATIONS,
    bootstrap_block_length: int | None = None,
    seed: int = 0,
) -> dict[str, object]:
    """Replay a schema-v2 SessionBank snapshot without MLX or model execution.

    The exact preregistered defaults retain their canonical provenance.
    Overrides return ``exploratory_configuration`` with their numeric outcome
    recorded separately.  The fitted capture-candidate calculation can only
    return ``counterfactual_only`` when its lower bound clears; this CPU replay
    has no deployable GO status and ``runtime_go`` remains false.
    """
    block = _integer(block_size, "block_size", minimum=1)
    forgetting = _real(decay, "decay", minimum=0.0)
    if forgetting <= 0.0 or forgetting > 1.0:
        raise ValueError("decay must be in (0, 1]")
    minimum_events = _integer(min_events, "min_events", minimum=1)
    minimum_bucket = _integer(min_bucket_events, "min_bucket_events", minimum=1)
    iterations = _integer(
        bootstrap_iterations, "bootstrap_iterations", minimum=_MIN_BOOTSTRAP_ITERATIONS
    )
    bootstrap_length = (
        None
        if bootstrap_block_length is None
        else _integer(bootstrap_block_length, "bootstrap_block_length", minimum=1)
    )
    random_seed = _integer(seed, "seed")
    canonical_gate = _canonical_gate(
        block_size=block,
        decay=forgetting,
        min_events=minimum_events,
        min_bucket_events=minimum_bucket,
        bootstrap_iterations=iterations,
        bootstrap_block_length=bootstrap_length,
        seed=random_seed,
    )
    events, trace_contract = _parse_payload(payload)
    entry_identities = _validate_candidate_identities(events)
    dropped = trace_contract.events_dropped
    fit_count = (
        len(events) * _CANONICAL_HISTORY_NUMERATOR // _CANONICAL_HISTORY_DENOMINATOR
    )
    fit_window = events[:fit_count]
    evaluation_window = events[fit_count:]
    fit_completed = [
        event for event in fit_window if event.terminal_status == "completed"
    ]
    evaluation_completed = [
        event for event in evaluation_window if event.terminal_status == "completed"
    ]
    histories = _fitted_history(fit_completed)
    truncated_candidate_events = sum(
        event.compatible_entry_count > len(event.candidates) for event in events
    )
    terminal_zero_benefit_rows = [
        {
            "sequence": event.sequence,
            "bank_epoch": event.bank_epoch,
            "partition": partition,
            "terminal_status": event.terminal_status,
            "cache_case": event.cache_case,
            "reason": "terminal request is a workload-wide zero-benefit row",
        }
        for partition, window in (
            ("fit", fit_window),
            ("evaluation", evaluation_window),
        )
        for event in window
        if event.terminal_status != "completed"
    ]

    selections_by_event: list[dict[str, _Selection]] = []
    event_views: list[dict[str, object]] = []
    for event in evaluation_window:

        def fitted(
            candidate: _Candidate,
        ) -> tuple[Sequence[int], int, int, int] | str | None:
            if not candidate.has_recurrent:
                return (), 0, 0, 0
            history_key = (event.bank_epoch, candidate.entry_ordinal)
            history = histories.get(history_key, ())
            if len(history) < _MIN_HISTORY_OBSERVATIONS:
                return "insufficient_fit_history"
            if entry_identities[history_key] != _candidate_identity(candidate):
                return "fit_identity_changed"
            captured_interiors = tuple(
                position
                for position in candidate.capture_candidate_tokens
                if position < candidate.stored_prefix_tokens
            )
            if not captured_interiors:
                return "missing_capture_candidate_tokens"
            positions = _fitted_positions_on_capture_candidates(
                candidate, history, decay=forgetting
            )
            effective = min(
                candidate.incumbent_interior_budget, len(captured_interiors)
            )
            return (
                positions,
                candidate.incumbent_interior_budget,
                effective,
                len(positions),
            )

        policies = {
            "prompt_end_only": _select(event, placement=_placement_prompt_end),
            "incumbent": _select_incumbent(event),
            "observed_fitted": _select(event, placement=fitted),
            "balanced": _select(
                event,
                placement=lambda candidate: _placement_balanced(candidate, block),
            ),
            "geometric_tail": _select(
                event,
                placement=lambda candidate: _placement_geometric(candidate, block),
            ),
            "future_known_upper": _select_future_known_upper(event),
        }
        selections_by_event.append(policies)
        deltas = {
            policy: policies["incumbent"].work_tokens - selection.work_tokens
            for policy, selection in policies.items()
            if policy != "incumbent"
        }
        event_views.append(
            {
                "sequence": event.sequence,
                "bank_epoch": event.bank_epoch,
                "prompt_tokens": event.prompt_tokens,
                "cache_case": event.cache_case,
                "cache_source": event.cache_source,
                "lane": event.lane,
                "terminal_status": event.terminal_status,
                "compatible_entry_count": event.compatible_entry_count,
                "recorded_candidate_count": len(event.candidates),
                "token_work_included": True,
                "zero_benefit_reason": (
                    "terminal request"
                    if event.terminal_status != "completed"
                    else (
                        "not a RAM SessionBank hit"
                        if event.cache_source != "ram"
                        else None
                    )
                ),
                "policies": {
                    policy: _selection_view(selection)
                    for policy, selection in policies.items()
                },
                "paired_delta_work_tokens_vs_incumbent": deltas,
            }
        )

    bucket_counts: Counter[str] = Counter()
    eligible_rows: list[dict[str, _Selection]] = []
    for event, row in zip(evaluation_window, selections_by_event, strict=True):
        selected = row["observed_fitted"]
        candidate = selected.candidate
        if candidate is None:
            continue
        eligible_rows.append(row)
        divergence = candidate.stored_prefix_tokens - candidate.common_prefix_tokens
        if divergence > 0 and (bucket := _bucket_for(divergence)) is not None:
            bucket_counts[bucket] += 1
    complete_bucket_counts = {name: bucket_counts[name] for name, _, _ in _BUCKETS}
    bootstrap_event_count = len(selections_by_event)
    block_length = bootstrap_length or max(
        1, math.ceil(bootstrap_event_count ** (1.0 / 3.0))
    )
    incumbent_sanity = _incumbent_sanity(events)
    evaluation_incumbent_sanity = _incumbent_sanity(evaluation_window)
    if (
        dropped == 0
        and truncated_candidate_events == 0
        and incumbent_sanity["verified"] == incumbent_sanity["eligible_ram_hits"]
        and evaluation_incumbent_sanity["verified"] >= minimum_events
        and len(eligible_rows) >= minimum_events
        and sum(count >= minimum_bucket for count in complete_bucket_counts.values())
        >= _MIN_POPULATED_NONAPPEND_BUCKETS
    ):
        observed_bootstrap: dict[str, object] | None = _bootstrap(
            selections_by_event,
            policy="observed_fitted",
            iterations=iterations,
            block_length=block_length,
            seed=random_seed,
        )
        optimistic_bootstrap: dict[str, object] | None = _bootstrap(
            selections_by_event,
            policy="future_known_upper",
            iterations=iterations,
            block_length=block_length,
            seed=random_seed,
        )
    else:
        observed_bootstrap = None
        optimistic_bootstrap = None
    numeric_status, numeric_reasons = _status(
        dropped=dropped,
        truncated_candidate_events=truncated_candidate_events,
        incumbent_sanity=incumbent_sanity,
        eligible=len(eligible_rows),
        min_events=minimum_events,
        bucket_counts=complete_bucket_counts,
        min_bucket_events=minimum_bucket,
        observed_bootstrap=observed_bootstrap,
        optimistic_bootstrap=optimistic_bootstrap,
    )
    if canonical_gate:
        status = numeric_status
        reasons = numeric_reasons
        exploratory_assessment: dict[str, object] | None = None
    else:
        numeric_outcome = numeric_status
        status = "exploratory_configuration"
        reasons = ["noncanonical configuration cannot emit a deployable GO status"]
        reasons.extend(numeric_reasons)
        exploratory_assessment = {
            "numeric_outcome": numeric_outcome,
            "numeric_reasons": numeric_reasons,
        }
    cache_case_counts = Counter(event.cache_case for event in evaluation_window)
    cache_source_counts = Counter(event.cache_source for event in evaluation_window)
    terminal_status_counts = Counter(event.terminal_status for event in events)
    summaries = {policy: _summary(selections_by_event, policy) for policy in _POLICIES}
    for policy, summary in summaries.items():
        work_total = int(summary["recompute_work_tokens"]["total"])  # type: ignore[index]
        incumbent_total = int(summaries["incumbent"]["recompute_work_tokens"]["total"])  # type: ignore[index]
        prompt_total = int(
            summaries["prompt_end_only"]["recompute_work_tokens"]["total"]
        )  # type: ignore[index]
        summary["workload_wide_gain_vs_incumbent_percent"] = _gain_percent(
            incumbent_total, work_total
        )
        summary["workload_wide_gain_vs_prompt_end_only_percent"] = _gain_percent(
            prompt_total, work_total
        )
        summary["conditional_nonappend_gain"] = _conditional_gain(
            selections_by_event, policy
        )
        summary["cache_hit_rate"] = (
            0.0
            if not evaluation_window
            else cache_case_counts["hit"] / len(evaluation_window)
        )
    return {
        "analysis_schema_version": 2,
        "status": status,
        "canonical_gate": canonical_gate,
        "runtime_go": False,
        "deployable_go_available": False,
        "status_reasons": reasons,
        "exploratory_assessment": exploratory_assessment,
        "gate_config": _gate_config(
            block_size=block,
            decay=forgetting,
            min_events=minimum_events,
            min_bucket_events=minimum_bucket,
            bootstrap_iterations=iterations,
            bootstrap_block_length=bootstrap_length,
            seed=random_seed,
            fit_window_event_count=len(fit_window),
            evaluation_window_event_count=len(evaluation_window),
            completed_evaluation_event_count=len(evaluation_completed),
            bootstrap_event_count=bootstrap_event_count,
            derived_bootstrap_base_block_length=block_length,
        ),
        "labels": {
            "incumbent": (
                "actual selected RAM restore, using only retained checkpoint payloads; "
                "recorded baseline rather than an alternative policy"
            ),
            "observed_fitted": (
                "capture-candidate counterfactual; later-history fitting may choose "
                "successful capture positions whose snapshot payloads were geometrically "
                "discarded, so those payloads are unavailable and this is not deployable"
            ),
            "future_known_upper": (
                "same-budget oracle over RAM candidates, limited to recorded capture "
                "boundaries at or below the event's known common prefix; optimistic "
                "no-go falsifier only"
            ),
        },
        "capture_candidate_counterfactual": {
            "policy": "observed_fitted",
            "payload_availability": (
                "unavailable by contract: selected capture positions need not remain "
                "among retained checkpoint payloads"
            ),
            "may_falsify": True,
            "may_authorize_go": False,
            "future_go_prerequisite": (
                "implement policy-before-capture, bounded retention, or measured "
                "rematerialization before a deployable GO gate can exist"
            ),
        },
        "caveats": [
            "Optimistic same-trace token-work replay, not M5 timing evidence.",
            "The schema cannot infer cache admission, eviction, or LRU behavior.",
            "Cache misses and bypasses contribute incumbent-equal zero benefit to the workload bootstrap.",
            "The fitted capture-candidate arm is non-deployable because its selected "
            "snapshot payloads may have been discarded by geometric retention.",
        ],
        "trace": {
            "input_schema_version": _SCHEMA_VERSION,
            "events_total": len(events),
            "enabled": True,
            "max_events": trace_contract.max_events,
            "events_collected": trace_contract.events_collected,
            "events_dropped": dropped,
            "pending_probe_count": trace_contract.pending_probe_count,
            "sequence_high_watermark": trace_contract.sequence_high_watermark,
            "bank_epoch": trace_contract.bank_epoch,
            "truncated_candidate_event_count": truncated_candidate_events,
            "fit_window_event_count": len(fit_window),
            "evaluation_window_event_count": len(evaluation_window),
            "fit_completed_event_count": len(fit_completed),
            "evaluation_completed_event_count": len(evaluation_completed),
            "eligible_evaluation_event_count": len(eligible_rows),
            "evaluation_ram_hit_provenance": evaluation_incumbent_sanity,
            "bootstrap_event_count": bootstrap_event_count,
            "cache_case_counts": {
                case: cache_case_counts[case] for case in ("hit", "miss", "bypass")
            },
            "cache_source_counts": {
                source: cache_source_counts[source]
                for source in ("ram", "ssd", "none", "other")
            },
            "terminal_status_counts": dict(sorted(terminal_status_counts.items())),
            "terminal_zero_benefit_rows": terminal_zero_benefit_rows,
            "history_observations_by_epoch_and_entry_ordinal": {
                f"{epoch}:{ordinal}": len(history)
                for (epoch, ordinal), history in sorted(histories.items())
            },
            "minimum_history_observations": _MIN_HISTORY_OBSERVATIONS,
            "chronological_fit_fraction": _HISTORY_FRACTION,
        },
        "nonappend_divergence_buckets": complete_bucket_counts,
        "incumbent_sanity": incumbent_sanity,
        "incumbent_provenance": {
            "completed_ram_hit_rows": evaluation_incumbent_sanity["eligible_ram_hits"],
            "verified_rows": evaluation_incumbent_sanity["verified"],
            "minimum_verified_rows": minimum_events,
            "unverified_rows": (
                evaluation_incumbent_sanity["mismatches"]
                + evaluation_incumbent_sanity["not_reported"]
                + evaluation_incumbent_sanity["impossible"]
            ),
            "gate_eligible": (
                incumbent_sanity["verified"] == incumbent_sanity["eligible_ram_hits"]
                and incumbent_sanity["mismatches"] == 0
                and incumbent_sanity["not_reported"] == 0
                and incumbent_sanity["impossible"] == 0
                and evaluation_incumbent_sanity["verified"] >= minimum_events
                and evaluation_incumbent_sanity["verified"]
                == evaluation_incumbent_sanity["eligible_ram_hits"]
            ),
        },
        "policy_summaries": summaries,
        "bootstrap": {
            "bootstrap_event_count": bootstrap_event_count,
            "observed_fitted": observed_bootstrap,
            "future_known_upper": optimistic_bootstrap,
            "gate_uses": (
                "lower=min across base and 2x block lengths; future-known upper=max"
            ),
        },
        "events": event_views,
    }
