"""On/off identity gates for the greedy-trio ports (#313 / #315c1 / #318).

The knobs default ON since the night-20260822 round-4 ruling (n=4
counterbalanced ABBA blend +2.7% mean; byte-identity held on greedy and
sampled-seed lanes); "0" opts out. Each gate pins knobs explicitly and proves
the off state and the on state are token- and receipt-identical on the
model-free TinyMTP harness, on both the all-accept lane (mtp_token=1) and the
all-reject lane (mtp_token=2). Tie-break exactness for #315c1 lives in
test_batched_greedy_argmax_tiebreak; real-model byte-identity and ABBA speed
are the measured phases (MEASUREMENTS.md 08-22).
"""

from __future__ import annotations

import pytest

from tests.test_graphbank_compiled_verify import _run_tiny_mtpk

_KNOBS = [
    "MTPLX_GREEDY_DRAFT_CHAIN",
    "MTPLX_BATCHED_GREEDY_ACCEPT",
    "MTPLX_BATCH_PAGED_OFFSETS",
]


def _clear(monkeypatch):
    """Pin every knob OFF — the identity baseline arm (defaults are ON now)."""
    for knob in _KNOBS:
        monkeypatch.setenv(knob, "0")
    import mtplx.graphbank as gb

    monkeypatch.setattr(gb, "_BATCH_PAGED_OFFSETS", False)


def test_trio_defaults_resolve_on(monkeypatch):
    """The default-flip pin: unset env resolves ON for all three knobs."""
    for knob in _KNOBS:
        monkeypatch.delenv(knob, raising=False)
    from mtplx.generation import _env_enabled_default_on
    import mtplx.graphbank as gb

    assert _env_enabled_default_on("MTPLX_GREEDY_DRAFT_CHAIN")
    assert _env_enabled_default_on("MTPLX_BATCHED_GREEDY_ACCEPT")
    assert gb._batch_paged_offsets_enabled()
    monkeypatch.setenv("MTPLX_GREEDY_DRAFT_CHAIN", "0")
    assert not _env_enabled_default_on("MTPLX_GREEDY_DRAFT_CHAIN")


def test_trio_context_fence_resolver(monkeypatch):
    """Fence default 12288; 0/off = unlimited; garbage falls back."""
    from mtplx.generation import _trio_max_context

    monkeypatch.delenv("MTPLX_GREEDY_TRIO_MAX_CONTEXT", raising=False)
    assert _trio_max_context() == 12288
    monkeypatch.setenv("MTPLX_GREEDY_TRIO_MAX_CONTEXT", "0")
    assert _trio_max_context() == 0
    monkeypatch.setenv("MTPLX_GREEDY_TRIO_MAX_CONTEXT", "off")
    assert _trio_max_context() == 0
    monkeypatch.setenv("MTPLX_GREEDY_TRIO_MAX_CONTEXT", "32768")
    assert _trio_max_context() == 32768
    monkeypatch.setenv("MTPLX_GREEDY_TRIO_MAX_CONTEXT", "junk")
    assert _trio_max_context() == 12288


def test_trio_fence_disarms_chain_above_context(monkeypatch, ):
    """A prompt at/above the fence must not run the chain lane (and output
    stays identical — the fence only routes, never changes tokens), and the
    graphbank stamp must read False for the fenced request."""
    _clear(monkeypatch)
    monkeypatch.setenv("MTPLX_GREEDY_DRAFT_CHAIN", "1")
    monkeypatch.setenv("MTPLX_GREEDY_TRIO_MAX_CONTEXT", "1")
    fenced, _ = _run_tiny_mtpk(max_tokens=8, mtp_token=1)
    import mtplx.graphbank as gb

    assert gb.paged_offsets_context_ok() is False
    chain_drafts = [
        d
        for e in (fenced.stats.events or [])
        for d in e.get("drafts", [])
        if d.get("draft_core") == "greedy-chain"
    ]
    assert not chain_drafts, "fence set but the chain lane still ran"

    monkeypatch.setenv("MTPLX_GREEDY_TRIO_MAX_CONTEXT", "0")
    unfenced, _ = _run_tiny_mtpk(max_tokens=8, mtp_token=1)
    assert gb.paged_offsets_context_ok() is True
    assert list(fenced.tokens) == list(unfenced.tokens)
    assert any(
        d.get("draft_core") == "greedy-chain"
        for e in (unfenced.stats.events or [])
        for d in e.get("drafts", [])
    ), "fence=0 must leave the chain lane armed"


def _fingerprint(out):
    return {
        "tokens": list(out.tokens),
        "drafted_by_depth": list(out.stats.drafted_by_depth or []),
        "accepted_by_depth": list(out.stats.accepted_by_depth or []),
        "verify_calls": out.stats.verify_calls,
    }


@pytest.mark.parametrize("mtp_token", [1, 2], ids=["all-accept", "all-reject"])
@pytest.mark.parametrize("knob", _KNOBS)
def test_trio_knob_on_matches_off(monkeypatch, knob, mtp_token):
    _clear(monkeypatch)
    baseline, _ = _run_tiny_mtpk(max_tokens=8, mtp_token=mtp_token)
    monkeypatch.setenv(knob, "1")
    if knob == "MTPLX_BATCH_PAGED_OFFSETS":
        # Module-resolved flag: force re-resolution for the test process.
        import mtplx.graphbank as gb

        monkeypatch.setattr(gb, "_BATCH_PAGED_OFFSETS", True)
    on, _ = _run_tiny_mtpk(max_tokens=8, mtp_token=mtp_token)
    assert _fingerprint(on) == _fingerprint(baseline), (
        f"{knob} on-arm diverged from off-arm on the {mtp_token=} lane"
    )


@pytest.mark.parametrize("mtp_token", [1, 2], ids=["all-accept", "all-reject"])
def test_trio_full_stack_matches_pin(monkeypatch, mtp_token):
    _clear(monkeypatch)
    baseline, _ = _run_tiny_mtpk(max_tokens=8, mtp_token=mtp_token)
    for knob in _KNOBS:
        monkeypatch.setenv(knob, "1")
    import mtplx.graphbank as gb

    monkeypatch.setattr(gb, "_BATCH_PAGED_OFFSETS", True)
    on, _ = _run_tiny_mtpk(max_tokens=8, mtp_token=mtp_token)
    assert _fingerprint(on) == _fingerprint(baseline)


def test_greedy_chain_engages_and_marks_events(monkeypatch):
    """#313: the chain lane must actually run (engagement counter law) and
    stamp its discriminator, while producing identical output."""
    _clear(monkeypatch)
    baseline, _ = _run_tiny_mtpk(max_tokens=8, mtp_token=1)
    monkeypatch.setenv("MTPLX_GREEDY_DRAFT_CHAIN", "1")
    on, _ = _run_tiny_mtpk(max_tokens=8, mtp_token=1)
    assert list(on.tokens) == list(baseline.tokens)
    chain_drafts = [
        draft
        for event in (on.stats.events or [])
        for draft in event.get("drafts", [])
        if draft.get("draft_core") == "greedy-chain"
    ]
    assert chain_drafts, "greedy chain never engaged — dead-switch scar (#314)"
    baseline_drafts = [
        draft
        for event in (baseline.stats.events or [])
        for draft in event.get("drafts", [])
    ]
    assert len(chain_drafts) == len(baseline_drafts) or baseline_drafts
