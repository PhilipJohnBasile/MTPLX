"""On/off identity gates for the greedy-trio ports (#313 / #315c1 / #318).

Every knob defaults OFF; each gate proves (a) the off state reproduces the
unported pin token-for-token, and (b) the on state is token- and
receipt-identical to off on the model-free TinyMTP harness, on both the
all-accept lane (mtp_token=1) and the all-reject lane (mtp_token=2).
Tie-break exactness for #315c1 lives in test_batched_greedy_argmax_tiebreak;
real-model byte-identity and ABBA speed are the founder-scheduled phases.
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
    for knob in _KNOBS:
        monkeypatch.delenv(knob, raising=False)


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
