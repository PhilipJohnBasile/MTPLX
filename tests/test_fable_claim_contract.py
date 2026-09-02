"""The request-time DECLINE / install-time RAISE contract.

Pure Python: no MLX, no model, no device.  These pin the rule that a fable
flag's precondition check must not turn an ineligible REQUEST into an outage,
while an install-time contract violation still fails loudly.
"""

from __future__ import annotations

import pytest

from mtplx import fable_claim_contract as contract


class _Ineligible(RuntimeError):
    pass


FLAG = "MTPLX_FABLE_TEST_LANE"


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setattr(contract, "_STRICT", False)
    contract.reset_for_test()
    yield
    contract.reset_for_test()


def _claim(*, ok: bool, receipt: dict | None = None):
    """A miniature claim shaped like the real ones."""

    try:
        if not ok:
            contract.decline("greedy_request", "this lane wants temperature > 0")
        return "plan"
    except contract.ClaimDeclined as declined:
        stamped = contract.declined_receipt(
            FLAG, declined, ineligible=_Ineligible
        )
        if receipt is not None:
            receipt.clear()
            receipt.update(stamped)
        return None


def test_an_eligible_request_still_claims():
    receipt: dict = {}
    assert _claim(ok=True, receipt=receipt) == "plan"
    assert receipt == {}
    assert contract.decline_counts(FLAG) == {}


def test_an_ineligible_request_declines_without_raising():
    receipt: dict = {}
    assert _claim(ok=False, receipt=receipt) is None
    assert receipt["installed"] is False
    assert receipt["declined"] == "greedy_request"
    assert receipt["declined_detail"] == "this lane wants temperature > 0"
    assert receipt["declines"] == {"greedy_request": 1}


def test_declines_accumulate_per_flag_and_reason():
    for _ in range(3):
        _claim(ok=False)
    assert contract.decline_counts(FLAG) == {"greedy_request": 3}
    # The returned tally is a copy: a caller cannot corrupt the ledger.
    tally = contract.decline_counts(FLAG)
    tally["greedy_request"] = 99
    assert contract.decline_counts(FLAG) == {"greedy_request": 3}


def test_one_warning_per_reason_per_process(capsys):
    """A 164-problem eval must not print 164 identical warnings.

    The line goes to stderr, the channel the lanes' own install receipts use,
    so it lands in ``server.log`` beside them.
    """

    for _ in range(5):
        _claim(ok=False)
    lines = [
        line
        for line in capsys.readouterr().err.splitlines()
        if FLAG in line
    ]
    assert len(lines) == 1
    assert "this request runs the shipped path" in lines[0]
    assert contract.STRICT_ENV in lines[0]


def test_strict_claims_turns_a_decline_into_the_lanes_own_error(monkeypatch):
    monkeypatch.setattr(contract, "_STRICT", True)
    with pytest.raises(_Ineligible, match="temperature > 0"):
        _claim(ok=False)
    # Nothing is tallied: strict mode never reaches the ledger.
    assert contract.decline_counts(FLAG) == {}


def test_strict_message_names_the_env_that_caused_the_failure(monkeypatch):
    monkeypatch.setattr(contract, "_STRICT", True)
    with pytest.raises(_Ineligible) as excinfo:
        _claim(ok=False)
    assert contract.STRICT_ENV in str(excinfo.value)


def test_the_strict_gate_is_off_by_default(monkeypatch):
    """Serving must never fail closed by accident.

    Deliberately does NOT reload the module: every lane holds ``except
    ClaimDeclined`` against THIS class object, and a reload would rebind the
    name to a new class that those handlers no longer catch -- which is
    exactly the outage this whole module exists to prevent.
    """

    monkeypatch.delenv(contract.STRICT_ENV, raising=False)
    assert contract._env_truthy(contract.STRICT_ENV) is False
    monkeypatch.setenv(contract.STRICT_ENV, "0")
    assert contract._env_truthy(contract.STRICT_ENV) is False
    for value in ("1", "true", "yes", "on", "ON"):
        monkeypatch.setenv(contract.STRICT_ENV, value)
        assert contract._env_truthy(contract.STRICT_ENV) is True


def test_decline_raises_the_internal_signal_not_a_lane_error():
    with pytest.raises(contract.ClaimDeclined) as excinfo:
        contract.decline("key", "detail")
    assert excinfo.value.key == "key"
    assert excinfo.value.detail == "detail"
