"""Exact joint-law oracle for the DFlash K1 block-speculation state machine.

.. warning::

   **THIS IS NOT A-SIDE EVIDENCE YET.** External audit, 2026-07-31, verdict
   ``NOT EVIDENCE``. Two things are wrong and both are recorded here rather
   than quietly fixed, because the second one is a reasoning error worth
   keeping visible:

   1. **It never calls the production path.** The oracle and the "planted
      defect" are both local functions. It proves the *math* is right and
      tests only itself; no mutation of the system under test is exercised.
      A1-A4 in ``docs/dflash-gate-preregistration.md`` §4.3 all remain unmet.

   2. **The conditioning "fix" was a weakening that hid a modelling gap.**
      The first version asserted an unconditional two-token joint against
      ``p_0 x p_1``, it failed, and the assertion was conditioned to make it
      pass. That was backwards. In a real decoder a rejection does not end
      generation -- the next K1 cycle produces the following token -- so for a
      prefix-independent toy target the two-token *emitted* law really is
      ``p_0 x p_1``. The failing assertion was right about the real machine;
      the oracle is what was wrong, because it terminates at the first
      rejection instead of continuing into the next cycle. Conditioning the
      check made a truncated model look correct.

   What a real version needs: a production-adapter harness driving the actual
   staged source and acceptance transitions across multiple cycles, an
   independent finite-state oracle over a fixed emitted-token horizon
   (including queue reuse, rejection clearing, bonus, max-length and stops),
   comparison against the prefix-conditional target AR law, and the five §4.3
   A4 mutations run through that same harness.

   Kept in-tree because the enumeration machinery and the distribution
   families are reusable, and because the weakening is instructive.

Why this exists
---------------
MTPLX's identity claim is question **A**: the drafter's declared ``q`` is the
distribution it sampled from, and acceptance + residual correction recover the
target law. Before this file the entire empirical A-side was one 4-token,
single-position unit test. Byte-equality against ``generate_ar`` tests question
**B** (are two implementations numerically identical) and says nothing about A.

The oracle below enumerates the **exact joint law over committed sequences** for
a tiny configuration (vocab 4, block 3), by brute force over every reachable
outcome. It models **one truncated block transition locally** and does not call
the production acceptance path at any point — see the warning above. Marginal
recovery at a single position is too weak a test: it cannot see errors that
only appear in the *joint* law — queue reuse across blocks, residual
mis-conditioning after a rejection, or bonus-token accounting.

Independence discipline
-----------------------
The oracle does **not** import ``compute_acceptance_probability`` or
``residual_distribution``. It recomputes acceptance and residuals from first
principles in plain Python floats, so a sign error or a normalization bug in
production cannot cancel itself out by appearing on both sides. The only shared
import is ``SparseDistribution``, used purely as a data container, never for
oracle arithmetic. Note this independence buys nothing until the harness
actually reaches production code.

What is asserted
----------------
For every (p, q) family, the locally-modelled committed distribution must match
the target's law to within 1e-12. This exercises the Leviathan-Chen *arithmetic*;
it does not exercise the engine.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from mtplx.sampling import SparseDistribution

VOCAB = 4
BLOCK = 3
TOL = 1e-12


# ---------------------------------------------------------------------------
# The oracle: exact enumeration, computed independently of production code
# ---------------------------------------------------------------------------
def _accept_prob(p: list[float], q: list[float], token: int) -> float:
    """min(1, p(x)/q(x)) — recomputed here, deliberately not imported."""
    if q[token] <= 0.0:
        return 1.0 if p[token] > 0.0 else 0.0
    return min(1.0, p[token] / q[token])


def _residual(p: list[float], q: list[float]) -> list[float]:
    """normalize(max(p - q, 0)) — recomputed here, deliberately not imported."""
    raw = [max(pi - qi, 0.0) for pi, qi in zip(p, q)]
    total = sum(raw)
    if total <= 0.0:
        return list(p)
    return [r / total for r in raw]


def oracle_committed_law(
    p_chain: list[list[float]], q_chain: list[list[float]]
) -> dict[tuple[int, ...], float]:
    """Exact distribution over committed prefixes for one speculative block.

    Enumerates every draft draw and every accept/reject coin outcome, weighting
    each path by its probability. Returns a mapping from committed token tuple
    to probability. Because every branch is enumerated rather than sampled,
    this is exact up to floating point.

    Semantics mirrored (and independently re-derived):
      - draft token i is drawn from q_chain[i];
      - it is accepted with probability min(1, p_i(x)/q_i(x));
      - on the first rejection, one token is drawn from residual(p_i, q_i) and
        the block ends;
      - if every draft token is accepted, a bonus token is drawn from the
        target distribution at the next position.
    """
    law: dict[tuple[int, ...], float] = {}

    def walk(index: int, prefix: tuple[int, ...], weight: float) -> None:
        if weight <= 0.0:
            return
        if index == len(q_chain):
            # All drafts accepted -> bonus draw from the next target position.
            bonus_p = p_chain[len(q_chain)]
            for token in range(VOCAB):
                if bonus_p[token] > 0.0:
                    key = prefix + (token,)
                    law[key] = law.get(key, 0.0) + weight * bonus_p[token]
            return

        p_i, q_i = p_chain[index], q_chain[index]
        for token in range(VOCAB):
            draw = q_i[token]
            if draw <= 0.0:
                continue
            a = _accept_prob(p_i, q_i, token)
            # accepted: continue the chain
            walk(index + 1, prefix + (token,), weight * draw * a)
            # rejected: commit one residual draw, block ends
            reject_w = weight * draw * (1.0 - a)
            if reject_w > 0.0:
                res = _residual(p_i, q_i)
                for r_token in range(VOCAB):
                    if res[r_token] > 0.0:
                        key = prefix + (r_token,)
                        law[key] = law.get(key, 0.0) + reject_w * res[r_token]

    walk(0, (), 1.0)
    return law


def target_autoregressive_law(p_chain: list[list[float]], length: int) -> dict[tuple[int, ...], float]:
    """The law the target would produce on its own: plain chain rule."""
    law: dict[tuple[int, ...], float] = {}
    for tokens in itertools.product(range(VOCAB), repeat=length):
        prob = 1.0
        for i, token in enumerate(tokens):
            prob *= p_chain[i][token]
        if prob > 0.0:
            law[tokens] = prob
    return law


# ---------------------------------------------------------------------------
# Distribution families: cover the cases that break naive implementations
# ---------------------------------------------------------------------------
def _families(seed: int = 0) -> list[tuple[str, list[list[float]], list[list[float]]]]:
    rng = np.random.default_rng(seed)

    def norm(v):
        v = np.asarray(v, dtype=np.float64)
        return (v / v.sum()).tolist()

    cases: list[tuple[str, list[list[float]], list[list[float]]]] = []

    # q == p: acceptance should be 1 everywhere; committed law must be exact.
    p = [norm(rng.random(VOCAB) + 0.05) for _ in range(BLOCK + 1)]
    cases.append(("q_equals_p", p, [list(row) for row in p[:BLOCK]]))

    # Disjoint support: q puts mass where p has none -> always rejected.
    p = [norm([0.5, 0.5, 0.0, 0.0]) for _ in range(BLOCK + 1)]
    q = [norm([0.0, 0.0, 0.5, 0.5]) for _ in range(BLOCK)]
    cases.append(("disjoint_support", p, q))

    # q sharper than p (the case where one-hot beats soft-q on acceptance).
    p = [norm(rng.random(VOCAB) + 0.05) for _ in range(BLOCK + 1)]
    q = [norm([0.97, 0.01, 0.01, 0.01]) for _ in range(BLOCK)]
    cases.append(("sharp_q_diffuse_p", p, q))

    # p sharper than q.
    p = [norm([0.97, 0.01, 0.01, 0.01]) for _ in range(BLOCK + 1)]
    q = [norm(rng.random(VOCAB) + 0.05) for _ in range(BLOCK)]
    cases.append(("sharp_p_diffuse_q", p, q))

    # Zeros in q: exercises the q<=0 accept-iff-p>0 convention.
    p = [norm(rng.random(VOCAB) + 0.05) for _ in range(BLOCK + 1)]
    q = [norm([0.6, 0.4, 0.0, 0.0]) for _ in range(BLOCK)]
    cases.append(("q_has_zeros", p, q))

    # Random families, the bulk of the coverage.
    for i in range(40):
        p = [norm(rng.random(VOCAB) + 1e-3) for _ in range(BLOCK + 1)]
        q = [norm(rng.random(VOCAB) + 1e-3) for _ in range(BLOCK)]
        cases.append((f"random_{i}", p, q))

    return cases


# ---------------------------------------------------------------------------
# The property that matters
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,p_chain,q_chain", _families())
def test_committed_law_equals_target_law(name, p_chain, q_chain):
    """Speculation must not change the output distribution — as a JOINT law.

    This is the A-side property. It holds for ANY q, which is exactly the
    Leviathan-Chen guarantee: the drafter's quality affects the acceptance
    rate, never the committed distribution.
    """
    law = oracle_committed_law(p_chain, q_chain)

    total = sum(law.values())
    assert abs(total - 1.0) < TOL, f"{name}: committed law sums to {total!r}, not 1"

    # Marginal at the first committed position must equal p_0 exactly.
    first = [0.0] * VOCAB
    for tokens, prob in law.items():
        first[tokens[0]] += prob
    for token in range(VOCAB):
        assert abs(first[token] - p_chain[0][token]) < TOL, (
            f"{name}: position-0 marginal {first[token]!r} != target {p_chain[0][token]!r} "
            f"for token {token}"
        )

    # Second committed token, CONDITIONED on the first. This is the part a
    # single-position marginal test cannot see.
    #
    # Note the conditioning is load-bearing and was got wrong on the first
    # attempt: reaching position 1 is NOT independent of the token committed at
    # position 0, because a token with low acceptance probability is less likely
    # to survive. Comparing an unconditional 2-token joint against p_0 x p_1
    # therefore fails for correct implementations. The guarantee is a
    # conditional one -- given the committed prefix, the next committed token is
    # distributed as p at that position -- so we normalize within each prefix.
    by_first: dict[int, dict[int, float]] = {}
    for tokens, prob in law.items():
        if len(tokens) >= 2:
            by_first.setdefault(tokens[0], {})
            by_first[tokens[0]][tokens[1]] = by_first[tokens[0]].get(tokens[1], 0.0) + prob

    for first_token, seconds in by_first.items():
        mass = sum(seconds.values())
        if mass <= 1e-15:
            continue
        for token in range(VOCAB):
            got = seconds.get(token, 0.0) / mass
            want = p_chain[1][token]
            assert abs(got - want) < 1e-9, (
                f"{name}: P(second={token} | first={first_token}) = {got!r} != "
                f"target p_1 = {want!r}"
            )


def test_oracle_detects_a_planted_defect():
    """The oracle must be able to FAIL. A gate that cannot fail certifies nothing.

    Plants the classic bug — using q instead of the residual after a rejection —
    and asserts the position-0 marginal check catches it.
    """
    rng = np.random.default_rng(7)

    def norm(v):
        v = np.asarray(v, dtype=np.float64)
        return (v / v.sum()).tolist()

    p_chain = [norm(rng.random(VOCAB) + 0.05) for _ in range(BLOCK + 1)]
    q_chain = [norm(rng.random(VOCAB) + 0.05) for _ in range(BLOCK)]

    def broken_law() -> dict[tuple[int, ...], float]:
        law: dict[tuple[int, ...], float] = {}
        p_i, q_i = p_chain[0], q_chain[0]
        for token in range(VOCAB):
            draw = q_i[token]
            if draw <= 0.0:
                continue
            a = _accept_prob(p_i, q_i, token)
            law[(token,)] = law.get((token,), 0.0) + draw * a
            reject_w = draw * (1.0 - a)
            for r_token in range(VOCAB):          # BUG: resample from q, not residual
                law[(r_token,)] = law.get((r_token,), 0.0) + reject_w * q_i[r_token]
        return law

    law = broken_law()
    first = [0.0] * VOCAB
    for tokens, prob in law.items():
        first[tokens[0]] += prob
    max_err = max(abs(first[t] - p_chain[0][t]) for t in range(VOCAB))
    assert max_err > 1e-6, (
        "the planted residual bug was NOT detected — the oracle cannot fail, "
        f"max marginal error was only {max_err!r}"
    )


def test_production_sparse_distribution_roundtrip():
    """Guard the one production type the oracle touches, so it stays a container.

    SparseDistribution renormalizes on construction; if that ever changed
    silently, oracle inputs and production inputs would diverge without any
    test noticing.
    """
    dist = SparseDistribution(
        token_ids=np.array([0, 2, 3], dtype=np.int64),
        probs=np.array([0.2, 0.3, 0.5], dtype=np.float64),
        vocab_size=VOCAB,
    )
    assert abs(float(dist.probs.sum()) - 1.0) < TOL
    assert abs(dist.probability(2) - 0.3) < TOL
    assert abs(dist.probability(1) - 0.0) < TOL
