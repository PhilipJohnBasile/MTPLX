# MLX-LM #1709 exactness audit

Date: 2026-08-10. Reviewed head:
[`a318a32`](https://github.com/ml-explore/mlx-lm/commit/a318a32c44c1fe7666160122d26e11b319248fd3),
base `254d153fdeb6f150edd4fc5a54f9828638481fa8`.

## Verdict

**Reject as merge-ready.** The residual and block-verification formulas are
sound, and the unchanged default path remains intact. The new integration is
not distribution-exact across its advertised plain-temperature and
logits-processor surface.

Distributional exactness and seeded byte identity are different contracts.
The supported residual/block rules intentionally consume random numbers
differently, so a correct implementation should reproduce the target law, not
the same seeded stream.

The CPU-only
[`research_mlx_lm_1709_exactness.py`](../../benchmarks/research_mlx_lm_1709_exactness.py)
reproducer and its
[`JSON receipt`](../../benchmarks/results/mlx-m5-research-20260810/mlx_lm_1709_exactness_reproducer.json)
materialize the three counterexamples below plus a positive rational,
prefix-independent K=1 oracle. The negative cases copy the pinned PR semantics; they do not import or
execute the PR. The positive case is independently derived arithmetic, not PR
integration evidence. Script SHA256 is
`e1cfc24513e7ac4f1e7d993a3197a6617a5b3c0972940ca494bd198bd7e71311`.

## Blocking findings

### 1. Temperature reconstruction changes dtype order

The sampler multiplies logits by inverse temperature in their current dtype
before categorical sampling
([`sample_utils.py:68-72`](https://github.com/ml-explore/mlx-lm/blob/a318a32c44c1fe7666160122d26e11b319248fd3/mlx_lm/sample_utils.py#L68-L72),
[`sample_utils.py:290-292`](https://github.com/ml-explore/mlx-lm/blob/a318a32c44c1fe7666160122d26e11b319248fd3/mlx_lm/sample_utils.py#L290-L292)).
Verification instead casts to float32 and then applies temperature
([`generate.py:472-481`](https://github.com/ml-explore/mlx-lm/blob/a318a32c44c1fe7666160122d26e11b319248fd3/mlx_lm/generate.py#L472-L481)).

Those are different numerical distributions for bfloat16/float16 logits. An
exact four-token bfloat16 counterexample at temperature 0.7 produced one-step
total variation error `0.0026887`. This is analytic error, not Monte Carlo
noise.

### 2. History-dependent processors follow the wrong prefix

The target model computes all verification rows along the draft prefix. The
processor loop then samples each target row independently, appends that sampled
token to `prev_tokens`, and processes the next already-computed row
([`generate.py:695-719`](https://github.com/ml-explore/mlx-lm/blob/a318a32c44c1fe7666160122d26e11b319248fd3/mlx_lm/generate.py#L695-L719),
[`generate.py:782-788`](https://github.com/ml-explore/mlx-lm/blob/a318a32c44c1fe7666160122d26e11b319248fd3/mlx_lm/generate.py#L782-L788)).

That mixes two incompatible conditionals. In an adversarial processor where
the first token is uniform and every later token must repeat the committed
previous token, the target law permits only `00` and `11`. Both new rules
produced all four pairs, with only 51.7% repeats instead of 100%.

### 3. The custom-sampler guard is forgeable

The capability check trusts any callable with a `.temp` attribute
([`generate.py:646-658`](https://github.com/ml-explore/mlx-lm/blob/a318a32c44c1fe7666160122d26e11b319248fd3/mlx_lm/generate.py#L646-L658)).
A custom top-1 sampler can set `sampler.temp = 1.0` and pass as plain
categorical sampling. In an adversarial target/drafter fixture, both new rules
then emitted a token that the target sampler could never emit in 748 of 2,000
trials.

## Secondary correctness gaps

- The linear-space residual can underflow to zero and falls back to ordinary
  target sampling, which can sample outside the mathematical residual support.
- `num_draft_tokens=0` leaves draft log-probabilities as `None` and crashes both
  new rules.
- Batched helper inputs fail through low-level shape errors instead of an
  explicit single-stream guard.
- Acceptance uses `u <= threshold`; `u < threshold` avoids accepting a
  zero-probability event if the finite PRNG emits exactly zero.

## What passed

The algorithmic statements below are from source inspection; test-count and
seeded-stream claims are upstream PR evidence, not results from an exact local
checkout. The rational oracle is the independent reproducer linked above.

- The residual rule matches Leviathan-Chen rejection sampling.
- `_block_verify` correctly expresses the cumulative ratio, longest passing
  prefix, and scaled correction from Block Verification.
- The independent exact rational four-token oracle had zero error for both
  algorithms at K=1. It is prefix-independent and does not exercise histories
  with one or two prior failures; a K>1 conditional-history oracle is required
  before making a later-prefix-rescue claim.
- Upstream reports all 12 new focused tests passing.
- Upstream reports greedy ties matching across all acceptance rules.
- Upstream EOS simulations pass under the narrow float32, no-processor contract.
- Upstream reports that the default exact path retained the base seeded output hash.

## Smallest defensible contract today

The new rules are distribution-exact only when all of these hold:

- one sequence, positive draft depth, equal vocabularies, and exact expected
  row shapes;
- a trusted plain categorical sampler with no hidden filtering;
- temperature 1.0, or already-float32 log-probabilities;
- no history-, state-, or randomness-dependent logits processors;
- target rows conditioned on the actual draft prefix;
- representable residual mass;
- exact cache rollback for both target and draft states.

## Required remediation

1. Expose a trusted sampler descriptor carrying the exact transformed
   distribution; do not infer capability from `.temp`.
2. Make sampling and verification use the same dtype and operation order.
3. Apply history-dependent processors along the draft prefix used by the
   target forward, or reject them.
4. Add explicit rank, row, vocabulary, temperature, and positive-depth guards.
5. Sample residuals stably in log space without a target-distribution fallback.
6. Add exhaustive joint-law tests for conditional histories, bfloat16/float16
   temperature, sampler spoofing, multiple block failures, EOS, ties, RNG
   advancement, and malformed shapes.

The algorithm is worth keeping. The current capability surface is not.
