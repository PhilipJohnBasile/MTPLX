# The target-prefix vs `generate_ar` divergence — static analysis

Status: hypothesis, code-read only, **no runs**. Written to make the eventual
verification run short and targeted instead of exploratory.

## The blocker

The staged-K1 DFlash lane proved it adds no divergence of its own: it is
24/24 byte-identical to the compiled target-prefix control. But that control is
only **13/24 byte-identical to `generate_ar`**. So the release blocker predates
DFlash — DFlash only made it visible.

## The two paths are not the same code

| | forward used | GDN recurrence |
|---|---|---|
| `generate_ar` (generation.py:4763) | `rt.forward_ar` → `self.model(...)` (runtime.py:146-180, and the plain override at :458-476) | stock model forward |
| compiled target-prefix M1 (a3b_compiled_target_prefix.py:276-312) | `runtime._forward_ar_capture_a3b_postconv` → `forward_with_a3b_gdn_postconv_capture` (runtime.py:299-315) | **fused post-conv implementation**, selected by `_a3b_gdn_postconv_impl_selection()` (gdn_capture.py:115-133) |

The compiled step additionally assigns cache leaves directly into a shadow
cache and is traced under `mx.compile`, so both the arithmetic *and* the
evaluation order differ from the AR path.

> **CORRECTION (2026-07-31, from `docs/dflash-gate-preregistration.md`).** Two
> claims below do not survive a read of the dispatch code, and the
> pre-registration supersedes them:
>
> 1. **The supporting evidence is from the wrong lane.** The 0.125-margin flip
>    and the failed scalar replay come from the **wide** lane's eight-row verify.
>    Verify widths outside {1,2,3} fall through to the **stock** MoE
>    (`a3b_whole_moe.py:1024-1047`), so the wide lane differs on an axis
>    staged-K1 does not have. Transferring that evidence here was invalid.
> 2. **"Different-but-valid path" is not established.** At staged-K1 both arms
>    are rows==1 in a decode phase and therefore run the *same* fused M1 route;
>    prefill runs the same stock path in both. Byte-equality is therefore not
>    unattainable-by-construction, and the argument for retiring the gate fails.
>
> H1 remains *a* hypothesis, but is no longer the "leading" one on this
> evidence. Use the L0–L6 axis ladder in the pre-registration, which adds one
> axis per rung so the first nonzero rung names the mechanism. Also note the
> probe as written cannot decide this: it compares two stock-MoE prefill
> forwards with no `attention_phase`, no `mx.compile`, and no shadow cache.

## Hypothesis H1 (superseded framing): numerics, not logic

The divergence is a floating-point accumulation-order difference in the GDN
recurrence between the fused post-conv route and the stock forward — not a bug
in speculation, acceptance, or cache rollback.

Supporting evidence already on record:

- The wide lane's first full-suite mismatch was classified precisely: scalar AR
  selected token 1724 while the eight-row target verify selected 1752 **at a
  0.125-logit margin**. That is a near-tie argmax flip, the signature of a
  numerics delta rather than a state bug.
- A low-margin scalar replay fixed the immediate token but not delayed state
  drift, and replaying enough cycles erased the speedup — consistent with a
  small per-step delta that compounds through the recurrence, not a single
  wrong branch.
- `docs/turbo-verify.md` documents the same class for NAX: different
  accumulation order, argmax-identical on probed positions but explicitly not
  bit-exact.

If H1 holds, 13/24 is roughly "11 of 24 prompts contained at least one
near-tie within the delta" — which is plausible for 160-192 token generations,
and would predict the mismatch rate rises with generation length.

## Competing hypotheses, cheap to separate

- **H2 — shadow-cache aliasing.** The M1 step writes `entry.cache[0..2]` from
  traced state and clears `rollback_state`. If any leaf aliases a buffer the
  stock path also owns, state could differ across steps. Predicts divergence
  that *starts* at a specific step and never recovers, rather than scattered
  near-tie flips.
- **H3 — sampler/RNG path difference.** Predicts divergence only at
  temperature > 0 and none under greedy. The receipts in question are greedy,
  so this is already weakly disfavoured.
- **H4 — logits_keep / emit_logits shaping.** `forward_ar` has
  `emit_logits`/`logits_keep` capability negotiation (runtime.py:168-180) that
  the compiled path does not go through. Predicts a systematic offset rather
  than sporadic flips.

## The minimal experiment (when GPU is free)

Ordered so the first result discriminates the most:

1. **Single-forward logit delta.** Same prompt, same cache state, one step:
   run `rt.forward_ar` and `rt._forward_ar_capture_a3b_postconv` and report
   `max|Δlogit|` plus whether argmax agrees. **H1 predicts a small but nonzero
   delta (~1e-2..1e-1) with argmax agreeing on most positions.** ~2 minutes.
2. **Impl A/B.** `MTPLX_A3B_GDN_POSTCONV_IMPL` selects `inline_g` (default) or
   `headquarter`. If the two disagree with each other, the fused route is the
   source and H1 is confirmed without touching the AR path at all.
3. **Length sweep.** If H1 holds, the mismatch fraction should grow with
   max_tokens. 24 prompts at 64 / 128 / 192 tokens.
4. Only if 1-3 come back clean: instrument per-step state hashes to chase H2.

## What follows from each outcome

- **H1 confirmed:** byte equality against `generate_ar` is the wrong gate for
  this lane — the compiled route is a different (not worse) numerical path, the
  same status NAX already has. The honest gate becomes a distributional one
  (argmax agreement rate + a sampled-q equivalence test), and that should be
  argued explicitly rather than quietly redefined. Note this does **not**
  weaken the DFlash exactness requirement, which is about the *proposal*
  distribution, not about matching a particular target implementation bit-for-bit.
- **H2 confirmed:** a real state bug; fix it and byte equality should return.
- **H4 confirmed:** a shaping mismatch; align the two call paths.
