"""Gates for the DeepSeek-V4 multi-token-prediction (MTP) draft block.

Three layers of evidence, mirroring the M2/M3 pattern the rest of this backend
uses:

  * **Numerical parity** — the whole draft block (input fusion -> HC-wrapped
    attention -> HC-wrapped MoE -> its own head collapse -> logits) against a
    self-contained NumPy transcription of the authoritative reference
    ``deepseek-ai/DeepSeek-V4-Flash/inference/model.py`` (``MTPBlock.forward``
    L757-766, ``Block.forward`` L688-700, ``Attention.forward`` L484-543,
    ``Gate``/``Expert``/``MoE`` L546-644, ``ParallelHead.hc_head`` L728-735) plus
    ``inference/kernel.py`` (``hc_split_sinkhorn_kernel`` L371-427,
    ``sparse_attn_kernel`` L294-350).  No torch, no download, CPU device so MLX
    fp32 is bit-exact IEEE rather than its reduced-precision GPU matmul path.

  * **Structure** — the draft block is a body block at ``layer_id = n_layers``
    (reference L791), which on this architecture means compress_ratio 0 (pure
    sliding window, no compressor/indexer) and a score-routed gate, and it holds
    no copy of the embedding or lm_head (reference L792-793 aliases the trunk's).

  * **Load path** — a checkpoint that ships ``mtp.0.*`` binds it through the
    ordinary ``sanitize`` -> ``quantize`` -> ``load_weights(strict=True)`` path
    with zero missing/extra keys, and one that does not (the published
    mlx-community conversions, which declare ``num_nextn_predict_layers: 1`` and
    ship no MTP tensor) drops it from the tree so the strict load still matches
    exactly and the runtime's degrade-to-autoregressive branch is reached
    unchanged.

Parity runs with ``swiglu_limit = 0``, like the trunk's own parity gate
(tests/test_deepseek_v4_parity.py) and like the reference goldens both were
captured at, so what this file measures is the block's algebra with the clamp
out of the way.  The clamp itself is now applied to the routed experts as well
as the shared one, and is gated -- with the branches driven into saturation, and
with this block's own MoE named as one of the covered sites -- in
tests/test_deepseek_v4_swiglu_clamp.py.
"""
import importlib.util
import math
import os
import sys

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

mx.set_default_device(mx.cpu)

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODEL = os.path.join(_HERE, "..", "mtplx", "models", "deepseek_v4.py")
_spec = importlib.util.spec_from_file_location("dsv4_mtp_undertest", _MODEL)
D = importlib.util.module_from_spec(_spec)
sys.modules["dsv4_mtp_undertest"] = D
_spec.loader.exec_module(D)


# --------------------------------------------------------------------------- #
# shrunk config: two trunk layers (so the MTP block is layer_id 2), a sequence
# longer than the sliding window, and every MTP-specific piece at a distinct size
# so a transposed/misrouted projection cannot broadcast its way to a pass.
# --------------------------------------------------------------------------- #
CFG = dict(
    vocab_size=61, hidden_size=32, num_hidden_layers=2, num_hash_layers=1,
    num_attention_heads=4, head_dim=16, qk_rope_head_dim=8,
    q_lora_rank=12, o_lora_rank=6, o_groups=2,
    moe_intermediate_size=10, n_routed_experts=6, num_experts_per_tok=2,
    index_n_heads=4, index_head_dim=16, index_topk=4,
    compress_ratios=[0, 4, 0], sliding_window=5,
    hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6, rms_norm_eps=1e-6,
    rope_theta=10000.0, routed_scaling_factor=1.5, scoring_func="sqrtsoftplus",
    swiglu_limit=0.0, num_nextn_predict_layers=1,
)
SEQ = 7


def _args(**over):
    c = dict(CFG)
    c.update(over)
    return D.ModelArgs(**c)


def n64(a):
    return np.asarray(a, dtype=np.float64)


def m2n(a):
    return np.array(a.astype(mx.float32)).astype(np.float64)


# --------------------------------------------------------------------------- #
# NumPy oracle (float64) transcribed from the reference
# --------------------------------------------------------------------------- #
def np_rmsnorm(x, w, eps):
    """reference RMSNorm.forward (model.py L191-196): fp32 var, then * weight."""
    v = (x**2).mean(-1, keepdims=True)
    return (x / np.sqrt(v + eps)) * w


def np_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def np_softmax(x, axis):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def np_sinkhorn(mixes, scale, base, hc, iters, eps):
    """kernel.py hc_split_sinkhorn_kernel L371-427."""
    pre = np_sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + eps
    post = 2.0 * np_sigmoid(mixes[..., hc: 2 * hc] * scale[1] + base[hc: 2 * hc])
    comb = mixes[..., 2 * hc:] * scale[2] + base[2 * hc:]
    comb = comb.reshape(*comb.shape[:-1], hc, hc)
    comb = np_softmax(comb, axis=-1) + eps
    comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + eps)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    return pre, post, comb


def np_hc_pre(x, fn, scale, base, hc, iters, hc_eps, norm_eps):
    """Block.hc_pre (model.py L673-681).

    Note the two epsilons are different fields in the reference: the rsqrt uses
    ``norm_eps`` and the Sinkhorn uses ``hc_eps``.  DeepSeek-V4-Flash ships both
    at 1e-6 so they coincide; they are kept separate here so the oracle stays a
    transcription rather than a copy of the port.
    """
    flat = x.reshape(*x.shape[:-2], -1)
    rsqrt = 1.0 / np.sqrt((flat**2).mean(-1, keepdims=True) + norm_eps)
    mixes = (flat @ fn.T) * rsqrt
    pre, post, comb = np_sinkhorn(mixes, scale, base, hc, iters, hc_eps)
    y = (pre[..., None] * x).sum(axis=-2)
    return y, post, comb


def np_hc_post(x, residual, post, comb):
    """Block.hc_post (L683-686): post*x + sum_j comb[j,k] * residual[j]."""
    term = post[..., None] * x[..., None, :]
    mixed = np.einsum("...jk,...jd->...kd", comb, residual)
    return term + mixed


def np_hc_head(x, fn, scale, base, hc, hc_eps, norm_eps):
    """ParallelHead.hc_head (L728-735): sigmoid pre-weights only, no Sinkhorn."""
    flat = x.reshape(*x.shape[:-2], -1)
    rsqrt = 1.0 / np.sqrt((flat**2).mean(-1, keepdims=True) + norm_eps)
    mixes = (flat @ fn.T) * rsqrt
    pre = np_sigmoid(mixes * scale + base) + hc_eps
    return (pre[..., None] * x).sum(axis=-2)


def np_inv_freq(dim, base):
    """precompute_freqs_cis (L199-229) with YaRN disabled (original_seq_len == 0),
    which is what Attention.__init__ L477-479 selects for a compress_ratio-0 layer."""
    return 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float64) / dim))


def np_rope(x, cos, sin, inverse=False):
    """apply_rotary_emb (L232-244): interleaved complex pairs on the last dim."""
    if inverse:
        sin = -sin
    x0, x1 = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = x0 * cos - x1 * sin
    out[..., 1::2] = x0 * sin + x1 * cos
    return out


def np_attention(x, P, c):
    """Attention.forward (L484-543) for a compress_ratio-0 (sliding-window) layer."""
    b, s, _ = x.shape
    nh, hd, rd = c["n_heads"], c["head_dim"], c["rope_head_dim"]
    eps, win = c["norm_eps"], c["window"]
    pos = np.arange(s, dtype=np.float64)
    ang = pos[:, None] * np_inv_freq(rd, c["rope_theta"])[None, :]
    cos, sin = np.cos(ang), np.sin(ang)

    qr = np_rmsnorm(x @ P["attn.wq_a.weight"].T, P["attn.q_norm.weight"], eps)
    q = (qr @ P["attn.wq_b.weight"].T).reshape(b, s, nh, hd)
    q = q / np.sqrt((q**2).mean(-1, keepdims=True) + eps)   # L498, per head
    q = np.concatenate(
        [q[..., :-rd], np_rope(q[..., -rd:], cos[None, :, None, :], sin[None, :, None, :])],
        axis=-1,
    )
    kv = np_rmsnorm(x @ P["attn.wkv.weight"].T, P["attn.kv_norm.weight"], eps)
    kv = np.concatenate(
        [kv[..., :-rd], np_rope(kv[..., -rd:], cos[None], sin[None])], axis=-1
    )

    scores = np.einsum("bshd,btd->bhst", q, kv) * (hd**-0.5)
    i = pos[:, None]
    j = pos[None, :]
    ok = (j <= i) & (j > i - win)                      # get_window_topk_idxs
    scores = np.where(ok[None, None], scores, -np.inf)
    # sparse_attn_kernel L345-348: the learned per-head sink logit joins the
    # softmax denominator only -- it contributes no value row.
    sink = P["attn.attn_sink"].reshape(1, nh, 1, 1)
    m = np.maximum(scores.max(-1, keepdims=True), sink)
    ex = np.exp(scores - m)
    den = ex.sum(-1, keepdims=True) + np.exp(sink - m)
    o = np.einsum("bhst,btd->bshd", ex / den, kv)
    o = np.concatenate(
        [o[..., :-rd],
         np_rope(o[..., -rd:], cos[None, :, None, :], sin[None, :, None, :], inverse=True)],
        axis=-1,
    )                                                   # L534, inverse rope
    # grouped output-LoRA (L536-542)
    g, r = c["o_groups"], c["o_lora_rank"]
    og = o.reshape(b, s, g, -1)
    wa = P["attn.wo_a.weight"].reshape(g, r, -1)
    return np.einsum("bsgp,grp->bsgr", og, wa).reshape(b, s, g * r) @ P["attn.wo_b.weight"].T


def np_moe(x, P, c):
    """Gate + Expert + MoE (L546-644), score-routed (the MTP layer is past
    n_hash_layers).  The reference multiplies the routing weight in *before* w2
    (L604-606); w2 is linear so that is the same number, computed differently."""
    b, s, d = x.shape
    flat = x.reshape(-1, d)
    scores = np.sqrt(np.log1p(np.exp(-np.abs(flat @ P["ffn.gate.weight"].T)))
                     + np.maximum(flat @ P["ffn.gate.weight"].T, 0.0))  # sqrt(softplus)
    biased = scores + P["ffn.gate.e_score_correction_bias"]
    idx = np.argsort(-biased, axis=-1, kind="stable")[:, : c["topk"]]
    w = np.take_along_axis(scores, idx, axis=-1)
    w = w / w.sum(-1, keepdims=True) * c["route_scale"]

    y = np.zeros_like(flat)
    for t in range(flat.shape[0]):
        for k in range(c["topk"]):
            e = idx[t, k]
            gate = flat[t] @ P["ffn.switch_mlp.gate_proj.weight"][e].T
            up = flat[t] @ P["ffn.switch_mlp.up_proj.weight"][e].T
            act = (gate / (1.0 + np.exp(-gate))) * up
            y[t] += (w[t, k] * act) @ P["ffn.switch_mlp.down_proj.weight"][e].T
    sg = flat @ P["ffn.shared_experts.gate_proj.weight"].T
    su = flat @ P["ffn.shared_experts.up_proj.weight"].T
    y = y + ((sg / (1.0 + np.exp(-sg))) * su) @ P["ffn.shared_experts.down_proj.weight"].T
    return y.reshape(b, s, d)


def np_mtp_block(h, ids, P, embed, head, c, *, order=("enorm", "hnorm"),
                 use_e_proj=True, stale_hc=False):
    """MTPBlock.forward (L757-766) -> draft logits.

    The keyword switches exist for the mutation gate at the bottom of this file;
    the default path is the faithful transcription.
    """
    hc, iters = c["hc"], c["iters"]
    heps, neps = c["hc_eps"], c["norm_eps"]

    e = embed[ids]                                          # [b, s, dim]
    e = np_rmsnorm(e, P["enorm.weight"], neps) if "enorm" in order else e
    xh = np_rmsnorm(h, P["hnorm.weight"], neps) if "hnorm" in order else h
    ep = (e @ P["e_proj.weight"].T)[:, :, None, :] if use_e_proj else 0.0
    x = ep + xh @ P["h_proj.weight"].T                      # [b, s, hc, dim]

    residual = x
    y, post, comb = np_hc_pre(x, P["attn_hc.fn"], P["attn_hc.scale"], P["attn_hc.base"],
                              hc, iters, heps, neps)
    y = np_rmsnorm(y, P["attn_norm.weight"], neps)
    y = np_attention(y, P, c)
    x = np_hc_post(y, residual, post, comb)

    residual = residual if stale_hc else x                  # mutation: stale stream
    y, post, comb = np_hc_pre(x, P["ffn_hc.fn"], P["ffn_hc.scale"], P["ffn_hc.base"],
                              hc, iters, heps, neps)
    y = np_rmsnorm(y, P["ffn_norm.weight"], neps)
    y = np_moe(y, P, c)
    x = np_hc_post(y, residual, post, comb)

    z = np_hc_head(x, P["hc_head.fn"], P["hc_head.scale"], P["hc_head.base"],
                   hc, heps, neps)
    z = np_rmsnorm(z, P["norm.weight"], neps)
    return z @ head.T


# --------------------------------------------------------------------------- #
def _build(seed=0):
    """Seeded model + the oracle's view of the same parameters."""
    args = _args()
    rng = np.random.default_rng(seed)
    model = D.Model(args)

    flat = dict(tree_flatten(model.parameters()))
    new = {}
    for k, v in flat.items():
        if k.endswith("tid2eid"):
            new[k] = mx.array(
                rng.integers(0, args.n_routed_experts,
                             size=v.shape).astype(np.int32))
        else:
            # small values keep sqrt(softplus) and the Sinkhorn away from
            # saturation, where an oracle mismatch could hide behind a plateau
            new[k] = mx.array((rng.standard_normal(v.shape) * 0.25).astype(np.float32))
    model.update(_unflatten(new))
    mx.eval(model.parameters())

    P = {k[len("mtp.0."):]: m2n(v) for k, v in new.items() if k.startswith("mtp.0.")}
    embed = m2n(new["model.embed_tokens.weight"])
    head = m2n(new["lm_head.weight"])
    c = dict(
        n_heads=args.num_attention_heads, head_dim=args.head_dim,
        rope_head_dim=args.qk_rope_head_dim, norm_eps=args.rms_norm_eps,
        win=args.window_size, window=args.window_size, rope_theta=args.rope_theta,
        o_groups=args.o_groups, o_lora_rank=args.o_lora_rank,
        topk=args.num_experts_per_tok, route_scale=args.routed_scaling_factor,
        hc=args.hc_mult, iters=args.hc_sinkhorn_iters, hc_eps=args.hc_eps,
    )
    ids = rng.integers(0, args.vocab_size, size=(1, SEQ)).astype(np.int32)
    h = (rng.standard_normal((1, SEQ, args.hc_mult, args.hidden_size)) * 0.5)
    return args, model, P, embed, head, c, ids, h


def _unflatten(flat):
    from mlx.utils import tree_unflatten
    return tree_unflatten(list(flat.items()))


def _run(model, h, ids):
    out = model.mtp_forward(mx.array(h.astype(np.float32)), mx.array(ids))
    mx.eval(out)
    return m2n(out)


# --------------------------------------------------------------------------- #
# 1. numerical parity
# --------------------------------------------------------------------------- #
def test_mtp_block_matches_reference_oracle():
    args, model, P, embed, head, c, ids, h = _build()
    got = _run(model, h, ids)
    ref = np_mtp_block(h, ids, P, embed, head, c)
    mad = float(np.max(np.abs(got - ref)))
    scale = float(np.max(np.abs(ref)))
    assert mad / scale < 1e-5, f"draft logits diverge: max_abs={mad:.3e} scale={scale:.3e}"
    assert np.array_equal(got.argmax(-1), ref.argmax(-1)), "draft argmax disagrees"


def test_mtp_parity_holds_on_a_second_seed():
    """One seed can pass by luck on a routing boundary; two cannot."""
    for seed in (1, 2):
        args, model, P, embed, head, c, ids, h = _build(seed)
        got = _run(model, h, ids)
        ref = np_mtp_block(h, ids, P, embed, head, c)
        rel = float(np.max(np.abs(got - ref))) / float(np.max(np.abs(ref)))
        assert rel < 1e-5, f"seed {seed}: max_rel={rel:.3e}"


def test_mtp_input_fusion_is_the_reference_shape():
    """``e_proj(enorm(embed(ids)))`` broadcasts over the hc copies; ``h_proj``
    does not (model.py L763).  A per-copy embedding term would change every
    copy identically and is caught by the projection being applied to the
    hc-shaped tensor only."""
    args, model, P, embed, head, c, ids, h = _build()
    blk = model.mtp[0]
    e = blk.enorm(model.model.embed_tokens(mx.array(ids)))
    fused = blk.e_proj(e)[:, :, None, :] + blk.h_proj(blk.hnorm(mx.array(h.astype(np.float32))))
    mx.eval(fused)
    ref_e = np_rmsnorm(embed[ids], P["enorm.weight"], args.rms_norm_eps)
    ref = (ref_e @ P["e_proj.weight"].T)[:, :, None, :] + \
        np_rmsnorm(h, P["hnorm.weight"], args.rms_norm_eps) @ P["h_proj.weight"].T
    assert m2n(fused).shape == (1, SEQ, args.hc_mult, args.hidden_size)
    assert np.max(np.abs(m2n(fused) - ref)) < 1e-5


# --------------------------------------------------------------------------- #
# 2. structure
# --------------------------------------------------------------------------- #
def test_mtp_block_is_a_body_block_at_layer_n_layers():
    """Reference L791 builds ``MTPBlock(n_layers + i)``, so compress_ratios and
    n_hash_layers are read at index 43 on the real config: sliding-window
    attention and a score-routed gate."""
    args = _args()
    model = D.Model(args)
    blk = model.mtp[0]
    assert blk.attn.layer_id == args.num_hidden_layers
    assert blk.attn.compress_ratio == 0
    assert not hasattr(blk.attn, "compressor")
    assert not hasattr(blk.attn, "indexer")
    assert blk.ffn.gate.hash is False
    keys = {k for k, _ in tree_flatten(blk.parameters())}
    assert "ffn.gate.e_score_correction_bias" in keys
    assert "ffn.gate.tid2eid" not in keys


def test_mtp_block_holds_no_copy_of_embedding_or_lm_head():
    """Reference L792-793 aliases the trunk's embed/head onto the block; the port
    passes them in instead, so the 129280-row tensors are never duplicated."""
    model = D.Model(_args())
    keys = {k for k, _ in tree_flatten(model.mtp[0].parameters())}
    assert not any("embed" in k or "lm_head" in k for k in keys)
    n_mtp = len(keys)
    total = {k for k, _ in tree_flatten(model.parameters())}
    assert sum(k.startswith("mtp.0.") for k in total) == n_mtp


def test_mtp_layer_ratio_falls_back_when_config_omits_the_entry():
    """The shipped config carries 44 ratios for 43 layers (trailing 0).  A config
    trimmed to the trunk length must not IndexError out of Attention.__init__."""
    model = D.Model(_args(compress_ratios=[0, 4]))
    assert model.mtp[0].attn.compress_ratio == 0
    # the trunk's own ratios are untouched by the pad
    assert [layer.attn.compress_ratio for layer in model.layers] == [0, 4]


def test_mtp_cache_is_separate_and_window_only():
    model = D.Model(_args())
    trunk, draft = model.make_cache(), model.make_mtp_cache()
    assert len(draft) == 1 and len(trunk) == model.args.num_hidden_layers
    assert draft[0] is not trunk[0]
    assert draft[0].compress_ratio == 0
    assert draft[0].window_size == model.args.window_size
    assert draft[0].n_compressed == 0


def test_hc_hidden_and_collapse_recompose_into_the_plain_forward():
    """The trunk split the draft block needs must not change what ``__call__``
    returns, or every existing gate on this backend is measuring a different
    model than serving runs."""
    _, model, _, _, _, _, ids, _ = _build()
    ids = mx.array(ids)
    a = model(ids)
    b = model.logits_from_hc_hidden(model.hc_hidden(ids))
    mx.eval(a, b)
    assert bool(mx.array_equal(a, b))


def test_shared_expert_applies_the_swiglu_clamp():
    """``DeepseekV4MLP`` implements the reference's ``swiglu_limit`` clamp
    (Expert.forward L600-602) for the shared expert.  The routed experts get the
    same clamp from ``ClampedSwiGLU``; that half is gated in
    tests/test_deepseek_v4_swiglu_clamp.py."""
    args = _args(swiglu_limit=0.5)
    mlp = D.DeepseekV4MLP(args, args.moe_intermediate_size)
    mlp.gate_proj.weight = mx.full(mlp.gate_proj.weight.shape, 8.0)
    mlp.up_proj.weight = mx.full(mlp.up_proj.weight.shape, 8.0)
    mlp.down_proj.weight = mx.zeros(mlp.down_proj.weight.shape)
    x = mx.ones((1, 1, args.hidden_size))
    pre_gate = mx.minimum(mlp.gate_proj(x), 0.5)
    pre_up = mx.clip(mlp.up_proj(x), -0.5, 0.5)
    mx.eval(pre_gate, pre_up)
    assert float(mx.max(pre_gate).item()) == 0.5
    assert float(mx.max(pre_up).item()) == 0.5
    assert float(mx.max(mx.abs(mlp(x))).item()) == 0.0


# --------------------------------------------------------------------------- #
# 3. load path
# --------------------------------------------------------------------------- #
def _synthetic_weights(args, with_mtp: bool):
    src = D.Model(args if with_mtp else _args(num_nextn_predict_layers=0))
    return {k: mx.zeros(v.shape, v.dtype)
            for k, v in tree_flatten(src.parameters())}


def test_missing_mtp_weights_degrade_to_a_trunk_only_tree():
    """The published mlx-community conversions declare ``num_nextn_predict_layers:
    1`` and ship no ``mtp.*`` tensor; that must load, not raise 58 missing keys."""
    args = _args()
    model = D.Model(args)
    assert model.has_mtp
    weights = _synthetic_weights(args, with_mtp=False)
    weights = model.sanitize(weights)
    assert not model.has_mtp
    assert not [k for k, _ in tree_flatten(model.parameters()) if k.startswith("mtp")]
    model.load_weights(list(weights.items()), strict=True)   # zero missing/extra
    out = model(mx.array([[1, 2, 3]]))
    mx.eval(out)
    assert out.shape == (1, 3, args.vocab_size)
    with pytest.raises(RuntimeError, match="no MTP block"):
        model.mtp_forward(mx.zeros((1, 3, args.hc_mult, args.hidden_size)),
                          mx.array([[1, 2, 3]]))


def test_present_mtp_weights_bind_with_zero_missing_or_extra():
    args = _args()
    model = D.Model(args)
    weights = _synthetic_weights(args, with_mtp=True)
    assert any(k.startswith("mtp.") for k in weights)
    weights = model.sanitize(weights)
    assert model.has_mtp
    tree = {k for k, _ in tree_flatten(model.parameters())}
    assert tree == set(weights), {
        "missing_from_weights": sorted(tree - set(weights))[:8],
        "extra_in_weights": sorted(set(weights) - tree)[:8],
    }
    model.load_weights(list(weights.items()), strict=True)


def test_mtp_binds_through_the_quantized_load_path():
    """The merged checkpoint ships the draft head quantized (affine 8-bit /
    group_size 64, declared per-path in ``config["quantization"]``), so the tree
    must survive ``nn.quantize`` with the same predicate mlx-lm applies."""
    args = _args(hidden_size=64, q_lora_rank=64, o_lora_rank=64, head_dim=32,
                 num_attention_heads=4, moe_intermediate_size=64,
                 qk_rope_head_dim=8, o_groups=2)
    model = D.Model(args)
    weights = _synthetic_weights(args, with_mtp=True)
    weights = model.sanitize(weights)
    stems = {f"mtp.0.{s}" for s in (
        "attn.wq_a", "attn.wq_b", "attn.wkv", "attn.wo_a", "attn.wo_b",
        "e_proj", "h_proj", "ffn.switch_mlp.gate_proj", "ffn.switch_mlp.up_proj",
        "ffn.switch_mlp.down_proj", "ffn.shared_experts.gate_proj",
        "ffn.shared_experts.up_proj", "ffn.shared_experts.down_proj")}
    nn.quantize(model, group_size=64, bits=8,
                class_predicate=lambda p, m: p in stems and hasattr(m, "to_quantized"))
    tree = {k for k, _ in tree_flatten(model.parameters())}
    for stem in stems:
        assert f"{stem}.scales" in tree and f"{stem}.biases" in tree, stem
    # rebuild the weight dict the way the checkpoint stores it and bind strictly
    qw = {k: mx.zeros(v.shape, v.dtype) for k, v in tree_flatten(model.parameters())}
    model.load_weights(list(qw.items()), strict=True)
    assert isinstance(model.mtp[0].e_proj, nn.QuantizedLinear)
    assert model.mtp[0].e_proj.bits == 8 and model.mtp[0].e_proj.group_size == 64


# --------------------------------------------------------------------------- #
# 4. the real merged checkpoint (index + config only -- no weights are read)
# --------------------------------------------------------------------------- #
def _merged_dir():
    for cand in (os.environ.get("MTPLX_DSV4_MTP_MODEL"),
                 os.path.expanduser("~/models/DeepSeek-V4-Flash-2bit-DQ-mtp")):
        if cand and os.path.exists(os.path.join(cand, "model.safetensors.index.json")):
            return cand
    return None


def _vanilla_dir():
    import glob as _glob
    for hit in sorted(_glob.glob(os.path.expanduser(
            "~/.cache/huggingface/hub/models--mlx-community--DeepSeek-V4-Flash-2bit-DQ"
            "/snapshots/*/"))):
        if os.path.exists(os.path.join(hit, "model.safetensors.index.json")):
            return hit
    return None


_MERGED = _merged_dir()
_VANILLA = _vanilla_dir()
_needs_merged = pytest.mark.skipif(
    _MERGED is None, reason="merged DeepSeek-V4 MTP model dir not built")


@_needs_merged
def test_merged_checkpoint_mtp_keys_match_the_module_tree_exactly():
    """Structural counts from the real config, tiny per-unit dims, so the key
    NAMES are the real ones: the shipped ``mtp.0.*`` set must equal what the tree
    expects once the declared per-path quantization is expanded -- zero missing,
    zero extra."""
    import json
    cfg = json.load(open(os.path.join(_MERGED, "config.json")))
    wmap = json.load(open(os.path.join(
        _MERGED, "model.safetensors.index.json")))["weight_map"]
    ckpt = {k for k in wmap if k.startswith("mtp.")}
    assert ckpt, "merged dir ships no mtp.* tensors"
    assert cfg["num_nextn_predict_layers"] >= 1

    args = _args(num_hidden_layers=cfg["num_hidden_layers"],
                 num_hash_layers=cfg["num_hash_layers"],
                 n_routed_experts=cfg["n_routed_experts"],
                 o_groups=cfg["o_groups"], compress_ratios=cfg["compress_ratios"],
                 num_nextn_predict_layers=cfg["num_nextn_predict_layers"])
    model = D.Model(args)
    quantizable = {n for n, m in model.named_modules() if hasattr(m, "to_quantized")}
    q = cfg["quantization"]
    expected = set()
    for path, _ in tree_flatten(model.parameters()):
        if not path.startswith("mtp."):
            continue
        if path.endswith(".weight"):
            stem = path[: -len(".weight")]
            if stem in quantizable and stem in q:
                assert q[stem]["bits"] >= 8, (
                    f"{stem} is quantized below the MTP precision floor: {q[stem]}")
                expected |= {f"{stem}.weight", f"{stem}.scales", f"{stem}.biases"}
                continue
        expected.add(path)
    assert expected == ckpt, {
        "missing_from_ckpt": sorted(expected - ckpt)[:8],
        "extra_in_ckpt": sorted(ckpt - expected)[:8],
    }
    assert len({wmap[k] for k in ckpt}) == 1, "mtp weights split across shards"


@_needs_merged
def test_merged_checkpoint_is_seen_as_mtp_bearing_by_the_degrade_guard():
    """The c54a2d1 guard decides raise-vs-degrade off the shard index.  The merged
    dir must read as weights-present so a genuine injection failure still raises;
    the unmodified mlx-community snapshot must still read as weights-absent so it
    keeps degrading to autoregressive."""
    from mtplx.artifacts import mtp_weights_present_on_disk
    assert mtp_weights_present_on_disk(_MERGED) is True
    if _VANILLA is not None:
        assert mtp_weights_present_on_disk(_VANILLA) is False


@_needs_merged
def test_merged_dir_did_not_mutate_the_hf_cache_snapshot():
    """The merge hardlinks the trunk; the source snapshot must be untouched and
    must still be missing its MTP block."""
    if _VANILLA is None:
        pytest.skip("mlx-community snapshot not in the HF cache")
    import json
    src = json.load(open(os.path.join(
        _VANILLA, "model.safetensors.index.json")))["weight_map"]
    assert not any(k.startswith("mtp.") for k in src)
    src_cfg = json.load(open(os.path.join(_VANILLA, "config.json")))
    assert not any(k.startswith("mtp.") for k in src_cfg.get("quantization", {}))


# --------------------------------------------------------------------------- #
# 5. mutation gate -- each of these must make the parity test fail
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kw", [
    pytest.param(dict(order=("hnorm",)), id="dropped-enorm"),
    pytest.param(dict(order=("enorm",)), id="dropped-hnorm"),
    pytest.param(dict(use_e_proj=False), id="dropped-e-projection"),
    pytest.param(dict(stale_hc=True), id="stale-hc-stream"),
])
def test_oracle_mutations_are_detected(kw):
    """Sensitivity check on the parity gate itself: a transcription error in the
    norm order, the embedding projection, or the Hyper-Connection residual stream
    must move the logits well past the 1e-5 bound, not hide inside it."""
    args, model, P, embed, head, c, ids, h = _build()
    got = _run(model, h, ids)
    bad = np_mtp_block(h, ids, P, embed, head, c, **kw)
    rel = float(np.max(np.abs(got - bad))) / float(np.max(np.abs(got)))
    assert rel > 1e-3, f"mutation {kw} not detected (max_rel={rel:.3e})"
