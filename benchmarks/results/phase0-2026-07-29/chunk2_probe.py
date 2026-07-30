"""Phase-0 chunk 2: direct AR baseline + stock-forward T_V(M) curve.

Measures, per model body:
  (a) steady AR decode tok/s at ~1k context (stock mlx_lm path, no engine,
      no MTP sidecar — the 'stock AR' baseline; the engine-path AR arm runs
      separately via mtplx --generation-mode ar),
  (b) T_V(M): wall time of a [1, M] verify-shaped forward at a ~1k-token
      cache, M = 1..16, stock MLX kernels (no NAX — engine-only). This bounds
      verify-row physics; the production-path variant is a later arm.
"""
import json, sys, time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache


def steady_ar(model, tok, prompt_tokens, n_gen=512, reps=2):
    rates = []
    for _ in range(reps):
        cache = make_prompt_cache(model)
        # prefill
        logits = model(mx.array([prompt_tokens]), cache=cache)
        mx.eval(logits)
        y = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(y)
        # warmup decode steps
        for _ in range(8):
            logits = model(y[None], cache=cache)
            y = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(y)
        t0 = time.perf_counter()
        for _ in range(n_gen):
            logits = model(y[None], cache=cache)
            y = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(y)
        dt = time.perf_counter() - t0
        rates.append(n_gen / dt)
    return rates


def t_v_curve(model, prompt_tokens, reps=6):
    cache = make_prompt_cache(model)
    logits = model(mx.array([prompt_tokens]), cache=cache)
    mx.eval(logits)
    curve = {}
    dummy = [42] * 16
    for m in range(1, 17):
        rows = mx.array([dummy[:m]])
        # warmup at this M (kernel selection/compile)
        for _ in range(2):
            out = model(rows, cache=cache)
            mx.eval(out)
            trim_prompt_cache(cache, m)
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            out = model(rows, cache=cache)
            mx.eval(out)
            times.append((time.perf_counter() - t0) * 1000)
            trim_prompt_cache(cache, m)
        times.sort()
        curve[m] = {"ms_median": round(times[len(times) // 2], 3),
                    "ms_min": round(times[0], 3), "ms_max": round(times[-1], 3)}
    return curve


def main():
    model_path = sys.argv[1]
    tag = sys.argv[2]
    no_tok = len(sys.argv) > 3 and sys.argv[3] == "notok"
    if no_tok:
        # Timing needs token IDs, not text: skip the tokenizer entirely
        # (transformers' rope validation rejects Laguna's YaRN config).
        from mlx_lm.utils import load_model

        loaded = load_model(Path(model_path))
        model = loaded[0] if isinstance(loaded, tuple) else loaded
        tok = None  # unused by the timing paths
        ptok = [(1000 + i * 7) % 90000 for i in range(1000)]
    else:
        model, tok = load(model_path)
        prompt = "def fibonacci(n):\n    " * 80  # ~1k tokens of code-ish context
        ptok = tok.encode(prompt)[:1000]

    ar_rates = steady_ar(model, tok, ptok)
    curve = t_v_curve(model, ptok)

    result = {
        "tag": tag, "model": model_path,
        "condition": "stock mlx_lm forward, no engine, no NAX, greedy, ~1k ctx",
        "ar_tok_s_runs": [round(r, 2) for r in ar_rates],
        "ar_tok_s_best": round(max(ar_rates), 2),
        "t_v_ms_by_M": curve,
        "t_v_ratio_M16_over_M1": round(
            curve[16]["ms_median"] / curve[1]["ms_median"], 3),
        "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2),
    }
    out = Path(__file__).parent / f"chunk2_{tag}.json"
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
