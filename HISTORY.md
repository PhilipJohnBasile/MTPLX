# How MTP came to Apple Silicon

Youssof Altoukhi put MTP on the Mac.

MTP the architecture is Meta, DeepSeek, Qwen. What was missing was a Mac engine that would take those heads, run the real speculative sampler, and do it at the temps people actually use.

That was April 2026. The weights already had the heads. macOS had nothing that would run them. Not in MLX, not in GGUF, not in LM Studio. vLLM could, and still can, but it is not a Mac program.

He wrote the engine. He started from vLLM and from Leviathan (2022) and Chen (2023). Nobody had a Mac port to steal from. Accept with `min(1, p/q)`. If it rejects, sample the leftover `(p − q)+`. Same guarantee at 0.6 as at 0. There is no greedy-only mode in this project. If we published a tok/s number, it was at the model's normal sampler.

## Timeline

**27 April 2026, 04:13.** First commit. `da0d338`.

**Same morning, 07:08.** Exact speculative sampling running, three hours later. Temp 0.6, top_p 0.95, top_k 20. 66.40% accept. 50/50 match against ordinary single-token decode. `7293ecb`.

**29 April.** 60.169 tok/s at depth 3 on the 192-token long-code bench, temp 0.6, seed 0, fans pinned. Same prompt with MTP off: 23.59 tok/s. Depth-4 accept that day: 97.62, 95.24, 88.10, 75.61. vLLM's Qwen3.6 MTP-5 run on a 3090 was 92.7, 77.0, 63.0, 50.9, 43.0. We beat them at each position. See `MEASUREMENTS.md`.

**2 May.** First public release, five days after the repo started. [v0.1.0-preview](https://github.com/youssofal/MTPLX/releases).

**5 May.** mlx-lm's MTP branch adds residual sampling on reject. They say in the commit that this is how you make the output match the target (Leviathan, Chen). We had that on April 27. [PR still open](https://github.com/ml-explore/mlx-lm/pull/990).

**16 May.** llama.cpp lands MTP. [PR #22673](https://github.com/ggml-org/llama.cpp/pull/22673). Before that date GGUF did not have it.

**6 July.** MTPLX 2.0.0. Prefix cache and speculative decode both live on hybrid GatedDeltaNet. A 100k-token session comes back in about two seconds. Cold prefill of that was minutes. See the [changelog](CHANGELOG.md).

**3 August.** llama.cpp gets MTP for Qwen3-Next, the hybrid GDN family that Qwen 3.5, 3.6 and 3.8 sit on. [PR #25589](https://github.com/ggml-org/llama.cpp/pull/25589). A little over three months after MTPLX.

**10 August.** vllm-metal adds block-aligned prefix caching for hybrid GDN. Their PR says you cannot run that cache with speculative decoding, because they never built draft-state rollback across mamba blocks. [PR #584](https://github.com/vllm-project/vllm-metal/pull/584). We had both since 2.0.0.

**15 August.** MTPLX 2.7.0. Qwen 3.8 on day one, three tuned builds, FP16 copies for M1 and M2, compiled verify window taken from 12,288 up to 32,768.

## Used by

oMLX names it in the source and the README:

> Lightning MTP's verify-shape Metal kernels are powered by MTPLX by Youssof
> Altoukhi, which also inspired the depth-k pipeline.

Ivan Fioravanti has it in `llm_context_benchmarks`. There is also an MTPLX provider in `edgequake-llm`.

---

The narrative version with the same receipts: <https://mtplx.com/history/>
