#!/usr/bin/env python3
"""Build ``ngram-hotness.npy``: the PLE sidecar rows a corpus actually gathers.

CPU only.  No MLX, no model weights, no GPU: the three hash buffers are read
straight out of the checkpoint with ``numpy.memmap``, the row arithmetic is
COMPILED OUT OF THE SHIPPED SOURCE (``_ngram_rows_np`` in
``mtplx/models/qwen4_exp.py``) rather than copied here, and the only heavy
dependency is the pack's own tokenizer.

Why it exists
-------------
``MTPLX_NGRAM_PREWARM`` reads the 29.8 GiB n-gram table into the page cache at
model load, but on a 128 GB Mac carrying ~85 GB of wired weights the budget is
usually a fraction of the table.  At a given budget every order warms the same
number of PAGES -- rows are ~100 B and hash-scattered, so a row costs a whole
16 KiB page either way -- so the only question is WHICH pages.  This produces
the answer: row ids in descending gather frequency, which
``ple_row_gather.load_hotness_order`` reads.

Nothing on disk already answers it.  The fable receipts carry only aggregate
`ple_hot_rows` hit/miss counts, the server request logs
(``~/.mtplx/logs/request-log-*.jsonl``) record token COUNTS and never ids,
``mtplx/request_capture.py`` would record ids but is off by default and has
never been run on this box, and the ``k20-rows-*.npz`` files hold 1,113
DECODE steps' candidate tokens, not prompts.  So the frequencies are produced
the only way left: hash a corpus through the model's own n-gram hash.

Usage
-----
    python scripts/fable/ngram_row_hotness.py \\
        --model ~/.mtplx/models/Youssofal--Qwen3.8-Flash-Next-MTPLX-Optimized-Speed \\
        --prompt-tokens 65536 --top-rows 8000000

Writes ``<model>/ngram-hotness.npy`` (int64, most-gathered first) unless
``--out`` says otherwise, and prints a coverage line: how many distinct rows
the corpus touched, what share of the table that is, and what share of all
gathers the written head accounts for.
"""

from __future__ import annotations

import argparse
import ast
import json
import struct
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_SOURCE = REPO_ROOT / "mtplx" / "models" / "qwen4_exp.py"

#: Tensor names of the three hash buffers.  They are registered buffers, so
#: the CHECKPOINT's values are authoritative -- `NGramEmbedding.__init__`
#: derives them from the config and `load_weights` then overwrites them.
#: Deriving them here instead would silently diverge on any pack whose
#: buffers were regenerated.
BUFFERS = {
    "layer_multipliers": "layer_multipliers",
    "ngram_heads_vocab_sizes": "ngram_heads_vocab_sizes",
    "ngram_heads_offsets": "ngram_heads_offsets",
}

_SAFETENSORS_DTYPES = {
    "I64": np.int64,
    "I32": np.int32,
    "U32": np.uint32,
    "BF16": np.uint16,
    "F16": np.uint16,
}


def shipped_rows_fn():
    """``_ngram_rows_np`` compiled out of the shipped model source.

    Extracted rather than vendored: this script's whole value is that its row
    ids are the ones the runtime will gather, and a copied hash drifts the
    first time the real one changes.  ``qwen4_exp`` imports MLX at module
    scope, so the function is lifted out with ``ast`` instead of imported.
    """

    source = MODEL_SOURCE.read_text("utf-8")
    node = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "_ngram_rows_np"
    )
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    scope: dict = {}
    exec(compile(module, "<qwen4_exp>", "exec"), scope)
    return scope["_ngram_rows_np"]


def read_safetensors_header(path: Path):
    with open(path, "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
    return header, 8 + size


def read_tensor(path: Path, info, data_start: int) -> np.ndarray:
    dtype = _SAFETENSORS_DTYPES[info["dtype"]]
    offset = data_start + int(info["data_offsets"][0])
    shape = tuple(int(n) for n in info["shape"])
    return np.array(
        np.memmap(path, mode="r", dtype=dtype, offset=offset, shape=shape)
    )


def load_hash_constants(model: Path):
    """``(mult, sizes, offs)`` read from the checkpoint, numpy only."""

    index_path = model / "model.safetensors.index.json"
    if not index_path.exists():
        raise SystemExit(f"no model.safetensors.index.json under {model}")
    weight_map = json.loads(index_path.read_text("utf-8"))["weight_map"]
    found: dict[str, np.ndarray] = {}
    for name, shard in weight_map.items():
        for key, suffix in BUFFERS.items():
            if name.endswith("ple_embedding." + suffix):
                shard_path = model / shard
                header, data_start = read_safetensors_header(shard_path)
                found[key] = read_tensor(shard_path, header[name], data_start)
    missing = sorted(set(BUFFERS) - set(found))
    if missing:
        raise SystemExit(f"checkpoint has no PLE hash buffers: {missing}")
    return (
        found["layer_multipliers"].astype(np.int64),
        found["ngram_heads_vocab_sizes"].astype(np.int64),
        found["ngram_heads_offsets"].astype(np.int64),
    )


def load_text_config(model: Path) -> dict:
    config = json.loads((model / "config.json").read_text("utf-8"))
    return config.get("text_config", config)


def table_row_count(model: Path) -> int | None:
    table = model / "ngram-table.safetensors"
    if not table.exists():
        return None
    header, _ = read_safetensors_header(table)
    meta = header.get("__metadata__", {})
    if "rows" in meta:
        return int(meta["rows"])
    info = header.get("ngram.weight")
    return int(info["shape"][0]) if info else None


# ---------------------------------------------------------------------------
# The corpus
# ---------------------------------------------------------------------------


def load_tokenizer(model: Path):
    """The pack's own tokenizer, CPU only.  No MLX, no model weights."""

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(model), trust_remote_code=False)


def _jsonl_texts(path: Path) -> list[str]:
    texts = []
    for line in path.read_text("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            texts.append(line)
            continue
        if isinstance(row, str):
            texts.append(row)
            continue
        for key in ("prompt", "text", "content", "instruction", "problem"):
            value = row.get(key)
            if isinstance(value, str) and value:
                texts.append(value)
                break
    return texts


def corpus_texts(paths) -> list[str]:
    texts: list[str] = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            print(f"[hotness] skipping missing corpus file {path}", file=sys.stderr)
            continue
        if path.suffix == ".jsonl":
            texts.extend(_jsonl_texts(path))
        else:
            texts.append(path.read_text("utf-8"))
    return [text for text in texts if text.strip()]


def prompt_id_streams(tokenizer, texts, *, prompt_tokens: int):
    """Token-id streams of ``prompt_tokens`` each, tiled from the corpus.

    Tiling mirrors what the fable driver does to hit an exact prompt length
    (``build_exact_coding_prompt_ids``): a long-context prompt is the same
    context repeated, and repetition is exactly what makes n-gram rows hot,
    so the frequencies this produces are the ones a long prompt really has.
    """

    for text in texts:
        ids = np.asarray(tokenizer.encode(text), dtype=np.int64).reshape(-1)
        if ids.size == 0:
            continue
        if prompt_tokens and ids.size < prompt_tokens:
            repeats = -(-prompt_tokens // ids.size)
            ids = np.tile(ids, repeats)[:prompt_tokens]
        elif prompt_tokens:
            ids = ids[:prompt_tokens]
        yield ids


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------


def accumulate(state, rows: np.ndarray):
    """Merge one prompt's row ids into a running ``(ids, counts)`` pair.

    Vectorised, not a dict loop: a 64K-token prompt hashes to ~1M rows and
    ~1M distinct ids, and a per-id Python round trip turns a whole corpus
    into minutes of interpreter time for arithmetic NumPy does in one pass.
    """

    uniq, freq = np.unique(rows, return_counts=True)
    if state is None:
        return uniq.astype(np.int64), freq.astype(np.int64)
    ids = np.concatenate([state[0], uniq])
    counts = np.concatenate([state[1], freq.astype(np.int64)])
    order = np.argsort(ids, kind="stable")
    ids, counts = ids[order], counts[order]
    first = np.ones(ids.size, dtype=bool)
    first[1:] = ids[1:] != ids[:-1]
    starts = np.flatnonzero(first)
    return ids[first], np.add.reduceat(counts, starts)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument(
        "--prompts",
        nargs="*",
        default=[],
        help="Text or .jsonl corpus files. Defaults to the fable fixtures.",
    )
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=65536,
        help="Tile each corpus text to exactly this many tokens (0 = as-is).",
    )
    parser.add_argument(
        "--top-rows",
        type=int,
        default=8_000_000,
        help="How many rows to write, most-gathered first (0 = all seen).",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    model = args.model.expanduser().resolve()
    config = load_text_config(model)
    ngram_size = int(config["ngram_size"])
    heads_per_ngram = int(config["heads_per_ngram"])
    eos = config.get("eos_token_id", config.get("eos_id"))
    eos_id = int(eos[0] if isinstance(eos, list) else eos)

    mult, sizes, offs = load_hash_constants(model)
    rows_np = shipped_rows_fn()
    total_rows = table_row_count(model)

    paths = args.prompts or [
        str(p)
        for p in (
            REPO_ROOT.parent
            / "qwen4-queue-first-draft"
            / "mtplx"
            / "benchmarks"
            / "prompts"
            / "qwen38_generation_context.py",
            Path.home()
            / "Library/Caches/evalplus/HumanEvalPlus-v0.1.10.jsonl",
        )
    ]
    texts = corpus_texts(paths)
    if not texts:
        raise SystemExit(f"no usable corpus text in {paths}")

    tokenizer = load_tokenizer(model)
    state = None
    prompts = 0
    gathers = 0
    context = ngram_size - 1
    for ids in prompt_id_streams(
        tokenizer, texts, prompt_tokens=max(0, args.prompt_tokens)
    ):
        prev = np.full((1, context), eos_id, dtype=np.int64)
        rows, _hist = rows_np(
            ids.reshape(1, -1),
            prev,
            mult=mult,
            sizes=sizes,
            offs=offs,
            eos=eos_id,
            ngram_size=ngram_size,
            heads_per_ngram=heads_per_ngram,
        )
        flat = rows.reshape(-1)
        gathers += int(flat.size)
        state = accumulate(state, flat)
        prompts += 1

    if state is None or state[0].size == 0:
        raise SystemExit("corpus produced no rows")
    ids, freq = state
    order = np.argsort(-freq, kind="stable")
    ids, freq = ids[order], freq[order]
    head = ids if args.top_rows <= 0 else ids[: args.top_rows]
    covered = int(freq[: head.size].sum())

    out = args.out or (model / "ngram-hotness.npy")
    np.save(str(out), head)

    share = "" if not total_rows else f" ({head.size / total_rows:.3%} of {total_rows:,} table rows)"
    print(
        f"[hotness] {prompts} prompts, {gathers:,} gathers, "
        f"{ids.size:,} distinct rows; wrote {head.size:,} rows{share} to {out}"
    )
    print(
        f"[hotness] coverage: the written head accounts for "
        f"{covered / gathers:.2%} of this corpus's gathers; "
        f"a 16 KiB page holds ~{16384 // 100} rows, so warming them costs "
        f"~{head.size * 16384 / 1024**3:.1f} GiB of page cache worst case"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
