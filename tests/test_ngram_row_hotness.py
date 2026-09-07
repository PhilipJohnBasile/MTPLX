"""tests/ngram_row_hotness.py: the hotness file, built CPU-only.

No MLX, no model weights, no tokenizer download: the checkpoint reader is
exercised against a safetensors shard written here, and the row arithmetic is
the SHIPPED ``_ngram_rows_np`` lifted out of ``mtplx/models/qwen4_exp.py`` by
the script itself -- which is the property most worth testing, because a
vendored copy of that hash would drift and produce a hotness file for rows the
runtime never gathers.
"""

from __future__ import annotations

import importlib.util
import json
import struct
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tests" / "ngram_row_hotness.py"


@pytest.fixture(scope="module")
def hotness():
    spec = importlib.util.spec_from_file_location("ngram_row_hotness", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


NGRAM_SIZE, HEADS = 3, 8
HEAD_COUNT = (NGRAM_SIZE - 1) * HEADS


def _write_shard(path: Path, tensors: dict[str, np.ndarray]) -> dict:
    """One safetensors file, written by hand (no safetensors dependency)."""

    header: dict = {}
    blob = bytearray()
    dtypes = {np.dtype(np.int64): "I64", np.dtype(np.uint32): "U32"}
    for name, array in tensors.items():
        start = len(blob)
        blob.extend(array.tobytes())
        header[name] = {
            "dtype": dtypes[array.dtype],
            "shape": list(array.shape),
            "data_offsets": [start, len(blob)],
        }
    raw = json.dumps(header).encode("utf-8")
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(raw)))
        handle.write(raw)
        handle.write(bytes(blob))
    return header


@pytest.fixture
def model_dir(tmp_path):
    prefix = "language_model.model.layers.1.ple.ple_embedding."
    tensors = {
        prefix + "layer_multipliers": np.array(
            [2_654_435_761, 40_503, 1_337], dtype=np.int64
        ),
        prefix + "ngram_heads_vocab_sizes": np.full(HEAD_COUNT, 9_973, np.int64),
        prefix + "ngram_heads_offsets": (
            np.arange(HEAD_COUNT, dtype=np.int64) * 9_973
        ),
    }
    _write_shard(tmp_path / "model-00001-of-00001.safetensors", tensors)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    name: "model-00001-of-00001.safetensors" for name in tensors
                }
            }
        ),
        "utf-8",
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "text_config": {
                    "ngram_size": NGRAM_SIZE,
                    "heads_per_ngram": HEADS,
                    "eos_token_id": [248_044],
                }
            }
        ),
        "utf-8",
    )
    # The table header carries the row count the coverage line quotes.
    table = tmp_path / "ngram-table.safetensors"
    raw = json.dumps(
        {
            "__metadata__": {"rows": "159568", "dim": "160"},
            "ngram.weight": {
                "dtype": "U32",
                "shape": [159568, 20],
                "data_offsets": [0, 4],
            },
        }
    ).encode("utf-8")
    with open(table, "wb") as handle:
        handle.write(struct.pack("<Q", len(raw)))
        handle.write(raw)
        handle.write(b"\0\0\0\0")
    return tmp_path


class _StubTokenizer:
    """Deterministic ids, so the test measures the hash and not a download."""

    def encode(self, text):
        return [(ord(ch) * 7919) % 40_000 for ch in text]


# --------------------------------------------------------------------------


def test_the_row_arithmetic_is_the_shipped_one_not_a_copy(hotness):
    fn = hotness.shipped_rows_fn()
    assert fn.__name__ == "_ngram_rows_np"
    source = (ROOT / "mtplx" / "models" / "qwen4_exp.py").read_text("utf-8")
    # The script must not carry its own hash.
    script = SCRIPT.read_text("utf-8")
    assert "def _ngram_rows_np" not in script
    assert "_ngram_rows_np" in source


def test_the_hash_constants_come_from_the_checkpoint(hotness, model_dir):
    mult, sizes, offs = hotness.load_hash_constants(model_dir)
    assert mult.tolist() == [2_654_435_761, 40_503, 1_337]
    assert sizes.shape == (HEAD_COUNT,)
    assert offs.tolist() == (np.arange(HEAD_COUNT) * 9_973).tolist()


def test_a_checkpoint_without_the_buffers_is_named_not_guessed(hotness, tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {}}), "utf-8"
    )
    with pytest.raises(SystemExit) as excinfo:
        hotness.load_hash_constants(tmp_path)
    assert "no PLE hash buffers" in str(excinfo.value)


def test_the_table_row_count_is_read_from_the_table_header(hotness, model_dir):
    assert hotness.table_row_count(model_dir) == 159_568
    assert hotness.table_row_count(model_dir / "nope") is None


def test_counts_merge_across_prompts_without_a_python_loop(hotness):
    state = hotness.accumulate(None, np.array([5, 5, 7], np.int64))
    assert state[0].tolist() == [5, 7]
    assert state[1].tolist() == [2, 1]
    state = hotness.accumulate(state, np.array([7, 9, 9, 9], np.int64))
    assert state[0].tolist() == [5, 7, 9]
    assert state[1].tolist() == [2, 2, 3]


def test_jsonl_and_plain_text_corpora_are_both_read(hotness, tmp_path):
    (tmp_path / "a.jsonl").write_text(
        '{"prompt": "one"}\n{"text": "two"}\n"three"\nnot json\n\n', "utf-8"
    )
    (tmp_path / "b.py").write_text("four", "utf-8")
    texts = hotness.corpus_texts([tmp_path / "a.jsonl", tmp_path / "b.py"])
    assert texts == ["one", "two", "three", "not json", "four"]


def test_a_missing_corpus_file_is_skipped_not_fatal(hotness, tmp_path):
    assert hotness.corpus_texts([tmp_path / "nope.jsonl"]) == []


def test_prompts_are_tiled_to_the_requested_length(hotness):
    streams = list(
        hotness.prompt_id_streams(_StubTokenizer(), ["abc"], prompt_tokens=10)
    )
    assert streams[0].size == 10
    assert streams[0][:3].tolist() == streams[0][3:6].tolist()
    long = list(
        hotness.prompt_id_streams(_StubTokenizer(), ["abcdef"], prompt_tokens=4)
    )
    assert long[0].size == 4
    as_is = list(
        hotness.prompt_id_streams(_StubTokenizer(), ["abcdef"], prompt_tokens=0)
    )
    assert as_is[0].size == 6


def test_end_to_end_writes_rows_in_descending_frequency(
    hotness, model_dir, monkeypatch, capsys
):
    monkeypatch.setattr(hotness, "load_tokenizer", lambda model: _StubTokenizer())
    corpus = model_dir / "corpus.jsonl"
    corpus.write_text('{"prompt": "the quick brown fox"}\n', "utf-8")
    out = model_dir / "ngram-hotness.npy"
    assert (
        hotness.main(
            [
                "--model",
                str(model_dir),
                "--prompts",
                str(corpus),
                "--prompt-tokens",
                "128",
                "--top-rows",
                "64",
            ]
        )
        == 0
    )
    rows = np.load(out)
    assert rows.dtype == np.int64
    assert rows.size == 64
    assert len(set(rows.tolist())) == 64  # distinct row ids, not repeats
    printed = capsys.readouterr().out
    assert "distinct rows" in printed
    assert "coverage:" in printed
    assert "of this corpus's gathers" in printed


def test_the_file_it_writes_is_what_the_pre_read_reads(hotness, model_dir, monkeypatch):
    """The whole point: this file must satisfy load_hotness_order's contract."""

    from mtplx import ple_row_gather

    monkeypatch.setattr(hotness, "load_tokenizer", lambda model: _StubTokenizer())
    corpus = model_dir / "corpus.jsonl"
    corpus.write_text('{"prompt": "hello world"}\n', "utf-8")
    hotness.main(
        ["--model", str(model_dir), "--prompts", str(corpus), "--prompt-tokens", "64"]
    )
    found = ple_row_gather.hotness_path_for(model_dir / "ngram-table.safetensors")
    assert found is not None
    assert found.name == ple_row_gather.HOTNESS_FILENAME
    order = ple_row_gather.load_hotness_order(found)
    assert order is not None and order.dtype == np.int64 and order.size > 0
