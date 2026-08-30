"""MLX-free structural checks for the operator-controlled QSA runtime gates.

The numeric scripts intentionally remain unexecuted until the operator releases
the GPU.  These tests parse their source, proving that the requested geometry,
boundaries, oracles, tolerances, anti-vacuity checks, and machine preflight are
present without importing MLX or compiling/dispatching Metal.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEXER_PATH = ROOT / "scripts" / "qsa_indexer_prefill_numeric_check.py"
FLASH_PATH = ROOT / "scripts" / "qsa_prefill_flash_numeric_check.py"
MODEL_PROCESS_PATTERN = "mtplx(\\.cli)? (serve|bench prefill-ladder)|mtplx.server.openai|mlx_lm"


def _source_and_tree(path: Path) -> tuple[str, ast.Module]:
    source = path.read_text(encoding="utf-8")
    return source, ast.parse(source, filename=str(path))


def _assignment(tree: ast.Module, name: str) -> ast.expr:
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(
                isinstance(target, ast.Name) and target.id == name for target in targets
            ):
                assert node.value is not None
                return node.value
    raise AssertionError(f"missing assignment {name}")


def _literal(tree: ast.Module, name: str):
    return ast.literal_eval(_assignment(tree, name))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1
    return matches[0]


def _function_source(source: str, tree: ast.Module, name: str) -> str:
    return ast.get_source_segment(source, _function(tree, name)) or ""


def _case_calls(tree: ast.Module, assignment: str, constructor: str) -> list[ast.Call]:
    value = _assignment(tree, assignment)
    assert isinstance(value, (ast.Tuple, ast.List))
    calls = list(value.elts)
    assert calls and all(isinstance(node, ast.Call) for node in calls)
    typed_calls = [node for node in calls if isinstance(node, ast.Call)]
    assert all(
        isinstance(node.func, ast.Name) and node.func.id == constructor
        for node in typed_calls
    )
    return typed_calls


def _calls_name(node: ast.AST, name: str) -> bool:
    return any(
        isinstance(item, ast.Call)
        and isinstance(item.func, ast.Name)
        and item.func.id == name
        for item in ast.walk(node)
    )


def _imports_mlx(node: ast.AST) -> bool:
    for item in ast.walk(node):
        if isinstance(item, ast.Import) and any(
            alias.name == "mlx" or alias.name.startswith("mlx.") for alias in item.names
        ):
            return True
        if isinstance(item, ast.ImportFrom) and (
            item.module == "mlx" or str(item.module).startswith("mlx.")
        ):
            return True
    return False


def test_runtime_gate_files_parse_without_top_level_mlx_imports():
    for path in (INDEXER_PATH, FLASH_PATH):
        _, tree = _source_and_tree(path)
        direct_imports = [
            node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        assert not any(_imports_mlx(node) for node in direct_imports)


def test_machine_preflight_dominates_mlx_import_and_runtime_dispatch():
    for path, runner in ((INDEXER_PATH, "_run_case"), (FLASH_PATH, "_run_case")):
        _, tree = _source_and_tree(path)
        safety = _function(tree, "_machine_safety_gate")
        pgrep_argv = []
        for node in ast.walk(safety):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "_command_output"
                and node.args
            ):
                argv = ast.literal_eval(node.args[0])
                if argv and argv[0] == "pgrep":
                    pgrep_argv.append(argv)
        assert pgrep_argv == [["pgrep", "-fl", MODEL_PROCESS_PATTERN]]

        main = _function(tree, "main")
        safety_indices = [
            index
            for index, node in enumerate(main.body)
            if _calls_name(node, "_machine_safety_gate")
        ]
        dangerous_indices = [
            index
            for index, node in enumerate(main.body)
            if _imports_mlx(node) or _calls_name(node, runner)
        ]
        assert len(safety_indices) == 1
        assert dangerous_indices
        assert safety_indices[0] < min(dangerous_indices)


def test_indexer_gate_covers_boundaries_seeds_ties_and_forced_chunks():
    source, tree = _source_and_tree(INDEXER_PATH)
    assert _literal(tree, "HEADS") == 4
    assert _literal(tree, "HEAD_DIM") == 128
    assert _literal(tree, "BLOCK_TOPK") == 512
    assert _literal(tree, "COMPRESS_RATIO") == 4
    assert _literal(tree, "SCORE_ATOL") == 1.0e-4

    calls = _case_calls(tree, "INDEXER_CASES", "IndexerCase")
    values = [ast.literal_eval(ast.Tuple(elts=call.args)) for call in calls]
    names = {row[0] for row in values}
    seeds = {row[1] for row in values}
    rows = {row[2] for row in values}
    totals = {row[3] for row in values}
    assert len(seeds) >= 8
    assert {2, 3, 7, 33, 65, 129, 257, 2048}.issubset(rows)
    assert {
        2048,
        2049,
        2052,
        2053,
        2054,
        2055,
        2056,
        8196,
        262144,
        1_000_000,
        1_048_576,
    }.issubset(totals)
    assert {total % 4 for total in totals} == {0, 1, 2, 3}
    assert "exact_zero_ties" in names
    tie_call = next(
        call for call in calls if ast.literal_eval(call.args[0]) == "exact_zero_ties"
    )
    assert ast.literal_eval(tie_call.args[-1]) is True
    for call in calls:
        case_rows = ast.literal_eval(call.args[2])
        forced_chunk_rows = ast.literal_eval(call.args[5])
        assert forced_chunk_rows < case_rows

    oracle = _function_source(source, tree, "_eager_oracle")
    assert "q.astype(mx.float32)" in oracle
    assert "pooled.astype(mx.float32)" in oracle
    assert "mx.maximum(per_head, 0.0).sum(axis=2)" in oracle
    assert "block.astype(mx.float32)[None, :] * 1.0e-12" in oracle
    assert "mx.argpartition" in oracle

    runner = _function_source(source, tree, "_run_case")
    assert "qsa_indexer_prefill_blocks_metal" in runner
    assert "qsa_indexer_prefill_score_chunk_rows" in runner
    assert "qsa_indexer_prefill_scores_mpp" in runner
    assert "recorded_scores" in runner
    assert '("mpp", scores)' in runner
    assert '("fallback", scores)' in runner
    assert 'expected_path = "fallback" if case.dtype_name == "float32" else "mpp"' in (
        runner
    )
    assert 'producer = "mlx" if expected_path == "fallback" else "mpp"' in runner
    assert 'score_planes = 1 if producer == "mpp" else HEADS + 1' in runner
    assert "bytes_per_row = logical_blocks * 4 * score_planes" in runner
    assert "producer=producer" in runner
    assert "actual_ids.tolist() != expected_ids.tolist()" in runner
    assert "actual_valid.tolist() != expected_valid.tolist()" in runner
    assert "raw_max_abs > SCORE_ATOL" in runner
    assert "selected_max_abs > SCORE_ATOL" in runner
    assert "list(range(BLOCK_TOPK))" in runner
    assert "case.positive_tie" in runner
    assert "stable final-K cutoff" in runner

    main = _function_source(source, tree, "main")
    assert 'path_counts = {"mpp": 0, "fallback": 0}' in main
    assert 'path_counts["mpp"] == 0 or path_counts["fallback"] == 0' in main


def test_flash_gate_uses_exact_production_geometry_and_explicit_tolerances():
    source, tree = _source_and_tree(FLASH_PATH)
    expected = {
        "BATCH": 1,
        "Q_HEADS": 24,
        "KV_HEADS": 2,
        "HEAD_DIM": 256,
        "BLOCK_TOPK": 512,
        "COMPRESS_RATIO": 4,
        "SCALE": 0.0625,
    }
    for name, value in expected.items():
        assert _literal(tree, name) == value
    assert _literal(tree, "TOLERANCES") == {
        "float16": (1.0e-2, 2.0e-2),
        "bfloat16": (2.0e-2, 2.0e-2),
    }
    assert _literal(tree, "FLASH_DTYPES") == ("float16", "bfloat16")

    calls = _case_calls(tree, "FLASH_CASES", "FlashCase")
    values = [ast.literal_eval(ast.Tuple(elts=call.args)) for call in calls]
    assert {row[4] for row in values} == {0, 1, 2, 3}
    assert {row[3] for row in values} == {2052, 2053, 2054, 2055}
    assert all(1 < row[2] <= 6 for row in values)
    assert all(row[3] > 2048 and row[3] % 4 == row[4] for row in values)
    assert {row[5] for row in values if len(row) > 5 and row[5] != "contiguous"} == {
        "offset1",
        "feature_stride2",
        "token_stride2",
    }
    assert {row[6] for row in values if len(row) > 6} == {"active_edges"}

    runner = _function_source(source, tree, "_run_case")
    assert "qsa_prefill_flash_supported" in runner
    assert "qsa_prefill_flash(" in runner
    assert ".transpose(0, 2, 1, 3)" in runner
    assert runner.count("_dense_sdpa(") >= 3
    assert "selected_mask" in runner
    assert "causal_mask" in runner
    assert "block_only_mask" in runner
    assert "full_gap <= SENSITIVITY_MIN" in runner
    assert "tail_gap <= SENSITIVITY_MIN" in runner
    assert "difference <= limit" in runner
    assert "atol + rtol * mx.abs(reference_f32)" in runner
    assert "_kv_view(mx, np, k_np, dtype, case.kv_layout)" in runner

    kv_view = _function_source(source, tree, "_kv_view")
    assert "[..., 1:]" in kv_view
    assert "[..., ::2]" in kv_view
    assert "[:, :, ::2, :]" in kv_view

    fixture = _function_source(source, tree, "_fixture")
    assert "v[:, :, :COMPRESS_RATIO, :] = -8.0" in fixture
    for token in (2047, 2052, 2053, 2054, 2055):
        assert str(token) in fixture

    selection = _function_source(source, tree, "_selection_and_masks")
    assert "ids[row, chosen.size] = complete" in selection
    assert "valid[row, chosen.size] = True" in selection
    assert "ids[row, 17] = chosen[1]" in selection
    assert "ids[row, -1] = complete" in selection


def test_runtime_gates_are_tiny_fixture_only_and_do_not_touch_system_limits():
    for path in (INDEXER_PATH, FLASH_PATH):
        source = path.read_text(encoding="utf-8")
        assert "/Users/mac/models" not in source
        assert "mx.load" not in source
        assert "load_model" not in source
        assert "iogpu.wired_limit_mb" not in source
        # The safety gate's pgrep pattern must still name the serve form.
        assert "mtplx(\\\\.cli)? (serve|bench prefill-ladder)" in source
