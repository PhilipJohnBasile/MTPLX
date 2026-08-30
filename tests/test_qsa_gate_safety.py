"""MLX-free static guards for the standalone Qwen4-Exp gates."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = {
    "numeric": ROOT / "scripts" / "qwen4exp_numeric_check.py",
    "mtp": ROOT / "scripts" / "qwen4exp_mtp_tiny_smoke.py",
}
MODEL_PROCESS_PATTERN = "mtplx(\\.cli)? (serve|bench prefill-ladder)|mtplx.server.openai|mlx_lm"


def _tree(name: str) -> ast.Module:
    return ast.parse(SCRIPTS[name].read_text(encoding="utf-8"))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1
    return matches[0]


def _calls(node: ast.AST, names: set[str]) -> bool:
    return any(
        isinstance(item, ast.Call)
        and isinstance(item.func, ast.Name)
        and item.func.id in names
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


def _top_level_statement_index(function: ast.FunctionDef, predicate) -> int:
    matches = [index for index, node in enumerate(function.body) if predicate(node)]
    assert len(matches) == 1
    return matches[0]


def test_both_gates_match_every_live_model_process_form() -> None:
    expected = ["pgrep", "-fl", MODEL_PROCESS_PATTERN]

    for name in SCRIPTS:
        safety = _function(_tree(name), "_machine_safety_gate")
        pgrep_argv = []
        for node in ast.walk(safety):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "_command_output"
                and node.args
            ):
                continue
            argv = ast.literal_eval(node.args[0])
            if argv and argv[0] == "pgrep":
                pgrep_argv.append(argv)
        assert pgrep_argv == [expected], f"{name} gate process pattern drifted"


def test_numeric_safety_dominates_parent_and_direct_worker_paths() -> None:
    main = _function(_tree("numeric"), "main")
    safety_index = _top_level_statement_index(
        main,
        lambda node: _calls(node, {"_machine_safety_gate"}),
    )
    safety_statement = main.body[safety_index]
    assert isinstance(safety_statement, ast.If)
    assert any(
        isinstance(node, ast.Return)
        and isinstance(node.value, ast.Constant)
        and node.value.value == 2
        for node in ast.walk(safety_statement)
    )

    dangerous = {"parent", "torch_worker", "mlx_worker"}
    dangerous_indices = [
        index for index, node in enumerate(main.body) if _calls(node, dangerous)
    ]
    assert dangerous_indices
    assert safety_index < min(dangerous_indices)


def test_mtp_safety_dominates_mlx_import_and_model_construction() -> None:
    main = _function(_tree("mtp"), "main")
    safety_index = _top_level_statement_index(
        main,
        lambda node: _calls(node, {"_machine_safety_gate"}),
    )
    dangerous_indices = [
        index
        for index, node in enumerate(main.body)
        if _imports_mlx(node)
        or _calls(
            node,
            {
                "_install_indexer_route_probe",
                "_install_mtp_precompute_route_probe",
                "_build_tiny_model",
            },
        )
    ]
    assert dangerous_indices
    assert safety_index < min(dangerous_indices)
