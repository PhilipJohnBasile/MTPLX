"""Pure-Python gates for fixed-M4 host-owned n-gram inputs."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_function(path: Path, name: str):
    tree = ast.parse(path.read_text())
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        ),
        None,
    )
    assert function is not None, f"missing {name} in {path}"
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def test_fixed_m4_previous_tokens_come_from_committed_host_ledger():
    previous = _load_function(
        ROOT / "mtplx/qwen4_fixed_verify.py", "_fixed_m4_previous_tokens"
    )
    prompt_tail = (101, 102)

    assert previous(prompt_tail, [201], 0) == prompt_tail
    assert previous(prompt_tail, [201, 202], 1) == (102, 201)
    assert previous(prompt_tail, [201, 202, 203], 2) == (201, 202)
    # A deferred correction or bonus is present in the emitted ledger but is
    # the current primary, so committed_count excludes it from prior history.
    assert previous(prompt_tail, [201, 202, 999], 2) == (201, 202)
    # A correction already re-forwarded into the target cache remains inside
    # the committed prefix when the next fresh primary is appended.
    assert previous(prompt_tail, [201, 202, 999, 301], 3) == (202, 999)


def test_fixed_m4_host_entrypoint_is_explicitly_plumbed():
    graphbank = ast.parse((ROOT / "mtplx/graphbank.py").read_text())
    bank = next(
        node
        for node in ast.walk(graphbank)
        if isinstance(node, ast.ClassDef) and node.name == "CompiledVerifyBank"
    )
    entrypoint = next(
        (
            node
            for node in bank.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward_fixed_m4"
        ),
        None,
    )
    assert entrypoint is not None
    arguments = {
        argument.arg for argument in (*entrypoint.args.args, *entrypoint.args.kwonlyargs)
    }
    assert {
        "input_ids",
        "host_input_ids",
        "completion_tokens",
        "committed_count",
        "cache",
    } <= arguments

    generation = ast.parse((ROOT / "mtplx/generation.py").read_text())
    calls = [
        node
        for node in ast.walk(generation)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "forward_fixed_m4"
    ]
    assert len(calls) == 1
    keywords = {keyword.arg for keyword in calls[0].keywords}
    assert {
        "host_input_ids",
        "completion_tokens",
        "committed_count",
        "cache",
    } <= keywords


def test_fixed_m4_sidecar_uses_the_installed_dispatch_contract():
    module = ast.parse((ROOT / "mtplx/qwen4_fixed_verify.py").read_text())
    sidecar = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "_FixedM4SidecarAux"
    )
    call = next(
        node
        for node in sidecar.body
        if isinstance(node, ast.FunctionDef) and node.name == "__call__"
    )
    assert [argument.arg for argument in call.args.args] == [
        "self",
        "_input_ids",
        "host_input_ids",
        "completion_tokens",
        "committed_count",
    ]
