"""Static safety gates for default-off QSA/MTP generation wiring.

``mtplx.generation`` imports MLX, so the live-server safety window cannot load
it.  Parsing its source still pins the phase-order contract: every new staging
or replay-reconciliation call must sit below the explicit opt-in, while the
three original ``cycle_offset + 1`` rollback paths remain present as the dark
branch.
"""

from __future__ import annotations

import ast
from pathlib import Path

GENERATION = Path(__file__).parents[1] / "mtplx" / "generation.py"


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _contains_name(node: ast.AST, name: str) -> bool:
    return any(
        isinstance(item, ast.Name) and item.id == name for item in ast.walk(node)
    )


def test_new_generation_calls_are_below_the_default_off_gate():
    tree = ast.parse(GENERATION.read_text(encoding="utf-8"))
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    guarded_calls = {
        "precompute_and_stage_qsa_replay_caches",
        "reconcile_mtp_indexer_history",
    }
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _call_name(node) in guarded_calls
    ]
    assert calls
    for call in calls:
        ancestor = parents.get(call)
        guarded = False
        while ancestor is not None:
            if isinstance(ancestor, ast.If) and (
                _contains_name(ancestor.test, "qsa_mtp_precompute_active")
                or any(
                    isinstance(item, ast.Call)
                    and _call_name(item) == "qsa_mtp_precompute_enabled"
                    for item in ast.walk(ancestor.test)
                )
            ):
                guarded = True
                break
            ancestor = parents.get(ancestor)
        assert guarded, f"unguarded Phase-3 call at line {call.lineno}"


def test_original_cycle_plus_one_reconciliation_paths_remain_available():
    tree = ast.parse(GENERATION.read_text(encoding="utf-8"))
    original_rollbacks = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _call_name(node) != "_rollback_mtp_cache":
            continue
        if len(node.args) < 2 or not isinstance(node.args[1], ast.BinOp):
            continue
        offset = node.args[1]
        if (
            isinstance(offset.op, ast.Add)
            and isinstance(offset.left, ast.Name)
            and offset.left.id == "cycle_mtp_offset"
            and isinstance(offset.right, ast.Constant)
            and offset.right.value == 1
        ):
            original_rollbacks += 1

    assert original_rollbacks == 3


def test_qsa_cache_is_excluded_from_both_outer_device_compile_routes():
    tree = ast.parse(GENERATION.read_text(encoding="utf-8"))
    eligibility = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in {
            "device_d2_eligible",
            "device_core_eligible",
        }:
            eligibility[target.id] = node.value

    assert set(eligibility) == {"device_d2_eligible", "device_core_eligible"}
    for name, expression in eligibility.items():
        assert _contains_name(
            expression,
            "qsa_mtp_outer_device_core_supported",
        ), f"{name} does not fail closed for QSA cache state"


def test_device_d2_builder_defensively_rejects_qsa_before_model_execution():
    tree = ast.parse(GENERATION.read_text(encoding="utf-8"))
    builders = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_make_device_d2_draft_core"
    ]
    assert len(builders) == 1
    builder = builders[0]

    guard_index = next(
        index
        for index, statement in enumerate(builder.body)
        if isinstance(statement, ast.If)
        and _contains_name(statement.test, "qsa_mtp_outer_device_core_supported")
    )
    model_call_index = next(
        index
        for index, statement in enumerate(builder.body)
        if any(
            isinstance(item, ast.Call) and _call_name(item) == "draft_mtp"
            for item in ast.walk(statement)
        )
    )
    assert guard_index < model_call_index
