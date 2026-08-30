"""MLX-free wiring gates for the end-to-end QSA large-prefill lane."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "mtplx" / "models" / "qwen4_exp.py"
PROFILES_PATH = ROOT / "mtplx" / "profiles.py"
MODEL_TEXT = MODEL_PATH.read_text(encoding="utf-8")
MODEL_TREE = ast.parse(MODEL_TEXT, filename=str(MODEL_PATH))


def _top_function(name: str) -> ast.FunctionDef:
    return next(
        node
        for node in MODEL_TREE.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _class_method(class_name: str, method_name: str) -> ast.FunctionDef:
    cls = next(
        node
        for node in MODEL_TREE.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )


def _source(node: ast.AST) -> str:
    value = ast.get_source_segment(MODEL_TEXT, node)
    assert value is not None
    return value


def test_large_prefill_is_phase_gated_with_device_scoped_auto_default():
    enabled = _source(_top_function("_qsa_prefill_enabled"))
    route = _source(_top_function("_qsa_large_prefill_enabled"))
    selector_floor = _source(_top_function("_qsa_prefill_min_context"))
    flash_floor = _source(_top_function("_qsa_prefill_flash_min_context"))
    flash_route = _source(_top_function("_qsa_prefill_flash_attention_enabled"))
    # Explicit env wins both ways; unset resolves auto through the flash
    # kernel's own device gate (NAX machines on, everything else off until
    # the portable tier carries its own hardware receipts).
    assert 'raw in {"1", "true", "yes", "on"}' in enabled
    assert 'raw in {"0", "false", "no", "off"}' in enabled
    assert "qsa_prefill_lane_auto_supported()" in enabled
    assert 'current_attention_phase() == "prefill"' in route
    assert "int(rows) >= _qsa_prefill_min_rows()" in route
    assert "int(total_tokens) - int(rows) >= _qsa_prefill_min_context()" in route
    assert 'os.environ.get("MTPLX_QSA_PREFILL_MIN_CONTEXT") or 32768' in (
        selector_floor
    )
    # 32768 flash crossover: 2026-08-30 ABBA receipts (flat at the 32K rung
    # both orders, full win beyond) retired the conservative 65536.
    assert 'os.environ.get("MTPLX_QSA_PREFILL_FLASH_MIN_CONTEXT") or 32768' in (
        flash_floor
    )
    assert "_qsa_large_prefill_enabled(rows, total_tokens)" in flash_route
    assert (
        "int(total_tokens) - int(rows) >= _qsa_prefill_flash_min_context()"
        in flash_route
    )


def test_prefill_lane_env_resolution_wins_over_auto(monkeypatch):
    import mtplx.models.qwen4_exp as qwen4_exp

    for auto in (True, False):
        monkeypatch.setattr(
            qwen4_exp, "qsa_prefill_lane_auto_supported", lambda a=auto: a
        )
        monkeypatch.setenv("MTPLX_QSA_PREFILL", "1")
        assert qwen4_exp._qsa_prefill_enabled() is True
        monkeypatch.setenv("MTPLX_QSA_PREFILL", "0")
        assert qwen4_exp._qsa_prefill_enabled() is False
        monkeypatch.delenv("MTPLX_QSA_PREFILL", raising=False)
        assert qwen4_exp._qsa_prefill_enabled() is auto


def test_serve_admission_prices_transients_by_resolved_lane():
    """The #393 dense transient term must be dropped when the lane serves.

    Measured 2026-08-30: 262K cold prefill peaked at 87.4 GB with the lane
    (weights + KV + aux + flat reserve), against the dense-priced ~115 GB
    that originally wedged the machine. Pricing the dense lane while the
    sparse lane serves would refuse a window the machine demonstrably holds.
    """

    server_text = (ROOT / "mtplx" / "server" / "openai.py").read_text(
        encoding="utf-8"
    )
    anchor = server_text.index("_plan_transient_per_token = _plan_transient_from_config")
    window = server_text[anchor : anchor + 700]
    assert "_qsa_prefill_enabled as _qsa_prefill_lane_resolved" in window
    assert "_qsa_prefill_lane_resolved()" in window
    assert "_plan_transient_per_token = 0" in window
    assert (
        '"prefill_transient_bytes_per_token": _plan_transient_per_token'
        in server_text
    )


def test_indexer_routes_large_prefill_to_compact_blocks_in_both_paths():
    eager = _source(_class_method("QSAIndexer", "_select_eager"))
    mode = _source(_class_method("QSAIndexer", "_compiled_mode"))
    compiled = _source(_class_method("QSAIndexer", "_call_rows_compiled"))
    rows = _source(_class_method("QSAIndexer", "_call_rows"))

    assert '("flash_prefill", block_ids, block_valid)' in eager
    assert "block_ids = mx.where(" in eager
    assert 'return "prefill_blocks"' in mode
    assert 'if mode == "prefill_blocks"' in compiled
    assert '("flash_prefill", block_ids, block_valid)' in compiled
    assert "qsa_indexer_prefill_blocks_metal(" in rows
    assert 'return ("flash_prefill", block_ids, block_valid)' in rows

    # The scalar row selector must never become the accidental large-S route.
    assert "and (decode or S < _qsa_prefill_min_rows())" in rows


def test_compile_capture_is_limited_to_one_canonical_prefill_width():
    supported = _source(_class_method("QSAIndexer", "_compiled_route_supported"))
    constructor = _source(_class_method("QSAIndexer", "_get_compiled_indexer_core"))
    assert (
        'mode not in ("prefill_blocks", "update_only")' in supported
        and "rows != _qsa_prefill_compile_rows()" in supported
    )
    assert 'mode == "update_only"' in supported
    assert 'current_attention_phase() == "prefill"' in supported
    assert (
        "prefill_score_workspace_bytes=_qsa_prefill_score_workspace_bytes()"
        in constructor
    )


def test_attention_consumes_blocks_directly_before_any_dense_ndim_path():
    attention = _source(_class_method("Attention", "__call__"))
    branch = attention.index('sel_mask[0] == "flash_prefill"')
    dense_ndim = attention.index("sel_mask.ndim == 1")
    assert branch < dense_ndim
    assert "qsa_prefill_flash_supported(" in attention
    assert "qsa_prefill_flash(" in attention
    assert "_qsa_prefill_flash_attention_enabled(S, T)" in attention
    assert "cache.kv.keys" in attention
    assert "cache.kv.values" in attention
    assert "_qsa_blocks_to_dense_mask(" in attention


def test_dense_fallback_uses_a_sentinel_and_row_specific_causal_tail():
    fallback = _source(_top_function("_qsa_blocks_to_dense_mask"))
    assert "logical_blocks + 1" in fallback
    assert "safe_ids = mx.where(valid, block_ids, sentinel)" in fallback
    assert "block_ids < complete_for_row[:, None]" in fallback
    assert ")[:, :logical_blocks]" in fallback
    assert "complete_for_row = (qpos + 1) // ratio" in fallback
    assert "tail_start = complete_for_row * ratio" in fallback
    assert "tpos[None, :] <= qpos[:, None]" in fallback
    assert "(token_selected | tail) & causal" in fallback


def test_all_prefill_knobs_are_registered_for_validated_operator_overrides():
    profiles = PROFILES_PATH.read_text(encoding="utf-8")
    for key in (
        "MTPLX_QSA_PREFILL",
        "MTPLX_QSA_PREFILL_MIN_ROWS",
        "MTPLX_QSA_PREFILL_MIN_CONTEXT",
        "MTPLX_QSA_PREFILL_FLASH_MIN_CONTEXT",
        "MTPLX_QSA_PREFILL_SCORE_MB",
        "MTPLX_QSA_PREFILL_COMPILE_ROWS",
    ):
        assert f'"{key}"' in profiles
