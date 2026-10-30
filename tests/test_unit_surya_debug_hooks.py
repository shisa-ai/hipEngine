"""Surya debug hooks follow the runtime's span-aware KV writer."""

import ast
from pathlib import Path


def test_debug_hooks_resolve_runtime_operations():
    import hipengine.runtime.surya as runtime

    source = Path("scripts/surya_gpu_debug_attn.py").read_text()
    tree = ast.parse(source)
    hooks = next(
        ast.literal_eval(node.iter)
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "_fname"
    )
    assert "surya_scatter_kv_f32_spans" in hooks
    for name in hooks:
        assert callable(getattr(runtime, name))


def test_only_span_aware_scatter_is_exported():
    from hipengine.kernels.hip_gfx1100 import surya
    from hipengine.kernels.hip_gfx1100.surya import surya_ops

    assert callable(surya.surya_scatter_kv_f32_spans)
    assert not hasattr(surya, "surya_scatter_kv_f32")
    assert not hasattr(surya_ops, "surya_scatter_kv_f32")
