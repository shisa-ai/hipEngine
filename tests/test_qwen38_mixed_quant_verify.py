from __future__ import annotations

from types import SimpleNamespace

from scripts.qwen38_mixed_quant_verify import _shape_manifest, _type_manifest


def _tensor(name: str, qtype: str, shape: tuple[int, ...]):
    return SimpleNamespace(name=name, ggml_type_name=qtype, shape=shape)


def test_mixed_quant_verify_manifests_are_order_independent() -> None:
    first = {
        "b.weight": _tensor("b.weight", "Q6_K", (4, 8)),
        "a.weight": _tensor("a.weight", "Q4_K", (8, 4)),
    }
    second = dict(reversed(tuple(first.items())))

    assert _shape_manifest(first) == _shape_manifest(second)
    assert _type_manifest(first) == _type_manifest(second)


def test_mixed_quant_verify_type_manifest_binds_type_not_only_shape() -> None:
    q4 = {"a.weight": _tensor("a.weight", "Q4_K", (8, 4))}
    q6 = {"a.weight": _tensor("a.weight", "Q6_K", (8, 4))}

    assert _shape_manifest(q4) == _shape_manifest(q6)
    assert _type_manifest(q4) != _type_manifest(q6)
