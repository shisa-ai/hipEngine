"""CPU protocol tests for a diagnostic two-consumer F16 cast replay."""
from dataclasses import replace

import pytest

from scripts.qwen4exp_shared_f16_cast_owner import Consumer, checked_pair, paired_operation


def consumers():
    common = dict(layer=24, rows=512, k=2560, x=1_000_000, half=20_000_000,
                  stream=0, runtime=object())
    return (Consumer(role="attn_qkv", n=10240, out=30_000_000, **common),
            Consumer(role="attn_gate", n=6144, out=60_000_000, **common))


@pytest.mark.parametrize("arm,expected", [
    ("two_casts", ["cast", "a", "cast", "b"]),
    ("one_cast", ["cast", "a", "b"]),
])
def test_complete_operation_includes_both_consumers(arm, expected):
    calls = []
    paired_operation(arm, lambda: calls.append("cast"),
                     lambda: calls.append("a"), lambda: calls.append("b"))
    assert calls == expected


def test_unknown_arm_does_not_launch():
    calls = []
    with pytest.raises(ValueError, match="arm"):
        paired_operation("estimate", lambda: calls.append(1), lambda: calls.append(2),
                         lambda: calls.append(3))
    assert not calls


def test_matching_pair_has_one_scoped_source():
    assert checked_pair(*consumers()) == "attn_qkv+attn_gate"


@pytest.mark.parametrize("change", [
    {"layer": 25}, {"rows": 1024}, {"k": 10240}, {"x": 1_000_004},
    {"half": 20_000_002}, {"stream": 1}, {"runtime": object()},
    {"role": "shared_up"}, {"out": 30_000_000}, {"out": 1_000_000},
    {"out": 20_000_000},
])
def test_pair_rejects_identity_mismatch_or_alias(change):
    a, b = consumers()
    with pytest.raises(ValueError):
        checked_pair(a, replace(b, **change))


def test_shared_expert_pair_and_reversed_order():
    a, b = consumers()
    a, b = replace(a, role="shared_gate", n=640), replace(b, role="shared_up", n=640)
    assert checked_pair(a, b) == "shared_gate+shared_up"
    with pytest.raises(ValueError):
        checked_pair(b, a)
