"""Host-only verifier MLP reservation policy and threshold transitions."""

from dataclasses import fields
from types import SimpleNamespace

import pytest

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.tensor import Tensor
from hipengine.runtime import qwen35_paro_runner as runner


_CLASSES = {
    "c1": runner.Qwen35ParoMoeScratch,
    "grouped": runner.Qwen35ParoGroupedMoeScratch,
    "dense": runner.Qwen35ParoDenseMlpScratch,
}


def _session(num_experts):
    session = runner.Qwen35ParoResidentSession.__new__(runner.Qwen35ParoResidentSession)
    session.config = SimpleNamespace(num_experts=num_experts)
    session._verify_mlp_scratch_cache = {}
    session._verify_linear_scratch_cache = {}
    session._verify_scratch_cache_generation = 0
    reservations = []

    def reserve(policy, *, tokens, activation_dtype):
        assert activation_dtype == DType.FP16
        tensor = Tensor.from_handle(0x10000 + len(reservations) * 0x1000, (tokens, 8), activation_dtype, Device("hip", 0))
        scratch = _CLASSES[policy](**{field.name: tensor for field in fields(_CLASSES[policy])})
        reservations.append((policy, tokens, scratch))
        return scratch

    state = SimpleNamespace(
        reserve_moe_c1_scratch=lambda **kwargs: reserve("c1", **kwargs),
        reserve_moe_grouped_prefill_scratch=lambda **kwargs: reserve("grouped", **kwargs),
        reserve_dense_mlp_scratch=lambda **kwargs: reserve("dense", **kwargs),
    )
    return session, state, reservations


@pytest.mark.parametrize("retired", [None, "0", "false", "1"])
@pytest.mark.parametrize("threshold", [None, "5", "11", "1"])
@pytest.mark.parametrize("num_experts", [0, 128])
def test_verify_mlp_policy_always_matches_runtime_threshold(monkeypatch, retired, threshold, num_experts):
    for name, value in (
        ("HIPENGINE_VERIFY_MLP_SCRATCH_POLICY_ALIGNED", retired),
        ("HIPENGINE_VERIFY_MOE_GROUPED_MIN_TOKENS", threshold),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    monkeypatch.setenv("HIPENGINE_VERIFY_SCRATCH_GENERATION_STAMP", "1")
    boundary = 16 if threshold is None else max(2, int(threshold))
    session, state, reservations = _session(num_experts)
    for rows in sorted({1, 3, 7, boundary - 1, boundary, boundary + 1, 23}):
        expected = "dense" if num_experts == 0 else ("grouped" if rows >= boundary else "c1")
        assert session._verify_mlp_scratch_policy(rows) == expected
        scratch = session._verify_mlp_scratch(2, state, rows=rows)
        assert isinstance(scratch, _CLASSES[expected])
        assert reservations[-1] == (expected, rows, scratch)
        assert session._verify_mlp_scratch(2, state, rows=rows) is scratch
        assert session._verify_mlp_scratch_cache[(2, rows, expected)] == (0, scratch)
    assert len(reservations) == len(session._verify_mlp_scratch_cache)


def test_verify_mlp_threshold_changes_keep_separate_cache_keys(monkeypatch):
    monkeypatch.setenv("HIPENGINE_VERIFY_MLP_SCRATCH_POLICY_ALIGNED", "0")
    monkeypatch.setenv("HIPENGINE_VERIFY_SCRATCH_GENERATION_STAMP", "1")
    session, state, reservations = _session(128)
    monkeypatch.setenv("HIPENGINE_VERIFY_MOE_GROUPED_MIN_TOKENS", "11")
    c1 = session._verify_mlp_scratch(2, state, rows=7)
    monkeypatch.setenv("HIPENGINE_VERIFY_MOE_GROUPED_MIN_TOKENS", "5")
    grouped = session._verify_mlp_scratch(2, state, rows=7)
    assert isinstance(c1, _CLASSES["c1"])
    assert isinstance(grouped, _CLASSES["grouped"])
    assert grouped is not c1
    monkeypatch.setenv("HIPENGINE_VERIFY_MOE_GROUPED_MIN_TOKENS", "11")
    assert session._verify_mlp_scratch(2, state, rows=7) is c1
    assert set(session._verify_mlp_scratch_cache) == {(2, 7, "c1"), (2, 7, "grouped")}
    assert len(reservations) == 2
    session._clear_verify_scratch_caches()
    assert session._verify_mlp_scratch(2, state, rows=7) is not c1
    assert len(reservations) == 3
