"""CPU lifetime contracts for cached verifier scratch objects."""

from dataclasses import fields
from types import SimpleNamespace

import pytest

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.tensor import Tensor
from hipengine.runtime import qwen35_paro_runner as runner


_CLASSES = {
    "linear": runner.Qwen35ParoLinearAttentionScratch,
    "c1": runner.Qwen35ParoMoeScratch,
    "grouped": runner.Qwen35ParoGroupedMoeScratch,
    "dense": runner.Qwen35ParoDenseMlpScratch,
}
_NAMES = {
    "linear": ("linear_attn.attn_input", "linear_attn.qkv_z", "linear_attn.tree_recurrent_state"),
    "c1": ("moe.normed",), "grouped": ("moe.grouped.normed",),
    "dense": ("dense_mlp.normed",),
}


def _session(family):
    session = runner.Qwen35ParoResidentSession.__new__(runner.Qwen35ParoResidentSession)
    session._verify_linear_scratch_cache = {}
    session._verify_mlp_scratch_cache = {}
    session._verify_scratch_cache_generation = 0
    session._verify_mlp_scratch_policy = lambda rows: family
    reservations = []
    allocations = {}
    lookups = []

    def reserve(*, tokens, activation_dtype):
        assert activation_dtype == DType.FP16
        tensor = Tensor.from_handle(0x10000 + len(reservations) * 0x1000, (tokens, 8), activation_dtype, Device("hip", 0))
        scratch = _CLASSES[family](**{field.name: tensor for field in fields(_CLASSES[family])})
        reservations.append(scratch)
        for name in _NAMES[family]:
            allocations[name] = SimpleNamespace(tensor=tensor)
        return scratch

    def allocation(name):
        lookups.append(name)
        return allocations[name]

    state = SimpleNamespace(
        workspace=SimpleNamespace(allocation=allocation),
        reserve_linear_attention_scratch=reserve, reserve_dense_mlp_scratch=reserve,
        reserve_moe_grouped_prefill_scratch=reserve, reserve_moe_c1_scratch=reserve,
    )
    method = session._verify_linear_attention_scratch if family == "linear" else session._verify_mlp_scratch
    get = lambda rows: method(2, state, rows=rows)
    return session, state, get, reservations, allocations, lookups


@pytest.mark.parametrize("retired", [None, "0", "false", "1"])
@pytest.mark.parametrize("family", list(_CLASSES))
@pytest.mark.parametrize("rows", [2, 7, 19])
@pytest.mark.parametrize("stamp", ["0", "1"])
def test_verifier_scratch_always_reuses_and_invalidates(monkeypatch, retired, family, rows, stamp):
    if retired is None:
        monkeypatch.delenv("HIPENGINE_VERIFY_SCRATCH_CACHE", raising=False)
    else:
        monkeypatch.setenv("HIPENGINE_VERIFY_SCRATCH_CACHE", retired)
    monkeypatch.setenv("HIPENGINE_VERIFY_SCRATCH_GENERATION_STAMP", stamp)
    session, state, get, reservations, allocations, lookups = _session(family)
    first = get(rows)
    assert get(rows) is first
    assert len(reservations) == 1
    assert bool(lookups) is (stamp == "0")
    other_rows = get(rows + 1)
    assert other_rows is not first
    # Shared workspace names now hold another row bucket's allocations. Only
    # the explicit validation mode must replace the old bucket on this hit.
    if stamp == "0":
        assert get(rows) is not first
    else:
        assert get(rows) is first
    session._clear_verify_scratch_caches()
    assert session._verify_scratch_cache_generation == 1
    assert session._verify_linear_scratch_cache == {}
    assert session._verify_mlp_scratch_cache == {}
    after = get(rows)
    assert after is not first and after is not other_rows
    assert get(rows) is after
    cache = session._verify_linear_scratch_cache if family == "linear" else session._verify_mlp_scratch_cache
    assert next(iter(cache.values())) == (1, after)


@pytest.mark.parametrize("family", list(_CLASSES))
@pytest.mark.parametrize("mismatch", ["missing", "pointer", "shape", "dtype", "device"])
def test_verifier_scratch_validates_workspace_without_stamp(monkeypatch, family, mismatch):
    monkeypatch.setenv("HIPENGINE_VERIFY_SCRATCH_GENERATION_STAMP", "0")
    session, state, get, reservations, allocations, lookups = _session(family)
    first = get(4)
    name = _NAMES[family][-1]
    tensor = allocations[name].tensor
    if mismatch == "missing":
        del allocations[name]
    else:
        values = dict(ptr=tensor.ptr, shape=tensor.shape, dtype=tensor.dtype, device=tensor.device)
        if mismatch == "pointer":
            values["ptr"] += 0x8000
        elif mismatch == "shape":
            values["shape"] = (8, 4)
        elif mismatch == "dtype":
            values["dtype"] = DType.BF16
        if mismatch == "device":
            values["device"] = Device("hip", 1)
        allocations[name] = SimpleNamespace(tensor=Tensor.from_handle(**values))
    assert get(4) is not first
    assert len(reservations) == 2
    assert name in lookups


@pytest.mark.parametrize("family", list(_CLASSES))
def test_verifier_scratch_rebuilds_on_generation_mismatch(monkeypatch, family):
    monkeypatch.setenv("HIPENGINE_VERIFY_SCRATCH_GENERATION_STAMP", "1")
    session, state, get, reservations, allocations, lookups = _session(family)
    first = get(4)
    session._verify_scratch_cache_generation += 1
    assert get(4) is not first
    assert len(reservations) == 2
    assert lookups == []


@pytest.mark.parametrize("family", list(_CLASSES))
def test_verifier_scratch_cache_keeps_layers_separate(monkeypatch, family):
    monkeypatch.setenv("HIPENGINE_VERIFY_SCRATCH_GENERATION_STAMP", "1")
    session, state, get, reservations, allocations, lookups = _session(family)
    first = get(4)
    method = session._verify_linear_attention_scratch if family == "linear" else session._verify_mlp_scratch
    other = method(5, state, rows=4)
    assert other is not first
    assert method(5, state, rows=4) is other
    assert get(4) is first
    assert len(reservations) == 2


def test_verifier_mlp_cache_keeps_policy_keys_separate(monkeypatch):
    monkeypatch.setenv("HIPENGINE_VERIFY_SCRATCH_GENERATION_STAMP", "1")
    session, *_ = _session("c1")
    cached = {}
    for family in ("c1", "grouped", "dense"):
        _, state, *_ = _session(family)
        session._verify_mlp_scratch_policy = lambda rows, policy=family: policy
        cached[family] = session._verify_mlp_scratch(2, state, rows=4)
    assert set(session._verify_mlp_scratch_cache) == {
        (2, 4, "c1"), (2, 4, "grouped"), (2, 4, "dense"),
    }
    for family, scratch in cached.items():
        session._verify_mlp_scratch_policy = lambda rows, policy=family: policy
        assert session._verify_mlp_scratch(2, state, rows=4) is scratch
