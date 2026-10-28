"""CPU contracts for immutable decode-state weight lookup memoization."""

from types import SimpleNamespace

import pytest

from hipengine.core.device import Device
from hipengine.core.dtype import DType
from hipengine.core.tensor import Tensor
from hipengine.loading.materialize import DeviceWeightMap
from hipengine.runtime import qwen35_paro as runtime


class _Weights(dict):
    def __init__(self, *args):
        super().__init__(*args)
        self.reads = []

    def __getitem__(self, name):
        self.reads.append(name)
        return super().__getitem__(name)


def _state():
    names = ("layers.0.self_attn.q_proj.weight", "layers.0.self_attn.k_proj.weight")
    tensors = _Weights({
        name: SimpleNamespace(tensor=Tensor.from_handle(0x1000 + i * 0x1000, (2, 8), DType.BF16, Device("hip", 0)))
        for i, name in enumerate(names)
    })
    layer = SimpleNamespace(layer_id=0, weights=DeviceWeightMap(tensors))
    state = runtime.Qwen35ParoDecodeState(layer_weights=layer, workspace=object())
    return state, tensors, names


@pytest.mark.parametrize("retired", [None, "0", "false", "1"])
@pytest.mark.parametrize("prefix", ["", "model.", "language_model.", "model.language_model."])
def test_weight_lookup_always_caches_exact_caller_name(monkeypatch, retired, prefix):
    if retired is None:
        monkeypatch.delenv("HIPENGINE_WEIGHT_TENSOR_LOOKUP_CACHE", raising=False)
    else:
        monkeypatch.setenv("HIPENGINE_WEIGHT_TENSOR_LOOKUP_CACHE", retired)
    state, tensors, names = _state()
    normalizer = runtime.normalize_qwen35_weight_name
    normalized = []

    def normalize(name):
        normalized.append(name)
        return normalizer(name)

    monkeypatch.setattr(runtime, "normalize_qwen35_weight_name", normalize)
    raw = prefix + names[0]
    first = state.tensor(raw)
    assert state.tensor(raw) is first
    assert normalized == [raw]
    assert tensors.reads == [names[0]]
    assert state._tensor_lookup_cache == {raw: first}
    assert first is dict.__getitem__(tensors, names[0]).tensor
    assert state.has_tensor(raw)
    assert state._tensor_lookup_cache == {raw: first}


def test_weight_lookup_aliases_have_independent_raw_keys_and_same_handle():
    state, tensors, names = _state()
    aliases = tuple(prefix + names[0] for prefix in ("", "model.", "language_model.", "model.language_model."))
    handles = [state.tensor(alias) for alias in aliases]
    assert all(handle is handles[0] for handle in handles)
    assert set(state._tensor_lookup_cache) == set(aliases)
    assert tensors.reads == [names[0]] * len(aliases)
    for alias in aliases:
        assert state.tensor(alias) is handles[0]
    assert len(tensors.reads) == len(aliases)
    assert state.tensor(names[1]) is not handles[0]


def test_weight_lookup_missing_key_raises_and_is_not_negative_cached():
    state, tensors, names = _state()
    missing = "model.layers.0.missing.weight"
    for _ in range(2):
        with pytest.raises(KeyError) as error:
            state.tensor(missing)
        assert error.value.args == ("layers.0.missing.weight",)
        assert missing not in state._tensor_lookup_cache
    assert not state.has_tensor(missing)
    # Missing lookups do not install a sentinel that can hide a later binding.
    tensors["layers.0.missing.weight"] = dict.__getitem__(tensors, names[1])
    assert state.tensor(missing) is dict.__getitem__(tensors, names[1]).tensor
    assert state.tensor(missing) is state._tensor_lookup_cache[missing]


def test_weight_lookup_cache_is_private_to_each_decode_state():
    first, first_tensors, names = _state()
    second, second_tensors, _ = _state()
    assert first.tensor(names[0]) is not second.tensor(names[0])
    assert first._tensor_lookup_cache is not second._tensor_lookup_cache
    assert first_tensors.reads == second_tensors.reads == [names[0]]


def test_weight_lookup_does_not_add_an_allocation_owner():
    state, tensors, names = _state()
    freed = []
    state.workspace = SimpleNamespace(free=lambda: freed.append("workspace"))
    state.layer_weights.free = lambda **kwargs: freed.append(("weights", kwargs))
    state.tensor(names[0])
    state.tensor("model." + names[0])
    state.free()
    assert freed == ["workspace", ("weights", {"runtime": None})]
