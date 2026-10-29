"""The unfused LM-head route remains a live FP16 and W8A16 consumer."""

from types import SimpleNamespace

import pytest

from hipengine.kernels.hip_gfx1100.fused import moonshine_glue as glue
from hipengine.kernels.hip_gfx1100.linear import moonshine_projection as projection
from hipengine.kernels.hip_gfx1100.linear import moonshine_w8a16 as w8
from hipengine.kernels.hip_gfx1100.norm import moonshine_layernorm as norm
from hipengine.runtime.moonshine import (
    MOONSHINE_DEFAULT_LM_HEAD_ROUTE,
    MoonshineResidentRuntime,
    normalize_moonshine_lm_head_route,
)


@pytest.mark.parametrize("route,quantized,expected", [
    ("wave8_top1", False, ["fp16_top1"]),
    ("wave8_argmax", False, ["fp16", "argmax"]),
    ("wave8_argmax", True, ["w8a16", "argmax"]),
])
def test_lm_head_real_token_dag_keeps_default_and_unfused_quantized_routes(monkeypatch, route, quantized, expected):
    calls = []
    def capture(name):
        return lambda *a, **kw: calls.append((name, a, kw))
    monkeypatch.setattr(glue, "moonshine_embedding_lookup_fp16", lambda *a, **kw: None)
    monkeypatch.setattr(norm, "moonshine_layernorm_fp16", lambda *a, **kw: None)
    monkeypatch.setattr(projection, "moonshine_f16_lm_head_projection_wave8_top1", capture("fp16_top1"))
    monkeypatch.setattr(projection, "moonshine_f16_lm_head_projection_wave8", capture("fp16"))
    monkeypatch.setattr(w8, "moonshine_w8a16_lm_head_wave8", capture("w8a16"))
    monkeypatch.setattr(glue, "moonshine_argmax_fp16", capture("argmax"))
    session = object.__new__(MoonshineResidentRuntime)
    session.closed = False
    session.spec = SimpleNamespace(self_cache_capacity=194, decoder_layers=0,
        embedding_weight_name="embed", lm_head_alias_name="lm_head", hidden_size=16,
        vocab_size=512, layer_norm_epsilon=1e-5)
    session.weights = {name: SimpleNamespace(ptr=ptr) for name, ptr in (
        ("embed", 100), ("lm_head", 101), ("model.decoder.norm.weight", 102))}
    session.decoder_libraries = SimpleNamespace(glue=object(), layernorm=object(),
        projection=object(), w8a16=object())
    session.runtime = object()
    session.cross_cache_valid = session.encoder_state_valid = True
    session.lm_head_route = normalize_moonshine_lm_head_route(route)
    tensors = {name: SimpleNamespace(ptr=index+1) for index, name in enumerate(
        ("hidden", "normalized", "query", "key", "value", "attention", "projection",
         "position", "token", "logits", "lm_head_partial_values", "lm_head_partial_indices"))}
    session.tensor = tensors.__getitem__
    quant = SimpleNamespace(qweight=SimpleNamespace(tensor=SimpleNamespace(ptr=201)),
        scales=SimpleNamespace(tensor=SimpleNamespace(ptr=202)))
    session.loaded_model = SimpleNamespace(w8a16={"embed": quant} if quantized else None)
    session._enqueue_token_step(route_position=1, stream=7)
    assert [name for name, _, _ in calls] == expected
    first = calls[0][1]
    assert first[0] == tensors["normalized"].ptr
    assert first[1] == (201 if quantized else 101)
    assert all(kwargs["stream"] == 7 and kwargs["runtime"] is session.runtime for _, _, kwargs in calls)
    if route == "wave8_top1":
        assert first[2:6] == tuple(tensors[name].ptr for name in (
            "logits", "lm_head_partial_values", "lm_head_partial_indices", "token"))
    else:
        assert calls[-1][1][:2] == (tensors["logits"].ptr, tensors["token"].ptr)
    contract = session.lm_head_contract()
    assert contract["route"] == route
    if route == "wave8_top1":
        assert contract["partial_count"] > 0
    else:
        assert contract["partial_count"] == 0
    assert contract["materializes_full_fp16_logits"] and contract["stable_lowest_id_top1"]


def test_top1_is_default_and_unknown_route_fails_loudly():
    assert MOONSHINE_DEFAULT_LM_HEAD_ROUTE == "wave8_top1"
    with pytest.raises(ValueError, match="lm_head_route"):
        normalize_moonshine_lm_head_route("missing")
