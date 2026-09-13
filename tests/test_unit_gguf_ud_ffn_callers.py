"""CPU-only mixed UD FFN chain ownership; numerical leaves are captured."""
from types import SimpleNamespace as NS

import pytest

from hipengine.runtime import gguf_linear
from hipengine.runtime import qwen35_gguf_runner as runner_module
from hipengine.loading.qwen35_gguf_materialize import LAYOUT_RAW_GGUF


@pytest.mark.parametrize("backend", ("hip_gfx1100", "hip_gfx1151"))
@pytest.mark.parametrize("quants", (
    ("iq4_xs", "q3_k", "iq4_xs"),
    ("q3_k", "iq4_xs", "iq3_s"),
    ("iq3_s", "q3_k", "iq4_xs"),
    ("iq3_xxs", "iq3_s", "iq3_s"),
    ("q3_k", "iq3_xxs", "iq3_s"),
    ("iq4_xs", "iq4_nl", "iq4_xs"),
))
@pytest.mark.parametrize("rows", (1, 2, 4, 7, 8, 16, 32, 511, 512, 513))
def test_mixed_raw_ffn_chain(monkeypatch, backend, quants, rows):
    h, f = 5120, 17408
    events = []

    def capture(label):
        def call(*args, **kwargs):
            events.append((label, args, kwargs))
        return call

    def weight(quant, ptr):
        allocation = NS(tensor=NS(ptr=ptr))
        def get_allocation(name="raw"):
            return {"raw": allocation}[name]
        return NS(backend=backend, spec=NS(layout=LAYOUT_RAW_GGUF, quant_key="gguf_"+quant),
                  allocations={"raw": allocation}, allocation=get_allocation)

    weights = dict(zip(("ffn_gate", "ffn_up", "ffn_down"),
                       (weight(q, p) for q, p in zip(quants, (101, 102, 103)))))
    weights["post_attention_norm"] = weight("f32", 104)
    scratch = NS(**{name: NS(ptr=ptr) for name, ptr in (
        ("post_norm", 201), ("residual", 202), ("ffn_gate_up", 100000000),
        ("ffn_intermediate", 200000000), ("ffn_down", 300000000))})
    runtime = object()
    runner = NS(runtime=runtime, hidden_size=h, ffn_size=f,
                weights=NS(layer=lambda _: NS(weight=weights.__getitem__),
                           config=NS(is_moe=False, rms_norm_eps=1e-6)),
                _dense_down_residual_decode_c1=False)
    monkeypatch.setattr(runner_module, "_gguf_norm_residual_decode_kernel",
                        lambda *a, **kw: capture("norm"))
    # Select the strict unfused policy, without substituting pair resolution.
    monkeypatch.setattr(runner_module, "_gguf_dense_pair_silu_decode_variant",
                        lambda *a, **kw: None)
    monkeypatch.setattr(runner_module, "silu_mul_separate_out_bf16", capture("silu"))
    monkeypatch.setattr(runner_module, "gguf_bf16_add", capture("add"))
    original_resolve = gguf_linear.resolve

    def resolve(**key):
        assert callable(original_resolve(**key))
        assert key == dict(backend=backend, layer="linear", quant=key["quant"],
                           variant=("gemv" if rows == 1 else "prefill")+"_bf16_bf16_out")
        return capture(key["quant"])

    monkeypatch.setattr(gguf_linear, "resolve", resolve)
    gguf_linear.clear_gguf_linear_dispatch_cache()
    try:
        runner_module.Qwen35GGUFFullStackRunner._run_post_attention_ffn_rows(
            runner, 0, 1, 2, 3, scratch, rows=rows, stream=19)
        common = dict(stream=19, runtime=runtime)
        up_ptr = scratch.ffn_gate_up.ptr + f*rows*2
        assert events == [
            ("norm", (1, 2, 104, 201, 202),
             dict(common, rows=rows, hidden_size=h, eps=1e-6)),
            ("gguf_"+quants[0], (201, 101, scratch.ffn_gate_up.ptr, rows, h, f), common),
            ("gguf_"+quants[1], (201, 102, up_ptr, rows, h, f), common),
            ("silu", (scratch.ffn_gate_up.ptr, up_ptr, scratch.ffn_intermediate.ptr),
             dict(common, rows=rows, features=f)),
            ("gguf_"+quants[2], (scratch.ffn_intermediate.ptr, 103, scratch.ffn_down.ptr, rows, f, h), common),
            ("add", (202, scratch.ffn_down.ptr, 3, rows*h), common),
        ]
    finally:
        gguf_linear.clear_gguf_linear_dispatch_cache()
