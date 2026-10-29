"""Host route ownership for the source-F16 boundary-fusion comparison."""

from types import SimpleNamespace

import pytest

from hipengine.loading.laguna_gguf import FULL_ATTENTION
from hipengine.runtime import laguna_gguf_runner as runner
from scripts import laguna_f16_boundary_ab as comparison


@pytest.mark.parametrize("rows", [3, 127, 129])
@pytest.mark.parametrize("mode", ["hipblaslt_range_direct", "hipblaslt"])
@pytest.mark.parametrize("fused", [False, True])
def test_f16_projection_and_output_routes_preserve_cast_and_pointer_ownership(monkeypatch, rows, mode, fused):
    calls = []
    session = object.__new__(runner.LagunaGGUFResidentSession)
    session.runtime = object()
    session.backend = "hip_gfx1151"
    session.f16_prefill_mode = mode
    session.fuse_f16_boundaries = fused
    session.weights = SimpleNamespace(config=SimpleNamespace(hidden_size=8))
    session.libraries = SimpleNamespace(cast=object())
    session._ensure_f16_hipblaslt = lambda: SimpleNamespace(launch=lambda *a, **kw: calls.append(("matmul", a, kw)))
    scratch = SimpleNamespace(**{
        name: SimpleNamespace(ptr=0x1000 + index * 0x100)
        for index, name in enumerate((
            "norm", "gated_context", "token_ids", "query", "key", "value",
            "gate_logits", "attention_output", "context",
        ))
    })
    weights = {name: index + 1 for index, name in enumerate(("attn_q", "attn_k", "attn_v", "attn_gate", "attn_output"))}
    layer = SimpleNamespace(
        attention_type=FULL_ATTENTION,
        weight=lambda name: SimpleNamespace(allocation=lambda key: SimpleNamespace(tensor=SimpleNamespace(ptr=weights[name]))),
    )
    for name in ("bf16_to_fp16", "bf16_to_fp16_scaled_rows", "f32_scale_rows", "f32_to_bf16", "f32_scale_rows_to_bf16"):
        monkeypatch.setattr(runner, name, lambda *a, _name=name, **kw: calls.append((_name, a, kw)))
    session._launch_attention_projections_rows(layer, scratch, rows=rows, q_width=12, kv_width=4, heads=3, stream=7)
    direct = mode == "hipblaslt_range_direct"
    expected_casts = ([] if fused else ["bf16_to_fp16"]) if direct else ["bf16_to_fp16_scaled_rows"]
    assert [name for name, _, _ in calls if name.startswith("bf16")] == expected_casts
    matmuls = [(a, kw) for name, a, kw in calls if name == "matmul"]
    assert len(matmuls) == 4
    for (args, kw), slot, dest, width in zip(matmuls, ("attn_q", "attn_k", "attn_v", "attn_gate"), (scratch.query, scratch.key, scratch.value, scratch.gate_logits), (12, 4, 4, 3), strict=True):
        assert args == (scratch.gated_context.ptr, weights[slot], dest.ptr, rows, 8, width)
        assert kw == {"stream": 7}
    assert sum(name == "f32_scale_rows" for name, _, _ in calls) == (0 if direct else 4)
    assert all(kw["runtime"] is session.runtime and kw["stream"] == 7 for name, _, kw in calls if name != "matmul")
    calls.clear()
    session._launch_attention_output_rows(layer, scratch, rows=rows, q_width=12, stream=7)
    assert [name for name, _, _ in calls] == (
        ([] if fused else ["bf16_to_fp16"]) + ["matmul", "f32_to_bf16"] if direct
        else ["bf16_to_fp16_scaled_rows", "matmul", "f32_scale_rows_to_bf16"]
    )
    args = next(a for name, a, _ in calls if name == "matmul")
    assert args == (scratch.gated_context.ptr, weights["attn_output"], scratch.context.ptr, rows, 12, 8)
    output_args = calls[-1][1]
    assert output_args[0] == scratch.context.ptr
    assert scratch.attention_output.ptr in output_args


@pytest.mark.parametrize("mode", comparison._MODES)
@pytest.mark.parametrize("failure", [None, "setter", "prefill"])
def test_f16_comparison_uses_real_setter_and_closes_on_each_exit(monkeypatch, mode, failure):
    child = object.__new__(runner.LagunaGGUFResidentSession)
    child.position = 511
    events = []
    child.runtime = SimpleNamespace(device_synchronize=lambda: events.append("sync"))
    child.close = lambda: events.append("close")
    if failure == "setter":
        child.set_f16_boundary_fusion = lambda enabled: (_ for _ in ()).throw(RuntimeError("synthetic setter failure"))

    def prefill(tokens):
        assert tokens == [1, 2, 3]
        assert child.fuse_f16_boundaries is comparison._MODES[mode]
        if failure == "prefill":
            raise RuntimeError("synthetic prefill failure")
        return SimpleNamespace(next_token_id=42, next_token_logit=1.5, logits="logits", final_hidden="final", post_layer_hidden="post")

    child.prefill = prefill
    monkeypatch.setattr(comparison, "_child_session", lambda owner: child)
    monkeypatch.setattr(comparison, "_device_digest", lambda runtime, value, dtype: value)
    monkeypatch.setattr(comparison, "_kv_digest", lambda selected: "kv" if selected is child else pytest.fail("wrong child"))
    ticks = iter((10.0, 11.0))
    monkeypatch.setattr(comparison.time, "perf_counter", lambda: next(ticks))
    if failure:
        with pytest.raises(RuntimeError, match="synthetic"):
            comparison._run(object(), mode, [1, 2, 3])
        assert events == ["close"]
    else:
        record = comparison._run(object(), mode, [1, 2, 3])
        assert record["next_token"] == 42 and record["final_position"] == 511
        assert record["logits_sha256"] == "logits" and record["kv_sha256"] == "kv"
        assert record["final_hidden_sha256"] == "final" and record["post_layer_hidden_sha256"] == "post"
        assert events == ["sync", "close"]
