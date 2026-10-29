"""Host execution contracts for packed attention output and its live comparison."""

from types import SimpleNamespace

import pytest

from hipengine.runtime import laguna_gguf_runner as runner
from scripts import laguna_attention_packed_output_gate_ab as comparison


@pytest.mark.parametrize("mode", tuple(comparison._COMPARISONS["packed_output_gate"]))
@pytest.mark.parametrize("rows", [128, 257, 384])
@pytest.mark.parametrize("direct_fp16", [False, True])
def test_output_comparison_arms_reach_distinct_real_layer_gate_routes(monkeypatch, mode, rows, direct_fp16):
    calls = []
    settings = comparison._COMPARISONS["packed_output_gate"][mode]
    session = object.__new__(runner.LagunaGGUFResidentSession)
    layer = SimpleNamespace(
        attention_type=runner.FULL_ATTENTION, mlp_type=runner.DENSE_MLP,
        weight=lambda name: SimpleNamespace(allocation=lambda key: SimpleNamespace(tensor=SimpleNamespace(ptr=0x1000))),
    )
    config = SimpleNamespace(
        head_count=lambda index: 48, head_count_kv=8, key_length=128,
        value_length=128, hidden_size=64, rms_norm_eps=1e-5,
    )
    session.weights = SimpleNamespace(config=config, layer=lambda index: layer)
    session.rows_scratch = SimpleNamespace(**{
        name: SimpleNamespace(ptr=0x2000 + index * 0x100000)
        for index, name in enumerate((
            "hidden", "gated_context", "norm", "query", "key", "positions",
            "query_rotated", "key_rotated", "value", "context", "gate_logits",
            "attention_output", "post_attention",
        ))
    })
    scratch = session.rows_scratch
    state = SimpleNamespace(q_heads=48, key_cache=SimpleNamespace(ptr=0xD000000), value_cache=SimpleNamespace(ptr=0xE000000))
    session.kv_cache = SimpleNamespace(
        can_dense_initial_prefill=lambda layer, count, **kw: count == 128,
        dense_initial_prefill_view=lambda layer, count, **kw: (state, object(), kw.get("row_offset", 0)),
        can_preappend_attention_prefill=lambda layer, count, **kw: count == 128,
        append_rows=lambda *a, **kw: calls.append(("append", a, kw)),
        attend_prefill_cached=lambda *a, **kw: calls.append(("cached", a, kw)),
        attend_prefill=lambda *a, **kw: calls.append(("fallback", a, kw)),
    )
    record = lambda label: lambda *a, **kw: calls.append((label, a, kw))
    session.kernel_plan = SimpleNamespace(
        rmsnorm=record("norm"), rmsnorm_fp16_via_bf16=record("norm_fp16"),
        attention_gate=record("gate"), attention_gate_fp16_via_bf16=record("gate_fp16"),
        add_rmsnorm=record("add_norm"),
    )
    session.libraries = SimpleNamespace(gguf_ops=object(), kv_attention=object(), attention_gate=object())
    session.full_rope = session.swa_rope = object()
    session.runtime = object()
    session.backend = "hip_gfx1151"
    session.f16_prefill_mode = "hipblaslt_range_direct" if direct_fp16 else "hipblaslt"
    session.fuse_f16_boundaries = True
    session.prefill_global_attention_chunk_size = 2048
    session.prefill_chunk_policy = SimpleNamespace(attention_rows=128)
    session.context_length = 1024
    session.prefill_kv_preappend = True
    session.prefill_attention_hipblaslt = True
    session.prefill_attention_hipblaslt_packed_queries = True
    session.prefill_attention_hipblaslt_packed_query_producer = settings["packed_query_producer"]
    session.prefill_attention_hipblaslt_packed_output_gate = settings["packed_output_gate"]
    session.prefill_long_attention_hipblaslt = session.prefill_block_attention_hipblaslt = False
    session.prefill_swa_attention_hipblaslt = session.prefill_dense_contiguous_cache = False
    session._launch_attention_projections_rows = record("projection")
    session._launch_attention_output_rows = record("output")
    session._run_dense_ffn_rows = record("ffn")
    session._ensure_attention_hipblaslt = lambda: SimpleNamespace(launch=record("blas"))
    monkeypatch.setattr(runner, "launch_laguna_head_rmsnorm_rope", record("head"))
    monkeypatch.setattr(runner, "laguna_softplus_head_gate_f32_bf16_packed_tiles_out", record("packed_gate"))
    monkeypatch.setattr(runner, "laguna_softplus_head_gate_f32_fp16_via_bf16_packed_tiles_out", record("packed_gate_fp16"))
    session._run_layer_rows(0, rows=rows, stream=7)

    eligible_offsets = [offset for offset in range(128, rows - 127, 128)]
    blas_calls = [entry for entry in calls if entry[0] == "blas"]
    assert len(blas_calls) == len(eligible_offsets)
    expected_packed = settings["packed_output_gate"] and bool(eligible_offsets)
    for (_, args, kwargs), offset in zip(blas_calls, eligible_offsets, strict=True):
        assert kwargs["query_is_packed"]
        assert kwargs["unpack_output"] is not settings["packed_output_gate"]
        assert args[0] == scratch.query_rotated.ptr + offset * 48 * 128 * 4
        assert args[3] == scratch.context.ptr + offset * 48 * 128 * 4
        assert kwargs["stream"] == 7
    gate_calls = [entry for entry in calls if "gate" in entry[0]]
    label, args, kwargs = gate_calls.pop()
    assert label == ("packed_gate" if expected_packed else "gate") + ("_fp16" if direct_fp16 else "")
    assert args[:6] == (scratch.context.ptr, scratch.gate_logits.ptr, scratch.gated_context.ptr, rows, 48, 128)
    if expected_packed:
        assert args[6:] == (128, eligible_offsets[-1] + 128)
    assert kwargs["stream"] == 7 and kwargs["runtime"] is session.runtime
    head = next(entry for entry in calls if entry[0] == "head")
    assert head[2]["packed_query_begin"] == (128 if eligible_offsets else None)
    assert head[2]["packed_query_end"] == (eligible_offsets[-1] + 128 if eligible_offsets else None)
    assert len([entry for entry in calls if entry[0] == "fallback"]) == (rows % 128 != 0)
    assert calls[-1][0] == "ffn"


@pytest.mark.parametrize("failure", [None, "prefill", "digest"])
def test_output_comparison_captures_state_and_closes_on_failure(monkeypatch, failure):
    calls = []

    def prefill(tokens):
        calls.append(("prefill", tuple(tokens)))
        if failure == "prefill":
            raise RuntimeError("synthetic prefill failure")
        return SimpleNamespace(next_token_id=9, next_token_logit=0.5, logits="logits", final_hidden="final", post_layer_hidden="post")

    child = SimpleNamespace(
        prefill=prefill, position=512, close=lambda: calls.append(("close",)),
        runtime=SimpleNamespace(device_synchronize=lambda: calls.append(("sync",))),
    )
    settings = comparison._COMPARISONS["packed_output_gate"]["packed_output_gate_candidate"]
    owner = object()

    def construct(actual_owner, actual_settings):
        assert actual_owner is owner and actual_settings is settings
        return child

    def digest(runtime, value, dtype):
        assert runtime is child.runtime
        if failure == "digest":
            raise RuntimeError("synthetic digest failure")
        return "digest:" + value

    monkeypatch.setattr(comparison, "_child", construct)
    monkeypatch.setattr(comparison, "_device_digest", digest)
    monkeypatch.setattr(comparison, "_kv_digest", lambda session: "digest:kv")
    clock = iter((1.0, 2.0))
    monkeypatch.setattr(comparison.time, "perf_counter", lambda: next(clock))
    if failure:
        with pytest.raises(RuntimeError, match=f"synthetic {failure} failure"):
            comparison._run(owner, "packed_output_gate_candidate", [1] * 512, settings)
    else:
        record = comparison._run(owner, "packed_output_gate_candidate", [1] * 512, settings)
        assert record == {
            "tok_s": 512.0, "elapsed_s": 1.0, "next_token": 9,
            "next_token_logit_hex": float(0.5).hex(), "logits_sha256": "digest:logits",
            "final_hidden_sha256": "digest:final", "post_layer_hidden_sha256": "digest:post",
            "kv_sha256": "digest:kv", "final_position": 512,
        }
        assert set(comparison._STATE_KEYS) == set(record) - {"tok_s", "elapsed_s"}
    assert calls[-1] == ("close",)


def test_paired_result_names_repaired_comparison_basis(monkeypatch, capsys):
    monkeypatch.setattr(comparison, "_parse_args", lambda: SimpleNamespace(
        comparison="packed_output_gate", repetitions=1, require_cached_build=False, output=None,
    ))
    monkeypatch.setattr(comparison, "GGUFReader", lambda path: SimpleNamespace(info=object()))
    monkeypatch.setattr(comparison, "LagunaGGUFTokenizer", SimpleNamespace(from_gguf_info=lambda info: object()))
    monkeypatch.setattr(comparison, "_load_prompts", lambda *a: [])
    monkeypatch.setattr(comparison, "_profile_token_stream", lambda *a: ([1] * 512, None))
    monkeypatch.setattr(comparison, "_compiler_version", lambda path: "test")
    runtime = SimpleNamespace(device_synchronize=lambda: None)
    owner = SimpleNamespace(close=lambda: None)
    child = SimpleNamespace(prefill=lambda tokens: None, close=lambda: None)
    monkeypatch.setattr(comparison, "get_hip_runtime", lambda: runtime)
    monkeypatch.setattr(comparison, "LagunaGGUFResidentSession", lambda *a, **kw: owner)
    monkeypatch.setattr(comparison, "_child", lambda *a: child)

    def run(actual_owner, mode, tokens, settings):
        assert actual_owner is owner and settings["packed_query_producer"]
        return {"tok_s": 1.0, **dict.fromkeys(comparison._STATE_KEYS, "same"), "next_token": 9}

    monkeypatch.setattr(comparison, "_run", run)
    assert comparison.main() == 0
    output = capsys.readouterr().out
    import json

    result = json.loads(output[output.index("{"):])
    assert result["comparison"]["basis"] == "packed_query_producer_on_output_gate_only"
    assert result["comparison"]["complete_state_exact"]
    for mode in comparison._COMPARISONS["packed_output_gate"]:
        assert result[mode]["packed_query_producer"]
