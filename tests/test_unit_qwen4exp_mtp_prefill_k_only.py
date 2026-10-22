"""Cache-only prompt attention computes keys/values, not unused queries."""
from types import SimpleNamespace as NS

import pytest

from hipengine.runtime import qwen4_exp_runner as module


@pytest.mark.parametrize("rows,start", [(1, 0), (17, 39), (64, 1023), (65, 2052)])
@pytest.mark.parametrize("rope_positions", [None, 9876])
def test_cache_only_prefill_omits_query_projection(monkeypatch, rows, start, rope_positions):
    runtime = object()
    events = []
    buffer = lambda ptr: NS(ptr=ptr)
    scratch = NS(closed=False, runtime=runtime, **{
        name: buffer(i + 100) for i, name in enumerate((
            "q_projected", "key_projected", "value_projected", "index_q_projected",
            "index_k_projected", "query", "key", "gate", "index_query",
        ))
    })
    projections = {name: name for name in (
        "attn_q", "attn_k", "attn_v", "attn_output", "index_q", "index_k",
    )}
    weights = NS(projections=projections, q_norm_weight_ptr=10, k_norm_weight_ptr=20,
                 index_q_norm_weight_ptr=30, index_k_norm_weight_ptr=40)
    attention = NS(closed=False, runtime=runtime, max_positions=8192,
                   block_table=buffer(200), block_size=16,
                   key_cache=buffer(201), value_cache=buffer(202))
    index = NS(closed=False, runtime=runtime, count=start, index_heads=2, index_dim=8,
               append_rows=lambda *a, **kw: events.append(("index", a, kw)))
    spans = object()
    metadata = NS(closed=False, runtime=runtime, rows=128, positions=buffer(300),
                  set_contiguous=lambda *a: events.append(("positions", a, {})),
                  spans=lambda **kw: spans)
    monkeypatch.setattr(module, "launch_gguf_linear_decode_rows",
                        lambda *a, **kw: events.append(("linear", a, kw)))
    for name in ("qwen4_exp_qsa_norm_rope_rows_f32", "qwen4_exp_qsa_norm_mrope_rows_f32",
                 "qwen35_write_paged_kv_f32_batch_spans"):
        monkeypatch.setattr(module, name,
                            lambda *a, _name=name, **kw: events.append((_name, a, kw)))

    def unused(*a, **kw):
        raise AssertionError("cache-only prefill must not compute queries")

    monkeypatch.setattr(module, "qwen4_exp_qsa_split_norm_rope_rows_f32", unused)
    monkeypatch.setattr(module, "qwen4_exp_qsa_split_norm_mrope_rows_f32", unused)
    result = module.run_qwen4_exp_qsa_prefill_token_mixer(
        1, weights, attention_state=attention, index_state=index, scratch=scratch,
        metadata=metadata, start_position=start, rows=rows, hidden=16,
        query_heads=4, kv_heads=2, head_dim=8, rotary_dim=4, theta=10000.,
        index_heads=2, index_dim=8, index_rotary_dim=4, runtime=runtime,
        rope_positions_ptr=rope_positions, cache_only=True, stream=7,
    )
    assert result is None
    assert [a[0] for name, a, _ in events if name == "linear"] == ["attn_k", "attn_v", "index_k"]
    name = "qwen4_exp_qsa_norm_rope_rows_f32" if rope_positions is None else "qwen4_exp_qsa_norm_mrope_rows_f32"
    norm = next(event for event in events if event[0] == name)
    assert norm[1][:8] == (scratch.key_projected.ptr, 20, rope_positions or 300,
                          scratch.key.ptr, rows, 2, 8, 4)
    assert norm[2] == {"stream": 7, "runtime": runtime}
    assert [e[0] for e in events][-2:] == ["index", "qwen35_write_paged_kv_f32_batch_spans"]
