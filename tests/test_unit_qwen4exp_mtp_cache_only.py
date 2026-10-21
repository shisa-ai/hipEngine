"""Prompt priming writes every draft cache row without discarded layer output."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from hipengine.runtime import qwen4_exp_mtp as draft_module
from hipengine.runtime import qwen4_exp_runner as qsa
from tests.test_unit_qwen4exp_prefill_chunks import _draft


@pytest.mark.parametrize("chunks", [(1,), (5,), (2, 2, 1)])
def test_priming_only_finishes_the_last_row_of_each_chunk(chunks):
    draft = _draft()
    calls = []
    original = draft.forward

    def forward(token, hidden, **kwargs):
        calls.append(kwargs)
        original(token, hidden, **kwargs)

    draft.forward = forward
    offset = 0
    hidden = np.arange(sum(chunks) * 2, dtype=np.float32).reshape(-1, 2)
    for size in chunks:
        draft.prime_prompt_chunk(tuple(range(offset, offset + size)), hidden[offset:offset + size],
                                 previous_hidden=None if offset == 0 else hidden[offset - 1])
        offset += size
    assert [call.get("cache_only", False) for call in calls] == [
        flag for size in chunks for flag in [True] * (size - 1) + [False]
    ]
    assert all(call["run_head"] is False for call in calls)
    assert draft.position == sum(chunks)


def mixer_fixture(monkeypatch, position):
    events = []
    owner = object()
    names = ("q_projected", "key_projected", "value_projected", "index_k_projected",
             "index_q_projected", "index_query", "query", "key", "gate", "context", "gated", "output")
    scratch = NS(closed=False, runtime=owner, **{name: NS(ptr=i + 100) for i, name in enumerate(names)})
    state = NS(closed=False, runtime=owner, max_positions=4096, position=NS(ptr=10),
               key_cache=NS(ptr=11), value_cache=NS(ptr=12), block_size=256,
               append_spans=object(), decode_spans=object(),
               set_position=lambda pos: events.append(("position", pos)))
    index = NS(index_heads=1, index_dim=2, count=position, dense_equivalent_limit=1024)

    def append(ptr, *, position, stream):
        events.append(("index_append", ptr, position))
        index.count = position + 1

    def select(*args, **kwargs):
        events.append(("index_select",))
        return 1, NS(ptr=14)

    index.append, index.select = append, select
    weights = NS(projections={name: name for name in
        ("attn_q", "attn_k", "attn_v", "attn_output", "index_q", "index_k")},
        q_norm_weight_ptr=21, k_norm_weight_ptr=22, index_q_norm_weight_ptr=23, index_k_norm_weight_ptr=24)
    monkeypatch.setattr(qsa, "launch_gguf_linear", lambda slot, *args, **kw: events.append(("linear", slot)))
    for name in ("qwen4_exp_qsa_norm_rope_f32", "qwen4_exp_qsa_split_norm_rope_f32",
                 "qwen35_write_paged_kv_f32_spans", "qwen35_paged_full_attn_decode_context_bf16_spans",
                 "qwen4_exp_qsa_sparse_attention_paged_bf16_f32", "qwen4_exp_qsa_gate_context_f32"):
        monkeypatch.setattr(qsa, name, lambda *a, _name=name, **kw: events.append((_name,)))
    args = dict(attention_state=state, scratch=scratch, position=position, rows=1, hidden=4,
                query_heads=2, kv_heads=1, head_dim=2, rotary_dim=2, theta=10000.0,
                index_state=index, index_heads=1, index_dim=2, index_rotary_dim=2, runtime=owner)
    return events, weights, args, index


@pytest.mark.parametrize("position", [0, 997, 1023, 1025, 2051])
def test_cache_only_mixer_keeps_kv_and_index_writes_but_skips_attention(monkeypatch, position):
    events, weights, args, index = mixer_fixture(monkeypatch, position)
    result = qsa.run_qwen4_exp_dense_qsa_token_mixer(1, weights, cache_only=True, **args)
    assert result is None
    assert index.count == position + 1
    assert events[-1] == ("qwen35_write_paged_kv_f32_spans",)
    assert ("index_append", args["scratch"].index_k_projected.ptr, position) in events
    assert ("index_select",) not in events
    assert ("linear", "index_q") not in events
    assert ("linear", "attn_output") not in events
    cached_events = list(events)
    events.clear()
    qsa.run_qwen4_exp_dense_qsa_token_mixer(1, weights, **args)
    # All operations that construct cache state keep their order and arguments.
    cache_names = {"position", "index_append", "qwen4_exp_qsa_split_norm_rope_f32",
                   "qwen35_write_paged_kv_f32_spans"}
    def cache_events(trace):
        return [event for event in trace if event[0] in cache_names or
                (event[0] == "linear" and event[1] in ("attn_q", "attn_k", "attn_v", "index_k"))]
    assert cache_events(events) == cached_events
    assert events[-1] == ("linear", "attn_output")


def test_cache_only_layer_stops_before_residual_and_feed_forward(monkeypatch):
    owner, calls = object(), []
    scratch = NS(closed=False, runtime=owner, attention_gr=object(), qsa=object())
    gr = NS(norm_weight_ptr=1, down=object(), up=object(), inject=object())
    weights = NS(attention_gr=gr, mixer=object())
    monkeypatch.setattr(qsa, "run_qwen4_exp_gr_read", lambda *a, **kw: NS(mixed=NS(ptr=2)))
    monkeypatch.setattr(qsa, "run_qwen4_exp_dense_qsa_token_mixer", lambda *a, **kw: calls.append(kw))
    result = qsa.run_qwen4_exp_dense_qsa_layer(1, weights, attention_state=object(), scratch=scratch,
        position=0, rows=1, branches=2, hidden=4, low_rank=1, query_heads=2, kv_heads=1,
        head_dim=2, rotary_dim=2, theta=10000., ffn=8, experts=4, top_k=2,
        runtime=owner, cache_only=True)
    assert result is None and calls[0]["cache_only"] is True


def test_cache_only_forward_does_not_copy_an_uncomputed_hidden_row(monkeypatch):
    draft = object.__new__(draft_module.Qwen4ExpGGUFMTPDraftRunner)
    draft.position, draft.max_sequence_length, draft.closed = 0, 8, False
    draft.config = NS(**{name: 2 for name in ("residual_branch_count", "hidden_size", "residual_low_rank",
        "attention_head_count", "attention_kv_head_count", "attention_key_length", "rope_dimension_count",
        "rope_freq_base", "indexer_head_count", "indexer_key_length", "expert_feed_forward_length",
        "expert_count", "expert_used_count")})
    draft._buffers = [NS(ptr=i, nbytes=4) for i in range(12)]
    draft.runtime = NS(memcpy=lambda *a: pytest.fail("cache-only must not copy layer output"))
    draft.binding = draft.attention = draft.index = draft.layer_scratch = object()
    draft._fuse_inputs = lambda *a, **kw: None
    calls = []
    monkeypatch.setattr(draft_module, "run_qwen4_exp_dense_qsa_layer", lambda *a, **kw: calls.append(kw))
    result = draft.forward(1, np.zeros(2), run_head=False, cache_only=True)
    assert result.token_id == -1 and result.hidden_seed is None and result.logits is None
    assert draft.position == 1 and calls[0]["cache_only"] is True
    with pytest.raises(ValueError, match="head"):
        draft.forward(1, np.zeros(2), run_head=True, cache_only=True)
    assert draft.position == 1 and len(calls) == 1
