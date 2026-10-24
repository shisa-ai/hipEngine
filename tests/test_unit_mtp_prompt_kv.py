"""Prompt-only KV appends must not truncate hidden-producing repair steps.

A device-token row must also leave the int64 chain token alone: the token
scalar it reads belongs to the chain, not to the host metadata upload.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.generation.qwen35_paro_mtp2 import Qwen35ParoMTP2Adapter, _ParoMTP2RequestState
from hipengine.speculative import mtp_native


def test_prompt_kv_append_forwards_kv_write_only_to_advance():
    proposer = object.__new__(mtp_native.NativeMtpChainProposer)
    calls = []
    proposer.advance = lambda **kwargs: calls.append(kwargs)

    proposer.append_prompt_kv(
        input_token=13, target_hidden_ptr=0x1234, position=9, stream=7,
    )

    assert calls == [{
        "input_token": 13, "target_hidden_ptr": 0x1234, "position": 9,
        "need_result": False, "kv_write_only": True,
        "read_token_id": False, "read_expert_topk": False,
        "read_lm_head_value": False, "stream": 7,
    }]


@pytest.mark.parametrize("prompt", [(10,), (10, 11, 12), tuple(range(17))])
def test_only_nonfinal_prompt_rows_are_kv_only(prompt):
    calls = []
    proposer = SimpleNamespace(
        append_prompt_kv=lambda **kw: calls.append(("kv", kw)),
        advance_with_target_hidden=lambda **kw: calls.append(("full", kw)),
    )
    row = SimpleNamespace(prompt_ids=prompt, mtp2_prompt_prime_ms=0.0)
    owner = SimpleNamespace(generator=SimpleNamespace(backend="hip_gfx1100"), _row=lambda _rid: row)
    adapter = Qwen35ParoMTP2Adapter(owner)
    adapter._states[7] = _ParoMTP2RequestState(7, proposer)
    # Individual calls are the chunk/lifetime boundary; only the real final
    # prompt index can carry a seed, regardless of how calls are partitioned.
    for index in range(len(prompt)):
        adapter.consume_prompt_row(
            7, prompt_index=index, target_hidden_ptr=0x1000 + 16 * index,
            seed_token=99 if index == len(prompt) - 1 else None,
        )
    assert [kind for kind, _ in calls] == ["kv"] * (len(prompt) - 1) + ["full"]
    assert [kw["input_token"] for _, kw in calls] == [*prompt[1:], 99]
    assert [kw["position"] for _, kw in calls] == list(range(1, len(prompt) + 1))
    assert all(kw.get("read_token_id", False) is False for _, kw in calls)
    assert calls[-1][1]["need_result"] is True
    assert adapter._states[7].prompt_rows_consumed == len(prompt)


def _prefix_proposer(monkeypatch, *, indexed):
    proposer = object.__new__(mtp_native.NativeMtpChainProposer)
    proposer.closed = False
    proposer.scoring_head = None
    proposer.cache_len = 2
    proposer.position = 2
    proposer.max_mtp_tokens = 64
    proposer.max_positions = 128
    proposer.hidden = 4
    proposer.vocab = 16
    proposer.kv_features = 4
    proposer.q_proj_features = 8
    proposer.q_heads = proposer.kv_heads = 1
    proposer.head_dim = proposer.rotary_dim = 4
    proposer.eps = 1e-6
    proposer.current = mtp_native.NativeMtpStepResult(token=77, logit=1.0, topk_experts=(), topk_logits=())
    proposer.token_position_host = np.zeros(3, dtype=np.int64)
    proposer.token_host = proposer.token_position_host[:1]
    proposer.position_host = proposer.token_position_host[1:2]
    proposer.cache_slot_host = proposer.token_position_host[2:]
    buffers = ("token_position_buf", "token_buf", "position_buf", "cache_slot_buf", "key_cache_buf", "value_cache_buf", "final_hidden_buf", "fused_buf", "fc_buf", "attn_in_buf", "q_proj_buf", "k_proj_buf", "query_buf", "gate_buf", "query_rot_buf", "q_norm_oneplus", "k_norm_oneplus", "cos_buf", "sin_buf")
    for index, name in enumerate(buffers):
        setattr(proposer, name, SimpleNamespace(ptr=0x1000 * (index + 1)))
    proposer.mtp_lib = proposer.dflash_lib = object()
    proposer.weights = {name: SimpleNamespace(ptr=0x100000) for name in (
        "embed_tokens.weight", "mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight", "mtp.fc.weight", "mtp.layers.0.input_layernorm.weight", "mtp.layers.0.self_attn.q_proj.weight", "mtp.layers.0.self_attn.k_proj.weight", "mtp.layers.0.self_attn.v_proj.weight",
    )}
    calls = []
    proposer.runtime = SimpleNamespace(memcpy_async=lambda *args: calls.append(("copy", args)))
    for name in ("mtp_fuse_inputs_f16_bf16", "dflash_dense_bf16_to_bf16", "mtp_rmsnorm_bf16_oneplus", "dflash_qkv_proj_bf16_mixed_indexed_v", "dflash_qkv_proj_bf16_mixed", "mtp_split_q_gate_f32_bf16", "dflash_head_rmsnorm_rotary_indexed_key_f32", "dflash_head_rmsnorm_rotary_f32"):
        monkeypatch.setattr(mtp_native, name, lambda *args, _name=name, **kwargs: calls.append((_name, args)))
    monkeypatch.setattr(mtp_native, "_indexed_kv_write_enabled", lambda: indexed)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("attention suffix executed")
    monkeypatch.setattr(mtp_native, "dflash_gqa_attention_f32_bf16", forbidden)
    return proposer, calls


@pytest.mark.parametrize("indexed", [False, True])
def test_kv_only_appends_existing_prefix_and_advances_cursors(monkeypatch, indexed):
    proposer, calls = _prefix_proposer(monkeypatch, indexed=indexed)
    proposer.append_prompt_kv(input_token=13, target_hidden_ptr=0xABCD, position=3)
    assert proposer.cache_len == 3
    assert proposer.position == 3
    assert proposer.current.token == -1  # No stale prediction is consumable.
    assert proposer.final_hidden_valid is False
    assert proposer.result_valid is False
    names = [name for name, _ in calls]
    assert "mtp_fuse_inputs_f16_bf16" in names
    assert any("rotary" in name for name in names)
    # A host-token row uploads token, position, and slot in one packed copy, and
    # the K/V projection writes the cache directly.
    assert names.count("copy") == 1
    assert proposer.token_position_host.tolist() == [13, 3, 2]


def test_hidden_producing_repair_is_not_kv_only(monkeypatch):
    proposer, _ = _prefix_proposer(monkeypatch, indexed=True)
    proposer.attn_out_buf = SimpleNamespace(ptr=0xBEEF)
    with pytest.raises(AssertionError, match="attention suffix executed"):
        proposer.advance(input_token=13, target_hidden_ptr=0xABCD, position=3, need_result=False)


@pytest.mark.parametrize("indexed", [False, True])
def test_device_token_row_publishes_metadata_without_uploading_the_token(monkeypatch, indexed):
    """A device-chain row must not have its token scalar overwritten by a copy."""

    proposer, calls = _prefix_proposer(monkeypatch, indexed=indexed)

    class _Launched(Exception):
        pass

    def launched(*_args, **_kwargs):
        raise _Launched

    monkeypatch.setattr(mtp_native, "mtp_fuse_inputs_f16_bf16", launched)
    with pytest.raises(_Launched):
        proposer.advance(
            input_token_ptr=0xCAFE,
            target_hidden_ptr=0xABCD,
            position=3,
            need_result=False,
            read_token_id=False,
            read_expert_topk=False,
            read_lm_head_value=False,
        )
    destinations = [args[0] for name, args in calls if name == "copy"]
    # Position (and the slot an indexed write reads) go out separately; neither
    # the packed record nor the token scalar is uploaded for a device token.
    assert destinations == [
        proposer.position_buf.ptr,
        *([proposer.cache_slot_buf.ptr] if indexed else []),
    ]
    assert proposer.token_position_host.tolist() == [0, 3, 2]
