"""CPU contract tests for K/V-only historical prompt priming in the native MTP proposer.

Historical prompt rows hand the proposer a target-supplied hidden state. For
those rows the proposer only has to append the row's K/V and publish the cache
cursor: nothing downstream of the attention read feeds the cache, the router, or
the proposal head. ``append_prompt_kv()`` (``advance(kv_write_only=True)``) is
that bounded operation.

The tests here are CPU-only. Every HIP kernel entry point, the runtime, and the
weight loader are replaced with recorders, so the assertions are about the exact
call boundary, the K/V destination addresses, the published cursor, and the
invalidation of the proposal/hidden surfaces -- not about device numerics.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.core.hip import HipMemcpyKind
from hipengine.core.memory import DeviceBuffer
from hipengine.speculative import mtp_native


_HIDDEN = 32
_VOCAB = 64
_Q_HEADS = 4
_KV_HEADS = 2
_HEAD_DIM = 8
_ROTARY_DIM = 8
_EXPERTS = 8
_TOP_K = 2
_INTERMEDIATE = 16
_SHARED_INTERMEDIATE = 16
_MAX_POSITIONS = 64
_MAX_MTP_TOKENS = 8
_LAYER = "mtp.layers.0"

_TARGET_HIDDEN_PTR = 0x500000

# Kernel entry points replaced with recorders. The list is the whole set of
# device work ``advance()`` can issue, so a cache-only step that calls anything
# outside the K/V producer prefix is visible as an unexpected record.
_KERNEL_NAMES = (
    "mtp_fuse_inputs_f16_bf16",
    "dflash_dense_bf16_to_bf16",
    "mtp_rmsnorm_bf16_oneplus",
    "dflash_qkv_proj_bf16_mixed",
    "dflash_qkv_proj_bf16_mixed_indexed_v",
    "mtp_split_q_gate_f32_bf16",
    "dflash_head_rmsnorm_rotary_f32",
    "dflash_head_rmsnorm_rotary_indexed_key_f32",
    "dflash_gqa_attention_f32_bf16",
    "mtp_gate_mul_bf16",
    "mtp_add_rmsnorm_bf16_oneplus",
    "dflash_dense_bf16_to_f32",
    "mtp_router_topk_softmax_f32",
    "topk_f32_rows_i32",
    "mtp_softmax_topk_f32",
    "dflash_dense_bf16_to_bf16_expert_routes",
    "dflash_silu_mul_gate_up_routes_bf16",
    "mtp_accumulate_routes_bf16_to_f32",
    "dflash_dense_bf16_to_bf16_expert",
    "dflash_silu_mul_bf16",
    "mtp_accumulate_route_bf16_to_f32",
    "mtp_accumulate_sigmoid_gate_bf16_to_f32",
    "mtp_finalize_f32_to_bf16",
    "lm_head_fp16_argmax_bf16",
    "w8a16_lm_head_argmax_rows_bf16",
    "dense_dual_gemv_out_bf16_wmma",
)

# Downstream-of-the-K/V-write work. None of it may run for a cache-only row.
_DOWNSTREAM_KERNELS = (
    "dflash_gqa_attention_f32_bf16",
    "mtp_gate_mul_bf16",
    "mtp_add_rmsnorm_bf16_oneplus",
    "dflash_dense_bf16_to_f32",
    "mtp_router_topk_softmax_f32",
    "topk_f32_rows_i32",
    "mtp_softmax_topk_f32",
    "dflash_dense_bf16_to_bf16_expert_routes",
    "dflash_silu_mul_gate_up_routes_bf16",
    "mtp_accumulate_routes_bf16_to_f32",
    "dflash_dense_bf16_to_bf16_expert",
    "dflash_silu_mul_bf16",
    "mtp_accumulate_route_bf16_to_f32",
    "mtp_accumulate_sigmoid_gate_bf16_to_f32",
    "mtp_finalize_f32_to_bf16",
    "lm_head_fp16_argmax_bf16",
    "w8a16_lm_head_argmax_rows_bf16",
    "dense_dual_gemv_out_bf16_wmma",
)

# Downstream work this fake model shape always runs on a full result step.
_ALWAYS_DOWNSTREAM = (
    "dflash_gqa_attention_f32_bf16",
    "mtp_gate_mul_bf16",
    "mtp_add_rmsnorm_bf16_oneplus",
    "dflash_dense_bf16_to_f32",
    "topk_f32_rows_i32",
    "mtp_softmax_topk_f32",
    "dflash_dense_bf16_to_bf16_expert_routes",
    "dflash_silu_mul_gate_up_routes_bf16",
    "mtp_accumulate_routes_bf16_to_f32",
    "dflash_silu_mul_bf16",
    "mtp_accumulate_sigmoid_gate_bf16_to_f32",
    "mtp_finalize_f32_to_bf16",
    "dense_dual_gemv_out_bf16_wmma",
    "w8a16_lm_head_argmax_rows_bf16",
)

_WEIGHT_NAMES = (
    "embed_tokens.weight",
    "lm_head.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
    "mtp.fc.weight",
    f"{_LAYER}.input_layernorm.weight",
    f"{_LAYER}.self_attn.q_proj.weight",
    f"{_LAYER}.self_attn.k_proj.weight",
    f"{_LAYER}.self_attn.v_proj.weight",
    f"{_LAYER}.self_attn.o_proj.weight",
    f"{_LAYER}.self_attn.q_norm.weight",
    f"{_LAYER}.self_attn.k_norm.weight",
    f"{_LAYER}.post_attention_layernorm.weight",
    f"{_LAYER}.mlp.gate.weight",
    f"{_LAYER}.mlp.experts.gate_up_proj",
    f"{_LAYER}.mlp.experts.down_proj",
    f"{_LAYER}.mlp.shared_expert_gate.weight",
    f"{_LAYER}.mlp.shared_expert.gate_proj.weight",
    f"{_LAYER}.mlp.shared_expert.up_proj.weight",
    f"{_LAYER}.mlp.shared_expert.down_proj.weight",
    "mtp.norm.weight",
)

_ENV_NAMES = (
    "HIPENGINE_MTP_PROPOSER_PACK_TOKEN_POSITION",
    "HIPENGINE_MTP_PROPOSER_DIRECT_KV_WRITE",
    "HIPENGINE_MTP_PROPOSER_INDEXED_KV_WRITE",
    "HIPENGINE_MTP_PROPOSER_ROUTER_TOPK_FUSED",
    "HIPENGINE_MTP_PROPOSER_ROUTE_BATCHED_EXPERT",
    "HIPENGINE_MTP_PROPOSER_SHARED_GATE_UP_DUAL",
    "HIPENGINE_MTP_DRAFT_VOCAB_CAP",
)


class _Recorder:
    def __init__(self) -> None:
        self.kernels: list[tuple[str, tuple, dict]] = []
        self.h2d: list[tuple[int, int]] = []
        self.d2h: list[tuple[int, int]] = []

    def clear(self) -> None:
        self.kernels.clear()
        self.h2d.clear()
        self.d2h.clear()

    def names(self) -> list[str]:
        return [name for name, _args, _kwargs in self.kernels]

    def calls(self, name: str) -> list[tuple]:
        return [args for kernel, args, _kwargs in self.kernels if kernel == name]


class _FakeRuntime:
    def __init__(self) -> None:
        self.copies: list[tuple[int, int, int, object, int]] = []
        self.memsets: list[tuple[int, int, int, int]] = []
        self.synchronizes = 0

    def memcpy_async(self, dst, src, nbytes, kind, stream=0) -> None:
        self.copies.append((int(dst), int(src), int(nbytes), kind, int(stream)))

    def memset_async(self, ptr, value, nbytes, stream=0) -> None:
        self.memsets.append((int(ptr), int(value), int(nbytes), int(stream)))

    def device_synchronize(self) -> None:
        self.synchronizes += 1

    def stream_synchronize(self, stream) -> None:
        return None


class _Harness:
    def __init__(self, rec: _Recorder, runtime: _FakeRuntime, make) -> None:
        self.rec = rec
        self.runtime = runtime
        self.make = make

    @property
    def d2d_copies(self) -> list[tuple[int, int, int, object, int]]:
        return [copy for copy in self.runtime.copies if copy[3] is HipMemcpyKind.DEVICE_TO_DEVICE]


def _recorder(rec: _Recorder, name: str):
    def record(*args, **kwargs):
        rec.kernels.append((name, args, kwargs))
        return None

    return record


def _fake_config() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=_HIDDEN,
        vocab_size=_VOCAB,
        num_attention_heads=_Q_HEADS,
        num_key_value_heads=_KV_HEADS,
        head_dim=_HEAD_DIM,
        rotary_dim=_ROTARY_DIM,
        num_experts=_EXPERTS,
        num_experts_per_tok=_TOP_K,
        moe_intermediate_size=_INTERMEDIATE,
        shared_expert_intermediate_size=_SHARED_INTERMEDIATE,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
    )


@pytest.fixture(autouse=True)
def _isolate_proposer_environment(monkeypatch: pytest.MonkeyPatch):
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    yield


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> _Harness:
    rec = _Recorder()
    runtime = _FakeRuntime()
    next_ptr = [0x100000]

    def fake_malloc(nbytes, *, runtime=None):
        ptr = next_ptr[0]
        next_ptr[0] += max(int(nbytes), 64) + 256
        return DeviceBuffer(ptr=ptr, nbytes=int(nbytes))

    infos = {
        name: SimpleNamespace(name=name, shape=(4, 4), dtype="BF16")
        for name in _WEIGHT_NAMES
    }

    def fake_load_info(info, *, device=None, runtime=None):
        ptr = fake_malloc(64).ptr
        return SimpleNamespace(
            name=info.name,
            tensor=SimpleNamespace(ptr=ptr),
            buffer=DeviceBuffer(ptr=ptr, nbytes=64),
            free=lambda *, runtime=None: None,
        )

    monkeypatch.setattr(mtp_native, "malloc", fake_malloc)
    monkeypatch.setattr(mtp_native, "free", lambda buffer, *, runtime=None: None)
    monkeypatch.setattr(
        mtp_native,
        "copy_host_to_device",
        lambda buffer, host_ptr, nbytes=None, *, runtime=None: rec.h2d.append(
            (int(buffer.ptr), int(nbytes if nbytes is not None else buffer.nbytes))
        ),
    )
    monkeypatch.setattr(
        mtp_native,
        "copy_device_to_host",
        lambda host_ptr, buffer, nbytes=None, *, runtime=None: rec.d2h.append(
            (int(buffer.ptr), int(nbytes if nbytes is not None else buffer.nbytes))
        ),
    )
    monkeypatch.setattr(
        mtp_native,
        "_normalized_infos",
        lambda model: (SimpleNamespace(config=None), infos),
    )
    monkeypatch.setattr(
        mtp_native,
        "validate_qwen35_mtp_model",
        lambda model: SimpleNamespace(raise_for_errors=lambda: None),
    )
    monkeypatch.setattr(mtp_native, "qwen35_paro_config_from_hf", lambda config: _fake_config())
    monkeypatch.setattr(mtp_native, "load_tensor_info_to_device", fake_load_info)
    monkeypatch.setattr(
        mtp_native,
        "read_tensor_storage_bytes",
        lambda info: b"\x00" * (4 * 4 * 2),
    )
    monkeypatch.setattr(mtp_native, "build_mtp_speculative", lambda **kwargs: object())
    monkeypatch.setattr(mtp_native, "build_dflash_drafter", lambda **kwargs: object())
    monkeypatch.setattr(mtp_native, "build_lm_head", lambda **kwargs: object())
    monkeypatch.setattr(mtp_native, "build_dense_gemv", lambda **kwargs: object())
    for name in _KERNEL_NAMES:
        monkeypatch.setattr(mtp_native, name, _recorder(rec, name))

    def make(*, scoring: bool = True):
        head = None
        if scoring:
            head = mtp_native.NativeMtpW8A16Head(
                weight_int8_ptr=0x90000,
                scale_f32_ptr=0xA0000,
                vocab_size=_VOCAB,
                threads=256,
            )
        return mtp_native.NativeMtpChainProposer(
            "/nonexistent-mtp-model",
            max_positions=_MAX_POSITIONS,
            max_mtp_tokens=_MAX_MTP_TOKENS,
            runtime=runtime,
            scoring_head=head,
        )

    return _Harness(rec, runtime, make)


def _key_dst(proposer: mtp_native.NativeMtpChainProposer, row: int) -> int:
    return proposer.key_cache_buf.ptr + row * proposer.kv_features * 4


def _value_dst(proposer: mtp_native.NativeMtpChainProposer, row: int) -> int:
    return proposer.value_cache_buf.ptr + row * proposer.kv_features * 2


@pytest.mark.parametrize(
    "direct,indexed",
    [(True, False), (False, False), (True, True)],
    ids=["direct-kv", "memcpy-kv", "indexed-kv"],
)
def test_kv_write_only_reproduces_full_step_prefix_and_appends_identical_kv(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch, direct: bool, indexed: bool
) -> None:
    monkeypatch.setenv("HIPENGINE_MTP_PROPOSER_DIRECT_KV_WRITE", "1" if direct else "0")
    monkeypatch.setenv("HIPENGINE_MTP_PROPOSER_INDEXED_KV_WRITE", "1" if indexed else "0")
    proposer = harness.make()
    kv_features = proposer.kv_features

    # Reference: a full result-producing advance from a fresh state.
    harness.rec.clear()
    harness.runtime.copies.clear()
    proposer.reset()
    proposer.advance_with_target_hidden(
        input_token=101,
        target_hidden_ptr=_TARGET_HIDDEN_PTR,
        position=9,
        need_result=True,
        read_expert_topk=False,
        read_lm_head_value=False,
    )
    full_kernels = list(harness.rec.kernels)
    full_names = [name for name, _args, _kwargs in full_kernels]
    full_copies = list(harness.runtime.copies)

    # Candidate: the same row through the K/V-only operation.
    harness.rec.clear()
    harness.runtime.copies.clear()
    proposer.reset()
    returned = proposer.append_prompt_kv(
        input_token=101,
        target_hidden_ptr=_TARGET_HIDDEN_PTR,
        position=9,
    )
    kv_kernels = list(harness.rec.kernels)
    kv_names = [name for name, _args, _kwargs in kv_kernels]

    assert returned is None
    assert kv_kernels, "a cache-only row must still issue its K/V producer work"
    # Identical call sequence: the cache-only step is exactly the prefix of the
    # full step that produces this row's K/V. Same pointers, same sizes, same
    # order, so the appended K/V bytes cannot depend on the skipped work.
    assert kv_kernels == full_kernels[: len(kv_kernels)]
    for name in _ALWAYS_DOWNSTREAM:
        assert name in full_names, f"{name} must run on a full step"
    assert set(kv_names).isdisjoint(_DOWNSTREAM_KERNELS)

    # The shared DFlash GEMV runs for fc (prefix) but never for o_proj or the
    # shared expert on a cache-only row.
    fc_weight_ptr = proposer.weights["mtp.fc.weight"].ptr
    for args in harness.rec.calls("dflash_dense_bf16_to_bf16"):
        assert args[1] == fc_weight_ptr

    # Cursor publication.
    assert proposer.cache_len == 1
    assert proposer.position == 9

    # K/V append destination and width, per write route.
    if indexed:
        assert proposer.cache_slot_host[0] == 0
        qkv_args = harness.rec.calls("dflash_qkv_proj_bf16_mixed_indexed_v")
        rotary_args = harness.rec.calls("dflash_head_rmsnorm_rotary_indexed_key_f32")
        assert len(qkv_args) == 1 and len(rotary_args) == 1
        assert proposer.value_cache_buf.ptr in qkv_args[0]
        assert proposer.cache_slot_buf.ptr in qkv_args[0]
        assert proposer.key_cache_buf.ptr in rotary_args[0]
        assert proposer.cache_slot_buf.ptr in rotary_args[0]
    elif direct:
        qkv_args = harness.rec.calls("dflash_qkv_proj_bf16_mixed")
        rotary_args = harness.rec.calls("dflash_head_rmsnorm_rotary_f32")
        assert len(qkv_args) == 1 and len(rotary_args) == 1
        assert qkv_args[0][6] == _value_dst(proposer, 0)
        assert rotary_args[0][9] == _key_dst(proposer, 0)
    else:
        d2d = [copy for copy in harness.runtime.copies if copy[3] is HipMemcpyKind.DEVICE_TO_DEVICE]
        assert d2d == [
            (
                _key_dst(proposer, 0),
                proposer.key_rot_buf.ptr,
                kv_features * 4,
                HipMemcpyKind.DEVICE_TO_DEVICE,
                0,
            ),
            (
                _value_dst(proposer, 0),
                proposer.v_proj_buf.ptr,
                kv_features * 2,
                HipMemcpyKind.DEVICE_TO_DEVICE,
                0,
            ),
        ]
        # The full step issues exactly the same two K/V copies before attention.
        full_d2d = [copy for copy in full_copies if copy[3] is HipMemcpyKind.DEVICE_TO_DEVICE]
        assert full_d2d == d2d

    # No host readback: a cache-only row never scores and never syncs to host.
    assert harness.rec.d2h == []


def test_kv_write_only_publishes_cursor_for_the_following_row(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HIPENGINE_MTP_PROPOSER_DIRECT_KV_WRITE", "0")
    proposer = harness.make()
    proposer.reset()

    harness.runtime.copies.clear()
    proposer.append_prompt_kv(input_token=11, target_hidden_ptr=_TARGET_HIDDEN_PTR, position=9)
    proposer.append_prompt_kv(input_token=12, target_hidden_ptr=_TARGET_HIDDEN_PTR + 64, position=10)

    assert proposer.cache_len == 2
    assert proposer.position == 10
    d2d = harness.d2d_copies
    assert d2d[0][0] == _key_dst(proposer, 0)
    assert d2d[1][0] == _value_dst(proposer, 0)
    assert d2d[2][0] == _key_dst(proposer, 1)
    assert d2d[3][0] == _value_dst(proposer, 1)

    # The final full row must attend over every primed row.
    harness.rec.clear()
    proposer.advance_with_target_hidden(
        input_token=13,
        target_hidden_ptr=_TARGET_HIDDEN_PTR + 128,
        position=11,
        read_expert_topk=False,
        read_lm_head_value=False,
    )
    attention = harness.rec.calls("dflash_gqa_attention_f32_bf16")
    assert len(attention) == 1
    assert attention[0][6] == 3
    assert proposer.cache_len == 3
    assert proposer.final_hidden_valid is True
    assert proposer.result_valid is True


def test_kv_write_only_requires_an_explicit_target_hidden(harness: _Harness) -> None:
    proposer = harness.make()

    with pytest.raises(ValueError, match="target hidden"):
        proposer.append_prompt_kv(input_token=11, target_hidden_ptr=0, position=9)
    with pytest.raises(ValueError, match="target hidden"):
        proposer.advance_with_target_hidden(
            input_token=11,
            target_hidden_ptr=proposer.final_hidden_buf.ptr,
            position=9,
            need_result=False,
            kv_write_only=True,
        )
    with pytest.raises(ValueError, match="target hidden"):
        proposer.advance(
            input_token=11,
            target_hidden_ptr=0,
            position=9,
            need_result=False,
            kv_write_only=True,
        )

    # Nothing was appended by a rejected call.
    assert proposer.cache_len == 0
    assert proposer.position == -1


def test_kv_write_only_cannot_produce_a_result(harness: _Harness) -> None:
    proposer = harness.make()

    with pytest.raises(ValueError, match="proposal result"):
        proposer.advance_with_target_hidden(
            input_token=11,
            target_hidden_ptr=_TARGET_HIDDEN_PTR,
            position=9,
            need_result=True,
            kv_write_only=True,
        )
    assert proposer.cache_len == 0


def test_kv_write_only_step_reports_no_prediction(harness: _Harness) -> None:
    proposer = harness.make()
    result = proposer.advance(
        input_token=11,
        target_hidden_ptr=_TARGET_HIDDEN_PTR,
        position=9,
        need_result=False,
        kv_write_only=True,
    )
    assert result.token == -1
    assert result.topk_experts == ()
    assert proposer.current.token == -1
    assert proposer.final_hidden_valid is False
    assert proposer.result_valid is False


def test_legacy_head_path_is_also_skipped(harness: _Harness) -> None:
    proposer = harness.make(scoring=False)
    harness.rec.clear()
    proposer.append_prompt_kv(input_token=11, target_hidden_ptr=_TARGET_HIDDEN_PTR, position=9)
    assert "lm_head_fp16_argmax_bf16" not in harness.rec.names()
    assert proposer.cache_len == 1


def test_previous_hidden_carry_is_rejected_after_a_cache_only_row(harness: _Harness) -> None:
    proposer = harness.make()
    proposer.append_prompt_kv(input_token=11, target_hidden_ptr=_TARGET_HIDDEN_PTR, position=9)

    with pytest.raises(RuntimeError, match="final hidden"):
        proposer.advance_with_previous_hidden(input_token=12, position=10, need_result=False)

    # The rejected carry must not move the cursor.
    assert proposer.cache_len == 1
    assert proposer.position == 9


def test_proposal_surfaces_are_invalid_after_a_cache_only_row(harness: _Harness) -> None:
    proposer = harness.make()
    proposer.append_prompt_kv(input_token=11, target_hidden_ptr=_TARGET_HIDDEN_PTR, position=9)

    with pytest.raises(RuntimeError, match="result-producing advance"):
        proposer.device_candidate_token_ids()
    with pytest.raises(RuntimeError, match="result-producing advance"):
        proposer.vocab_topk(k=4)
    with pytest.raises(RuntimeError, match="result-producing advance"):
        proposer.top1_prob_proxy()
    with pytest.raises(RuntimeError, match="final hidden"):
        proposer.read_final_hidden_bf16()


def test_full_step_after_cache_only_rows_restores_valid_state(harness: _Harness) -> None:
    proposer = harness.make()
    proposer.append_prompt_kv(input_token=11, target_hidden_ptr=_TARGET_HIDDEN_PTR, position=9)
    proposer.advance_with_target_hidden(
        input_token=12,
        target_hidden_ptr=_TARGET_HIDDEN_PTR + 64,
        position=10,
        read_expert_topk=False,
        read_lm_head_value=False,
    )

    assert proposer.final_hidden_valid is True
    assert proposer.result_valid is True
    proposer.device_candidate_token_ids()
    proposer.advance_with_previous_hidden(input_token=13, position=11, need_result=False)
    assert proposer.cache_len == 3


def test_need_result_false_still_carries_a_valid_hidden(harness: _Harness) -> None:
    """Existing ``need_result=False`` semantics are unchanged by this change."""

    proposer = harness.make()
    result = proposer.advance_with_target_hidden(
        input_token=11,
        target_hidden_ptr=_TARGET_HIDDEN_PTR,
        position=9,
        need_result=False,
        read_expert_topk=False,
        read_lm_head_value=False,
    )

    assert result.token == -1
    assert proposer.current.token == -1
    assert proposer.final_hidden_valid is True
    assert proposer.result_valid is False
    assert "w8a16_lm_head_argmax_rows_bf16" not in harness.rec.names()
    proposer.advance_with_previous_hidden(input_token=12, position=10, need_result=False)
    assert proposer.cache_len == 2


def test_reset_clears_hidden_validity(harness: _Harness) -> None:
    proposer = harness.make()
    proposer.advance_with_target_hidden(
        input_token=11,
        target_hidden_ptr=_TARGET_HIDDEN_PTR,
        position=9,
        need_result=False,
        read_expert_topk=False,
        read_lm_head_value=False,
    )
    assert proposer.final_hidden_valid is True

    proposer.reset()

    assert proposer.final_hidden_valid is False
    assert proposer.result_valid is False
    assert proposer.cache_len == 0
    assert proposer.position == -1
    with pytest.raises(RuntimeError, match="final hidden"):
        proposer.advance_with_previous_hidden(input_token=12, position=10, need_result=False)


def test_snapshot_round_trip_carries_hidden_validity(harness: _Harness) -> None:
    proposer = harness.make()
    proposer.advance_with_target_hidden(
        input_token=11,
        target_hidden_ptr=_TARGET_HIDDEN_PTR,
        position=9,
        need_result=False,
        read_expert_topk=False,
        read_lm_head_value=False,
    )
    snapshot = proposer.save_state(0)
    assert snapshot.hidden_valid is True

    # A cache-only row invalidates the hidden; restoring the snapshot revives it.
    proposer.append_prompt_kv(input_token=12, target_hidden_ptr=_TARGET_HIDDEN_PTR + 64, position=10)
    assert proposer.final_hidden_valid is False
    proposer.restore_state(snapshot)
    assert proposer.final_hidden_valid is True
    assert proposer.cache_len == 1
    assert proposer.position == 9
    proposer.advance_with_previous_hidden(input_token=13, position=10, need_result=False)

    # A snapshot taken while the hidden is invalid stays invalid on restore.
    proposer.append_prompt_kv(input_token=14, target_hidden_ptr=_TARGET_HIDDEN_PTR + 128, position=11)
    invalid = proposer.save_state(1)
    assert invalid.hidden_valid is False
    proposer.advance_with_target_hidden(
        input_token=15,
        target_hidden_ptr=_TARGET_HIDDEN_PTR + 192,
        position=12,
        need_result=True,
        read_expert_topk=False,
        read_lm_head_value=False,
    )
    assert proposer.final_hidden_valid is True
    proposer.restore_state(invalid)
    assert proposer.final_hidden_valid is False
    with pytest.raises(RuntimeError, match="result-producing"):
        proposer.device_candidate_token_ids()
    with pytest.raises(RuntimeError, match="final hidden"):
        proposer.advance_with_previous_hidden(input_token=16, position=13, need_result=False)


def test_advance_with_target_hidden_keeps_its_default_forwarding() -> None:
    """The default (non cache-only) call forwards exactly the previous keywords."""

    proposer = mtp_native.NativeMtpChainProposer.__new__(mtp_native.NativeMtpChainProposer)
    calls: list[dict] = []

    def fake_advance(**kwargs):
        calls.append(kwargs)
        return mtp_native.NativeMtpStepResult(token=3, logit=0.5, topk_experts=(), topk_logits=())

    proposer.advance = fake_advance  # type: ignore[method-assign]
    proposer.advance_with_target_hidden(
        input_token=19,
        target_hidden_ptr=0xABC000,
        position=33,
        need_result=False,
        read_expert_topk=False,
        read_lm_head_value=False,
        stream=7,
    )
    proposer.advance_with_target_hidden(
        input_token=20,
        target_hidden_ptr=0xABC000,
        position=34,
        need_result=False,
        kv_write_only=True,
        read_expert_topk=False,
        read_lm_head_value=False,
    )

    assert calls[0] == {
        "input_token": 19,
        "target_hidden_ptr": 0xABC000,
        "position": 33,
        "need_result": False,
        "read_token_id": True,
        "read_expert_topk": False,
        "read_lm_head_value": False,
        "stream": 7,
    }
    assert calls[1]["kv_write_only"] is True
