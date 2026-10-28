"""CPU contract tests for bulk historical-prompt K/V appends in the native MTP proposer.

``append_prompt_kv()`` appends one historical prompt row per call. The public
PARO adapter drives it once per prompt row, so a prompt of ``n`` rows costs
``n`` separate launches of the same K/V producer chain. The kernels in that
chain are already row-capable (each output row is an independent block over the
same per-row reduction), so the same work can be issued for a bounded block of
rows.

``append_prompt_kv_bulk()`` is that bounded block operation. These tests pin the
contract the serial primitive already has and the bulk primitive must keep:

* the same K/V producer chain, with no attention/expert/head work;
* K/V written to the same cache slots the serial calls would write;
* a fixed row workspace bounded independently of prompt length;
* one cursor publication per completed block, and no cursor movement for a
  rejected request;
* no proposal or hidden carry left behind.

Every HIP kernel entry point, the runtime, and the weight loader are replaced
with recorders, so the assertions are about launch boundaries, row counts,
destination addresses, workspace bounds, and cursor publication -- not device
numerics. Device parity between serial and bulk execution is a separate GPU
oracle (see ``benchmarks/results/``); nothing here claims it.
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
_MAX_MTP_TOKENS = 16
_LAYER = "mtp.layers.0"

_HIDDEN_ROW_BYTES = _HIDDEN * 2
_TARGET_HIDDEN_PTR = 0x500000

# The whole set of device work ``advance()`` can issue, so a bulk append that
# runs anything outside the K/V producer prefix is visible as an unexpected
# record.
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

# The K/V producer prefix, in the order the serial primitive issues it.
_KV_PREFIX_KERNELS = (
    "mtp_fuse_inputs_f16_bf16",
    "dflash_dense_bf16_to_bf16",
    "mtp_rmsnorm_bf16_oneplus",
    "dflash_qkv_proj_bf16_mixed",
    "mtp_split_q_gate_f32_bf16",
    "dflash_head_rmsnorm_rotary_f32",
)

_INDEXED_PREFIX_KERNELS = (
    "mtp_fuse_inputs_f16_bf16",
    "dflash_dense_bf16_to_bf16",
    "mtp_rmsnorm_bf16_oneplus",
    "dflash_qkv_proj_bf16_mixed_indexed_v",
    "mtp_split_q_gate_f32_bf16",
    "dflash_head_rmsnorm_rotary_indexed_key_f32",
)

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
        self.fail_on: dict[str, int] = {}
        self.seen: dict[str, int] = {}

    def clear(self) -> None:
        self.kernels.clear()
        self.h2d.clear()
        self.d2h.clear()

    def names(self) -> list[str]:
        return [name for name, _args, _kwargs in self.kernels]

    def calls(self, name: str) -> list[tuple]:
        return [args for kernel, args, _kwargs in self.kernels if kernel == name]

    def record(self, name: str, args: tuple, kwargs: dict) -> None:
        count = self.seen.get(name, 0)
        self.seen[name] = count + 1
        limit = self.fail_on.get(name)
        if limit is not None and count >= limit:
            raise RuntimeError(f"injected device failure in {name}")
        self.kernels.append((name, args, kwargs))


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
    def __init__(self, rec: _Recorder, runtime: _FakeRuntime, make, allocations: list[int]) -> None:
        self.rec = rec
        self.runtime = runtime
        self.make = make
        self.allocations = allocations

    @property
    def d2d_copies(self) -> list[tuple[int, int, int, object, int]]:
        return [copy for copy in self.runtime.copies if copy[3] is HipMemcpyKind.DEVICE_TO_DEVICE]

    @property
    def h2d_copies(self) -> list[tuple[int, int, int, object, int]]:
        return [copy for copy in self.runtime.copies if copy[3] is HipMemcpyKind.HOST_TO_DEVICE]


def _recorder(rec: _Recorder, name: str):
    def record(*args, **kwargs):
        rec.record(name, args, kwargs)
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
    allocations: list[int] = []
    next_ptr = [0x100000]

    def fake_malloc(nbytes, *, runtime=None):
        ptr = next_ptr[0]
        next_ptr[0] += max(int(nbytes), 64) + 256
        allocations.append(int(nbytes))
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
        "copy_host_array_to_device",
        lambda buffer, array, nbytes=None, *, runtime=None: rec.h2d.append(
            (int(buffer.ptr), int(nbytes if nbytes is not None else array.nbytes))
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

    return _Harness(rec, runtime, make, allocations)


def _key_dst(proposer: mtp_native.NativeMtpChainProposer, row: int) -> int:
    return proposer.key_cache_buf.ptr + row * proposer.kv_features * 4


def _value_dst(proposer: mtp_native.NativeMtpChainProposer, row: int) -> int:
    return proposer.value_cache_buf.ptr + row * proposer.kv_features * 2


def _rows(count: int, *, first_position: int = 1, base_ptr: int = _TARGET_HIDDEN_PTR):
    """A contiguous target-hidden slab with ``hidden``-wide rows."""

    return [
        (
            100 + index,
            base_ptr + index * _HIDDEN_ROW_BYTES,
            first_position + index,
        )
        for index in range(count)
    ]


# --------------------------------------------------------------------------
# Bounds
# --------------------------------------------------------------------------


def test_bulk_rejects_empty_row_sequence(harness: _Harness) -> None:
    proposer = harness.make()

    with pytest.raises(ValueError, match="non-empty"):
        proposer.append_prompt_kv_bulk([])

    assert harness.rec.kernels == []
    assert proposer.cache_len == 0
    assert proposer.position == -1


def test_bulk_rejects_chunk_rows_outside_the_bounded_workspace(harness: _Harness) -> None:
    proposer = harness.make()

    for chunk_rows in (0, -1, mtp_native._BULK_PROMPT_MAX_ROWS + 1):
        with pytest.raises(ValueError, match="chunk_rows"):
            proposer.append_prompt_kv_bulk(_rows(3), chunk_rows=chunk_rows)

    assert harness.rec.kernels == []
    assert proposer.cache_len == 0


def test_bulk_capacity_is_checked_before_any_device_work(harness: _Harness) -> None:
    proposer = harness.make()
    proposer.append_prompt_kv_bulk(_rows(2))

    harness.rec.clear()
    harness.runtime.copies.clear()
    with pytest.raises(RuntimeError, match="capacity"):
        proposer.append_prompt_kv_bulk(_rows(_MAX_MTP_TOKENS))

    # The rejected request appended nothing: the cursor still names the two
    # committed rows and no launch was issued for the overflow.
    assert proposer.cache_len == 2
    assert proposer.position == 2
    assert harness.rec.kernels == []
    assert harness.runtime.copies == []


def test_bulk_position_is_checked_before_any_device_work(harness: _Harness) -> None:
    proposer = harness.make()

    bad = _rows(3)
    bad[2] = (bad[2][0], bad[2][1], _MAX_POSITIONS)
    with pytest.raises(ValueError, match="position outside MTP RoPE table"):
        proposer.append_prompt_kv_bulk(bad)

    assert harness.rec.kernels == []
    assert harness.runtime.copies == []
    assert proposer.cache_len == 0
    assert proposer.position == -1


def test_bulk_rejects_malformed_rows(harness: _Harness) -> None:
    proposer = harness.make()

    with pytest.raises(ValueError, match="triples"):
        proposer.append_prompt_kv_bulk([(1, 2)])
    with pytest.raises(ValueError, match="triples"):
        proposer.append_prompt_kv_bulk([(1, 2, 3, 4)])

    assert harness.rec.kernels == []


# --------------------------------------------------------------------------
# Workspace lifecycle and bounds
# --------------------------------------------------------------------------


def test_bulk_workspace_is_absent_until_first_use(harness: _Harness) -> None:
    proposer = harness.make()

    assert proposer._bulk_workspace is None
    before = len(harness.allocations)
    proposer.append_prompt_kv_bulk(_rows(2))

    assert proposer._bulk_workspace is not None
    assert len(harness.allocations) > before


def test_bulk_workspace_has_one_fixed_capacity(harness: _Harness) -> None:
    proposer = harness.make()

    proposer.append_prompt_kv_bulk(_rows(2), chunk_rows=2)
    workspace = proposer._bulk_workspace
    assert workspace is not None
    # Capacity is the hard cap lowered to the cache capacity, not the request:
    # a two-row append still stages a whole workspace.
    assert workspace.rows == min(mtp_native._BULK_PROMPT_MAX_ROWS, _MAX_MTP_TOKENS)
    assert workspace.fused_buf.nbytes == workspace.rows * 2 * _HIDDEN * 2

    allocations = len(harness.allocations)
    for chunk_rows in (1, 4, _MAX_MTP_TOKENS, mtp_native._BULK_PROMPT_MAX_ROWS):
        proposer.reset()
        harness.rec.clear()
        proposer.append_prompt_kv_bulk(_rows(min(chunk_rows, _MAX_MTP_TOKENS)), chunk_rows=chunk_rows)
        # Same workspace, no new allocation, and the chunk is still what the
        # caller asked for (clamped only by the rows actually appended).
        assert proposer._bulk_workspace is workspace
        assert len(harness.allocations) == allocations
        widths = [call[6] for call in harness.rec.calls("mtp_fuse_inputs_f16_bf16")]
        expected = []
        remaining = min(chunk_rows, _MAX_MTP_TOKENS)
        while remaining > 0:
            step = min(chunk_rows, remaining)
            expected.append(step)
            remaining -= step
        assert widths == expected


def test_bulk_workspace_never_grows_or_abandons_allocations(harness: _Harness) -> None:
    proposer = harness.make()

    # A one-row append must not size the workspace to one row.
    proposer.append_prompt_kv_bulk(_rows(1), chunk_rows=1)
    workspace = proposer._bulk_workspace
    assert workspace is not None
    allocations = len(harness.allocations)

    # The widest chunk this proposer can ever ask for must reuse it, leaving no
    # abandoned allocation behind in ``self.buffers``.
    proposer.reset()
    proposer.append_prompt_kv_bulk(
        _rows(_MAX_MTP_TOKENS), chunk_rows=mtp_native._BULK_PROMPT_MAX_ROWS
    )

    assert proposer._bulk_workspace is workspace
    assert len(harness.allocations) == allocations
    assert proposer.cache_len == _MAX_MTP_TOKENS


def test_bulk_workspace_is_reused_across_calls(harness: _Harness) -> None:
    proposer = harness.make()
    proposer.append_prompt_kv_bulk(_rows(4), chunk_rows=4)
    workspace = proposer._bulk_workspace
    allocations = len(harness.allocations)

    # A smaller request, a reset, and a repeat all reuse the same workspace.
    proposer.append_prompt_kv_bulk(_rows(2), chunk_rows=4)
    proposer.reset()
    proposer.append_prompt_kv_bulk(_rows(3), chunk_rows=4)

    assert proposer._bulk_workspace is workspace
    assert len(harness.allocations) == allocations


def test_bulk_workspace_grows_never_after_first_use(harness: _Harness) -> None:
    proposer = harness.make()
    proposer.append_prompt_kv_bulk(_rows(2), chunk_rows=2)
    first = proposer._bulk_workspace
    allocations = len(harness.allocations)

    proposer.reset()
    proposer.append_prompt_kv_bulk(_rows(6), chunk_rows=6)

    # No wider workspace exists to allocate: the capacity is already the cap.
    assert proposer._bulk_workspace is first
    assert len(harness.allocations) == allocations


def test_bulk_workspace_is_released_by_close(harness: _Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    freed: list[int] = []
    monkeypatch.setattr(
        mtp_native,
        "free",
        lambda buffer, *, runtime=None: freed.append(int(buffer.ptr)),
    )
    proposer = harness.make()
    proposer.append_prompt_kv_bulk(_rows(2))
    workspace = proposer._bulk_workspace
    assert workspace is not None
    workspace_ptrs = {workspace.fused_buf.ptr, workspace.tokens_buf.ptr}

    proposer.close()

    assert workspace_ptrs.issubset(set(freed))
    with pytest.raises(RuntimeError, match="closed"):
        proposer.append_prompt_kv_bulk(_rows(2))


# --------------------------------------------------------------------------
# Target-hidden / token alignment
# --------------------------------------------------------------------------


def test_bulk_uploads_tokens_positions_and_slots_in_row_order(harness: _Harness) -> None:
    proposer = harness.make()
    rows = _rows(3, first_position=5)
    proposer.append_prompt_kv_bulk(rows, chunk_rows=3)

    workspace = proposer._bulk_workspace
    assert workspace is not None
    # One packed H2D carries the whole chunk's tokens, positions and slots.
    assert len(harness.h2d_copies) == 1
    dst, _src, nbytes, kind, stream = harness.h2d_copies[0]
    assert (dst, nbytes, kind, stream) == (
        workspace.tokens_buf.ptr,
        workspace.host.nbytes,
        HipMemcpyKind.HOST_TO_DEVICE,
        0,
    )
    assert workspace.host.nbytes == 2 * workspace.rows * 8
    assert [int(x) for x in workspace.tokens_host[:3]] == [100, 101, 102]
    assert [int(x) for x in workspace.positions_host[:3]] == [5, 6, 7]
    assert [int(x) for x in workspace.slots_host[:3]] == [0, 1, 2]

    fuse = harness.rec.calls("mtp_fuse_inputs_f16_bf16")
    assert len(fuse) == 1
    # Row-major target hidden slab starts at the first row's pointer.
    assert fuse[0][0] == workspace.tokens_buf.ptr
    assert fuse[0][2] == rows[0][1]
    assert fuse[0][6] == 3

    rotary = harness.rec.calls("dflash_head_rmsnorm_rotary_f32")
    assert len(rotary) == 1
    assert rotary[0][6] == workspace.positions_buf.ptr
    assert rotary[0][7] == workspace.positions_buf.ptr
    assert rotary[0][10] == 1
    assert rotary[0][11] == 3
    assert rotary[0][12] == 3


def test_bulk_rejects_a_non_contiguous_target_hidden_slab(harness: _Harness) -> None:
    proposer = harness.make()
    gapped = _rows(3)
    gapped[2] = (gapped[2][0], gapped[2][1] + _HIDDEN_ROW_BYTES, gapped[2][2])

    with pytest.raises(ValueError, match="contiguous"):
        proposer.append_prompt_kv_bulk(gapped)

    unordered = _rows(3)
    unordered[1], unordered[2] = unordered[2], unordered[1]
    with pytest.raises(ValueError, match="contiguous"):
        proposer.append_prompt_kv_bulk(unordered)

    assert harness.rec.kernels == []
    assert proposer.cache_len == 0


def test_bulk_requires_an_explicit_target_hidden_per_row(harness: _Harness) -> None:
    proposer = harness.make()
    rows = _rows(2)

    zeroed = list(rows)
    zeroed[1] = (zeroed[1][0], 0, zeroed[1][2])
    with pytest.raises(ValueError, match="target hidden"):
        proposer.append_prompt_kv_bulk(zeroed)

    own_hidden = list(rows)
    own_hidden[0] = (own_hidden[0][0], proposer.final_hidden_buf.ptr, own_hidden[0][2])
    with pytest.raises(ValueError, match="target hidden"):
        proposer.append_prompt_kv_bulk(own_hidden)

    assert harness.rec.kernels == []
    assert proposer.cache_len == 0


# --------------------------------------------------------------------------
# Chunked execution and cache destination
# --------------------------------------------------------------------------


def test_bulk_chunks_rows_and_publishes_the_cursor_once_per_chunk(harness: _Harness) -> None:
    proposer = harness.make()
    rows = _rows(5)

    returned = proposer.append_prompt_kv_bulk(rows, chunk_rows=2)

    assert returned == 5
    assert proposer.cache_len == 5
    assert proposer.position == rows[-1][2]
    # Three chunks: two, two, one.
    assert len(harness.rec.calls("mtp_fuse_inputs_f16_bf16")) == 3
    assert [call[6] for call in harness.rec.calls("mtp_fuse_inputs_f16_bf16")] == [2, 2, 1]
    assert len(harness.h2d_copies) == 3
    # Each chunk's K/V goes to the cache rows the serial calls would fill.
    qkv = harness.rec.calls("dflash_qkv_proj_bf16_mixed")
    rotary = harness.rec.calls("dflash_head_rmsnorm_rotary_f32")
    assert [call[6] for call in qkv] == [
        _value_dst(proposer, 0),
        _value_dst(proposer, 2),
        _value_dst(proposer, 4),
    ]
    assert [call[9] for call in rotary] == [
        _key_dst(proposer, 0),
        _key_dst(proposer, 2),
        _key_dst(proposer, 4),
    ]
    # Row widths match the chunk, so consecutive rows land in consecutive slots.
    assert [call[7] for call in qkv] == [2, 2, 1]
    assert [call[12] for call in rotary] == [2, 2, 1]
    # The last chunk held one row, so it republished slot 4 over the previous
    # chunk's first slot; the trailing entries are stale by design.
    assert int(proposer._bulk_workspace.slots_host[0]) == 4


def test_bulk_appends_the_same_kv_producer_chain_and_no_downstream_work(
    harness: _Harness,
) -> None:
    proposer = harness.make()
    proposer.append_prompt_kv_bulk(_rows(3), chunk_rows=3)

    names = harness.rec.names()
    assert tuple(names) == _KV_PREFIX_KERNELS
    assert set(names).isdisjoint(_DOWNSTREAM_KERNELS)
    # fc is the only dense GEMV in the prefix.
    fc_weight_ptr = proposer.weights["mtp.fc.weight"].ptr
    assert [call[1] for call in harness.rec.calls("dflash_dense_bf16_to_bf16")] == [fc_weight_ptr]
    # No proposal surface is produced and no host readback happens.
    assert harness.rec.d2h == []
    assert proposer.final_hidden_valid is False
    assert proposer.result_valid is False
    assert proposer.current.token == -1
    with pytest.raises(RuntimeError, match="result-producing advance"):
        proposer.device_candidate_token_ids()
    with pytest.raises(RuntimeError, match="final hidden"):
        proposer.read_final_hidden_bf16()


@pytest.mark.parametrize("retired", [None, "0", "false", "1"])
@pytest.mark.parametrize("chunk_rows", [1, 2, 3])
def test_bulk_always_writes_directly_to_each_chunk_cache_destination(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch, retired, chunk_rows,
) -> None:
    if retired is not None:
        monkeypatch.setenv("HIPENGINE_MTP_PROPOSER_DIRECT_KV_WRITE", retired)
    proposer = harness.make()
    proposer.cache_len = 1
    proposer.append_prompt_kv_bulk(_rows(5), chunk_rows=chunk_rows, stream=13)

    workspace = proposer._bulk_workspace
    assert not hasattr(workspace, "key_rot_buf")
    assert not hasattr(workspace, "v_proj_buf")
    assert harness.d2d_copies == []
    starts = list(range(1, 6, chunk_rows))
    widths = [min(chunk_rows, 6 - start) for start in starts]
    qkv = [entry for entry in harness.rec.kernels if entry[0] == "dflash_qkv_proj_bf16_mixed"]
    rotary = [entry for entry in harness.rec.kernels if entry[0] == "dflash_head_rmsnorm_rotary_f32"]
    assert [args[6] for _, args, _ in qkv] == [_value_dst(proposer, start) for start in starts]
    assert [args[9] for _, args, _ in rotary] == [_key_dst(proposer, start) for start in starts]
    assert [args[7] for _, args, _ in qkv] == widths
    assert [args[12] for _, args, _ in rotary] == widths
    assert all(kwargs["stream"] == 13 for _, _, kwargs in qkv + rotary)
    assert proposer.cache_len == 6


@pytest.mark.parametrize("retired", [None, "0", "false", "1"])
def test_bulk_indexed_route_writes_slots_from_the_chunk_base(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch, retired,
) -> None:
    if retired is not None:
        monkeypatch.setenv("HIPENGINE_MTP_PROPOSER_DIRECT_KV_WRITE", retired)
    monkeypatch.setenv("HIPENGINE_MTP_PROPOSER_INDEXED_KV_WRITE", "1")
    proposer = harness.make()
    proposer.append_prompt_kv_bulk(_rows(4), chunk_rows=2)

    assert tuple(harness.rec.names()) == _INDEXED_PREFIX_KERNELS * 2
    workspace = proposer._bulk_workspace
    assert workspace is not None
    qkv = harness.rec.calls("dflash_qkv_proj_bf16_mixed_indexed_v")
    rotary = harness.rec.calls("dflash_head_rmsnorm_rotary_indexed_key_f32")
    assert len(qkv) == 2 and len(rotary) == 2
    for args in qkv:
        assert args[6] == proposer.value_cache_buf.ptr
        assert args[7] == workspace.slots_buf.ptr
        assert args[8] == _MAX_MTP_TOKENS
        assert args[9] == 2
    for args in rotary:
        assert args[9] == proposer.key_cache_buf.ptr
        assert args[10] == workspace.slots_buf.ptr
        assert args[11] == _MAX_MTP_TOKENS
        assert args[12] == 1
        assert args[13] == 2
    # The indexed kernels resolve the slot from ``cache_slot[0] + row``, so the
    # chunk's slots must be consecutive from its first cache row.
    assert [int(x) for x in workspace.slots_host[:2]] == [2, 3]


def test_bulk_then_final_row_attends_over_every_primed_row(harness: _Harness) -> None:
    proposer = harness.make()
    rows = _rows(3)
    proposer.append_prompt_kv_bulk(rows, chunk_rows=2)

    harness.rec.clear()
    proposer.advance_with_target_hidden(
        input_token=200,
        target_hidden_ptr=rows[-1][1] + _HIDDEN_ROW_BYTES,
        position=rows[-1][2] + 1,
        read_expert_topk=False,
        read_lm_head_value=False,
    )

    attention = harness.rec.calls("dflash_gqa_attention_f32_bf16")
    assert len(attention) == 1
    assert attention[0][6] == 4
    assert proposer.cache_len == 4
    assert proposer.final_hidden_valid is True
    assert proposer.result_valid is True


# --------------------------------------------------------------------------
# Failure behavior
# --------------------------------------------------------------------------


def test_bulk_first_chunk_failure_clears_a_previous_result(harness: _Harness) -> None:
    proposer = harness.make()
    # A valid carry and proposal exist before the append starts.
    proposer.advance_with_target_hidden(
        input_token=11,
        target_hidden_ptr=_TARGET_HIDDEN_PTR,
        position=5,
    )
    assert proposer.final_hidden_valid is True
    assert proposer.result_valid is True
    cache_len = proposer.cache_len
    position = proposer.position

    # The very first launch of the first chunk fails: no chunk commits.
    harness.rec.fail_on["mtp_fuse_inputs_f16_bf16"] = 0
    with pytest.raises(RuntimeError, match="injected device failure"):
        proposer.append_prompt_kv_bulk(_rows(3), chunk_rows=3)

    # Nothing is published, and the pre-append carry/proposal is gone: a
    # partially launched chunk may already have written cache rows, so the old
    # hidden and proposal no longer describe a state the caller can observe.
    assert proposer.cache_len == cache_len
    assert proposer.position == position
    assert proposer.final_hidden_valid is False
    assert proposer.result_valid is False
    assert proposer.current.token == -1
    with pytest.raises(RuntimeError, match="result-producing advance"):
        proposer.device_candidate_token_ids()
    with pytest.raises(RuntimeError, match="final hidden"):
        proposer.read_final_hidden_bf16()


def test_bulk_device_failure_keeps_only_the_committed_chunks(harness: _Harness) -> None:
    proposer = harness.make()
    rows = _rows(4)
    # The second chunk's rotary launch fails.
    harness.rec.fail_on["dflash_head_rmsnorm_rotary_f32"] = 1

    with pytest.raises(RuntimeError, match="injected device failure"):
        proposer.append_prompt_kv_bulk(rows, chunk_rows=2)

    # The first chunk committed and published its cursor; the failing chunk
    # left no valid hidden, proposal, or cursor movement behind.
    assert proposer.cache_len == 2
    assert proposer.position == rows[1][2]
    assert proposer.final_hidden_valid is False
    assert proposer.result_valid is False
    assert proposer.current.token == -1

    # A retry from the published cursor appends the remaining rows.
    harness.rec.fail_on.clear()
    proposer.append_prompt_kv_bulk(rows[2:], chunk_rows=2)
    assert proposer.cache_len == 4
    assert [call[6] for call in harness.rec.calls("dflash_qkv_proj_bf16_mixed")][-1] == _value_dst(
        proposer, 2
    )


def test_bulk_rejects_a_closed_proposer(harness: _Harness) -> None:
    proposer = harness.make()
    proposer.close()

    with pytest.raises(RuntimeError, match="closed"):
        proposer.append_prompt_kv_bulk(_rows(2))


def test_bulk_rejects_a_closed_borrowed_scorer(harness: _Harness) -> None:
    owner = SimpleNamespace(closed=True)
    proposer = harness.make()
    proposer.scoring_head = mtp_native.NativeMtpW8A16Head(
        weight_int8_ptr=0x90000,
        scale_f32_ptr=0xA0000,
        vocab_size=_VOCAB,
        threads=256,
        owner=owner,
    )

    with pytest.raises(RuntimeError, match="closed"):
        proposer.append_prompt_kv_bulk(_rows(2))

    assert proposer.cache_len == 0
    assert harness.rec.kernels == []


# --------------------------------------------------------------------------
# Unchanged serial use
# --------------------------------------------------------------------------


def test_bulk_does_not_route_through_the_serial_primitive(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    proposer = harness.make()

    def forbidden(**kwargs):
        raise AssertionError("bulk append must not call advance()")

    monkeypatch.setattr(proposer, "advance", forbidden)
    proposer.append_prompt_kv_bulk(_rows(2), chunk_rows=2)

    assert proposer.cache_len == 2


def test_serial_primitive_still_forwards_exactly_one_cache_only_row(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    proposer = harness.make()
    calls: list[dict] = []

    def fake_advance(**kwargs):
        calls.append(kwargs)
        return mtp_native.NativeMtpStepResult(token=-1, logit=float("nan"), topk_experts=(), topk_logits=())

    monkeypatch.setattr(proposer, "advance", fake_advance)
    returned = proposer.append_prompt_kv(
        input_token=77,
        target_hidden_ptr=_TARGET_HIDDEN_PTR,
        position=9,
    )

    assert returned is None
    assert calls == [
        {
            "input_token": 77,
            "target_hidden_ptr": _TARGET_HIDDEN_PTR,
            "position": 9,
            "need_result": False,
            "kv_write_only": True,
            "read_token_id": False,
            "read_expert_topk": False,
            "read_lm_head_value": False,
            "stream": 0,
        }
    ]


def test_serial_and_one_row_bulk_issue_the_same_producer_chain(harness: _Harness) -> None:
    proposer = harness.make()

    proposer.append_prompt_kv(input_token=100, target_hidden_ptr=_TARGET_HIDDEN_PTR, position=1)
    serial_names = list(harness.rec.names())
    serial_rows = [call[6] for call in harness.rec.calls("mtp_fuse_inputs_f16_bf16")]
    serial_cache_len = proposer.cache_len
    serial_position = proposer.position

    proposer.reset()
    harness.rec.clear()
    proposer.append_prompt_kv_bulk([(100, _TARGET_HIDDEN_PTR, 1)], chunk_rows=1)

    # Same chain, same row width, same published cursor: the bulk primitive is
    # the serial primitive's work issued for a block of one.
    assert harness.rec.names() == serial_names
    assert [call[6] for call in harness.rec.calls("mtp_fuse_inputs_f16_bf16")] == serial_rows
    assert proposer.cache_len == serial_cache_len
    assert proposer.position == serial_position
    assert proposer.final_hidden_valid is False
    assert proposer.result_valid is False
