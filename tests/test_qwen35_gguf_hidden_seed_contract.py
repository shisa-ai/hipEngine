from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.dtype import DType
from hipengine.core.hip import HipMemcpyKind
from hipengine.core.memory import DeviceBuffer
from hipengine.runtime import qwen35_gguf_runner as gguf_runner
from hipengine.runtime.qwen35_gguf_runner import (
    Qwen35GGUFHiddenSeedContract,
    Qwen35GGUFMTPDraftSeed,
    Qwen35GGUFResidentSession,
    qwen35_gguf_current_hidden_seed_contract,
    qwen35_gguf_fp32_hidden_seed_contract,
    qwen35_gguf_fp32_verify_hidden_seed_contract,
)


def test_current_gguf_hidden_seed_contract_marks_bf16_tap_non_llama_compatible() -> None:
    contract = qwen35_gguf_current_hidden_seed_contract(hidden_size=4096)

    assert contract.provenance == "post_output_norm"
    assert contract.dtype is DType.BF16
    assert contract.rows == 1
    assert contract.hidden_size == 4096
    assert contract.source_buffer == "Qwen35GGUFResidentSession.scratch.norm"
    assert contract.requires_fp32_tap
    assert not contract.llama_cpp_compatible
    assert contract.as_dict() == {
        "provenance": "post_output_norm",
        "dtype": "BF16",
        "rows": 1,
        "hidden_size": 4096,
        "source_buffer": "Qwen35GGUFResidentSession.scratch.norm",
        "populated_by_decode": True,
        "llama_cpp_compatible": False,
        "requires_fp32_tap": True,
        "ready_for_mtp": False,
    }


def test_fp32_hidden_seed_contract_marks_m25_target_buffer_unpopulated() -> None:
    contract = qwen35_gguf_fp32_hidden_seed_contract(hidden_size=4096, rows=4)

    assert contract.provenance == "post_output_norm"
    assert contract.dtype is DType.FP32
    assert contract.rows == 4
    assert contract.hidden_size == 4096
    assert contract.source_buffer == "Qwen35GGUFResidentSession.scratch.hidden_seed_fp32"
    assert not contract.requires_fp32_tap
    assert not contract.populated_by_decode
    assert not contract.llama_cpp_compatible
    assert not contract.ready_for_mtp


def test_resident_session_reports_current_and_fp32_hidden_seed_contracts_without_gpu_init() -> None:
    session = object.__new__(Qwen35GGUFResidentSession)
    session.runner = SimpleNamespace(hidden_size=8192)
    session.scratch = SimpleNamespace(hidden_seed_fp32=SimpleNamespace(ptr=12345))
    session._hidden_seed_fp32_populated = False

    current = session.hidden_seed_contract(rows=2)
    fp32 = session.fp32_hidden_seed_contract(rows=2)

    assert current.provenance == "post_output_norm"
    assert current.dtype is DType.BF16
    assert current.rows == 2
    assert current.hidden_size == 8192
    assert current.requires_fp32_tap
    assert not current.llama_cpp_compatible
    assert fp32.provenance == "post_output_norm"
    assert fp32.dtype is DType.FP32
    assert fp32.rows == 2
    assert fp32.hidden_size == 8192
    assert not fp32.requires_fp32_tap
    assert not fp32.populated_by_decode
    assert not fp32.llama_cpp_compatible
    assert not fp32.ready_for_mtp
    with pytest.raises(RuntimeError, match="GGUF fp32 hidden seed is not populated"):
        session.fp32_hidden_seed_ptr()

    session._hidden_seed_fp32_populated = True
    populated = session.fp32_hidden_seed_contract(rows=2)
    assert populated.populated_by_decode
    assert populated.llama_cpp_compatible
    assert populated.ready_for_mtp
    assert session.fp32_hidden_seed_ptr() == 12345
    seed = session.mtp_draft_seed(token_id=99, position=7)
    assert seed.token_id == 99
    assert seed.position == 7
    assert seed.hidden_ptr == 12345
    assert seed.hidden_contract.ready_for_mtp


def test_mtp_draft_seed_rejects_unready_contract_or_invalid_fields() -> None:
    ready = qwen35_gguf_fp32_hidden_seed_contract(
        hidden_size=4096,
        populated_by_decode=True,
    )
    unready = qwen35_gguf_fp32_hidden_seed_contract(hidden_size=4096)

    with pytest.raises(ValueError, match="requires a ready fp32 hidden contract"):
        Qwen35GGUFMTPDraftSeed(
            token_id=1,
            position=2,
            hidden_ptr=123,
            hidden_contract=unready,
        )
    with pytest.raises(ValueError, match="token_id must be non-negative"):
        Qwen35GGUFMTPDraftSeed(
            token_id=-1,
            position=2,
            hidden_ptr=123,
            hidden_contract=ready,
        )
    with pytest.raises(ValueError, match="position must be non-negative"):
        Qwen35GGUFMTPDraftSeed(
            token_id=1,
            position=-2,
            hidden_ptr=123,
            hidden_contract=ready,
        )
    with pytest.raises(ValueError, match="hidden_ptr must be a non-zero"):
        Qwen35GGUFMTPDraftSeed(
            token_id=1,
            position=2,
            hidden_ptr=0,
            hidden_contract=ready,
        )

    seed = Qwen35GGUFMTPDraftSeed(
        token_id=1,
        position=2,
        hidden_ptr=123,
        hidden_contract=ready,
    )
    assert seed.as_dict()["hidden_contract"] == ready.as_dict()


def test_run_current_hidden_to_final_hidden_populates_fp32_seed_only_when_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, int, int, int]] = []

    def fake_bf16(src_ptr: int, weight_ptr: int, out_ptr: int, **kwargs: object) -> None:
        calls.append(("bf16", src_ptr, weight_ptr, out_ptr))

    def fake_f32(src_ptr: int, weight_ptr: int, out_ptr: int, **kwargs: object) -> None:
        calls.append(("f32", src_ptr, weight_ptr, out_ptr))

    monkeypatch.setattr(gguf_runner, "gguf_rmsnorm_bf16_f32_weight", fake_bf16)
    monkeypatch.setattr(gguf_runner, "gguf_rmsnorm_bf16_f32_weight_out_f32", fake_f32)

    session = object.__new__(Qwen35GGUFResidentSession)
    output_norm = SimpleNamespace(allocation=lambda: SimpleNamespace(tensor=SimpleNamespace(ptr=200)))
    weights = SimpleNamespace(
        config=SimpleNamespace(layer_types=(), rms_norm_eps=1.0e-6, is_moe=False),
        root=lambda name: output_norm if name == "output_norm" else None,
    )
    session.runner = SimpleNamespace(weights=weights, hidden_size=8)
    session.scratch = SimpleNamespace(
        position_host=np.zeros((1,), dtype=np.int64),
        context_host=np.zeros((1,), dtype=np.int64),
        norm=SimpleNamespace(ptr=300),
        hidden_seed_fp32=SimpleNamespace(ptr=400),
    )
    session.runtime = object()
    session._hidden_a = SimpleNamespace(ptr=100)
    session._hidden_b = SimpleNamespace(ptr=101)
    session._hidden_seed_fp32_populated = True
    monkeypatch.setattr(gguf_runner, "_gguf_moe_graph_enabled", lambda: False)

    ptr = session._run_current_hidden_to_final_hidden(position=5, capture_hidden_seed_fp32=False)

    assert ptr == 300
    assert calls == [("bf16", 100, 200, 300)]
    assert not session._hidden_seed_fp32_populated
    assert not session.fp32_hidden_seed_contract().ready_for_mtp

    calls.clear()
    ptr = session._run_current_hidden_to_final_hidden(position=6, capture_hidden_seed_fp32=True)

    assert ptr == 300
    assert calls == [("bf16", 100, 200, 300), ("f32", 100, 200, 400)]
    assert session._hidden_seed_fp32_populated
    assert session.fp32_hidden_seed_contract().ready_for_mtp


def test_resident_prefill_capture_marks_only_final_serial_prompt_token() -> None:
    session = object.__new__(Qwen35GGUFResidentSession)
    session.runner = SimpleNamespace(
        weights=SimpleNamespace(config=SimpleNamespace(ssm_conv_kernel=99))
    )
    session.scratch = SimpleNamespace(zero_states=lambda runtime, **kwargs: None)
    session._target_scratch_owner = session.scratch
    session.runtime = object()
    session._set_full_attention_position_device = lambda position, stream=0: None
    session._position = 17
    session._hidden_seed_fp32_populated = True
    calls: list[tuple[int, int, bool]] = []

    def fake_run_token_to_final_hidden(
        token_id: int,
        *,
        position: int,
        capture_hidden_seed_fp32: bool = False,
    ) -> int:
        calls.append((token_id, position, capture_hidden_seed_fp32))
        session._hidden_seed_fp32_populated = bool(capture_hidden_seed_fp32)
        return 1000 + token_id

    def fake_sample_from_hidden(
        hidden_ptr: int,
        *,
        return_logits: bool,
        stream: int = 0,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            token_id=5,
            hidden_ptr=hidden_ptr,
            return_logits=return_logits,
        )

    session._run_token_to_final_hidden = fake_run_token_to_final_hidden
    session._sample_from_hidden = fake_sample_from_hidden

    result = session.prefill(
        [3, 4, 7],
        use_bulk=False,
        return_logits=False,
        capture_hidden_seed_fp32=True,
    )

    assert calls == [(3, 0, False), (4, 1, False), (7, 2, True)]
    assert session._position == 3
    assert session._hidden_seed_fp32_populated
    assert result.hidden_ptr == 1007
    assert result.return_logits is False


def test_resident_prefill_forwards_capture_request_to_bulk_prefill() -> None:
    session = object.__new__(Qwen35GGUFResidentSession)
    session.runner = SimpleNamespace(
        backend="hip_gfx1100",
        weights=SimpleNamespace(config=SimpleNamespace(ssm_conv_kernel=2)),
    )
    session.use_wmma_prefill = None
    session.use_gemv_decode = None
    bulk_calls: list[dict[str, object]] = []

    def fake_bulk_prefill_and_sample(
        token_ids: list[int] | tuple[int, ...],
        *,
        bulk_attention_mode: str,
        return_logits: bool,
        capture_hidden_seed_fp32: bool,
        capture_layer_output_hidden: tuple[int, ...],
    ) -> SimpleNamespace:
        bulk_calls.append(
            {
                "token_ids": tuple(token_ids),
                "bulk_attention_mode": bulk_attention_mode,
                "return_logits": return_logits,
                "capture_hidden_seed_fp32": capture_hidden_seed_fp32,
                "capture_layer_output_hidden": capture_layer_output_hidden,
            }
        )
        return SimpleNamespace(token_id=8)

    session._run_bulk_prefill_and_sample = fake_bulk_prefill_and_sample
    session._q8_mmq_prefill_context = lambda: nullcontext()

    result = session.prefill(
        [10, 11],
        use_bulk=True,
        bulk_attention_mode="native",
        return_logits=False,
        capture_hidden_seed_fp32=True,
        capture_layer_output_hidden=(0, 1),
    )

    assert result.token_id == 8
    assert bulk_calls == [
        {
            "token_ids": (10, 11),
            "bulk_attention_mode": "native",
            "return_logits": False,
            "capture_hidden_seed_fp32": True,
            "capture_layer_output_hidden": (0, 1),
        }
    ]


def test_resident_prefill_forwards_target_hidden_rows_to_bulk_prefill() -> None:
    session = object.__new__(Qwen35GGUFResidentSession)
    session.runner = SimpleNamespace(
        backend="hip_gfx1100",
        weights=SimpleNamespace(config=SimpleNamespace(ssm_conv_kernel=2)),
    )
    session.use_wmma_prefill = None
    session.use_gemv_decode = None
    target_hidden_rows = DeviceBuffer(0xD000, 32)
    bulk_calls: list[dict[str, object]] = []

    def fake_bulk_prefill_and_sample(token_ids, **kwargs):
        bulk_calls.append({"token_ids": tuple(token_ids), **kwargs})
        return SimpleNamespace(token_id=8)

    session._run_bulk_prefill_and_sample = fake_bulk_prefill_and_sample
    session._q8_mmq_prefill_context = lambda: nullcontext()

    result = session.prefill(
        [10, 11],
        use_bulk=True,
        return_logits=False,
        capture_target_hidden_rows=target_hidden_rows,
    )

    assert result.token_id == 8
    assert bulk_calls == [
        {
            "token_ids": (10, 11),
            "bulk_attention_mode": "bulk",
            "return_logits": False,
            "capture_target_hidden_rows": target_hidden_rows,
        }
    ]


def test_bulk_prefill_capture_populates_all_prompt_hidden_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, ...]] = []

    class Runtime:
        def memcpy_async(self, dst, src, nbytes, kind, stream) -> None:
            calls.append(("memcpy_async", int(dst), int(src), int(nbytes), int(kind), int(stream)))

    class BulkScratch:
        def for_chunk(self, start, rows, total_tokens, *, runtime, stream=0):
            calls.append(("scratch", int(start), int(rows), int(total_tokens), runtime, int(stream)))
            return SimpleNamespace(norm=SimpleNamespace(ptr=0x7000))

    def fake_copy_host_to_device(buf, host_ptr, nbytes, *, runtime) -> None:
        calls.append(("copy_host_to_device", int(buf.ptr), int(nbytes), runtime))

    def fake_embedding(weight, token_ptr, out_ptr, **kwargs: object) -> None:
        calls.append(
            (
                "embedding",
                int(token_ptr),
                int(out_ptr),
                int(kwargs["rows"]),
                int(kwargs["hidden_size"]),
                int(kwargs["vocab_size"]),
            )
        )

    def fake_bf16(src_ptr: int, weight_ptr: int, out_ptr: int, **kwargs: object) -> None:
        calls.append(
            (
                "bf16",
                int(src_ptr),
                int(weight_ptr),
                int(out_ptr),
                int(kwargs["rows"]),
                int(kwargs["hidden_size"]),
            )
        )

    def fake_f32(src_ptr: int, weight_ptr: int, out_ptr: int, **kwargs: object) -> None:
        calls.append(
            (
                "f32",
                int(src_ptr),
                int(weight_ptr),
                int(out_ptr),
                int(kwargs["rows"]),
                int(kwargs["hidden_size"]),
            )
        )

    def fake_set_decode_position_i64(position_buf, context_buf, position, **kwargs: object) -> None:
        calls.append(("set_decode_position", int(position_buf), int(context_buf), int(position)))

    monkeypatch.setattr(gguf_runner, "copy_host_to_device", fake_copy_host_to_device)
    monkeypatch.setattr(gguf_runner, "launch_gguf_embedding", fake_embedding)
    monkeypatch.setattr(gguf_runner, "gguf_rmsnorm_bf16_f32_weight", fake_bf16)
    monkeypatch.setattr(gguf_runner, "gguf_rmsnorm_bf16_f32_weight_out_f32", fake_f32)
    monkeypatch.setattr(gguf_runner, "set_decode_position_i64", fake_set_decode_position_i64)

    runtime = Runtime()
    output_norm = SimpleNamespace(allocation=lambda: SimpleNamespace(tensor=SimpleNamespace(ptr=0x6000)))
    token_embedding = SimpleNamespace(name="token_embedding", allocations={"raw": object()})
    weights = SimpleNamespace(
        config=SimpleNamespace(layer_types=(), rms_norm_eps=1.0e-6, ssm_conv_kernel=2),
        root=lambda name: output_norm if name == "output_norm" else token_embedding,
    )
    session = object.__new__(Qwen35GGUFResidentSession)
    session.runner = SimpleNamespace(weights=weights, hidden_size=8, vocab_size=100)
    session.runtime = runtime
    session.scratch = SimpleNamespace(
        max_positions=16,
        zero_states=lambda active_runtime, **kwargs: calls.append(("zero_states", active_runtime)),
        hidden_seed_fp32=SimpleNamespace(ptr=0x9000),
        position_host=np.zeros((1,), dtype=np.int64),
        context_host=np.zeros((1,), dtype=np.int64),
        position_buf=SimpleNamespace(ptr=0xA000),
        context_buf=SimpleNamespace(ptr=0xB000),
    )
    session._target_scratch_owner = session.scratch
    session._prefill_token_buf = SimpleNamespace(ptr=0x3000)
    session._prefill_hidden_a = SimpleNamespace(ptr=0x1000, nbytes=16 * 8 * 2)
    session._prefill_hidden_b = SimpleNamespace(ptr=0x2000, nbytes=16 * 8 * 2)
    session._bulk_prefill_scratch = BulkScratch()
    session._runtime_state_library = None
    session._verify_hidden_seed_buf = None
    session._verify_hidden_seed_rows_populated = 0
    session._verify_block_rows_capacity = 0
    session._hidden_seed_fp32_populated = True
    session._int8_prefill_oracle_buffers = {}
    session.host_token_embedding_enabled = False

    def fake_ensure(rows: int, *, runtime) -> None:
        calls.append(("ensure_verify", int(rows), runtime))
        session._verify_hidden_seed_buf = SimpleNamespace(ptr=0x8000)
        session._verify_block_rows_capacity = int(rows)

    def fake_sample_from_hidden(
        hidden_ptr: int,
        *,
        return_logits: bool,
        stream: int = 0,
    ) -> SimpleNamespace:
        calls.append(("sample", int(hidden_ptr), bool(return_logits)))
        return SimpleNamespace(hidden_ptr=int(hidden_ptr), return_logits=bool(return_logits), token_id=5)

    session._ensure_verify_block_buffers = fake_ensure
    session._sample_from_hidden = fake_sample_from_hidden

    target_hidden_rows = DeviceBuffer(0xD000, 3 * 8 * DType.BF16.itemsize)
    result = session._run_bulk_prefill_and_sample(
        [3, 4, 7],
        bulk_attention_mode="bulk",
        return_logits=False,
        capture_hidden_seed_fp32=True,
        capture_target_hidden_rows=target_hidden_rows,
    )

    hidden_row_bytes = 8 * DType.FP32.itemsize
    last_bf16_hidden_ptr = 0x7000 + 2 * 8 * DType.BF16.itemsize
    assert result.hidden_ptr == last_bf16_hidden_ptr
    assert result.return_logits is False
    assert ("ensure_verify", 3, runtime) in calls
    assert ("bf16", 0x1000, 0x6000, 0x7000, 3, 8) in calls
    assert ("f32", 0x1000, 0x6000, 0x8000, 3, 8) in calls
    assert (
        "memcpy_async",
        0xD000,
        0x1000,
        3 * 8 * DType.BF16.itemsize,
        int(HipMemcpyKind.DEVICE_TO_DEVICE),
        0,
    ) in calls
    assert (
        "memcpy_async",
        0x9000,
        0x8000 + 2 * hidden_row_bytes,
        hidden_row_bytes,
        int(HipMemcpyKind.DEVICE_TO_DEVICE),
        0,
    ) in calls
    assert ("set_decode_position", 0xA000, 0xB000, 3) in calls
    assert session._verify_hidden_seed_rows_populated == 3
    assert session._hidden_seed_fp32_populated
    assert session._position == 3
    assert session.scratch.position_host[0] == 3
    assert session.scratch.context_host[0] == 4
    assert session.fp32_hidden_seed_ptr() == 0x9000
    assert session.fp32_verify_hidden_seed_ptr(2) == 0x8000 + 2 * hidden_row_bytes


def test_bulk_prefill_without_capture_keeps_last_row_output_norm(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, ...]] = []

    class Runtime:
        def stream_synchronize(self, stream: int) -> None:
            calls.append(("stream_synchronize", int(stream)))

    class BulkScratch:
        def for_chunk(self, start, rows, total_tokens, *, runtime, stream=0):
            calls.append(("scratch", int(start), int(rows), int(total_tokens), runtime, int(stream)))
            return SimpleNamespace(norm=SimpleNamespace(ptr=0x7000))

    def fake_copy_host_to_device(buf, host_ptr, nbytes, *, runtime) -> None:
        calls.append(("copy_host_to_device", int(buf.ptr), int(nbytes), runtime))

    def fake_embedding(weight, token_ptr, out_ptr, **kwargs: object) -> None:
        calls.append(("embedding", int(token_ptr), int(out_ptr), int(kwargs["rows"])))

    def fake_set_decode_position_i64(position_buf, context_buf, position, **kwargs: object) -> None:
        calls.append(("set_decode_position", int(position_buf), int(context_buf), int(position)))

    monkeypatch.setattr(gguf_runner, "copy_host_to_device", fake_copy_host_to_device)
    monkeypatch.setattr(gguf_runner, "launch_gguf_embedding", fake_embedding)
    monkeypatch.setattr(gguf_runner, "set_decode_position_i64", fake_set_decode_position_i64)

    runtime = Runtime()
    token_embedding = SimpleNamespace(name="token_embedding", allocations={"raw": object()})
    weights = SimpleNamespace(
        config=SimpleNamespace(
            layer_types=(gguf_runner.LINEAR_ATTENTION, gguf_runner.LINEAR_ATTENTION),
            rms_norm_eps=1.0e-6,
            ssm_conv_kernel=2,
            is_moe=False,
        ),
        root=lambda name: token_embedding,
    )

    def fake_linear_layer(layer_id, src_ptr, dst_ptr, scratch, **kwargs: object) -> None:
        calls.append(
            (
                "linear_layer",
                int(layer_id),
                int(src_ptr),
                int(dst_ptr),
                scratch.norm.ptr,
                int(kwargs["rows"]),
            )
        )

    session = object.__new__(Qwen35GGUFResidentSession)
    session.runner = SimpleNamespace(
        weights=weights,
        hidden_size=8,
        vocab_size=100,
        _run_linear_attention_prefill_layer_rows=fake_linear_layer,
    )
    session.runtime = runtime
    session.scratch = SimpleNamespace(
        max_positions=16,
        zero_states=lambda active_runtime, **kwargs: calls.append(("zero_states", active_runtime)),
        hidden_seed_fp32=SimpleNamespace(ptr=0x9000),
        position_host=np.zeros((1,), dtype=np.int64),
        context_host=np.zeros((1,), dtype=np.int64),
        position_buf=SimpleNamespace(ptr=0xA000),
        context_buf=SimpleNamespace(ptr=0xB000),
    )
    session._target_scratch_owner = session.scratch
    session._prefill_token_buf = SimpleNamespace(ptr=0x3000)
    session._prefill_hidden_a = SimpleNamespace(ptr=0x1000, nbytes=16 * 8 * 2)
    session._prefill_hidden_b = SimpleNamespace(ptr=0x2000, nbytes=16 * 8 * 2)
    session._bulk_prefill_scratch = BulkScratch()
    session._runtime_state_library = None
    session._verify_hidden_seed_buf = None
    session._verify_hidden_seed_rows_populated = 0
    session._verify_block_rows_capacity = 0
    session._hidden_seed_fp32_populated = True
    session._int8_prefill_oracle_buffers = {}
    session.host_token_embedding_enabled = False
    session.use_expert_sidecar = False
    session.prefill_queue_drain = "layer"
    session._linear_prefill_layer_chunk_size = lambda rows: int(rows)

    def fake_run_output_norm_hidden(
        src_ptr: int,
        out_ptr: int,
        *,
        stream: int = 0,
        capture_hidden_seed_fp32: bool = False,
    ) -> int:
        calls.append(("last_row_output_norm", int(src_ptr), int(out_ptr), int(stream), bool(capture_hidden_seed_fp32)))
        session._hidden_seed_fp32_populated = bool(capture_hidden_seed_fp32)
        return int(out_ptr)

    def fake_sample_from_hidden(
        hidden_ptr: int,
        *,
        return_logits: bool,
        stream: int = 0,
    ) -> SimpleNamespace:
        calls.append(("sample", int(hidden_ptr), bool(return_logits)))
        return SimpleNamespace(hidden_ptr=int(hidden_ptr), return_logits=bool(return_logits), token_id=5)

    session._run_output_norm_hidden = fake_run_output_norm_hidden
    session._sample_from_hidden = fake_sample_from_hidden

    result = session._run_bulk_prefill_and_sample(
        [3, 4, 7],
        bulk_attention_mode="bulk",
        return_logits=False,
        capture_hidden_seed_fp32=False,
    )

    last_src_ptr = 0x1000 + 2 * 8 * DType.BF16.itemsize
    assert [call[:4] for call in calls if call[0] == "scratch"] == [
        ("scratch", 0, 3, 3),
    ]
    assert ("last_row_output_norm", last_src_ptr, 0x7000, 0, False) in calls
    assert result.hidden_ptr == 0x7000
    assert result.return_logits is False
    assert [call for call in calls if call[0] == "stream_synchronize"] == [
        ("stream_synchronize", 0),
        ("stream_synchronize", 0),
    ]
    assert session._verify_hidden_seed_rows_populated == 0
    assert not session._hidden_seed_fp32_populated
    assert session._position == 3


def test_resident_output_norm_hidden_populates_fp32_seed_for_bulk_and_decode(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, int, int, int]] = []

    def fake_bf16(src_ptr: int, weight_ptr: int, out_ptr: int, **kwargs: object) -> None:
        calls.append(("bf16", src_ptr, weight_ptr, out_ptr))

    def fake_f32(src_ptr: int, weight_ptr: int, out_ptr: int, **kwargs: object) -> None:
        calls.append(("f32", src_ptr, weight_ptr, out_ptr))

    monkeypatch.setattr(gguf_runner, "gguf_rmsnorm_bf16_f32_weight", fake_bf16)
    monkeypatch.setattr(gguf_runner, "gguf_rmsnorm_bf16_f32_weight_out_f32", fake_f32)

    session = object.__new__(Qwen35GGUFResidentSession)
    output_norm = SimpleNamespace(allocation=lambda: SimpleNamespace(tensor=SimpleNamespace(ptr=200)))
    weights = SimpleNamespace(
        config=SimpleNamespace(rms_norm_eps=1.0e-6),
        root=lambda name: output_norm if name == "output_norm" else None,
    )
    session.runner = SimpleNamespace(weights=weights, hidden_size=8)
    session.scratch = SimpleNamespace(hidden_seed_fp32=SimpleNamespace(ptr=400))
    session.runtime = object()
    session._hidden_seed_fp32_populated = True

    ptr = session._run_output_norm_hidden(
        100,
        300,
        capture_hidden_seed_fp32=False,
    )

    assert ptr == 300
    assert calls == [("bf16", 100, 200, 300)]
    assert not session._hidden_seed_fp32_populated

    calls.clear()
    ptr = session._run_output_norm_hidden(
        101,
        301,
        capture_hidden_seed_fp32=True,
    )

    assert ptr == 301
    assert calls == [("bf16", 101, 200, 301), ("f32", 101, 200, 400)]
    assert session._hidden_seed_fp32_populated


def test_linear_attention_boundary_capture_runs_decode_tap_and_copies_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []

    class FakeRuntime:
        def device_synchronize(self) -> None:
            calls.append(("device_synchronize",))

    def fake_position(position: int, *, stream: int = 0) -> None:
        calls.append(("position", position, stream))

    def fake_token(token_id: int, *, stream: int = 0) -> None:
        calls.append(("token", token_id, stream))

    def fake_attn(
        layer_id: int,
        hidden_ptr: int,
        attn_out_ptr: int,
        scratch: object,
        **kwargs: object,
    ) -> None:
        calls.append(("attn", layer_id, hidden_ptr, attn_out_ptr, kwargs["stream"]))

    def fake_copy_bf16(ptr: int, elements: int, *, runtime: object) -> np.ndarray:
        calls.append(("copy_bf16", ptr, elements, runtime))
        payloads = {
            2000: np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            2100: np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32),
            2200: np.asarray([0.7, 0.8, 0.9, 1.0], dtype=np.float32),
            3000: np.asarray([0.25, 0.5], dtype=np.float32),
            4000: np.asarray([-0.25, -0.5], dtype=np.float32),
            2400: np.asarray([1.5, 1.6, 1.7, 1.8], dtype=np.float32),
            1000: np.asarray([5.0, 6.0, 7.0, 8.0], dtype=np.float32),
        }
        return payloads[int(ptr)]

    def fake_copy_f32(ptr: int, elements: int, *, runtime: object) -> np.ndarray:
        calls.append(("copy_f32", ptr, elements, runtime))
        payloads = {
            2300: np.asarray([1.1, 1.2, 1.3, 1.4, 1.5, 1.6], dtype=np.float32),
            2350: np.asarray([2.1, 2.2, 2.3, 2.4], dtype=np.float32),
        }
        return payloads[int(ptr)]

    monkeypatch.setattr(gguf_runner, "_copy_bf16_ptr_to_host_f32", fake_copy_bf16)
    monkeypatch.setattr(gguf_runner, "_copy_f32_ptr_to_host", fake_copy_f32)

    runtime = FakeRuntime()
    session = object.__new__(Qwen35GGUFResidentSession)
    session.runtime = runtime
    session._hidden_a = SimpleNamespace(ptr=1234)
    session._hidden_seed_fp32_populated = True
    session._set_full_attention_position_device = fake_position
    session._set_token_id_device = fake_token
    cfg = SimpleNamespace(
        layer_types=(gguf_runner.LINEAR_ATTENTION,),
        ssm_time_step_rank=2,
        ssm_inner_size=4,
        is_moe=True,
    )
    session.runner = SimpleNamespace(
        weights=SimpleNamespace(config=cfg),
        hidden_size=4,
        linear_qkv_width=6,
        _run_linear_attention_attn_only=fake_attn,
    )
    session.scratch = SimpleNamespace(
        norm=SimpleNamespace(ptr=2000),
        linear_qkv=SimpleNamespace(ptr=2100),
        linear_z=SimpleNamespace(ptr=2200),
        linear_alpha=SimpleNamespace(ptr=3000),
        linear_beta=SimpleNamespace(ptr=4000),
        linear_alpha_beta=SimpleNamespace(ptr=5000),
        conv_out=SimpleNamespace(ptr=2300),
        recurrent_out=SimpleNamespace(ptr=2350),
        recurrent_bf16=SimpleNamespace(ptr=2400),
        attn_out=SimpleNamespace(ptr=1000),
    )

    capture = session.capture_linear_attention_boundary(17, position=3, layer_id=0)

    assert not session._hidden_seed_fp32_populated
    assert capture.as_summary_dict() == {
        "layer_id": 0,
        "token_id": 17,
        "position": 3,
        "hidden_size": 4,
        "ssm_time_step_rank": 2,
        "linear_qkv_width": 6,
        "ssm_inner_size": 4,
        "attn_norm_shape": [4],
        "linear_qkv_shape": [6],
        "linear_z_shape": [4],
        "ssm_alpha_shape": [2],
        "ssm_beta_shape": [2],
        "conv_out_shape": [6],
        "recurrent_out_shape": [4],
        "recurrent_bf16_shape": [4],
        "attn_out_shape": [4],
        "finite": True,
    }
    np.testing.assert_allclose(capture.ssm_beta_f32, [-0.25, -0.5])
    np.testing.assert_allclose(capture.conv_out_f32[:3], [1.1, 1.2, 1.3])
    assert calls == [
        ("position", 3, 0),
        ("token", 17, 0),
        ("attn", 0, 1234, 1000, 0),
        ("device_synchronize",),
        ("copy_bf16", 2000, 4, runtime),
        ("copy_bf16", 2100, 6, runtime),
        ("copy_bf16", 2200, 4, runtime),
        ("copy_bf16", 3000, 2, runtime),
        ("copy_bf16", 4000, 2, runtime),
        ("copy_f32", 2300, 6, runtime),
        ("copy_f32", 2350, 4, runtime),
        ("copy_bf16", 2400, 4, runtime),
        ("copy_bf16", 1000, 4, runtime),
    ]


def test_linear_attention_layer_capture_runs_full_layer_and_copies_post_ffn_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []

    class FakeRuntime:
        def device_synchronize(self) -> None:
            calls.append(("device_synchronize",))

    def fake_position(position: int, *, stream: int = 0) -> None:
        calls.append(("position", position, stream))

    def fake_token(token_id: int, *, stream: int = 0) -> None:
        calls.append(("token", token_id, stream))

    def fake_full_layer(
        layer_id: int,
        src_ptr: int,
        dst_ptr: int,
        scratch: object,
        **kwargs: object,
    ) -> None:
        calls.append(("layer", layer_id, src_ptr, dst_ptr, kwargs["stream"]))

    def fake_copy_bf16(ptr: int, elements: int, *, runtime: object) -> np.ndarray:
        calls.append(("copy_bf16", ptr, elements, runtime))
        payloads = {
            90: np.asarray([0.9, 0.91, 0.92, 0.93], dtype=np.float32),
            100: np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            1000: np.asarray([1.0, 1.1, 1.2, 1.3], dtype=np.float32),
            1050: np.asarray([1.5, 1.6, 1.7, 1.8, 1.9], dtype=np.float32),
            1060: np.asarray([1.91, 1.92, 1.93, 1.94], dtype=np.float32),
            1070: np.asarray([1.95, 1.96], dtype=np.float32),
            1080: np.asarray([1.97, 1.98], dtype=np.float32),
            1090: np.asarray([1.99, 2.0, 2.01, 2.02], dtype=np.float32),
            1100: np.asarray([2.0, 2.1, 2.2, 2.3], dtype=np.float32),
            1200: np.asarray([3.0, 3.1, 3.2, 3.3], dtype=np.float32),
            1300: np.asarray(
                [
                    4.0,
                    4.1,
                    4.2,
                    4.3,
                    4.4,
                    4.5,
                    4.6,
                    4.7,
                    4.8,
                    4.9,
                    5.0,
                    5.1,
                ],
                dtype=np.float32,
            ),
            1350: np.asarray([4.0, 4.1, 4.2, 4.3, 4.4, 4.5], dtype=np.float32),
            1360: np.asarray([4.6, 4.7], dtype=np.float32),
            1400: np.asarray([5.0, 5.1, 5.2, 5.3], dtype=np.float32),
            200: np.asarray([6.0, 6.1, 6.2, 6.3], dtype=np.float32),
        }
        return payloads[int(ptr)]

    def fake_copy_f32(ptr: int, elements: int, *, runtime: object) -> np.ndarray:
        calls.append(("copy_f32", ptr, elements, runtime))
        payloads = {
            1500: np.asarray([0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float32),
            1510: np.asarray([0.9, 1.0, 1.1, 1.2], dtype=np.float32),
            1600: np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
            1700: np.asarray([0.9, 0.8, 0.7, 0.6], dtype=np.float32),
            1716: np.asarray([-0.25], dtype=np.float32),
        }
        return payloads[int(ptr)]

    def fake_copy_i64(ptr: int, elements: int, *, runtime: object) -> np.ndarray:
        calls.append(("copy_i64", ptr, elements, runtime))
        return np.asarray([7, 8, 9], dtype=np.int64)

    monkeypatch.setattr(gguf_runner, "_copy_bf16_ptr_to_host_f32", fake_copy_bf16)
    monkeypatch.setattr(gguf_runner, "_copy_f32_ptr_to_host", fake_copy_f32)
    monkeypatch.setattr(gguf_runner, "_copy_i64_ptr_to_host", fake_copy_i64)

    runtime = FakeRuntime()
    session = object.__new__(Qwen35GGUFResidentSession)
    session.runtime = runtime
    session._hidden_a = SimpleNamespace(ptr=100)
    session._hidden_b = SimpleNamespace(ptr=200)
    session._hidden_seed_fp32_populated = True
    session._set_full_attention_position_device = fake_position
    session._set_token_id_device = fake_token
    cfg = SimpleNamespace(
        layer_types=(gguf_runner.LINEAR_ATTENTION,),
        is_moe=True,
        expert_used_count=3,
        expert_count=4,
        expert_feed_forward_length=2,
        expert_shared_feed_forward_length=2,
        ssm_inner_size=4,
        ssm_time_step_rank=2,
    )
    session.runner = SimpleNamespace(
        weights=SimpleNamespace(config=cfg),
        hidden_size=4,
        linear_qkv_width=5,
        _run_linear_attention_layer=fake_full_layer,
    )
    session.scratch = SimpleNamespace(
        norm=SimpleNamespace(ptr=90),
        attn_out=SimpleNamespace(ptr=1000),
        linear_qkv=SimpleNamespace(ptr=1050),
        linear_z=SimpleNamespace(ptr=1060),
        linear_alpha=SimpleNamespace(ptr=1070),
        linear_beta=SimpleNamespace(ptr=1080),
        recurrent_bf16=SimpleNamespace(ptr=1090),
        post_norm=SimpleNamespace(ptr=1100),
        residual=SimpleNamespace(ptr=1200),
        moe_down_out=SimpleNamespace(ptr=1300),
        ffn_intermediate=SimpleNamespace(ptr=1350),
        moe_shared_intermediate=SimpleNamespace(ptr=1360),
        moe_shared_out=SimpleNamespace(ptr=1400),
        conv_out=SimpleNamespace(ptr=1500),
        recurrent_out=SimpleNamespace(ptr=1510),
        moe_routing_weights=SimpleNamespace(ptr=1600),
        moe_router_logits=SimpleNamespace(ptr=1700),
        moe_selected_experts=SimpleNamespace(ptr=1800),
        ffn_down=SimpleNamespace(ptr=1500),
    )

    capture = session.capture_linear_attention_layer(17, position=3, layer_id=0)

    assert not session._hidden_seed_fp32_populated
    summary = capture.as_summary_dict()
    expected_summary_items = {
        "layer_id": 0,
        "layer_type": gguf_runner.LINEAR_ATTENTION,
        "token_id": 17,
        "position": 3,
        "hidden_size": 4,
        "is_moe": True,
        "top_k": 3,
        "preceding_layer_count": 0,
        "hidden_in_shape": [4],
        "attn_norm_shape": [4],
        "attn_out_shape": [4],
        "post_norm_shape": [4],
        "post_norm_source": "bf16_scratch.post_norm",
        "residual_shape": [4],
        "ffn_or_moe_down_shape": [12],
        "moe_router_logits_shape": [4],
        "moe_selected_intermediate_shape": [6],
        "moe_shared_intermediate_shape": [2],
        "moe_shared_out_shape": [4],
        "moe_routing_weights_shape": [3],
        "moe_shared_gate_shape": [1],
        "moe_selected_experts_shape": [3],
        "linear_qkv_shape": [5],
        "linear_z_shape": [4],
        "ssm_alpha_shape": [2],
        "ssm_beta_shape": [2],
        "conv_out_shape": [5],
        "recurrent_out_shape": [4],
        "recurrent_bf16_shape": [4],
        "layer_out_shape": [4],
        "finite": True,
    }
    for key, value in expected_summary_items.items():
        assert summary[key] == value
    np.testing.assert_allclose(capture.layer_out_f32, [6.0, 6.1, 6.2, 6.3])
    np.testing.assert_allclose(capture.moe_routing_weights_f32, [0.1, 0.2, 0.3])
    np.testing.assert_array_equal(capture.moe_selected_experts_i64, [7, 8, 9])
    assert calls == [
        ("position", 3, 0),
        ("token", 17, 0),
        ("layer", 0, 100, 200, 0),
        ("device_synchronize",),
        ("copy_bf16", 1050, 5, runtime),
        ("copy_bf16", 1060, 4, runtime),
        ("copy_bf16", 1070, 2, runtime),
        ("copy_bf16", 1080, 2, runtime),
        ("copy_f32", 1500, 5, runtime),
        ("copy_f32", 1510, 4, runtime),
        ("copy_bf16", 1090, 4, runtime),
        ("copy_f32", 1700, 4, runtime),
        ("copy_bf16", 1350, 6, runtime),
        ("copy_bf16", 1360, 2, runtime),
        ("copy_bf16", 1400, 4, runtime),
        ("copy_f32", 1600, 3, runtime),
        ("copy_f32", 1716, 1, runtime),
        ("copy_i64", 1800, 3, runtime),
        ("copy_bf16", 100, 4, runtime),
        ("copy_bf16", 90, 4, runtime),
        ("copy_bf16", 1000, 4, runtime),
        ("copy_bf16", 1100, 4, runtime),
        ("copy_bf16", 1200, 4, runtime),
        ("copy_bf16", 1300, 12, runtime),
        ("copy_bf16", 200, 4, runtime),
    ]


def test_attention_layer_capture_runs_full_attention_layer_for_full_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []

    class FakeRuntime:
        def device_synchronize(self) -> None:
            calls.append(("device_synchronize",))

    def fake_position(position: int, *, stream: int = 0) -> None:
        calls.append(("position", position, stream))

    def fake_token(token_id: int, *, stream: int = 0) -> None:
        calls.append(("token", token_id, stream))

    def fake_full_attention_layer(
        layer_id: int,
        src_ptr: int,
        dst_ptr: int,
        scratch: object,
        **kwargs: object,
    ) -> None:
        calls.append(
            ("full_layer", layer_id, src_ptr, dst_ptr, kwargs["position"], kwargs["stream"])
        )

    def fake_copy_bf16(ptr: int, elements: int, *, runtime: object) -> np.ndarray:
        calls.append(("copy_bf16", ptr, elements, runtime))
        payloads = {
            90: np.asarray([0.9, 0.91, 0.92, 0.93], dtype=np.float32),
            100: np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            200: np.asarray([6.0, 6.1, 6.2, 6.3], dtype=np.float32),
            1000: np.asarray([1.0, 1.1, 1.2, 1.3], dtype=np.float32),
            1100: np.asarray([2.0, 2.1, 2.2, 2.3], dtype=np.float32),
            1200: np.asarray([3.0, 3.1, 3.2, 3.3], dtype=np.float32),
            1500: np.asarray([4.0, 4.1, 4.2, 4.3], dtype=np.float32),
        }
        return payloads[int(ptr)]

    monkeypatch.setattr(gguf_runner, "_copy_bf16_ptr_to_host_f32", fake_copy_bf16)

    runtime = FakeRuntime()
    session = object.__new__(Qwen35GGUFResidentSession)
    session.runtime = runtime
    session._hidden_a = SimpleNamespace(ptr=100)
    session._hidden_b = SimpleNamespace(ptr=200)
    session._hidden_seed_fp32_populated = True
    session._set_full_attention_position_device = fake_position
    session._set_token_id_device = fake_token
    cfg = SimpleNamespace(
        layer_types=(gguf_runner.FULL_ATTENTION,),
        is_moe=False,
        expert_used_count=1,
    )
    session.runner = SimpleNamespace(
        weights=SimpleNamespace(config=cfg),
        hidden_size=4,
        _run_full_attention_layer=fake_full_attention_layer,
    )
    session.scratch = SimpleNamespace(
        norm=SimpleNamespace(ptr=90),
        attn_out=SimpleNamespace(ptr=1000),
        post_norm=SimpleNamespace(ptr=1100),
        residual=SimpleNamespace(ptr=1200),
        ffn_down=SimpleNamespace(ptr=1500),
    )

    capture = session.capture_attention_layer(17, position=3, layer_id=0)

    assert capture.as_summary_dict()["layer_type"] == gguf_runner.FULL_ATTENTION
    np.testing.assert_allclose(capture.layer_out_f32, [6.0, 6.1, 6.2, 6.3])
    assert calls == [
        ("position", 3, 0),
        ("token", 17, 0),
        ("full_layer", 0, 100, 200, 3, 0),
        ("device_synchronize",),
        ("copy_bf16", 100, 4, runtime),
        ("copy_bf16", 90, 4, runtime),
        ("copy_bf16", 1000, 4, runtime),
        ("copy_bf16", 1100, 4, runtime),
        ("copy_bf16", 1200, 4, runtime),
        ("copy_bf16", 1500, 4, runtime),
        ("copy_bf16", 200, 4, runtime),
    ]


def test_attention_layer_capture_can_run_preceding_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []

    class FakeRuntime:
        def device_synchronize(self) -> None:
            calls.append(("device_synchronize",))

    def fake_position(position: int, *, stream: int = 0) -> None:
        calls.append(("position", position, stream))

    def fake_token(token_id: int, *, stream: int = 0) -> None:
        calls.append(("token", token_id, stream))

    def fake_linear_layer(
        layer_id: int,
        src_ptr: int,
        dst_ptr: int,
        scratch: object,
        **kwargs: object,
    ) -> None:
        calls.append(("linear_layer", layer_id, src_ptr, dst_ptr, kwargs["stream"]))

    def fake_full_attention_layer(
        layer_id: int,
        src_ptr: int,
        dst_ptr: int,
        scratch: object,
        **kwargs: object,
    ) -> None:
        calls.append(
            ("full_layer", layer_id, src_ptr, dst_ptr, kwargs["position"], kwargs["stream"])
        )

    def fake_copy_bf16(ptr: int, elements: int, *, runtime: object) -> np.ndarray:
        calls.append(("copy_bf16", ptr, elements, runtime))
        payloads = {
            90: np.asarray([0.9, 0.91, 0.92, 0.93], dtype=np.float32),
            100: np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            200: np.asarray([6.0, 6.1, 6.2, 6.3], dtype=np.float32),
            1000: np.asarray([1.0, 1.1, 1.2, 1.3], dtype=np.float32),
            1100: np.asarray([2.0, 2.1, 2.2, 2.3], dtype=np.float32),
            1200: np.asarray([3.0, 3.1, 3.2, 3.3], dtype=np.float32),
            1500: np.asarray([4.0, 4.1, 4.2, 4.3], dtype=np.float32),
        }
        return payloads[int(ptr)]

    monkeypatch.setattr(gguf_runner, "_copy_bf16_ptr_to_host_f32", fake_copy_bf16)

    runtime = FakeRuntime()
    session = object.__new__(Qwen35GGUFResidentSession)
    session.runtime = runtime
    session._hidden_a = SimpleNamespace(ptr=100)
    session._hidden_b = SimpleNamespace(ptr=200)
    session._hidden_seed_fp32_populated = True
    session._set_full_attention_position_device = fake_position
    session._set_token_id_device = fake_token
    cfg = SimpleNamespace(
        layer_types=(gguf_runner.LINEAR_ATTENTION, gguf_runner.FULL_ATTENTION),
        is_moe=False,
        expert_used_count=1,
    )
    session.runner = SimpleNamespace(
        weights=SimpleNamespace(config=cfg),
        hidden_size=4,
        _run_linear_attention_layer=fake_linear_layer,
        _run_full_attention_layer=fake_full_attention_layer,
    )
    session.scratch = SimpleNamespace(
        norm=SimpleNamespace(ptr=90),
        attn_out=SimpleNamespace(ptr=1000),
        post_norm=SimpleNamespace(ptr=1100),
        residual=SimpleNamespace(ptr=1200),
        ffn_down=SimpleNamespace(ptr=1500),
    )

    capture = session.capture_attention_layer(
        17,
        position=3,
        layer_id=1,
        run_preceding_layers=True,
    )

    assert capture.as_summary_dict()["preceding_layer_count"] == 1
    np.testing.assert_allclose(capture.hidden_in_f32, [6.0, 6.1, 6.2, 6.3])
    np.testing.assert_allclose(capture.layer_out_f32, [0.1, 0.2, 0.3, 0.4])
    assert calls == [
        ("position", 3, 0),
        ("token", 17, 0),
        ("linear_layer", 0, 100, 200, 0),
        ("full_layer", 1, 200, 100, 3, 0),
        ("device_synchronize",),
        ("copy_bf16", 200, 4, runtime),
        ("copy_bf16", 90, 4, runtime),
        ("copy_bf16", 1000, 4, runtime),
        ("copy_bf16", 1100, 4, runtime),
        ("copy_bf16", 1200, 4, runtime),
        ("copy_bf16", 1500, 4, runtime),
        ("copy_bf16", 100, 4, runtime),
    ]


def test_resident_session_reset_clears_hidden_seed_populated_flag_without_gpu_init() -> None:
    session = object.__new__(Qwen35GGUFResidentSession)
    session.scratch = SimpleNamespace(zero_states=lambda runtime, **kwargs: None)
    session._target_scratch_owner = session.scratch
    session.runtime = object()
    session._set_full_attention_position_device = lambda position, stream=0: None
    session._position = 7
    session._hidden_seed_fp32_populated = True

    session.reset()

    assert session._position == 0
    assert not session._hidden_seed_fp32_populated


def test_resident_session_hidden_seed_contract_rejects_closed_session() -> None:
    session = object.__new__(Qwen35GGUFResidentSession)
    session.runner = None

    with pytest.raises(RuntimeError, match="GGUF resident session is closed"):
        session.hidden_seed_contract()
    with pytest.raises(RuntimeError, match="GGUF resident session is closed"):
        session.fp32_hidden_seed_contract()
    with pytest.raises(RuntimeError, match="GGUF resident session is closed"):
        session.fp32_hidden_seed_ptr()


def test_fp32_hidden_seed_contract_is_llama_compatible() -> None:
    contract = Qwen35GGUFHiddenSeedContract(
        provenance="post_output_norm",
        dtype=DType.FP32,
        rows=3,
        hidden_size=4096,
        source_buffer="future_fp32_hidden_seed_tap",
        populated_by_decode=True,
        llama_cpp_compatible=True,
    )

    assert not contract.requires_fp32_tap
    assert contract.ready_for_mtp
    assert contract.as_dict()["dtype"] == "FP32"


def test_fp32_verify_hidden_seed_contract_uses_verifier_row_buffer() -> None:
    contract = qwen35_gguf_fp32_verify_hidden_seed_contract(
        hidden_size=4096,
        rows=2,
        populated_by_decode=True,
    )

    assert contract.provenance == "post_output_norm"
    assert contract.dtype is DType.FP32
    assert contract.rows == 2
    assert contract.hidden_size == 4096
    assert contract.source_buffer == "Qwen35GGUFResidentSession._verify_hidden_seed_buf"
    assert contract.ready_for_mtp


def test_resident_session_stages_current_hidden_seed_as_verify_row_without_gpu_init() -> None:
    class Runtime:
        def __init__(self) -> None:
            self.calls = []

        def memcpy_async(self, dst, src, nbytes, kind, stream) -> None:
            self.calls.append((int(dst), int(src), int(nbytes), int(kind), int(stream)))

    runtime = Runtime()
    session = object.__new__(Qwen35GGUFResidentSession)
    session.runner = SimpleNamespace(hidden_size=8)
    session.scratch = SimpleNamespace(hidden_seed_fp32=SimpleNamespace(ptr=0x1000))
    session.runtime = runtime
    session._hidden_seed_fp32_populated = True
    session._verify_hidden_seed_buf = None
    session._verify_hidden_seed_rows_populated = 0
    session._verify_block_rows_capacity = 0

    def ensure(rows, *, runtime) -> None:
        session._verify_hidden_seed_buf = SimpleNamespace(ptr=0x2000)
        session._verify_block_rows_capacity = int(rows)

    session._ensure_verify_block_buffers = ensure

    seed = session.stage_current_hidden_seed_as_verify_row(
        row_index=2,
        token_id=123,
        position=45,
        rows_capacity=4,
        stream=7,
    )

    assert runtime.calls == [
        (0x2000 + 2 * 8 * 4, 0x1000, 8 * 4, int(HipMemcpyKind.DEVICE_TO_DEVICE), 7)
    ]
    assert session._verify_hidden_seed_rows_populated == 3
    assert seed.token_id == 123
    assert seed.position == 45
    assert seed.hidden_ptr == 0x2000 + 2 * 8 * 4
    assert seed.hidden_contract.ready_for_mtp
    assert seed.hidden_contract.source_buffer == "Qwen35GGUFResidentSession._verify_hidden_seed_buf"
    assert session.fp32_verify_hidden_seed_ptr(2) == seed.hidden_ptr
    with pytest.raises(RuntimeError, match="not populated"):
        session.fp32_verify_hidden_seed_ptr(3)


def test_resident_session_describes_external_native_verify_seed_rows() -> None:
    session = object.__new__(Qwen35GGUFResidentSession)
    session.runner = SimpleNamespace(hidden_size=8)
    session._verify_hidden_seed_rows_populated = 0

    seed = session.mtp_verify_seed(
        2,
        token_id=123,
        position=45,
        hidden_seed_base_ptr=0x4000,
        hidden_seed_row_count=3,
    )

    assert seed.hidden_ptr == 0x4000 + 2 * 8 * 4
    assert seed.hidden_contract.ready_for_mtp
    assert seed.hidden_contract.source_buffer == "Qwen35GGUFNativeAcceptCommitResult.hidden_seed_rows_ptr"
    with pytest.raises(ValueError, match="outside external hidden rows"):
        session.mtp_verify_seed(
            3,
            token_id=123,
            position=45,
            hidden_seed_base_ptr=0x4000,
            hidden_seed_row_count=3,
        )


def test_hidden_seed_contract_rejects_pre_norm_or_wrong_compatibility() -> None:
    with pytest.raises(ValueError, match="provenance must be post_output_norm"):
        Qwen35GGUFHiddenSeedContract(
            provenance="pre_output_norm",
            dtype=DType.FP32,
            rows=1,
            hidden_size=4096,
            source_buffer="bad",
            populated_by_decode=True,
            llama_cpp_compatible=True,
        )

    with pytest.raises(ValueError, match="llama_cpp_compatible must reflect"):
        Qwen35GGUFHiddenSeedContract(
            provenance="post_output_norm",
            dtype=DType.BF16,
            rows=1,
            hidden_size=4096,
            source_buffer="bad",
            populated_by_decode=True,
            llama_cpp_compatible=True,
        )

    with pytest.raises(ValueError, match="llama_cpp_compatible must reflect"):
        Qwen35GGUFHiddenSeedContract(
            provenance="post_output_norm",
            dtype=DType.FP32,
            rows=1,
            hidden_size=4096,
            source_buffer="bad",
            populated_by_decode=False,
            llama_cpp_compatible=True,
        )
