from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.dtype import DType
from hipengine.runtime import qwen35_gguf_runner as gguf_runner
from hipengine.runtime.qwen35_gguf_runner import (
    Qwen35GGUFHiddenSeedContract,
    Qwen35GGUFMTPDraftSeed,
    Qwen35GGUFResidentSession,
    qwen35_gguf_current_hidden_seed_contract,
    qwen35_gguf_fp32_hidden_seed_contract,
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
        config=SimpleNamespace(layer_types=(), rms_norm_eps=1.0e-6),
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
    session.scratch = SimpleNamespace(zero_states=lambda runtime: None)
    session.runtime = object()
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

    def fake_sample_from_hidden(hidden_ptr: int, *, return_logits: bool) -> SimpleNamespace:
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
        weights=SimpleNamespace(config=SimpleNamespace(ssm_conv_kernel=2))
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
    ) -> SimpleNamespace:
        bulk_calls.append(
            {
                "token_ids": tuple(token_ids),
                "bulk_attention_mode": bulk_attention_mode,
                "return_logits": return_logits,
                "capture_hidden_seed_fp32": capture_hidden_seed_fp32,
            }
        )
        return SimpleNamespace(token_id=8)

    session._run_bulk_prefill_and_sample = fake_bulk_prefill_and_sample

    result = session.prefill(
        [10, 11],
        use_bulk=True,
        bulk_attention_mode="native",
        return_logits=False,
        capture_hidden_seed_fp32=True,
    )

    assert result.token_id == 8
    assert bulk_calls == [
        {
            "token_ids": (10, 11),
            "bulk_attention_mode": "native",
            "return_logits": False,
            "capture_hidden_seed_fp32": True,
        }
    ]


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
            5000: np.asarray([0.25, 0.5], dtype=np.float32),
            5004: np.asarray([-0.25, -0.5], dtype=np.float32),
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
        ("copy_bf16", 5000, 2, runtime),
        ("copy_bf16", 5004, 2, runtime),
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
            100: np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            1000: np.asarray([1.0, 1.1, 1.2, 1.3], dtype=np.float32),
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
            1400: np.asarray([5.0, 5.1, 5.2, 5.3], dtype=np.float32),
            200: np.asarray([6.0, 6.1, 6.2, 6.3], dtype=np.float32),
        }
        return payloads[int(ptr)]

    def fake_copy_f32(ptr: int, elements: int, *, runtime: object) -> np.ndarray:
        calls.append(("copy_f32", ptr, elements, runtime))
        payloads = {
            1600: np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
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
    )
    session.runner = SimpleNamespace(
        weights=SimpleNamespace(config=cfg),
        hidden_size=4,
        _run_linear_attention_layer=fake_full_layer,
    )
    session.scratch = SimpleNamespace(
        attn_out=SimpleNamespace(ptr=1000),
        post_norm=SimpleNamespace(ptr=1100),
        residual=SimpleNamespace(ptr=1200),
        moe_down_out=SimpleNamespace(ptr=1300),
        moe_shared_out=SimpleNamespace(ptr=1400),
        moe_routing_weights=SimpleNamespace(ptr=1600),
        moe_router_logits=SimpleNamespace(ptr=1700),
        moe_selected_experts=SimpleNamespace(ptr=1800),
        ffn_down=SimpleNamespace(ptr=1500),
    )

    capture = session.capture_linear_attention_layer(17, position=3, layer_id=0)

    assert not session._hidden_seed_fp32_populated
    assert capture.as_summary_dict() == {
        "layer_id": 0,
        "layer_type": gguf_runner.LINEAR_ATTENTION,
        "token_id": 17,
        "position": 3,
        "hidden_size": 4,
        "is_moe": True,
        "top_k": 3,
        "hidden_in_shape": [4],
        "attn_out_shape": [4],
        "post_norm_shape": [4],
        "residual_shape": [4],
        "ffn_or_moe_down_shape": [12],
        "moe_shared_out_shape": [4],
        "moe_routing_weights_shape": [3],
        "moe_shared_gate_shape": [1],
        "moe_selected_experts_shape": [3],
        "layer_out_shape": [4],
        "finite": True,
    }
    np.testing.assert_allclose(capture.layer_out_f32, [6.0, 6.1, 6.2, 6.3])
    np.testing.assert_allclose(capture.moe_routing_weights_f32, [0.1, 0.2, 0.3])
    np.testing.assert_array_equal(capture.moe_selected_experts_i64, [7, 8, 9])
    assert calls == [
        ("position", 3, 0),
        ("token", 17, 0),
        ("layer", 0, 100, 200, 0),
        ("device_synchronize",),
        ("copy_bf16", 1400, 4, runtime),
        ("copy_f32", 1600, 3, runtime),
        ("copy_f32", 1716, 1, runtime),
        ("copy_i64", 1800, 3, runtime),
        ("copy_bf16", 100, 4, runtime),
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
        ("copy_bf16", 1000, 4, runtime),
        ("copy_bf16", 1100, 4, runtime),
        ("copy_bf16", 1200, 4, runtime),
        ("copy_bf16", 1500, 4, runtime),
        ("copy_bf16", 200, 4, runtime),
    ]


def test_resident_session_reset_clears_hidden_seed_populated_flag_without_gpu_init() -> None:
    session = object.__new__(Qwen35GGUFResidentSession)
    session.scratch = SimpleNamespace(zero_states=lambda runtime: None)
    session.runtime = object()
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
