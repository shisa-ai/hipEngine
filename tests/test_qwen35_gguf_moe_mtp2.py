from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.core.dtype import DType
from hipengine.core.memory import DeviceBuffer
from hipengine.core.specdec2_scope import moe_physical_c2_numerics_session
from hipengine.generation.qwen35_gguf_moe_mtp2 import (
    Qwen35GGUFMoEMTP2Adapter,
    _MoeTargetHiddenSink,
    create_qwen35_gguf_moe_mtp2_adapter,
)
from hipengine.generation.qwen35_gguf_mtp2 import Qwen35GGUFMTP2Adapter
from hipengine.generation.qwen35_gguf_mtp2_registry import (
    register_builtin_gguf_mtp2_adapters,
    resolve_gguf_mtp2_adapter,
)
from hipengine.speculative.provider import SpeculativeRequestSemantics
from hipengine.runtime.qwen35_gguf_runner import _gguf_verify_f32_residual_enabled


class _Runtime:
    def __init__(self) -> None:
        self.copies = []
        self.syncs = []

    def memcpy_async(self, dst, src, nbytes, kind, stream):
        self.copies.append((int(dst), int(src), int(nbytes), kind, int(stream)))

    def stream_synchronize(self, stream):
        self.syncs.append(int(stream))


def test_moe_physical_c2_numerics_scope_is_request_local(monkeypatch) -> None:
    monkeypatch.setenv("HIPENGINE_GGUF_VERIFY_F32_RESIDUAL", "1")

    assert _gguf_verify_f32_residual_enabled() is True
    with moe_physical_c2_numerics_session(True):
        assert _gguf_verify_f32_residual_enabled() is False
    assert _gguf_verify_f32_residual_enabled() is True


def test_moe_target_hidden_sink_preserves_rows_and_publishes_final_target() -> None:
    runtime = _Runtime()
    target = SimpleNamespace(
        runtime=runtime,
        scratch=SimpleNamespace(hidden_seed_fp32=DeviceBuffer(0x7000, 16)),
        _hidden_seed_fp32_populated=False,
        _last_target_hidden_ptr=0,
    )
    norm_calls = []

    def run_norm(src, dst, *, stream, capture_hidden_seed_fp32):
        norm_calls.append((src, dst, stream, capture_hidden_seed_fp32))

    target._run_output_norm_hidden = run_norm
    sink = _MoeTargetHiddenSink(
        request_id=7,
        hidden_size=4,
        total_rows=3,
        target=target,
        buffer=DeviceBuffer(0x1000, 3 * 4 * DType.BF16.itemsize),
        normalized=DeviceBuffer(0x2000, 3 * 4 * DType.FP32.itemsize),
        normalized_bf16=DeviceBuffer(0x3000, 4 * DType.BF16.itemsize),
    )

    sink.consume(request_id=7, chunk_start=1, hidden_ptr=0x5000, rows=2, stream=9)
    sink.finish(request_id=7, total_rows=3, stream=9)

    assert runtime.copies[0][:3] == (0x1008, 0x5000, 16)
    assert [call[0] for call in norm_calls] == [0x1000, 0x1008, 0x1010]
    assert target._hidden_seed_fp32_populated is True
    assert target._last_target_hidden_ptr == 0x1010
    assert sink.finished is True


def test_moe_adapter_capability_is_c1_k2_and_short_context_only() -> None:
    adapter = object.__new__(Qwen35GGUFMoEMTP2Adapter)
    adapter.enabled = True
    adapter.candidate_budget = 2
    adapter.quant = "gguf_q4_k_m"
    adapter.target_verify_mode = "native"
    adapter._disabled_requests = set()
    adapter._states = {7: object()}
    target = SimpleNamespace(
        position=6,
        target_layout=SimpleNamespace(max_sequence_length=1024),
        kv_storage_dtype="bf16",
    )
    row = SimpleNamespace(
        native_greedy=True,
        first_token_emitted=True,
        lease=SimpleNamespace(session=target),
        slot=object(),
    )
    adapter.owner = SimpleNamespace(capacity=1, _row=lambda request_id: row)
    adapter.generator = SimpleNamespace(
        backend="hip_gfx1100",
        execution_profile=SimpleNamespace(value="strict"),
    )

    capability = adapter.capability(
        (SpeculativeRequestSemantics(7, "greedy", "verify_chain", 6, 8),)
    )

    assert capability is not None
    assert capability.target_key == "qwen_moe_gguf"
    assert capability.provider_key == "qwen_nextn_moe"
    assert capability.max_requests == 1
    assert capability.max_candidates_per_request == 2
    assert capability.proposal_widths == (1,)
    assert capability.graph_supported is True
    assert adapter.capability(
        (SpeculativeRequestSemantics(7, "greedy", "verify_chain", 96, 8),)
    ) is None
    assert adapter.capability(
        (SpeculativeRequestSemantics(7, "greedy", "verify_chain", 6, 2),)
    ) is not None
    assert adapter.capability(
        (SpeculativeRequestSemantics(7, "greedy", "verify_chain", 6, 3),)
    ) is not None


def test_moe_adapter_capability_exposes_exact_physical_c2() -> None:
    adapter = object.__new__(Qwen35GGUFMoEMTP2Adapter)
    adapter.enabled = True
    adapter.candidate_budget = 2
    adapter.quant = "gguf_q4_k_m"
    adapter.target_verify_mode = "native"
    adapter._disabled_requests = set()
    adapter._states = {7: object(), 8: object()}
    targets = {
        rid: SimpleNamespace(
            position=6,
            target_layout=SimpleNamespace(max_sequence_length=1024),
            kv_storage_dtype="bf16",
        )
        for rid in (7, 8)
    }
    rows = {
        rid: SimpleNamespace(
            native_greedy=True,
            first_token_emitted=True,
            lease=SimpleNamespace(session=targets[rid]),
            slot=object(),
        )
        for rid in targets
    }
    adapter.owner = SimpleNamespace(
        capacity=2,
        _row=lambda request_id: rows[int(request_id)],
    )
    adapter.generator = SimpleNamespace(
        backend="hip_gfx1100",
        execution_profile=SimpleNamespace(value="production"),
    )
    one = SpeculativeRequestSemantics(7, "greedy", "verify_chain", 6, 8)
    two = SpeculativeRequestSemantics(8, "greedy", "verify_chain", 6, 8)

    assert adapter.capability((one,)) is None
    capability = adapter.capability((one, two))
    assert capability is not None
    assert adapter.staged_frontier is True
    assert capability.max_requests == 2
    assert capability.max_frontier_rows == 6
    assert capability.proposal_widths == (1, 2)
    adapter.generator.execution_profile = SimpleNamespace(value="strict")
    assert adapter.capability((one, two)) is None


def test_moe_adapter_accepts_generic_static_eligibility_registration() -> None:
    adapter = object.__new__(Qwen35GGUFMoEMTP2Adapter)
    adapter.candidate_budget = 2
    adapter._intents = {}
    adapter._disabled_requests = {7}

    adapter.register_request(7, 3, static_eligibility=object())

    assert adapter._intents == {7: 2}
    assert adapter._disabled_requests == set()


def test_moe_prefill_missing_sink_fails_closed_only_beyond_context_limit() -> None:
    adapter = object.__new__(Qwen35GGUFMoEMTP2Adapter)
    adapter._intents = {7: 2}
    adapter._disabled_requests = set()
    adapter._states = {}
    adapter._prompt_sinks = {}
    target = SimpleNamespace()
    row = SimpleNamespace(lease=SimpleNamespace(session=target))
    adapter.owner = SimpleNamespace(_row=lambda request_id: row)

    adapter.observe_prefill_result(7, tuple(range(96)), SimpleNamespace(token_id=1))

    assert adapter._disabled_requests == {7}
    adapter._disabled_requests.clear()
    with pytest.raises(RuntimeError, match="lost its target-hidden sink"):
        adapter.observe_prefill_result(7, tuple(range(95)), SimpleNamespace(token_id=1))


def test_moe_adapter_factory_keeps_c1_and_selects_batched_c2() -> None:
    owner_c1 = SimpleNamespace(capacity=1, generator=SimpleNamespace())
    owner_c2 = SimpleNamespace(
        capacity=2,
        generator=SimpleNamespace(
            execution_profile=SimpleNamespace(value="production"),
            backend="hip_gfx1100",
        ),
    )
    kwargs = {
        "enabled": True,
        "target_verify_mode": "native",
        "candidate_budget": 2,
        "quant": "gguf_q4_k_m",
    }

    c1 = create_qwen35_gguf_moe_mtp2_adapter(owner_c1, **kwargs)
    c2 = create_qwen35_gguf_moe_mtp2_adapter(owner_c2, **kwargs)

    assert isinstance(c1, Qwen35GGUFMoEMTP2Adapter)
    assert isinstance(c2, Qwen35GGUFMTP2Adapter)
    assert c2.production_physical_extra_rowtiles is False
    assert c2.production_physical_q5_rowtile is False
    assert c2.production_physical_q6_rowtile is False
    assert c2.moe_physical_c2_numerics is True
    assert c2.target_key == "qwen_moe_gguf"
    assert c2.provider_key == "qwen_nextn_moe"
    assert c2.policy_prefix == "moe-nextn"


def test_builtin_registry_exposes_distinct_dense_and_moe_factories() -> None:
    register_builtin_gguf_mtp2_adapters()

    assert resolve_gguf_mtp2_adapter("dense_nextn").__name__ == "Qwen35GGUFMTP2Adapter"
    assert (
        resolve_gguf_mtp2_adapter("moe_nextn")
        is create_qwen35_gguf_moe_mtp2_adapter
    )
