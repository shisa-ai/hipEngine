from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.dtype import DType
from hipengine.runtime import qwen35_gguf_runner as gguf_runner
from hipengine.runtime.qwen35_gguf_runner import (
    Qwen35GGUFHiddenSeedContract,
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
    session.scratch = SimpleNamespace(hidden_seed_fp32=object())
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

    session._hidden_seed_fp32_populated = True
    populated = session.fp32_hidden_seed_contract(rows=2)
    assert populated.populated_by_decode
    assert populated.llama_cpp_compatible
    assert populated.ready_for_mtp


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
