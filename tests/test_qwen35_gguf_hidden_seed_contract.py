from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.core.dtype import DType
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
