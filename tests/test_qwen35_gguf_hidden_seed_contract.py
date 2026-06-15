from __future__ import annotations

import pytest

from hipengine.core.dtype import DType
from hipengine.runtime.qwen35_gguf_runner import (
    Qwen35GGUFHiddenSeedContract,
    qwen35_gguf_current_hidden_seed_contract,
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
        "llama_cpp_compatible": False,
        "requires_fp32_tap": True,
    }


def test_fp32_hidden_seed_contract_is_llama_compatible() -> None:
    contract = Qwen35GGUFHiddenSeedContract(
        provenance="post_output_norm",
        dtype=DType.FP32,
        rows=3,
        hidden_size=4096,
        source_buffer="future_fp32_hidden_seed_tap",
        llama_cpp_compatible=True,
    )

    assert not contract.requires_fp32_tap
    assert contract.as_dict()["dtype"] == "FP32"


def test_hidden_seed_contract_rejects_pre_norm_or_wrong_compatibility() -> None:
    with pytest.raises(ValueError, match="provenance must be post_output_norm"):
        Qwen35GGUFHiddenSeedContract(
            provenance="pre_output_norm",
            dtype=DType.FP32,
            rows=1,
            hidden_size=4096,
            source_buffer="bad",
            llama_cpp_compatible=True,
        )

    with pytest.raises(ValueError, match="llama_cpp_compatible must reflect"):
        Qwen35GGUFHiddenSeedContract(
            provenance="post_output_norm",
            dtype=DType.BF16,
            rows=1,
            hidden_size=4096,
            source_buffer="bad",
            llama_cpp_compatible=True,
        )
