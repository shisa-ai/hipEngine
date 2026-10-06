from __future__ import annotations

import pytest

from scripts.qwen4exp_ed3_mixed_quant_gate import (
    EXPECTED_ORDER,
    Q51_LAYERS,
    Q8_LAYERS,
    expected_chunks,
    expected_owner_calls,
    validate_quant_split,
)


def test_ed3_has_the_frozen_q51_q8_layer_partition():
    assert Q8_LAYERS == (2, 4, 30, 46, 47)
    assert len(Q51_LAYERS) == 43
    assert len(EXPECTED_ORDER) == 48
    assert [index for index, owner in enumerate(EXPECTED_ORDER) if owner == "q8"] == list(Q8_LAYERS)


def test_ed3_owner_call_contract_scales_by_nonempty_prefill_chunk():
    assert expected_chunks(1, 1024) == 1
    assert expected_chunks(4096, 1024) == 4
    assert expected_owner_calls(4096, 1024) == {"q51": 172, "q8": 20}
    with pytest.raises(ValueError):
        expected_chunks(0, 1024)


def test_ed3_quant_triplets_require_q4_gate_up_and_mixed_down():
    triplets = [
        (("gguf_q5_k", "gguf_q5_k") if layer == 2 else ("gguf_q4_k", "gguf_q4_k"))
        + (("gguf_q8_0",) if layer in Q8_LAYERS else ("gguf_q5_1",))
        for layer in range(48)
    ]
    report = validate_quant_split(triplets)
    assert report["q51_layers"] == list(Q51_LAYERS)
    assert report["q8_layers"] == list(Q8_LAYERS)
    bad = list(triplets)
    bad[2] = ("gguf_q5_k", "gguf_q5_k", "gguf_q5_1")
    with pytest.raises(ValueError, match="unexpected model quant split"):
        validate_quant_split(bad)
