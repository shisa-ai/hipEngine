from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from scripts import inspect_gguf
from tests.test_qwen35_gguf_mtp_mapping import _synthetic_qwen35moe_mtp_info


FIXTURE = Path("benchmarks/fixtures/qwen36_35b_a3b_ud_q4_k_m_mtp_inventory.json")


def test_inspect_gguf_reports_full_qwen35_mtp_inventory_and_optional_status() -> None:
    info = _synthetic_qwen35moe_mtp_info()

    summary = inspect_gguf.summarize(
        SimpleNamespace(info=info), check_dequant=False, smoke_rows=1
    )

    inventory = summary["qwen35_mtp_inventory"]
    assert inventory["declared_block_count"] == 3
    assert inventory["ar_block_count"] == 2
    assert inventory["ignored_block_ids"] == [2]
    (block,) = inventory["blocks"]
    assert block["layer_id"] == 2
    assert block["tensor_count"] == 20
    assert block["nextn_tensor_count"] == 4
    assert block["required_missing"] == []
    assert block["optional_missing"] == [
        "blk.2.nextn.embed_tokens.weight",
        "blk.2.nextn.shared_head_head.weight",
    ]
    assert block["unexpected_tensor_names"] == []

    status = block["nextn_optional_status"]
    assert status["nextn.embed_tokens"] == {
        "status": "fallback",
        "tensor_name": None,
        "fallback_slot": "token_embedding",
        "fallback_tensor_name": "token_embd.weight",
        "fallback_shape": [11, 8],
        "fallback_type": "F32",
        "fallback_nbytes": 352,
    }
    assert status["nextn.shared_head_head"] == {
        "status": "fallback",
        "tensor_name": None,
        "fallback_slot": "lm_head",
        "fallback_tensor_name": "output.weight",
        "fallback_shape": [11, 8],
        "fallback_type": "F32",
        "fallback_nbytes": 352,
    }
    assert status["nextn.shared_head_norm"] == {
        "status": "present",
        "tensor_name": "blk.2.nextn.shared_head_norm.weight",
        "shape": [8],
        "type": "F32",
        "nbytes": 32,
        "fallback_slot": None,
        "fallback_tensor_name": None,
    }

    tensors_by_name = {tensor["name"]: tensor for tensor in block["tensors"]}
    assert len(tensors_by_name) == 20
    assert tensors_by_name["blk.2.nextn.eh_proj.weight"]["shape"] == [8, 16]
    assert tensors_by_name["blk.2.attn_q.weight"]["shape"] == [16, 8]
    assert tensors_by_name["blk.2.ffn_down_exps.weight"]["shape"] == [3, 8, 5]


def test_committed_mtp_inventory_fixture_matches_local_ud_q4_k_m_contract() -> None:
    fixture = json.loads(FIXTURE.read_text())

    assert fixture["schema"] == 1
    assert fixture["source_model"].endswith("/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
    assert fixture["architecture"] == "qwen35moe"
    assert fixture["file_type_name"] == "MOSTLY_Q4_K_M"
    assert fixture["tensor_count"] == 753
    inventory = fixture["qwen35_mtp_inventory"]
    assert inventory["declared_block_count"] == 41
    assert inventory["ar_block_count"] == 40
    assert inventory["ignored_block_ids"] == [40]
    (block,) = inventory["blocks"]
    assert block["layer_id"] == 40
    assert block["tensor_count"] == 20
    assert block["nextn_tensor_count"] == 4
    assert block["required_missing"] == []
    assert block["optional_missing"] == [
        "blk.40.nextn.embed_tokens.weight",
        "blk.40.nextn.shared_head_head.weight",
    ]
    assert block["unexpected_tensor_names"] == []

    status = block["nextn_optional_status"]
    assert status["nextn.embed_tokens"]["status"] == "fallback"
    assert status["nextn.embed_tokens"]["fallback_tensor_name"] == "token_embd.weight"
    assert status["nextn.embed_tokens"]["fallback_type"] == "Q8_0"
    assert status["nextn.shared_head_head"]["status"] == "fallback"
    assert status["nextn.shared_head_head"]["fallback_tensor_name"] == "output.weight"
    assert status["nextn.shared_head_head"]["fallback_type"] == "Q6_K"
    assert status["nextn.shared_head_norm"] == {
        "fallback_slot": None,
        "fallback_tensor_name": None,
        "nbytes": 8192,
        "shape": [2048],
        "status": "present",
        "tensor_name": "blk.40.nextn.shared_head_norm.weight",
        "type": "F32",
    }

    tensors_by_name = {tensor["name"]: tensor for tensor in block["tensors"]}
    expected_qtypes = {
        "blk.40.nextn.eh_proj.weight": "Q8_0",
        "blk.40.nextn.enorm.weight": "F32",
        "blk.40.nextn.hnorm.weight": "F32",
        "blk.40.nextn.shared_head_norm.weight": "F32",
        "blk.40.attn_q.weight": "Q8_0",
        "blk.40.attn_k.weight": "Q8_0",
        "blk.40.attn_v.weight": "Q8_0",
        "blk.40.attn_output.weight": "Q8_0",
        "blk.40.ffn_gate_inp.weight": "BF16",
        "blk.40.ffn_gate_exps.weight": "Q4_K",
        "blk.40.ffn_up_exps.weight": "Q4_K",
        "blk.40.ffn_down_exps.weight": "Q5_K",
    }
    assert set(expected_qtypes).issubset(tensors_by_name)
    for name, qtype in expected_qtypes.items():
        assert tensors_by_name[name]["type"] == qtype
