"""Host-only pin: the Qwen3.8-27B GGUF's NextN draft block maps cleanly.

The artifact declares ``qwen35.nextn_predict_layers = 1`` and names its draft
block ``blk.64`` (not ``nextn.*``). Guards the Packet 5 MTP-route prerequisite
from worklog entries 76aed6 / 1b6b62: the in-tree NextN map builder must keep
consuming this naming.
"""

from __future__ import annotations

from pathlib import Path

import pytest

MODEL = Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason=f"missing {MODEL}")


def test_qwen38_nextn_draft_block_maps_from_blk64_naming() -> None:
    from hipengine.loading.gguf import scan_gguf
    from hipengine.loading.qwen35_gguf_nextn import build_qwen35_gguf_nextn_tensor_map

    info = scan_gguf(MODEL)
    assert info.metadata.get("qwen35.nextn_predict_layers") == 1

    nextn_map = build_qwen35_gguf_nextn_tensor_map(info, strict=False)
    assert nextn_map.block_id == 64
    assert set(nextn_map.nextn_tensors.keys()) >= {
        "eh_proj", "enorm", "hnorm", "shared_head_norm",
    }
    assert {
        "attn_q", "attn_k", "attn_v", "attn_output",
        "ffn_gate", "ffn_up", "ffn_down",
        "attn_norm", "post_attention_norm",
    } <= set(nextn_map.layer_tensors.keys())
