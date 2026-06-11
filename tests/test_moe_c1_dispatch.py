from __future__ import annotations

import hipengine.runtime.moe_c1_dispatch as moe_c1_dispatch


def test_moe_c1_dispatch_split_output_tiled_prefill_default_and_optout(monkeypatch) -> None:
    monkeypatch.delenv("HIPENGINE_W4_DUAL_OUTPUT_TILED_SPLIT_PREFILL", raising=False)

    assert moe_c1_dispatch._w4_dual_output_tiled_split_prefill_enabled()

    monkeypatch.setenv("HIPENGINE_W4_DUAL_OUTPUT_TILED_SPLIT_PREFILL", "0")

    assert not moe_c1_dispatch._w4_dual_output_tiled_split_prefill_enabled()
