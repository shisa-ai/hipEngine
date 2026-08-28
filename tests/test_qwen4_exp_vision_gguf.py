from __future__ import annotations

from pathlib import Path

import pytest

from hipengine.loading.gguf import GGUFReader
from hipengine.loading.qwen4_exp_vision_gguf import build_qwen4_exp_vision_gguf_map
from hipengine.loading.qwen4_exp_vision_materialize import plan_qwen4_exp_vision_residency

_MMPROJ = Path('/models/gguf/Qwen3.8-Flash-Next-mmproj/Qwen3.8-Flash-Next-BF16.gguf')


@pytest.mark.skipif(not _MMPROJ.exists(), reason='local Qwen3.8 mmproj is unavailable')
def test_real_qwen4_exp_vision_gguf_matches_frozen_contract() -> None:
    reader = GGUFReader(_MMPROJ)
    model_map = build_qwen4_exp_vision_gguf_map((reader.info,))
    plan = plan_qwen4_exp_vision_residency(model_map)

    assert model_map.validation.passed
    assert len(model_map.tensor_refs) == 334
    assert model_map.config.block_count == 27
    assert model_map.config.head_dim == 72
    assert model_map.weight('patch.weight0').tensor.shape == (1152, 3, 16, 16)
    assert model_map.weight('layers.26.ffn_up.weight').tensor.shape == (4304, 1152)
    assert model_map.weight('merge.fc2.weight').tensor.shape == (2560, 4608)
    assert plan.device_weight_bytes == 907_523_008
    assert len(plan.specs) == 334
