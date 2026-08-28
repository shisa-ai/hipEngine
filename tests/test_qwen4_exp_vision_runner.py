from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

from hipengine.loading.gguf import GGUFReader
from hipengine.loading.qwen4_exp_vision_gguf import build_qwen4_exp_vision_gguf_map
from hipengine.loading.qwen4_exp_vision_materialize import materialize_qwen4_exp_vision_weights,plan_qwen4_exp_vision_residency
from hipengine.runtime.qwen4_exp_vision import Qwen4ExpVisionRunner

_MMPROJ=Path('/models/gguf/Qwen3.8-Flash-Next-mmproj/Qwen3.8-Flash-Next-BF16.gguf')

def _hip_available():
    try:ctypes.CDLL('libamdhip64.so')
    except OSError:return False
    return True

@pytest.mark.skipif(not _hip_available(),reason='HIP unavailable')
@pytest.mark.skipif(not _MMPROJ.exists(),reason='local mmproj unavailable')
def test_real_qwen4_exp_vision_runner_is_finite_deterministic_and_image_sensitive():
    from hipengine.core.hip import get_hip_runtime
    rt=get_hip_runtime();reader=GGUFReader(_MMPROJ);m=build_qwen4_exp_vision_gguf_map((reader.info,));plan=plan_qwen4_exp_vision_residency(m);resident=materialize_qwen4_exp_vision_weights((reader,),plan=plan,backend='hip_gfx1151',runtime=rt);runner=None
    try:
        runner=Qwen4ExpVisionRunner(resident,patch_weight0=reader.tensor_data('v.patch_embd.weight'),patch_weight1=reader.tensor_data('v.patch_embd.weight.1'),patch_bias=reader.tensor_data('v.patch_embd.bias'),position_embedding=reader.tensor_data('v.position_embd.weight'))
        black=np.zeros((32,32,3),np.uint8);pattern=np.zeros_like(black);pattern[:16,:16,0]=255;pattern[16:,16:,1]=255
        first=runner.encode(black);repeat=runner.encode(black);other=runner.encode(pattern)
    finally:
        if runner is not None:runner.close()
        resident.close()
    assert first.shape==(1,2560);assert np.isfinite(first).all();np.testing.assert_array_equal(first,repeat);assert not np.array_equal(first,other)
