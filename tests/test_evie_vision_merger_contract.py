"""CPU dispatch contract: merger width is independent of vision MLP width."""

from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.memory import DeviceBuffer
from hipengine.runtime import evie


@pytest.mark.parametrize("vh,mlp_width,out_width", [(1024, 4096, 2048), (1152, 4304, 4096)])
def test_vision_merger_matches_checkpoint_geometry(monkeypatch, vh, mlp_width, out_width):
    # Checkpoint merger fc1 is (4*vh, 4*vh), fc2 is (out_width, 4*vh).
    # 8B's block MLP width 4304 must not replace its merger width 4608.
    # Exercise real host orchestration without HIP, weights, or transformer
    # blocks; fake launches record the geometry submitted to device kernels.
    runner = evie.EvieRunner.__new__(evie.EvieRunner)
    runner.spec = SimpleNamespace(
        vision_spatial_merge_size=2, vision_hidden_size=vh,
        vision_depth=0, vision_intermediate_size=mlp_width,
        vision_out_hidden_size=out_width,
    )
    runner.VISION_HEADS = 16
    runner.VISION_HEAD_DIM = vh // 16
    runner._misc_buffers = []
    runner._w = defaultdict(lambda: 1)
    runner._w["visual.merger.linear_fc1.weight"] = 101
    runner._w["visual.merger.linear_fc1.bias"] = 102
    runner._w["visual.merger.linear_fc2.weight"] = 103
    runner._to_dev = lambda array: DeviceBuffer(200, array.nbytes)
    runner._vision_pos_embed_host = lambda grid: np.zeros((16, vh), np.float32)
    runner._add = lambda *args: None
    gemms, launches = [], []
    runner._gemm = lambda *args: gemms.append(args)

    def bind(symbol, types):
        def launch(*args):
            launches.append((symbol, tuple(arg.value for arg in args)))
            return 0
        return launch

    runner._k = bind
    monkeypatch.setattr(evie, "_malloc_committed", lambda size: DeviceBuffer(300, size))
    scratch = SimpleNamespace(buffers=defaultdict(lambda: DeviceBuffer(400, 1 << 24)))
    runner.vision_forward(
        np.zeros((16, 1536), np.float32), np.array([[1, 4, 4]]), scratch,
    )
    merger_width = 4 * vh
    fc1 = next(call for call in gemms if call[1] == 101)
    fc2 = next(call for call in gemms if call[1] == 103)
    assert fc1[3:] == (4, merger_width, merger_width)
    assert fc2[3:] == (4, merger_width, out_width)
    bias = next(args for name, args in launches
                if name == "hipengine_evie_add_bias_f32" and args[1] == 102)
    assert bias[2:4] == (4 * merger_width, merger_width)
    gelu = next(args for name, args in launches if name == "hipengine_evie_gelu_erf_f32")
    assert gelu[2] == 4 * merger_width
