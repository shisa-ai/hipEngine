"""The dense Q8_0 projections of an Inkling prompt block through the wide F16 tiles.

`fast` projects a block of enough rows through `dense_wide64x128_f32_f32_out` and `production` through
the tile that adds a residual F16 part to each operand. Narrow blocks, narrow outputs and `strict`
keep the exact coltile kernel. The last test needs the GGUF of `HIPENGINE_INKLING2_MINI_GGUF` and
compares each tile with the exact kernel on the weights of the model.
"""

from __future__ import annotations

import ctypes
import os
from contextlib import closing
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from hipengine import LLM
from hipengine.core.hip import get_hip_runtime
from hipengine.core.rocblas import Rocblas
from hipengine.kernels.backends import resolve_backend
from hipengine.loading.inkling2_gguf import Inkling2GGUFWeights
from hipengine.generation.inkling2_gguf import DENSE_WIDE, DENSE_WIDE_RESIDUAL
from hipengine.loading.inkling2_gguf_device import (
    Q,
    _WIDE_DENSE_MIN_ROWS,
    _WIDE_DENSE_NARROW_MIN_ROWS,
    _WIDE_DENSE_WIDE_OUTPUTS,
    Inkling2GGUFDeviceProjections,
)


from tests.test_unit_inkling2_gguf import _file

# The rows each tile starts at: 1,024 or more outputs, then fewer.
STARTS = {DENSE_WIDE: (64, 384), DENSE_WIDE_RESIDUAL: (192, 512)}


def _backend():
    try:
        ctypes.CDLL("libamdhip64.so")
        ctypes.CDLL("librocblas.so")
    except OSError as exc:
        pytest.skip(f"HIP or rocBLAS unavailable: {exc}")
    backend = resolve_backend("auto", execution_device=True)
    if backend not in ("hip_gfx1100", "hip_gfx1151"):
        pytest.skip("test requires gfx1100 or gfx1151")
    return backend


def _route(owner, rows, outputs, inputs=256):
    """The launch a Q8_0 projection of this shape takes, with every launch replaced by a mock."""
    launches = {"wide": Mock(), "exact": Mock(), "other": Mock()}
    saved = owner._launch_wide_dense, owner._q8_prefill, owner._packed_launch
    owner._launch_wide_dense, owner._q8_prefill = launches["wide"], launches["exact"]
    owner._packed_launch = lambda quant: launches["other"]
    try:
        owner._launch_projection(0, 0, rows, 2, (Mock(ptr=1), (outputs, inputs), Q.Q8_0), None)
    finally:
        owner._launch_wide_dense, owner._q8_prefill, owner._packed_launch = saved
    taken = [name for name, launch in launches.items() if launch.called]
    assert len(taken) == 1, taken
    return taken[0]


@pytest.mark.parametrize("tile", [DENSE_WIDE, DENSE_WIDE_RESIDUAL])
@pytest.mark.parametrize("legacy", [False, True])
def test_prompt_rows_take_the_wide_tile_where_it_is_faster(legacy, tile):
    backend = _backend()
    blas = Rocblas.load()
    options = dict(runtime=get_hip_runtime(), blas=blas, backend=backend)
    with (Inkling2GGUFDeviceProjections(Inkling2GGUFWeights(Mock(info=_file(legacy))), **options) as plain,
          Inkling2GGUFDeviceProjections(Inkling2GGUFWeights(Mock(info=_file(legacy))), wide_dense=tile,
                                        **options) as wide):
        assert plain.wide_dense is None and plain._q8_wide is None
        assert wide._q8_wide is not None
        assert (_WIDE_DENSE_MIN_ROWS[DENSE_WIDE], _WIDE_DENSE_NARROW_MIN_ROWS[DENSE_WIDE]) == STARTS[DENSE_WIDE]
        assert (_WIDE_DENSE_MIN_ROWS[DENSE_WIDE_RESIDUAL],
                _WIDE_DENSE_NARROW_MIN_ROWS[DENSE_WIDE_RESIDUAL]) == STARTS[DENSE_WIDE_RESIDUAL]
        wide_outputs, narrow_outputs = _WIDE_DENSE_WIDE_OUTPUTS, _WIDE_DENSE_WIDE_OUTPUTS - 8
        for outputs, start in ((wide_outputs, STARTS[tile][0]), (narrow_outputs, STARTS[tile][1])):
            assert _route(wide, start, outputs) == "wide"
            assert _route(wide, 1024, outputs) == "wide"
            assert _route(wide, start - 1, outputs) == "exact"
            assert _route(wide, 4, outputs) == "exact"
        # An input width the tile cannot stage stays on the exact kernel.
        assert _route(wide, 1024, wide_outputs, inputs=200) == "exact"
        assert _route(plain, 1024, wide_outputs) == "exact"
    with pytest.raises(ValueError, match="unknown Inkling wide dense tile"):
        Inkling2GGUFDeviceProjections(Inkling2GGUFWeights(Mock(info=_file(legacy))),
                                      wide_dense="dense_wide256_f32_f32_out", **options)


def _skip_unless_runnable() -> str:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        pytest.skip("ROCm runtime is not available")
    model = os.environ.get("HIPENGINE_INKLING2_MINI_GGUF", "")
    if not model or not Path(model).is_file():
        pytest.skip("HIPENGINE_INKLING2_MINI_GGUF must point at the Inkling 2 GGUF model")
    return model


@pytest.mark.parametrize(("profile", "tile", "tolerance"),
                         [("fast", DENSE_WIDE, 2e-3), ("production", DENSE_WIDE_RESIDUAL, 1e-4)])
def test_the_model_weights_project_within_the_f16_envelope_of_the_exact_tile(profile, tile, tolerance):
    model = _skip_unless_runnable()
    with closing(LLM(model=model, backend="hip_gfx1100", execution_profile=profile, max_sequence_length=512,
                     prefix_cache="off", speculative_mtp_serving="off")) as llm:
        text = llm._get_text_generator().model
        projection = text.projection
        layer = text.mapping.layers[0]
        assert projection.wide_dense == tile and projection._q8_wide is not None
        wide_launch = projection._q8_wide
        rng = np.random.default_rng(5)
        for name, rows, taken in (("attn_q.weight", 256, True), ("attn_output.weight", 256, True),
                                  ("attn_rel.weight", 512, True), ("attn_q.weight", 32, False),
                                  ("attn_rel.weight", 256, False)):
            tensor = layer.tensors[name]
            assert Q(tensor.ggml_type) is Q.Q8_0
            hidden = rng.normal(0, 0.1, (rows, tensor.shape[1])).astype(np.float32)
            wide = projection.project(tensor.name, hidden)
            projection._q8_wide = None
            try:
                exact = projection.project(tensor.name, hidden)
            finally:
                projection._q8_wide = wide_launch
            if not taken:
                # Below the rows the tile starts at, both owners run the exact kernel.
                np.testing.assert_array_equal(wide, exact)
                continue
            # Of the RMS on these shapes: 2.9e-4 for the single-product tile, 7e-6 for the residual
            # one, 1.1e-7 for the exact kernel.
            np.testing.assert_allclose(wide, exact, rtol=tolerance, atol=tolerance * np.abs(exact).max())
