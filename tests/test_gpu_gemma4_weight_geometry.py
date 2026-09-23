"""GPU numerical coverage for unequal-width and untied Gemma GGUF artifacts."""

from dataclasses import replace

import numpy as np
import pytest

from tests._rocm_guard import hip_runtime_available
from tests._gemma4_gguf_fixture import (
    FIXTURE_HIDDEN, FIXTURE_VOCAB, default_fixture_tensors,
    fixture_metadata, write_fixture_gguf,
)

pytestmark = pytest.mark.skipif(not hip_runtime_available(), reason="HIP runtime unavailable")


@pytest.mark.parametrize("untied", [False, True])
def test_unequal_dense_widths_match_the_cpu_reference(tmp_path, untied):
    from hipengine.kernels.cpu_reference.gemma4 import gemma4_text_forward
    from hipengine.loading.gguf import GGUFReader
    from hipengine.loading.gemma4_gguf_materialize import materialize_gemma4_reference_weights
    from hipengine.quant.gguf import GGMLQuantizationType
    from hipengine.runtime.gemma4 import Gemma4Runner, load_gemma4_device_weights

    widths = (192, 320)
    tensors = []
    for name, shape, qtype in default_fixture_tensors():
        if name.endswith((".ffn_gate.weight", ".ffn_up.weight")):
            shape = (widths[int(name.split(".")[1])], shape[1])
        elif name.endswith(".ffn_down.weight"):
            shape = (shape[0], widths[int(name.split(".")[1])])
        tensors.append((name, shape, qtype))
    if untied:
        tensors.append(("output.weight", (FIXTURE_VOCAB, FIXTURE_HIDDEN), GGMLQuantizationType.Q8_0))
    metadata = [
        (key, 9, (4, list(widths))) if key == "gemma4.feed_forward_length" else (key, kind, value)
        for key, kind, value in fixture_metadata()
    ]
    reader = GGUFReader(write_fixture_gguf(tmp_path / "geometry.gguf", tensors, metadata))
    reference = materialize_gemma4_reference_weights(reader)
    config = replace(reference.config, final_logit_softcapping=None)
    tokens = [1, 5, 9, 13]
    expected = gemma4_text_forward(reference.weights, config, tokens).logits[-1]
    weights = load_gemma4_device_weights(reader)
    try:
        runner = Gemma4Runner(weights, capacity=32, max_block=3)
        try:
            got = runner.forward(tokens, apply_softcap=False)
            assert tuple(s.dense_intermediate for s in runner._scratches) == widths
            assert (weights.lm_head is not None) == untied
            assert np.isfinite(got).all()
            scale = float(np.abs(expected).max())
            assert scale > 0
            # Existing two-layer BF16-vs-F32 runner oracle envelope.
            np.testing.assert_allclose(got, expected, rtol=0.05, atol=0.05 * scale)
        finally:
            runner.close()
    finally:
        weights.free()
