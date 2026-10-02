"""Live public-surface gate for the Gemma 4 INT8 KV storage request.

The unit and GPU route tests prove the owner, the writer, the consumer and the
layer wiring. None of them reaches the surface a user does. This test runs one
real ``hipengine.LLM.generate()`` request with ``kv_storage`` set to
``int8_per_token_head`` and confirms the intended path actually ran -- the
runner holds the INT8 cache and the layer selected the registered writer and
consumer -- rather than inferring it from finite text. It then runs a second
request on the default (BF16) storage and confirms the runner rebuilt to the
comparison path instead of silently reusing the INT8 cache.

**One test function on purpose.** ``tests/conftest.py`` restores the kernel
registry to a collection-time snapshot after every test, which drops kernels a
model load registered lazily. Keeping both storage modes in one function avoids
relying on that harness artefact either way.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import pytest

_MODEL_DIR = Path("/mnt/nvme1/models/gemma-4-26B-A4B-it-GGUF")
_TARGET = _MODEL_DIR / "gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _hip_available() or not _TARGET.exists(),
    reason="needs ROCm (libamdhip64.so) and the gemma4 fixture artifact",
)


def test_live_generate_with_int8_kv_storage_runs_the_int8_route():
    import hipengine
    from hipengine import SamplingParams
    from hipengine.kernels.hip_gfx1100.gemma4.gemma4_layer import last_int8_kv_route

    llm = hipengine.LLM(model=str(_TARGET))
    # The whole body is guarded: an assertion failure must still release the
    # 17 GB of resident weights and the runner, or a failing run poisons the
    # device for every later test in the session.
    try:
        params = SamplingParams(
            max_tokens=4,
            temperature=0.0,
            top_p=1.0,
            ignore_eos=True,
            kv_storage="int8_per_token_head",
            kv_scale_dtype="fp16",
            kv_scale_granularity="per_token_head",
        )
        outputs = llm.generate("The capital of France is", params)
        assert outputs and outputs[0], "the INT8 KV request produced no text"

        generator = llm._get_text_generator()
        # ``_get_text_generator`` returns the submit/poll adapter; the model
        # generator that owns the runner is its ``_inner``.
        model_generator = generator._inner
        runner = model_generator._runner
        assert runner is not None, "the generator built no runner"
        assert runner.uses_int8_kv is True, "the runner did not take the INT8 KV cache"
        assert runner.kv_cache is not None
        assert runner.kv_storage_resolved == "int8_per_token_head"
        int8_owner = runner.kv_cache

        route = last_int8_kv_route()
        assert route is not None, "the layer recorded no INT8 route"
        # The final request step is a decode; its writer and consumer must be the
        # registered per-token/head kernels, not a fallback.
        assert route["writer"] == "int8_per_token_head/per_token_head_bf16_spans", route
        assert route["consumer"] == (
            "paged_attn_decode/int8_per_token_head/gemma4_direct_spans"
        ), route

        # A second request on the default storage must rebuild the runner to BF16,
        # not keep serving the INT8 cache the caller no longer asked for.
        default_outputs = llm.generate(
            "The capital of France is",
            SamplingParams(max_tokens=4, temperature=0.0, top_p=1.0, ignore_eos=True),
        )
        assert default_outputs and default_outputs[0]
        rebuilt = model_generator._runner
        assert rebuilt is not None
        assert rebuilt is not runner, "the runner was not rebuilt for the new storage"
        assert rebuilt.kv_storage_resolved == "bf16"
        assert rebuilt.uses_int8_kv is False
        # The replaced runner's INT8 owner was released with it, not leaked.
        assert int8_owner.closed is True, "the replaced INT8 owner was not closed"
    finally:
        llm.close()

    # Public close releases the runner and the resident weights, so a later
    # load in the same process starts from an empty device.
    assert model_generator._runner is None
    assert model_generator._weights is None
