from __future__ import annotations

import os

from hipengine.execution_profiles import ExecutionProfile, resolve_runtime_profile
from hipengine.generation import register_builtin_generators
from hipengine.generation.qwen38_gguf_profiles import (
    FP16_RECURRENT_STATE_ENV,
    QWEN38_GGUF_BACKEND,
    QWEN38_GGUF_MODEL,
    QWEN38_GGUF_QUANT,
    VERIFY_CAPTURE_PREFILL_GDN_ENV,
    qwen38_gguf_gfx1151_strict_registered,
)


def test_qwen38_strict_profile_resolves_and_disables_fp16_state() -> None:
    register_builtin_generators()
    assert qwen38_gguf_gfx1151_strict_registered()

    resolved = resolve_runtime_profile(
        model=QWEN38_GGUF_MODEL,
        backend=QWEN38_GGUF_BACKEND,
        quant=QWEN38_GGUF_QUANT,
        profile=ExecutionProfile.STRICT,
    )
    generator = object.__new__(type("Generator", (), {}))
    assert resolved.binder is not None
    resolved.binder(generator, resolved)

    assert os.environ[FP16_RECURRENT_STATE_ENV] == "0"
    assert os.environ[VERIFY_CAPTURE_PREFILL_GDN_ENV] == "1"
    assert resolved.profile is ExecutionProfile.STRICT
    assert resolved.manifest_sha256 == resolved.strict_manifest_sha256
    assert resolved.manifest["graph_policy"] == "specdec2_eager_c1"
    assert resolved.manifest["kv_policy"] == "paged_bf16"
