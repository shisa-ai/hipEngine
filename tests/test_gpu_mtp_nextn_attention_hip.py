"""RED/GREEN test for the native hip ``mtp_nextn_attention`` sub-kernel.

M3 deliverable: a *real* GPU NextN attention sublayer registered under
``KernelKey(backend, "mtp_nextn_attention", "gguf_f32", "qwen35_dense")``.
Without it the registry falls back to the ``cpu_reference`` numpy oracle
(``registry._candidate_keys`` appends ``cpu_reference`` last), so this test
asserts the exact hip key is registered -- otherwise it silently passes on the
numpy fallback (the drift that left M3 with no native runtime kernel).

Scope (M3, correctness-first): the F32 M3 fixture exercises the DEFAULT dense
attention path -- positions=arange(tokens), context_counts=pos+1, no RoPE, dense
cache = the current token's K/V (single token self-attending). The RoPE and
KVLiveSpans paged-cache branches are M6 work and are not asserted here; this
test gates the default-path math vs ``cpu_reference`` exactly.
"""

from __future__ import annotations

import ctypes
import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "benchmarks/fixtures/qwen35_gguf_mtp_nextn_cpu_reference_fixture.json"


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _hip_available(), reason="ROCm/HIP not available")

import hipengine.kernels.cpu_reference  # noqa: F401,E402 - registers cpu oracle
import hipengine.kernels.hip_gfx1151  # noqa: F401,E402 - registers hip aliases incl. mtp_nextn


def _f32(value) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=np.float32)


def _exact_key_registered(backend: str) -> bool:
    from hipengine.kernels.registry import KernelKey, registered_keys

    target = KernelKey(backend, "mtp_nextn_attention", "gguf_f32", "qwen35_dense")
    return target in registered_keys()


@pytest.fixture(scope="module")
def attn_inputs() -> dict:
    fixture = json.loads(FIXTURE.read_text())
    inputs = fixture["inputs"]
    # The attention sublayer takes the *projected* hidden (eh_proj output) as
    # its ``hidden`` input.  Recompute it from the cpu_reference eh_proj so the
    # attention kernel is gated on its own math, not eh_proj's.
    from hipengine.kernels.cpu_reference.ops import qwen35_gguf_mtp_eh_proj

    projected = qwen35_gguf_mtp_eh_proj(
        _f32(inputs["hidden_seed"]),
        _f32(inputs["token_embedding"]),
        _f32(inputs["eh_proj_weight"]),
        _f32(inputs["hnorm_weight"]),
        _f32(inputs["enorm_weight"]),
        eps=float(fixture["kwargs"].get("eps", 1e-6)),
    )
    return {
        "hidden": np.ascontiguousarray(projected, dtype=np.float32),
        "attn_norm_weight": _f32(inputs["attn_norm_weight"]),
        "wq_weight": _f32(inputs["wq_weight"]),
        "wk_weight": _f32(inputs["wk_weight"]),
        "wv_weight": _f32(inputs["wv_weight"]),
        "wo_weight": _f32(inputs["wo_weight"]),
        "q_norm_weight": _f32(inputs["q_norm_weight"]),
        "k_norm_weight": _f32(inputs["k_norm_weight"]),
        "num_heads": int(fixture["kwargs"]["num_heads"]),
        "num_kv_heads": int(fixture["kwargs"]["num_kv_heads"]),
        "eps": float(fixture["kwargs"].get("eps", 1e-6)),
    }


def test_hip_attention_key_registered():
    """RED gate: a real hip mtp_nextn_attention kernel must be registered, not
    the cpu_reference fallback."""
    assert _exact_key_registered("hip_gfx1100"), (
        "no hip_gfx1100 mtp_nextn_attention kernel registered (M3 native kernel missing; "
        "registry is falling back to cpu_reference)"
    )
    assert _exact_key_registered("hip_gfx1151"), (
        "no hip_gfx1151 mtp_nextn_attention alias registered"
    )


@pytest.mark.parametrize("backend", ["hip_gfx1100", "hip_gfx1151"])
def test_hip_attention_matches_cpu_reference(backend, attn_inputs):
    if not _exact_key_registered(backend):
        pytest.skip(f"{backend} mtp_nextn_attention not registered yet (RED)")

    from hipengine.kernels.cpu_reference.ops import (
        qwen35_gguf_mtp_attention_sublayer as cpu_oracle,
    )
    from hipengine.kernels.registry import resolve

    expected = cpu_oracle(
        attn_inputs["hidden"],
        attn_inputs["attn_norm_weight"],
        attn_inputs["wq_weight"],
        attn_inputs["wk_weight"],
        attn_inputs["wv_weight"],
        attn_inputs["wo_weight"],
        attn_inputs["q_norm_weight"],
        attn_inputs["k_norm_weight"],
        num_heads=attn_inputs["num_heads"],
        num_kv_heads=attn_inputs["num_kv_heads"],
        eps=attn_inputs["eps"],
    )

    kernel = resolve(
        backend=backend, layer="mtp_nextn_attention", quant="gguf_f32",
        variant="qwen35_dense",
    )
    assert kernel is not cpu_oracle, (
        f"resolve({backend}) returned the cpu_reference oracle via fallback; "
        "native hip kernel not registered"
    )

    actual = kernel(
        attn_inputs["hidden"],
        attn_inputs["attn_norm_weight"],
        attn_inputs["wq_weight"],
        attn_inputs["wk_weight"],
        attn_inputs["wv_weight"],
        attn_inputs["wo_weight"],
        attn_inputs["q_norm_weight"],
        attn_inputs["k_norm_weight"],
        num_heads=attn_inputs["num_heads"],
        num_kv_heads=attn_inputs["num_kv_heads"],
        eps=attn_inputs["eps"],
    )
    actual = np.asarray(actual, dtype=np.float32)

    assert actual.shape == expected.shape, f"shape {actual.shape} != {expected.shape}"
    max_abs = float(np.max(np.abs(actual - expected)))
    assert max_abs < 1e-3, (
        f"{backend} mtp_nextn_attention max_abs={max_abs} exceeds 1e-3 vs cpu_reference"
    )
