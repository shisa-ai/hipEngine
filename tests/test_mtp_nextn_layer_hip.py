"""RED test for the native hip ``mtp_nextn_layer`` (full NextN draft-head forward).

M3 acceptance gate: a *real* GPU ``mtp_nextn_layer`` registered under
``KernelKey(backend, "mtp_nextn_layer", "w4_gguf", "qwen35_dense_logits")`` whose
logits match the ``cpu_reference`` oracle on the F32 M3 fixture within
KL<=0.05 and top-1 agreement >= 90%.

Why this does not assert ``MissingKernelError``:
  ``hipengine/kernels/registry.py`` ``_candidate_keys`` appends ``cpu_reference``
  as a last-resort fallback for any non-cpu backend.  So ``resolve(hip_gfx1100,
  mtp_nextn_layer, ...)`` *never* raises ``MissingKernelError`` -- it silently
  returns the numpy oracle, which is exactly the drift that left M3 with no
  native runtime kernel.  The real RED condition here is therefore:

    1. the exact hip key is NOT in ``registered_keys()`` (a native kernel is
       missing), and
    2. ``resolve()`` returns the ``cpu_reference`` oracle via fallback (so any
       gate comparison would be a tautology, not a GPU run).

  Once a native kernel lands, the key appears and ``resolve`` returns the GPU
  kernel; the KL/top-1 gate then becomes a real correctness check.  The
  eh_proj sub-kernel already follows this pattern (see
  ``test_mtp_nextn_eh_proj_hip.py``).
"""

from __future__ import annotations

import ctypes
import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "benchmarks/fixtures/qwen35_gguf_mtp_nextn_cpu_reference_fixture.json"

# Gate thresholds mirror scripts/gguf_mtp_oracle_gate.py defaults.
MAX_KL = 0.05
MIN_TOP1 = 0.90


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _hip_available(), reason="ROCm/HIP not available")

# Import at module scope (collection time) so hip registrations are captured in
# tests/conftest.py's baseline snapshot (it clears _KERNELS and restores the
# collection-time snapshot after each test).  Importing is pure-Python.
import hipengine.kernels.cpu_reference  # noqa: F401,E402 - registers cpu oracle
import hipengine.kernels.hip_gfx1151  # noqa: F401,E402 - registers hip aliases incl. mtp_nextn


@pytest.fixture(scope="module")
def fixture() -> dict:
    return json.loads(FIXTURE.read_text())


def _f32(value) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    eps = np.finfo(np.float32).tiny
    p_safe = np.clip(p, eps, 1.0)
    q_safe = np.clip(q, eps, 1.0)
    return np.sum(p_safe * (np.log(p_safe) - np.log(q_safe)), axis=-1)


def _exact_key_registered(backend: str) -> bool:
    from hipengine.kernels.registry import KernelKey, registered_keys

    target = KernelKey(backend, "mtp_nextn_layer", "w4_gguf", "qwen35_dense_logits")
    return target in registered_keys()


def _native_layer_kernel(backend: str):
    """Resolve the mtp_nextn_layer for ``backend``; return None if only the
    cpu_reference fallback is available (i.e. no native kernel registered)."""
    from hipengine.kernels.cpu_reference.ops import qwen35_gguf_mtp_nextn_layer_logits
    from hipengine.kernels.registry import resolve

    kernel = resolve(
        backend=backend, layer="mtp_nextn_layer", quant="w4_gguf",
        variant="qwen35_dense_logits",
    )
    if kernel is qwen35_gguf_mtp_nextn_layer_logits:
        # Fallback masquerade: the exact hip key is missing and the resolver
        # handed back the cpu_reference numpy oracle.
        return None
    return kernel


def _run_layer(kernel, fixture: dict) -> np.ndarray:
    inputs = fixture["inputs"]
    from hipengine.quant.gguf import GGMLQuantizationType

    logits = kernel(
        _f32(inputs["hidden_seed"]),
        _f32(inputs["token_embedding"]),
        _f32(inputs["eh_proj_weight"]),
        _f32(inputs["hnorm_weight"]),
        _f32(inputs["enorm_weight"]),
        _f32(inputs["attn_norm_weight"]),
        _f32(inputs["wq_weight"]),
        _f32(inputs["wk_weight"]),
        _f32(inputs["wv_weight"]),
        _f32(inputs["wo_weight"]),
        _f32(inputs["q_norm_weight"]),
        _f32(inputs["k_norm_weight"]),
        _f32(inputs["attn_post_norm_weight"]),
        _f32(inputs["router_weight"]),
        _f32(inputs["gate_qweight"]),
        _f32(inputs["up_qweight"]),
        _f32(inputs["down_qweight"]),
        GGMLQuantizationType[str(inputs["gate_qtype"])],
        GGMLQuantizationType[str(inputs["up_qtype"])],
        GGMLQuantizationType[str(inputs["down_qtype"])],
        _f32(inputs["shared_gate_logit_weight"]),
        _f32(inputs["shared_gate_qweight"]),
        _f32(inputs["shared_up_qweight"]),
        _f32(inputs["shared_down_qweight"]),
        GGMLQuantizationType[str(inputs["shared_qtype"])],
        _f32(inputs["shared_head_norm_weight"]),
        _f32(inputs["shared_head_weight"]),
        **dict(fixture["kwargs"]),
    )
    return np.asarray(logits, dtype=np.float32)


@pytest.mark.parametrize("backend", ["hip_gfx1100", "hip_gfx1151"])
def test_hip_mtp_nextn_layer_key_registered(backend):
    """RED gate: a real native hip mtp_nextn_layer must be registered, not the
    cpu_reference fallback.  This is the M3 native-runtime-key check."""
    assert _exact_key_registered(backend), (
        f"no {backend} mtp_nextn_layer kernel registered "
        f"(KernelKey({backend!r}, 'mtp_nextn_layer', 'w4_gguf', 'qwen35_dense_logits')); "
        "M3 native GPU kernel missing -- registry falls back to cpu_reference"
    )


@pytest.mark.parametrize("backend", ["hip_gfx1100", "hip_gfx1151"])
def test_hip_mtp_nextn_layer_matches_cpu_reference_gate(backend, fixture):
    """KL<=0.05 and top-1>=90% vs cpu_reference, on a *native* hip kernel.

    Skipped (not failed) while RED: until the exact hip key exists, resolve()
    returns the cpu_reference oracle and the comparison would be a tautology.
    """
    if not _exact_key_registered(backend):
        pytest.skip(f"{backend} mtp_nextn_layer not registered yet (RED)")

    kernel = _native_layer_kernel(backend)
    assert kernel is not None, (
        f"resolve({backend}) returned the cpu_reference oracle via fallback; "
        "native hip kernel not registered"
    )

    from hipengine.kernels.cpu_reference.ops import (
        qwen35_gguf_mtp_nextn_layer_logits as cpu_oracle,
    )

    expected = _run_layer(cpu_oracle, fixture)
    actual = _run_layer(kernel, fixture)

    assert actual.shape == expected.shape, f"shape {actual.shape} != {expected.shape}"

    kl_values = _kl_divergence(_softmax(expected), _softmax(actual))
    max_kl = float(np.max(kl_values)) if kl_values.size else 0.0

    expected_top1 = np.argmax(expected, axis=-1)
    actual_top1 = np.argmax(actual, axis=-1)
    top1_agreement = float(np.mean(actual_top1 == expected_top1))

    assert max_kl <= MAX_KL, (
        f"{backend} mtp_nextn_layer max_kl={max_kl} exceeds {MAX_KL} vs cpu_reference"
    )
    assert top1_agreement >= MIN_TOP1, (
        f"{backend} mtp_nextn_layer top1_agreement={top1_agreement} "
        f"below {MIN_TOP1} vs cpu_reference"
    )
