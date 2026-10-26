"""RED/GREEN test for the hip_gfx1100/hip_gfx1151 ``mtp_nextn_eh_proj`` kernel.

M3 deliverable: a *real* GPU NextN eh_proj sub-kernel (RMSNorm x2 + concat +
F32 GEMV) registered under ``KernelKey(backend, "mtp_nextn_eh_proj", "gguf_f32",
"qwen35")``.  The registry otherwise auto-falls-back to the ``cpu_reference``
numpy oracle (``registry._candidate_keys`` appends ``cpu_reference`` last), so
this test asserts the *exact* hip key is registered -- otherwise it would
silently pass on the numpy fallback, which is exactly the drift that left M3
with no native runtime kernel.

Correctness gate: output matches ``cpu_reference.qwen35_gguf_mtp_eh_proj`` on
the committed F32 MTP fixture within a tight f32 tolerance.

The Q8_0/Q6_K routes are covered separately with synthetic block weights over
rows 1, 2 and 3 against ``cpu_reference.gguf_quant_gemv`` over the same bytes,
i.e. the dequantized-weight oracle. Row 1 alone cannot catch a concat whose
layout is ``[2*rows, hidden]`` instead of ``[rows, 2*hidden]``, so the row
sweep is the point. The module caches uploaded weights by name only, so these
cases clear that cache around every test.
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

# Import at module scope (collection time) so the hip mtp_nextn registrations
# are captured in tests/conftest.py's baseline snapshot.  Importing the package
# is pure-Python (no GPU touch); registration happens at import.
import hipengine.kernels.cpu_reference  # noqa: F401,E402 - registers cpu oracle
import hipengine.kernels.hip_gfx1151  # noqa: F401,E402 - registers hip aliases incl. mtp_nextn

from hipengine.kernels.cpu_reference.ops import (  # noqa: E402
    gguf_quant_gemv,
    rmsnorm,
)
from hipengine.quant.gguf import GGMLQuantizationType  # noqa: E402
from tests._gguf_synthetic_weights import (  # noqa: E402
    make_q6_k_weight,
    make_q8_0_weight,
)

# 2 * HIDDEN == QK_K, so one shape serves both block formats: Q6_K needs input
# features divisible by 256 and Q8_0 by 32; the Q6_K pack8 ABI also needs
# out_features divisible by 8.
HIDDEN = 128
EPS = 1e-6
_Q8_0 = GGMLQuantizationType.Q8_0
_Q6_K = GGMLQuantizationType.Q6_K
_F32_TOL = dict(atol=5e-3, rtol=5e-3)


@pytest.fixture(autouse=True)
def _fresh_weight_cache():
    """The module caches weights by name, so quant/shape cases must not share."""

    from hipengine.kernels.hip_gfx1100.speculative.mtp_nextn import clear_weight_cache

    clear_weight_cache()
    yield
    clear_weight_cache()


def _inputs(seed: int, rows: int) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "hidden_seed": rng.normal(0.0, 0.5, (rows, HIDDEN)).astype(np.float32),
        "token_embedding": rng.normal(0.0, 0.5, (rows, HIDDEN)).astype(np.float32),
        "hnorm_weight": rng.normal(1.0, 0.1, HIDDEN).astype(np.float32),
        "enorm_weight": rng.normal(1.0, 0.1, HIDDEN).astype(np.float32),
    }


def _synthetic_weight(qtype) -> np.ndarray:
    if qtype is _Q6_K:
        return make_q6_k_weight(HIDDEN, 2 * HIDDEN)
    return make_q8_0_weight(HIDDEN, 2 * HIDDEN)


def _oracle(inputs: dict, weight: np.ndarray, qtype) -> np.ndarray:
    """Dequantized-weight oracle for the same bytes the kernel reads.

    The Q6_K route consumes fp16 activations, so the oracle rounds the concat
    to fp16 first; that rounding is part of the declared contract, not error.
    """

    e_norm = rmsnorm(inputs["token_embedding"], inputs["enorm_weight"], EPS)
    h_norm = rmsnorm(inputs["hidden_seed"], inputs["hnorm_weight"], EPS)
    fused = np.concatenate([e_norm, h_norm], axis=-1)
    if qtype is _Q6_K:
        fused = fused.astype(np.float16).astype(np.float32)
    return gguf_quant_gemv(fused, weight, qtype)


@pytest.fixture(scope="module")
def eh_proj_inputs() -> dict:
    fixture = json.loads(FIXTURE.read_text())
    inputs = fixture["inputs"]
    return {
        "hidden_seed": np.ascontiguousarray(inputs["hidden_seed"], dtype=np.float32),
        "token_embedding": np.ascontiguousarray(inputs["token_embedding"], dtype=np.float32),
        "eh_proj_weight": np.ascontiguousarray(inputs["eh_proj_weight"], dtype=np.float32),
        "hnorm_weight": np.ascontiguousarray(inputs["hnorm_weight"], dtype=np.float32),
        "enorm_weight": np.ascontiguousarray(inputs["enorm_weight"], dtype=np.float32),
        "eps": float(fixture["kwargs"].get("eps", 1e-6)),
    }


def _exact_key_registered(backend: str) -> bool:
    from hipengine.kernels.registry import KernelKey, registered_keys

    import hipengine.kernels.cpu_reference  # noqa: F401 - registers cpu oracle
    import hipengine.kernels.hip_gfx1151  # noqa: F401 - registers hip aliases

    target = KernelKey(backend, "mtp_nextn_eh_proj", "gguf_f32", "qwen35")
    return target in registered_keys()


def test_hip_eh_proj_key_registered():
    """RED gate: a real hip mtp_nextn_eh_proj kernel must be registered, not the
    cpu_reference fallback."""
    assert _exact_key_registered("hip_gfx1100"), (
        "no hip_gfx1100 mtp_nextn_eh_proj kernel registered (M3 native kernel missing; "
        "registry is falling back to cpu_reference)"
    )
    assert _exact_key_registered("hip_gfx1151"), (
        "no hip_gfx1151 mtp_nextn_eh_proj alias registered"
    )


@pytest.mark.parametrize("backend", ["hip_gfx1100", "hip_gfx1151"])
def test_hip_eh_proj_matches_cpu_reference(backend, eh_proj_inputs):
    if not _exact_key_registered(backend):
        pytest.skip(f"{backend} mtp_nextn_eh_proj not registered yet (RED)")

    from hipengine.kernels.cpu_reference.ops import qwen35_gguf_mtp_eh_proj
    from hipengine.kernels.registry import resolve

    expected = qwen35_gguf_mtp_eh_proj(
        eh_proj_inputs["hidden_seed"],
        eh_proj_inputs["token_embedding"],
        eh_proj_inputs["eh_proj_weight"],
        eh_proj_inputs["hnorm_weight"],
        eh_proj_inputs["enorm_weight"],
        eps=eh_proj_inputs["eps"],
    )

    kernel = resolve(
        backend=backend, layer="mtp_nextn_eh_proj", quant="gguf_f32", variant="qwen35"
    )
    # Defense in depth: the resolved kernel must be the native hip kernel, not
    # the cpu_reference numpy oracle reached via fallback.
    assert kernel is not qwen35_gguf_mtp_eh_proj, (
        f"resolve({backend}) returned the cpu_reference oracle via fallback; "
        "native hip kernel not registered"
    )

    actual = kernel(
        eh_proj_inputs["hidden_seed"],
        eh_proj_inputs["token_embedding"],
        eh_proj_inputs["eh_proj_weight"],
        eh_proj_inputs["hnorm_weight"],
        eh_proj_inputs["enorm_weight"],
        eps=eh_proj_inputs["eps"],
    )
    actual = np.asarray(actual, dtype=np.float32)

    assert actual.shape == expected.shape, f"shape {actual.shape} != {expected.shape}"
    max_abs = float(np.max(np.abs(actual - expected)))
    assert max_abs < 1e-4, f"{backend} eh_proj max_abs={max_abs} exceeds 1e-4 vs cpu_reference"


@pytest.mark.parametrize("qtype", [_Q8_0, _Q6_K], ids=["q8_0", "q6_k"])
@pytest.mark.parametrize("rows", [1, 2, 3])
def test_quantized_eh_proj_matches_dequantized_oracle(qtype, rows):
    """Q8_0/Q6_K: device concat + quantized GEMV against the dequantized oracle.

    Synthetic block weights keep the oracle independent of any real GGUF file:
    ``gguf_quant_gemv`` dequantizes exactly the bytes the kernel reads.
    """
    from hipengine.kernels.hip_gfx1100.speculative.mtp_nextn import (
        qwen35_gguf_mtp_eh_proj_f32,
    )

    inputs = _inputs(0xE4 + 7 * rows + int(qtype), rows)
    weight = _synthetic_weight(qtype)

    actual = qwen35_gguf_mtp_eh_proj_f32(
        inputs["hidden_seed"],
        inputs["token_embedding"],
        weight,
        inputs["hnorm_weight"],
        inputs["enorm_weight"],
        eh_proj_qtype=qtype,
        eps=EPS,
    )
    actual = np.asarray(actual, dtype=np.float32)

    assert actual.shape == (rows, HIDDEN)
    np.testing.assert_allclose(actual, _oracle(inputs, weight, qtype), **_F32_TOL)


@pytest.mark.parametrize("qtype", [_Q8_0, _Q6_K], ids=["q8_0", "q6_k"])
def test_quantized_eh_proj_rows_are_numerically_consistent(qtype):
    """Each row of a 3-row call agrees numerically with that row alone.

    A concat laid out as ``[2*rows, hidden]`` instead of ``[rows, 2*hidden]``
    passes the rows == 1 case and fails here.
    """
    from hipengine.kernels.hip_gfx1100.speculative.mtp_nextn import (
        qwen35_gguf_mtp_eh_proj_f32,
    )

    weight = _synthetic_weight(qtype)
    batched = _inputs(0xB3 + int(qtype), 3)
    actual = qwen35_gguf_mtp_eh_proj_f32(
        batched["hidden_seed"],
        batched["token_embedding"],
        weight,
        batched["hnorm_weight"],
        batched["enorm_weight"],
        eh_proj_qtype=qtype,
        eps=EPS,
    )
    actual = np.asarray(actual, dtype=np.float32)
    assert actual.shape == (3, HIDDEN)

    for row in range(3):
        single = {
            "hidden_seed": batched["hidden_seed"][row : row + 1],
            "token_embedding": batched["token_embedding"][row : row + 1],
            "hnorm_weight": batched["hnorm_weight"],
            "enorm_weight": batched["enorm_weight"],
        }
        alone = qwen35_gguf_mtp_eh_proj_f32(
            single["hidden_seed"],
            single["token_embedding"],
            weight,
            single["hnorm_weight"],
            single["enorm_weight"],
            eh_proj_qtype=qtype,
            eps=EPS,
        )
        np.testing.assert_allclose(
            actual[row], np.asarray(alone, dtype=np.float32)[0], **_F32_TOL
        )
