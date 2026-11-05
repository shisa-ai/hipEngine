"""Live tier: Muse Glimmer GPU vision runner vs the f32-dequantized oracle.

Same oracle and distributional gates as the CPU-reference tier (cosine >=
0.998, mean/std drift <= 0.01, max |err| <= 25 catastrophic bound): the runner
computes the architecture's f32 semantics with rocBLAS sgemm + the registered
FP32 vision primitives. The model amplifies f32 accumulation-order noise
~10^4x on smooth inputs (pinned by the noise-injection experiment in the
reference tier), so per-element parity is not the semantic gate. Requires
ROCm, the local mmproj, and the local oracle dumps (regenerate with
scripts/muse_glimmer_vision_oracle.cpp against an F32 mmproj, flash attention
disabled, GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32).
"""

from __future__ import annotations

import ctypes
import json
from pathlib import Path

import numpy as np
import pytest

try:
    ctypes.CDLL("libamdhip64.so")
    _HIP_AVAILABLE = True
except OSError:
    _HIP_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _HIP_AVAILABLE, reason="ROCm runtime not available")

MMPROJ = Path(
    "/models/gguf/Muse-Glimmer-30B-GGUF/mmproj-Muse-Glimmer-30B-Q4_K_M.gguf"
)
ORACLE_DIR = Path(__file__).parent.parent / "fixtures" / "muse_glimmer" / "vision_oracle_e0f142d5d"


@pytest.fixture(scope="module")
def oracle() -> dict:
    return json.loads((ORACLE_DIR / "oracle.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def runner():
    if not MMPROJ.exists():
        pytest.skip(f"local Muse Glimmer mmproj not found: {MMPROJ}")
    from hipengine.kernels.cpu_reference.muse_glimmer_vision import MuseGlimmerVisionConfig
    from hipengine.loading.gguf import GGUFReader
    from hipengine.runtime.muse_glimmer_vision import MuseGlimmerVisionRunner

    reader = GGUFReader(MMPROJ)
    config = MuseGlimmerVisionConfig.from_gguf_info(reader.info)
    run = MuseGlimmerVisionRunner(reader, config, backend="hip_gfx1151")
    yield run
    run.close()


@pytest.mark.parametrize("stem", ["shapes_448", "blocks_672x448", "gradient_896"])
def test_gpu_projected_rows_match_oracle(stem, oracle, runner) -> None:
    from PIL import Image

    expected = oracle["images"][stem]
    full_path = ORACLE_DIR / f"{stem}.embd.f32"
    if not full_path.exists():
        pytest.skip(f"local oracle dump not found: {full_path}")
    pixels = np.asarray(
        Image.open(ORACLE_DIR.parent / "images" / f"{stem}.png").convert("RGB")
    )
    features = runner.encode(pixels)
    assert features.shape == (expected["n_out"], 6656)
    assert np.isfinite(features).all()
    reference = np.fromfile(full_path, dtype=np.float32).reshape(features.shape)
    err = float(np.abs(features - reference).max())
    cos = float(np.sum(features * reference) / (
        np.linalg.norm(features) * np.linalg.norm(reference)))
    assert cos >= 0.998, f"{stem}: GPU vs oracle cosine {cos}"
    assert err <= 25.0, f"{stem}: GPU vs oracle max |err| {err}"
    assert abs(float(features.mean()) - expected["global"]["mean"]) <= 0.01
    assert abs(float(features.std()) - expected["global"]["std"]) <= 0.01