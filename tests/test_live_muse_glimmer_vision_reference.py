"""Live tier: Muse Glimmer vision CPU reference vs the llama.cpp mtmd oracle.

The oracle (fixtures/muse_glimmer/vision_oracle_e0f142d5d/oracle.json) was
captured from llama.cpp build 12005 e0f142d5d with the same mmproj artifact via
scripts/muse_glimmer_vision_oracle.cpp. Full embedding dumps stay local
(*.embd.f32, git-ignored); when present the gate compares every row, otherwise
it compares the committed sampled rows.

Tolerances (declared before measurement, refined by localization): the
reference dequantizes the same weights to f32 (verified bit-exact vs ggml's
F32 conversion) but accumulates in a different order. Stage-by-stage bisection
against the oracle's graph nodes shows every stage matches to f32 rounding;
the residual is zero-mean noise (std ~6e-4 at pre_ln). This model amplifies
such noise ~10^4x on smooth inputs: injecting random noise of the same std
into the oracle's own pre_ln reproduces the observed output spread (2.9-14.9
vs our 14.4 on the 896x896 gradient). Per-element parity is therefore not a
semantic gate here; the gates are distributional: cosine >= 0.998, global
mean/std drift <= 0.01, and max |err| <= 25 as a catastrophic-failure bound
(a real semantic bug produces O(100%) divergence, cosine < 0.9).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hipengine.kernels.cpu_reference.muse_glimmer_vision import (
    MuseGlimmerVisionConfig,
    forward,
    gguf_weight_source,
    grid_size,
    normalize_pixels,
)
from hipengine.loading.gguf import GGUFReader

MMPROJ = Path(
    "/models/gguf/Muse-Glimmer-30B-GGUF/mmproj-Muse-Glimmer-30B-Q4_K_M.gguf"
)
ORACLE_DIR = Path(__file__).parent.parent / "fixtures" / "muse_glimmer" / "vision_oracle_e0f142d5d"


@pytest.fixture(scope="module")
def oracle() -> dict:
    return json.loads((ORACLE_DIR / "oracle.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def vision_setup():
    if not MMPROJ.exists():
        pytest.skip(f"local Muse Glimmer mmproj not found: {MMPROJ}")
    reader = GGUFReader(MMPROJ)
    config = MuseGlimmerVisionConfig.from_gguf_info(reader.info)
    return reader, config, gguf_weight_source(reader)


def _load_rgb(path: Path) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _encode(stem: str, setup) -> np.ndarray:
    reader, config, get = setup
    pixels = _load_rgb(ORACLE_DIR.parent / "images" / f"{stem}.png")
    h, w = pixels.shape[:2]
    patch_hw = config.patch_size * config.spatial_merge_size
    tw, th = grid_size(w, h, patch_hw, config.max_image_tokens)
    assert (tw, th) == (w, h), "fixture images must be grid-aligned (no resize)"
    return forward(normalize_pixels(pixels, config), get, config)


@pytest.mark.parametrize("stem", ["shapes_448", "blocks_672x448", "gradient_896"])
def test_projected_rows_match_oracle(stem, oracle, vision_setup) -> None:
    expected = oracle["images"][stem]
    features = _encode(stem, vision_setup)
    assert features.shape == (expected["n_out"], 6656)
    assert np.isfinite(features).all()

    full_path = ORACLE_DIR / f"{stem}.embd.f32"
    if full_path.exists():
        reference = np.fromfile(full_path, dtype=np.float32).reshape(features.shape)
        err = float(np.abs(features - reference).max())
        cos = float(np.sum(features * reference) / (
            np.linalg.norm(features) * np.linalg.norm(reference)))
        assert cos >= 0.998, f"{stem}: cosine {cos}"
        assert err <= 25.0, f"{stem}: full-dump max |err| {err}"
        assert abs(float(features.mean()) - expected["global"]["mean"]) <= 0.01
        assert abs(float(features.std()) - expected["global"]["std"]) <= 0.01
    else:
        for ri, row_meta in expected["sampled_rows"].items():
            row = features[int(ri)]
            assert abs(float(row.mean()) - row_meta["mean"]) <= 0.05
            assert float(row.min()) >= row_meta["min"] - 1.0
            assert float(row.max()) <= row_meta["max"] + 1.0