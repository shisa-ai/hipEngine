from __future__ import annotations

import glob
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.kernels.cpu_reference.surya import (
    SuryaSpec,
    SuryaWeights,
    vision_forward,
)

FIXTURES = Path("tests/fixtures/surya")


def _checkpoint() -> str | None:
    hits = sorted(
        glob.glob(
            str(
                Path(
                    "~/.cache/huggingface/hub/models--datalab-to--surya-ocr-2/snapshots/*/model.safetensors"
                ).expanduser()
            )
        )
    )
    return hits[0] if hits else None


@pytest.fixture(scope="module")
def weights() -> SuryaWeights:
    ckpt = _checkpoint()
    if ckpt is None:
        pytest.skip("datalab-to/surya-ocr-2 not in local HF cache")
    return SuryaWeights.load(ckpt)


@pytest.fixture(scope="module")
def oracle() -> dict[str, np.ndarray]:
    path = FIXTURES / "oracle_image.npz"
    if not path.exists():
        pytest.skip(f"{path} not present; run scripts/surya_oracle_torch.py")
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _grid(oracle: dict[str, np.ndarray]) -> list[tuple[int, int, int]]:
    t, h, w = oracle["image_grid_thw"][0]
    return [(int(t), int(h), int(w))]


# measured against the torch fp32 oracle (see worklog entry for this unit)
PATCH_EMBED_GATE = 1e-4
TOWER_GATE = 1e-3
MERGED_GATE = 1e-3


@pytest.mark.parametrize("activation", ["_gelu_tanh_np", "_gelu_erf_np"])
def test_vision_activation_preserves_fp32(activation):
    from hipengine.kernels.cpu_reference import surya

    output = getattr(surya, activation)(np.linspace(-4, 4, 17, dtype=np.float32))
    assert output.dtype == np.float32
    assert np.isfinite(output).all()


def test_vision_position_embedding_returns_fp32():
    from hipengine.kernels.cpu_reference.surya import vision_pos_embed

    output = vision_pos_embed(
        {"model.visual.pos_embed.weight": np.arange(4, dtype=np.float32)[:, None]},
        SimpleNamespace(vision_num_position_embeddings=4),
        [(1, 3, 3)], (np.arange(3), np.arange(3)),
    )
    assert output.dtype == np.float32
    np.testing.assert_array_equal(output[:, 0], [0.0, 1.5, 3.0])


def test_vision_patch_embed(weights: SuryaWeights, oracle: dict[str, np.ndarray]) -> None:
    spec = SuryaSpec()
    out = vision_patch_embed_h = vision_forward(
        weights, spec, oracle["pixel_values"], _grid(oracle)
    )[0]
    ref = oracle["vision_patch_embed"]
    assert out.shape == ref.shape == (256, 768)
    np.testing.assert_allclose(out, ref, atol=PATCH_EMBED_GATE)


def test_vision_tower(weights: SuryaWeights, oracle: dict[str, np.ndarray]) -> None:
    spec = SuryaSpec()
    _, tower, _ = vision_forward(weights, spec, oracle["pixel_values"], _grid(oracle))
    ref = oracle["vision_tower_out"]
    assert tower.shape == ref.shape == (256, 768)
    assert tower.dtype == np.float32
    np.testing.assert_allclose(tower, ref, atol=TOWER_GATE, rtol=1e-3)


def test_vision_merged(weights: SuryaWeights, oracle: dict[str, np.ndarray]) -> None:
    spec = SuryaSpec()
    _, _, merged = vision_forward(
        weights, spec, oracle["pixel_values"], _grid(oracle)
    )
    ref = oracle["vision_merged"]
    assert merged.shape == ref.shape == (64, 1024)
    assert merged.dtype == np.float32
    np.testing.assert_allclose(merged, ref, atol=MERGED_GATE, rtol=1e-3)
