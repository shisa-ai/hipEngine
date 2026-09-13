from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

FIXTURES = Path("tests/fixtures/surya")


def _load_oracle(name: str) -> dict[str, np.ndarray]:
    path = FIXTURES / name
    if not path.exists():
        pytest.skip(f"{path} not present; run scripts/surya_oracle_torch.py")
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


@pytest.fixture(scope="module")
def image_oracle() -> dict[str, np.ndarray]:
    return _load_oracle("oracle_image.npz")


@pytest.fixture(scope="module")
def text_oracle() -> dict[str, np.ndarray]:
    return _load_oracle("oracle_text.npz")


def test_image_case_inputs(image_oracle: dict[str, np.ndarray]) -> None:
    z = image_oracle
    assert z["input_ids"].shape == (1, 106)
    assert z["pixel_values"].shape == (256, 1536)  # 2x256 patch rows x 3*16*16
    assert z["image_grid_thw"].tolist() == [[1, 16, 16]]
    # 64 merged image tokens: (16/2)*(16/2)
    assert z["vision_merged"].shape == (64, 1024)
    assert z["vision_patch_embed"].shape == (256, 768)
    assert z["vision_tower_out"].shape == (256, 768)
    # one <|image_pad|> in the template expands to 64 positions
    pads = int((z["input_ids"][0] == 11).sum())
    assert pads == 64
    assert int((z["input_ids"][0] == 9).sum()) == 1  # vision_start
    assert int((z["input_ids"][0] == 10).sum()) == 1  # vision_end


def test_mm_token_type_ids_marks_image_span(
    image_oracle: dict[str, np.ndarray],
) -> None:
    z = image_oracle
    ids = z["mm_token_type_ids"][0]
    assert ids.shape == (106,)
    # image tokens flagged 1, text tokens 0; image span is contiguous
    assert int(ids.sum()) == 64
    flagged = np.flatnonzero(ids == 1)
    assert flagged[-1] - flagged[0] + 1 == 64
    # flag boundaries align with the <|image_pad|> run
    pad_idx = np.flatnonzero(z["input_ids"][0] == 11)
    assert np.array_equal(flagged, pad_idx)


def test_mrope_positions_are_three_axis(image_oracle: dict[str, np.ndarray]) -> None:
    pos = image_oracle["position_ids_prefill"]
    assert pos.shape == (3, 1, 106)  # (t, h, w) axes
    # spatial compression: image positions advance slower than token index
    assert int(pos[0, 0, -1]) < 106
    text_pos = _load_oracle_reuse_text_pos()
    assert text_pos.shape == (3, 1, 40)


def _load_oracle_reuse_text_pos() -> np.ndarray:
    with np.load(FIXTURES / "oracle_text.npz") as z:
        return z["position_ids_prefill"]


def test_gdn_cache_states(image_oracle: dict[str, np.ndarray]) -> None:
    z = image_oracle
    # conv state spans the fused qkv projection width (3*2048) x kernel 4
    assert z["prefill.cache.layer0.conv_states[0]"].shape == (1, 6144, 4)
    # recurrent state: 16 heads x 128 x 128, fp32 (mamba_ssm_dtype)
    assert z["prefill.cache.layer0.recurrent_states[0]"].shape == (1, 16, 128, 128)
    assert z["prefill.cache.layer0.recurrent_states[0]"].dtype == np.float32
    for layer in (20,):
        assert f"prefill.cache.layer{layer}.conv_states[0]" in z
        assert f"prefill.cache.layer{layer}.recurrent_states[0]" in z


def test_full_attention_cache_states(image_oracle: dict[str, np.ndarray]) -> None:
    z = image_oracle
    for layer in (3, 23):
        assert z[f"prefill.cache.layer{layer}.keys"].shape == (1, 2, 106, 256)
        assert z[f"prefill.cache.layer{layer}.values"].shape == (1, 2, 106, 256)


def test_cached_decode_steps_advance(
    image_oracle: dict[str, np.ndarray], text_oracle: dict[str, np.ndarray]
) -> None:
    z = image_oracle
    steps = 4
    for layer in (3, 23):
        assert z[f"steps.cache.layer{layer}.keys"].shape == (1, 2, 106 + steps, 256)
    for i in range(steps):
        assert f"logits_step{i}" in z
        assert z[f"logits_step{i}"].shape == (65425,)
    # text case mirrors the structure with no vision tensors
    assert "vision_merged" not in text_oracle
    assert text_oracle["steps.cache.layer3.keys"].shape == (1, 2, 44, 256)


def test_logits_are_finite_and_first_token_differs_by_case(
    image_oracle: dict[str, np.ndarray], text_oracle: dict[str, np.ndarray]
) -> None:
    for z in (image_oracle, text_oracle):
        for k in z:
            if k.startswith("logits"):
                assert np.isfinite(z[k]).all(), k
    assert not np.allclose(
        image_oracle["logits_first"], text_oracle["logits_first"]
    )


def test_meta_contract() -> None:
    path = FIXTURES / "meta.json"
    if not path.exists():
        pytest.skip("meta.json not present; run scripts/surya_oracle_torch.py")
    meta = json.loads(path.read_text())
    assert meta["eos_token_id"] == 2  # never the stale text_config value
    assert meta["image_token_id"] == 11
    assert meta["image_grid_thw"] == [[1, 16, 16]]
    assert len(meta["decode_step_ids"]) == meta["steps"]
    assert meta["prompt_image"].endswith("<|im_start|>assistant\n")
