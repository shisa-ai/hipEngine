from __future__ import annotations

import numpy as np
import pytest

from hipengine.generation.qwen4_exp_multimodal import (
    qwen4_exp_multimodal_token_control,
    render_qwen4_exp_multimodal_prompt,
)
from hipengine.runtime.qwen4_exp_vision import Qwen4ExpVisionFeatures


def _feature(rows: int, grid: tuple[int, int, int], kind: str):
    return Qwen4ExpVisionFeatures(
        np.arange(rows * 8, dtype=np.float32).reshape(rows, 8), grid, kind
    )


def test_qwen4_exp_multimodal_control_builds_exact_three_axis_positions() -> None:
    image = _feature(2, (1, 2, 4), "image")
    video = _feature(2, (2, 2, 2), "video")
    ids = [10, 248056, 248056, 11, 248057, 248057, 12]
    overrides, positions, next_position = qwen4_exp_multimodal_token_control(
        ids, (image, video)
    )

    assert tuple(overrides) == (1, 2, 4, 5)
    np.testing.assert_array_equal(
        positions,
        np.asarray(
            [
                [0, 1, 1, 3, 4, 5, 5],
                [0, 1, 1, 3, 4, 4, 5],
                [0, 1, 2, 3, 4, 4, 5],
            ],
            dtype=np.int64,
        ),
    )
    assert next_position == 6


def test_qwen4_exp_multimodal_prompt_expands_typed_markers() -> None:
    image = _feature(2, (1, 2, 4), "image")
    video = _feature(3, (3, 2, 2), "video")
    rendered = render_qwen4_exp_multimodal_prompt(
        "A <|vision_start|><|image_pad|><|vision_end|> B "
        "<|vision_start|><|video_pad|><|vision_end|>",
        (image, video),
    ).rendered
    assert rendered.count("<|image_pad|>") == 2
    assert rendered.count("<|video_pad|>") == 3


@pytest.mark.parametrize(
    ("prompt", "features", "match"),
    [
        ("<|image_pad|><|image_pad|>", (_feature(1, (1, 2, 2), "image"),), "markers"),
        ("<|video_pad|>", (_feature(1, (1, 2, 2), "image"),), "does not match"),
    ],
)
def test_qwen4_exp_multimodal_prompt_rejects_placeholder_mismatch(
    prompt, features, match
) -> None:
    with pytest.raises(ValueError, match=match):
        render_qwen4_exp_multimodal_prompt(prompt, features)


def test_qwen4_exp_multimodal_control_rejects_feature_count_mismatch() -> None:
    with pytest.raises(ValueError, match="features and tokens"):
        qwen4_exp_multimodal_token_control(
            [248056], (_feature(2, (1, 2, 4), "image"),)
        )
