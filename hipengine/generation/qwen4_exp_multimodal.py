"""Pure Qwen4Exp multimodal prompt and MRoPE control helpers."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

import numpy as np

from hipengine.runtime.qwen4_exp_vision import Qwen4ExpVisionFeatures

_IMAGE_PAD = "<|image_pad|>"
_VIDEO_PAD = "<|video_pad|>"
_VISION_START = "<|vision_start|>"
_VISION_END = "<|vision_end|>"
_MARKER = re.compile(r"<\|(image|video)_pad\|>")


@dataclass(frozen=True)
class Qwen4ExpMultimodalPrompt:
    rendered: str
    features: tuple[Qwen4ExpVisionFeatures, ...]


def render_qwen4_exp_multimodal_prompt(
    prompt: str,
    features: Sequence[Qwen4ExpVisionFeatures],
) -> Qwen4ExpMultimodalPrompt:
    """Expand one media marker per encoded item, or prepend generated spans."""

    values = tuple(features)
    if not values:
        raise ValueError("Qwen4Exp multimodal generation requires media")
    text = str(prompt)
    markers = list(_MARKER.finditer(text))
    if markers:
        if len(markers) != len(values):
            raise ValueError(
                f"Qwen4Exp media markers ({len(markers)}) do not match inputs "
                f"({len(values)})"
            )
        pieces: list[str] = []
        cursor = 0
        for marker, item in zip(markers, values, strict=True):
            kind = marker.group(1)
            if kind != item.modality:
                raise ValueError(
                    f"Qwen4Exp {kind} marker does not match {item.modality} input"
                )
            pieces.append(text[cursor : marker.start()])
            pad = _IMAGE_PAD if kind == "image" else _VIDEO_PAD
            pieces.append(pad * int(item.embeddings.shape[0]))
            cursor = marker.end()
        pieces.append(text[cursor:])
        body = "".join(pieces)
    else:
        spans = []
        for item in values:
            pad = _IMAGE_PAD if item.modality == "image" else _VIDEO_PAD
            spans.append(
                _VISION_START + pad * int(item.embeddings.shape[0]) + _VISION_END
            )
        body = "\n".join((*spans, text))
    rendered = (
        "<|im_start|>user\n"
        + body
        + "<|im_end|>\n<|im_start|>assistant\n"
    )
    return Qwen4ExpMultimodalPrompt(rendered, values)


def qwen4_exp_multimodal_token_control(
    token_ids: Sequence[int],
    features: Sequence[Qwen4ExpVisionFeatures],
    *,
    image_token_id: int = 248_056,
    video_token_id: int = 248_057,
    spatial_merge_size: int = 2,
) -> tuple[dict[int, np.ndarray], np.ndarray, int]:
    """Validate placeholders and return embedding overrides + 3-axis positions.

    Position construction follows Qwen3.5/Qwen4Exp ``get_rope_index``: text
    advances all axes by token count, while each vision group advances by the
    larger merged spatial side. Returned positions are axis-major ``[3,L]``.
    """

    ids = tuple(int(token) for token in token_ids)
    values = tuple(features)
    runs: list[tuple[int, int, int]] = []
    index = 0
    while index < len(ids):
        token = ids[index]
        if token not in (image_token_id, video_token_id):
            index += 1
            continue
        end = index + 1
        while end < len(ids) and ids[end] == token:
            end += 1
        runs.append((token, index, end))
        index = end
    if len(runs) != len(values):
        raise ValueError(
            f"Qwen4Exp placeholder groups ({len(runs)}) do not match encoded "
            f"media ({len(values)})"
        )

    overrides: dict[int, np.ndarray] = {}
    positions = np.empty((3, len(ids)), dtype=np.int64)
    current = 0
    media_index = 0
    cursor = 0
    for token, start, end in runs:
        if start > cursor:
            count = start - cursor
            scalar = np.arange(current, current + count, dtype=np.int64)
            positions[:, cursor:start] = scalar
            current += count
        item = values[media_index]
        expected_token = image_token_id if item.modality == "image" else video_token_id
        if token != expected_token:
            raise ValueError("Qwen4Exp placeholder token type does not match media")
        rows = int(item.embeddings.shape[0])
        if end - start != rows:
            raise ValueError(
                f"Qwen4Exp {item.modality} features and tokens do not match: "
                f"tokens={end-start}, features={rows}"
            )
        grid_t, grid_h, grid_w = (int(value) for value in item.grid_thw)
        if (
            grid_t <= 0
            or grid_h <= 0
            or grid_w <= 0
            or grid_h % spatial_merge_size
            or grid_w % spatial_merge_size
        ):
            raise ValueError("Qwen4Exp media grid is not merge-compatible")
        llm_h = grid_h // spatial_merge_size
        llm_w = grid_w // spatial_merge_size
        if grid_t * llm_h * llm_w != rows:
            raise ValueError("Qwen4Exp media grid does not match embedding rows")
        temporal, height, width = np.meshgrid(
            np.arange(grid_t, dtype=np.int64),
            np.arange(llm_h, dtype=np.int64),
            np.arange(llm_w, dtype=np.int64),
            indexing="ij",
        )
        vision_positions = np.stack(
            (temporal, height, width), axis=0
        ).reshape(3, -1)
        vision_positions += current
        positions[:, start:end] = vision_positions
        for offset, row in enumerate(item.embeddings):
            overrides[start + offset] = np.ascontiguousarray(row, dtype=np.float32)
        current += max(llm_h, llm_w)
        cursor = end
        media_index += 1
    if cursor < len(ids):
        count = len(ids) - cursor
        scalar = np.arange(current, current + count, dtype=np.int64)
        positions[:, cursor:] = scalar
        current += count
    next_position = int(positions.max()) + 1 if positions.size else 0
    return overrides, np.ascontiguousarray(positions), next_position


__all__ = [
    "Qwen4ExpMultimodalPrompt",
    "qwen4_exp_multimodal_token_control",
    "render_qwen4_exp_multimodal_prompt",
]
