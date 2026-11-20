"""Prompt preparation for Muse Glimmer image embedding spans.

Placement matches the mtmd muse-glimmer handler and the captured oracle:
``<|image_start|>`` + n_out ``<|image|>`` placeholder rows + ``<|image_end|>``.
The ATEM chat template renders ``<|patch|>`` for image content parts; that
marker (or the mtmd default ``<__media__>``) is accepted as the media position.
Bring-up tokenizes plain prompts as-is with BOS; full ATEM chat rendering is
the P5 serving closure.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

IMAGE_BEGIN = '<|image_start|>'
IMAGE_END = '<|image_end|>'
IMAGE_SOFT = '<|image|>'
MEDIA_MARKERS = ('<|patch|>', '<__media__>')


def muse_glimmer_media_images(media):
    """Accept one RGB array/path or the server's ordered image-item mapping."""

    if isinstance(media, Mapping):
        items = tuple(media.get('items', ()))
        if not items:
            raise ValueError('Muse Glimmer image input requires at least one image item')
        images = []
        for item in items:
            if not isinstance(item, Mapping) or item.get('type') != 'image':
                raise NotImplementedError('Muse Glimmer multimodal input implements images, not video or audio')
            if 'data' not in item:
                raise ValueError('Muse Glimmer image item requires data')
            images.append(item['data'])
        return tuple(images)
    return (media,)


def muse_glimmer_image_prompt(prompt, counts, tokenizer):
    """Expand ordered media markers and return IDs plus their soft-token spans."""

    counts = tuple(int(count) for count in counts)
    if not counts or any(count <= 0 for count in counts):
        raise ValueError('Muse Glimmer images must each produce positive embedding rows')
    text = str(prompt)
    marker = next((m for m in MEDIA_MARKERS if m in text), None)
    if marker is None:
        marker = MEDIA_MARKERS[0]
        text = '\n'.join((*(marker for _ in counts), text))
    elif text.count(marker) != len(counts):
        raise ValueError(
            f'Muse Glimmer image markers ({text.count(marker)}) do not match images ({len(counts)})'
        )
    pieces = text.split(marker)
    expanded = pieces[0]
    for count, suffix in zip(counts, pieces[1:], strict=True):
        expanded += IMAGE_BEGIN + IMAGE_SOFT * count + IMAGE_END + suffix
    ids = tuple(tokenizer.encode(expanded, add_special_tokens=True))
    begin_id = tokenizer.token_to_id.get(IMAGE_BEGIN)
    soft_id = tokenizer.token_to_id.get(IMAGE_SOFT)
    end_id = tokenizer.token_to_id.get(IMAGE_END)
    if any(value is None for value in (begin_id, soft_id, end_id)):
        raise ValueError('Muse Glimmer tokenizer lacks the image begin, soft or end token')
    spans = []
    cursor = 0
    for count in counts:
        try:
            begin = ids.index(begin_id, cursor)
        except ValueError as exc:
            raise ValueError('Muse Glimmer tokenizer did not preserve the begin-image token') from exc
        a, b = begin + 1, begin + 1 + count
        if ids[a:b] != (soft_id,) * count or b >= len(ids) or ids[b] != end_id:
            raise ValueError('Muse Glimmer tokenizer did not preserve the image placeholder span')
        spans.append((a, b))
        cursor = b + 1
    if sum(token == soft_id for token in ids) != sum(counts):
        raise ValueError('Muse Glimmer prompt contains unmatched image placeholders')
    return ids, tuple(spans)


def muse_glimmer_embedding_overrides(features, spans):
    """Bind projected rows to absolute prompt positions (pre embedding-norm)."""

    overrides = {}
    for value, (begin, end) in zip(features, spans, strict=True):
        array = np.asarray(value, dtype=np.float32)
        if array.ndim != 2 or len(array) != end - begin or not np.isfinite(array).all():
            raise ValueError('Muse Glimmer image features do not match the placeholder span')
        overrides.update((begin + offset, np.ascontiguousarray(row)) for offset, row in enumerate(array))
    return overrides


__all__ = [
    'IMAGE_BEGIN', 'IMAGE_END', 'IMAGE_SOFT', 'MEDIA_MARKERS',
    'muse_glimmer_media_images', 'muse_glimmer_image_prompt',
    'muse_glimmer_embedding_overrides',
]