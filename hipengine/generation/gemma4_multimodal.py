"""Prompt preparation for Gemma 4 image embedding spans."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

IMAGE_MARKER = '<|image|>'
IMAGE_BEGIN = '<|image>'
IMAGE_END = '<image|>'


def gemma4_media_images(media):
    """Accept one RGB array or the server's ordered image-item mapping."""

    if isinstance(media, Mapping):
        items = tuple(media.get('items', ()))
        if not items:
            raise ValueError('Gemma 4 image input requires at least one image item')
        images = []
        for item in items:
            if not isinstance(item, Mapping) or item.get('type') != 'image':
                raise NotImplementedError('Gemma 4 multimodal input implements images, not video or audio')
            if 'data' not in item:
                raise ValueError('Gemma 4 image item requires data')
            images.append(item['data'])
        return tuple(images)
    return (media,)


def gemma4_image_prompt(prompt, counts, tokenizer, render_chat):
    """Expand ordered image markers and return IDs plus their soft-token spans."""

    counts = tuple(int(count) for count in counts)
    if not counts or any(count <= 0 for count in counts):
        raise ValueError('Gemma 4 images must each produce positive embedding rows')
    text = str(prompt)
    markers = text.count(IMAGE_MARKER)
    if markers == 0:
        text = '\n'.join((*(IMAGE_MARKER for _ in counts), text))
    elif markers != len(counts):
        raise ValueError(f'Gemma 4 image markers ({markers}) do not match images ({len(counts)})')
    # A preformatted conversation is already rendered, including its thinking
    # policy. Plain text becomes one user turn through the artifact template.
    if '<|turn>' not in text:
        text = render_chat([{'role':'user', 'content':text}])
    if text.count(IMAGE_MARKER) != len(counts):
        raise ValueError('Gemma 4 chat template did not preserve image markers')
    pieces = text.split(IMAGE_MARKER)
    expanded = pieces[0]
    for count, suffix in zip(counts, pieces[1:], strict=True):
        expanded += IMAGE_BEGIN + IMAGE_MARKER*count + IMAGE_END + suffix
    ids = tuple(tokenizer.encode(expanded, add_special_tokens=False))
    image_id = tokenizer.token_to_id.get(IMAGE_MARKER)
    begin_id = tokenizer.token_to_id.get(IMAGE_BEGIN)
    end_id = tokenizer.token_to_id.get(IMAGE_END)
    if any(value is None for value in (image_id,begin_id,end_id)):
        raise ValueError('Gemma 4 tokenizer lacks the image begin, soft or end token')
    spans = []
    cursor = 0
    for count in counts:
        try:
            begin = ids.index(begin_id,cursor)
        except ValueError as exc:
            raise ValueError('Gemma 4 tokenizer did not preserve the begin-image token') from exc
        a,b = begin+1,begin+1+count
        if ids[a:b] != (image_id,)*count or b >= len(ids) or ids[b] != end_id:
            raise ValueError('Gemma 4 tokenizer did not preserve the image placeholder span')
        spans.append((a,b))
        cursor = b+1
    if sum(token == image_id for token in ids) != sum(counts):
        raise ValueError('Gemma 4 prompt contains unmatched image placeholders')
    return ids,tuple(spans)


def gemma4_embedding_overrides(features, spans):
    """Bind projected rows to absolute prompt positions without text scaling."""

    overrides = {}
    for value,(begin,end) in zip(features,spans,strict=True):
        array = np.asarray(value,dtype=np.float32)
        if array.ndim != 2 or len(array) != end-begin or not np.isfinite(array).all():
            raise ValueError('Gemma 4 image features do not match the placeholder span')
        overrides.update((begin+offset,np.ascontiguousarray(row)) for offset,row in enumerate(array))
    return overrides
