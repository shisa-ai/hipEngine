"""Bounded OpenAI chat multimodal parsing without hot-path image dependencies."""

from __future__ import annotations

import base64
import binascii
from io import BytesIO
import struct
from typing import Any, Mapping, Sequence
import zlib

import numpy as np

_IMAGE_MARKER = "<|vision_start|><|image_pad|><|vision_end|>"
_VIDEO_MARKER = "<|vision_start|><|video_pad|><|vision_end|>"
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# The OpenAI HTTP vision scope defaults to a 1 MP / 8 MiB payload, which is what
# the Qwen4Exp multimodal path was qualified against. An engine that declares a
# larger ``vision_max_pixels`` (a document OCR model does: a 300-DPI A4 page is
# 2480x3508 = 8.7 MP, and the Surya preprocessor ceiling is 16.8 MP) widens both
# bounds through :func:`resolve_vision_http_limits` rather than by relaxing the
# default for every engine.
DEFAULT_VISION_MAX_PIXELS = 1_048_576
DEFAULT_VISION_MAX_BYTES = 8 * 1024 * 1024
MAX_VISION_MAX_BYTES = 64 * 1024 * 1024


def resolve_vision_http_limits(
    engine: Any,
    *,
    max_pixels: int | None = None,
    max_bytes: int | None = None,
) -> tuple[int, int]:
    """Decoded-pixel and compressed-payload bounds for HTTP vision input.

    Returns ``(max_pixels, max_bytes)``. ``max_pixels`` bounds the decoded
    image area, so it also bounds the per-side limit at ``ceil(sqrt)`` and stops
    a small payload from decoding into a large buffer. ``max_bytes`` bounds the
    compressed request payload. An explicit argument wins; otherwise an engine
    that declares ``vision_max_pixels`` sets the pixel bound, and the byte bound
    scales with it because a 300-DPI document does not compress like a
    thumbnail.
    """

    declared = int(getattr(engine, "vision_max_pixels", 0) or 0)
    pixels = int(max_pixels) if max_pixels else (declared or DEFAULT_VISION_MAX_PIXELS)
    if pixels <= 0:
        raise ValueError("vision max_pixels must be positive")
    if max_bytes:
        resolved_bytes = int(max_bytes)
    elif pixels > DEFAULT_VISION_MAX_PIXELS:
        # One compressed byte per admitted pixel, floored at the default and
        # capped so a single request cannot ask for an unbounded buffer.
        resolved_bytes = min(MAX_VISION_MAX_BYTES, max(DEFAULT_VISION_MAX_BYTES, pixels))
    else:
        resolved_bytes = DEFAULT_VISION_MAX_BYTES
    if resolved_bytes <= 0:
        raise ValueError("vision max_bytes must be positive")
    return pixels, resolved_bytes


def vision_max_side(max_pixels: int) -> int:
    """Per-side bound implied by an area bound."""

    side = 1
    while side * side < int(max_pixels):
        side += 1
    return side


def decode_bounded_png_data_url(
    value: str,
    *,
    max_bytes: int = DEFAULT_VISION_MAX_BYTES,
    max_side: int = 1_024,
    max_pixels: int | None = None,
) -> np.ndarray:
    """Decode an 8-bit non-interlaced RGB/RGBA PNG data URL."""

    prefix = "data:image/png;base64,"
    if not isinstance(value, str) or not value.startswith(prefix):
        raise ValueError("HTTP vision requires a base64 image/png data URL")
    try:
        payload = base64.b64decode(value[len(prefix) :], validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("invalid base64 image payload") from exc
    if not payload or len(payload) > int(max_bytes):
        raise ValueError("PNG payload is empty or exceeds the HTTP vision limit")
    if not payload.startswith(_PNG_SIGNATURE):
        raise ValueError("invalid PNG signature")
    offset = len(_PNG_SIGNATURE)
    width = height = color_type = bit_depth = interlace = None
    compressed: list[bytes] = []
    saw_end = False
    while offset + 12 <= len(payload):
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        kind = payload[offset + 4 : offset + 8]
        end = offset + 12 + length
        if length > max_bytes or end > len(payload):
            raise ValueError("invalid PNG chunk length")
        data = payload[offset + 8 : offset + 8 + length]
        if kind == b"IHDR":
            if length != 13 or width is not None:
                raise ValueError("invalid PNG IHDR")
            width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(
                ">IIBBBBB", data
            )
            if (
                width <= 0
                or height <= 0
                or width > max_side
                or height > max_side
                or (max_pixels is not None and width * height > int(max_pixels))
                or bit_depth != 8
                or color_type not in (2, 6)
                or compression != 0
                or filtering != 0
                or interlace != 0
            ):
                raise ValueError("unsupported PNG geometry/format")
        elif kind == b"IDAT":
            compressed.append(data)
        elif kind == b"IEND":
            saw_end = True
            break
        offset = end
    if width is None or not compressed or not saw_end:
        raise ValueError("incomplete PNG payload")
    channels = 3 if color_type == 2 else 4
    stride = width * channels
    expected = height * (stride + 1)
    try:
        raw = zlib.decompress(b"".join(compressed))
    except zlib.error as exc:
        raise ValueError("invalid PNG compressed payload") from exc
    if len(raw) != expected:
        raise ValueError("PNG scanline size does not match IHDR")
    output = np.empty((height, stride), dtype=np.uint8)
    prior = np.zeros(stride, dtype=np.uint8)
    cursor = 0
    for row in range(height):
        filter_type = raw[cursor]
        scan = np.frombuffer(raw, dtype=np.uint8, count=stride, offset=cursor + 1).copy()
        cursor += stride + 1
        recon = output[row]
        for column in range(stride):
            left = int(recon[column - channels]) if column >= channels else 0
            up = int(prior[column])
            upper_left = int(prior[column - channels]) if column >= channels else 0
            value_at = int(scan[column])
            if filter_type == 0:
                value = value_at
            elif filter_type == 1:
                value = value_at + left
            elif filter_type == 2:
                value = value_at + up
            elif filter_type == 3:
                value = value_at + ((left + up) // 2)
            elif filter_type == 4:
                predictor = left + up - upper_left
                pa = abs(predictor - left)
                pb = abs(predictor - up)
                pc = abs(predictor - upper_left)
                nearest = left if pa <= pb and pa <= pc else up if pb <= pc else upper_left
                value = value_at + nearest
            else:
                raise ValueError("unsupported PNG scanline filter")
            recon[column] = value & 0xFF
        prior = recon
    pixels = output.reshape(height, width, channels)
    return np.ascontiguousarray(pixels[..., :3])


def vision_prompt_markers(engine: Any) -> tuple[str, str]:
    """Prompt text an engine wants inserted where its media sits.

    The Qwen4Exp renderer splices features at a marker, so its prompt needs
    ``<|vision_start|><|image_pad|><|vision_end|>``. A model that renders its
    own placeholder from the image geometry (Surya OCR: ``render_chat_prompt``
    derives the image pad span from the patch grid) wants the bare text, so it
    declares ``vision_prompt_marker = ""``.

    An engine that declares the marker sets it for both media kinds unless it
    also declares ``vision_video_prompt_marker``; ``None`` and ``""`` both mean
    "insert nothing".
    """

    missing = object()
    declared = getattr(engine, "vision_prompt_marker", missing)
    if declared is missing:
        return _IMAGE_MARKER, _VIDEO_MARKER
    video_declared = getattr(engine, "vision_video_prompt_marker", declared)
    return (
        "" if declared is None else str(declared),
        "" if video_declared is None else str(video_declared),
    )


_MEDIA_PART_TYPES = frozenset({"image_url", "input_image", "video_frames"})


def request_has_media(messages: Sequence[Any]) -> bool:
    """Whether any message carries a typed media part.

    A structural check with no decoding, so a caller can decide whether a
    request needs a vision-capable engine before paying to resolve one.
    """

    for message in messages:
        content = (
            message.get("content")
            if isinstance(message, Mapping)
            else getattr(message, "content", None)
        )
        if isinstance(content, list) and any(
            isinstance(part, Mapping) and part.get("type") in _MEDIA_PART_TYPES
            for part in content
        ):
            return True
    return False


def extract_chat_media(
    messages: Sequence[Any],
    *,
    max_bytes: int = DEFAULT_VISION_MAX_BYTES,
    max_side: int = 1_024,
    max_pixels: int | None = None,
    image_marker: str = _IMAGE_MARKER,
    video_marker: str = _VIDEO_MARKER,
) -> tuple[str, dict[str, Any]] | None:
    """Extract one user multipart message into prompt text and typed media.

    Model-neutral: the caller supplies the bounds, because the admitted image
    size is an engine property (a document OCR model admits a page; the
    Qwen4Exp path admits a thumbnail), and the prompt markers, because whether
    the prompt needs an inline placeholder is a rendering property. The
    returned media is a neutral
    ``{"items": [{"type": "image", "data": <RGB array>}, ...]}`` mapping; use
    :func:`media_for_engine` to adapt it to what a given engine accepts.
    """

    if not request_has_media(messages):
        return None
    if len(messages) != 1:
        raise ValueError("HTTP multimodal chat currently requires one user message")
    message = messages[0]
    role = message.get("role") if isinstance(message, Mapping) else getattr(message, "role", None)
    content = message.get("content") if isinstance(message, Mapping) else getattr(message, "content", None)
    if role != "user" or not isinstance(content, list):
        raise ValueError("HTTP multimodal chat requires multipart user content")
    prompt_parts: list[str] = []
    items: list[dict[str, Any]] = []
    for index, part in enumerate(content):
        if not isinstance(part, Mapping):
            raise ValueError(f"multimodal content part {index} must be an object")
        kind = str(part.get("type", ""))
        if kind == "text":
            text = part.get("text")
            if not isinstance(text, str):
                raise ValueError(f"multimodal text part {index} requires text")
            prompt_parts.append(text)
        elif kind in {"image_url", "input_image"}:
            image_value = part.get("image_url", part.get("image"))
            if isinstance(image_value, Mapping):
                image_value = image_value.get("url")
            image = decode_bounded_png_data_url(
                image_value, max_bytes=max_bytes, max_side=max_side,
                max_pixels=max_pixels,
            )
            prompt_parts.append(image_marker)
            items.append({"type": "image", "data": image})
        elif kind == "video_frames":
            frames = part.get("frames")
            if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)) or not frames:
                raise ValueError(f"video_frames part {index} requires frame data URLs")
            decoded = [
                decode_bounded_png_data_url(
                    str(frame), max_bytes=max_bytes, max_side=max_side,
                    max_pixels=max_pixels,
                )
                for frame in frames
            ]
            if any(frame.shape != decoded[0].shape for frame in decoded):
                raise ValueError("video frames must share one RGB shape")
            prompt_parts.append(video_marker)
            items.append({"type": "video", "data": np.stack(decoded)})
        else:
            raise ValueError(f"unsupported multimodal content part type {kind!r}")
    if not items:
        return None
    return "".join(prompt_parts), {"items": items}


def media_for_engine(engine: Any, media: dict[str, Any]) -> Any:
    """Adapt extracted media to the form an engine's vision path accepts.

    An engine declares ``vision_media_input``:

    - ``"items"`` (or unset) — the neutral ``{"items": [...]}`` mapping, which
      the Qwen4Exp path unpacks itself.
    - ``"image_array"`` — a single RGB array, for a model whose generator takes
      one image (Surya OCR: one page per request).
    """

    form = getattr(engine, "vision_media_input", None) or "items"
    if form == "items":
        return media
    if form == "image_array":
        items = list(media.get("items", ()))
        images = [item for item in items if item.get("type") == "image"]
        if len(items) != 1 or len(images) != 1:
            raise ValueError(
                "this model accepts exactly one image per request"
            )
        return images[0]["data"]
    raise ValueError(f"unknown vision_media_input {form!r}")


__all__ = [
    "DEFAULT_VISION_MAX_BYTES",
    "DEFAULT_VISION_MAX_PIXELS",
    "MAX_VISION_MAX_BYTES",
    "decode_bounded_png_data_url",
    "extract_chat_media",
    "media_for_engine",
    "request_has_media",
    "resolve_vision_http_limits",
    "vision_max_side",
    "vision_prompt_markers",
]
