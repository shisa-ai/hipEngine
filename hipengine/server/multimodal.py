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


def decode_bounded_png_data_url(
    value: str, *, max_bytes: int = 8 * 1024 * 1024, max_side: int = 1_024
) -> np.ndarray:
    """Decode an 8-bit non-interlaced RGB/RGBA PNG data URL."""

    prefix = "data:image/png;base64,"
    if not isinstance(value, str) or not value.startswith(prefix):
        raise ValueError("Qwen4Exp HTTP vision requires a base64 image/png data URL")
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


def extract_qwen4_exp_chat_media(
    messages: Sequence[Any],
) -> tuple[str, dict[str, Any]] | None:
    """Extract one user multipart message into prompt text and typed media."""

    multimedia = False
    for message in messages:
        content = message.get("content") if isinstance(message, Mapping) else getattr(message, "content", None)
        if isinstance(content, list) and any(
            isinstance(part, Mapping)
            and part.get("type") in {"image_url", "input_image", "video_frames"}
            for part in content
        ):
            multimedia = True
            break
    if not multimedia:
        return None
    if len(messages) != 1:
        raise ValueError("Qwen4Exp HTTP multimodal chat currently requires one user message")
    message = messages[0]
    role = message.get("role") if isinstance(message, Mapping) else getattr(message, "role", None)
    content = message.get("content") if isinstance(message, Mapping) else getattr(message, "content", None)
    if role != "user" or not isinstance(content, list):
        raise ValueError("Qwen4Exp HTTP multimodal chat requires multipart user content")
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
            image = decode_bounded_png_data_url(image_value)
            prompt_parts.append(_IMAGE_MARKER)
            items.append({"type": "image", "data": image})
        elif kind == "video_frames":
            frames = part.get("frames")
            if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)) or not frames:
                raise ValueError(f"video_frames part {index} requires frame data URLs")
            decoded = [decode_bounded_png_data_url(str(frame)) for frame in frames]
            if any(frame.shape != decoded[0].shape for frame in decoded):
                raise ValueError("video frames must share one RGB shape")
            prompt_parts.append(_VIDEO_MARKER)
            items.append({"type": "video", "data": np.stack(decoded)})
        else:
            raise ValueError(f"unsupported multimodal content part type {kind!r}")
    if not items:
        return None
    return "".join(prompt_parts), {"items": items}


__all__ = ["decode_bounded_png_data_url", "extract_qwen4_exp_chat_media"]
