"""Torch-free Gemma 4 vision preprocessing: image -> patches, grid, positions.

The reference processor resizes each image to the largest aspect-ratio-preserving
size inside a patch budget, rescales ``[0, 255]`` to ``[0, 1]``, and emits
channel-last patches in row-major patch order.  This module does that with NumPy
and, only when a resize is needed, Pillow.

It does not pad to the patch budget: the attention kernel that consumes these
patches has no padding to mask, so exactly the image's own patches are returned.

**Patch flatten order.** :attr:`Gemma4VisionImage.patches` is channel-last,
``(patch_row, patch_column, channel)``, the reference processor's order.  GGUF
stores the patch kernel as ``(hidden, channel, patch_row, patch_column)``, so a
consumer that wants the kernel's order reads
:attr:`Gemma4VisionImage.patches_chw` and reshapes the kernel with no transpose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hipengine.loading.gemma4_vision_gguf import (
    DEFAULT_MAX_SOFT_TOKENS,
    DEFAULT_POOLING_KERNEL_SIZE,
)

GEMMA4_VISION_CHANNELS = 3
_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif", ".tif", ".tiff")


@dataclass(frozen=True)
class Gemma4VisionImage:
    """One preprocessed image: patches, patch grid, and patch positions."""

    patches: np.ndarray
    pos_x: np.ndarray
    pos_y: np.ndarray
    grid_size: tuple[int, int]
    target_size: tuple[int, int]
    patch_size: int
    pooling_kernel_size: int

    @property
    def pixel_values(self) -> np.ndarray:
        """Alias for :attr:`patches` under the reference processor's name."""

        return self.patches

    @property
    def patches_chw(self) -> np.ndarray:
        """Patches flattened as ``(channel, patch_row, patch_column)``.

        This is the GGUF patch-embedding kernel's own flatten order, so
        ``kernel.reshape(hidden, -1)`` consumes it without a transpose.
        """

        return gemma4_vision_patches_as_chw(self.patches, self.patch_size)

    @property
    def num_patches(self) -> int:
        return int(self.patches.shape[0])

    @property
    def num_soft_tokens(self) -> int:
        grid_height, grid_width = self.grid_size
        return (grid_height // self.pooling_kernel_size) * (
            grid_width // self.pooling_kernel_size
        )


def gemma4_vision_resize_target(
    height: int,
    width: int,
    *,
    patch_size: int,
    max_soft_tokens: int = DEFAULT_MAX_SOFT_TOKENS,
    pooling_kernel_size: int = DEFAULT_POOLING_KERNEL_SIZE,
) -> tuple[int, int]:
    """Aspect-ratio-preserving resize target inside the patch budget.

    Both sides land on a multiple of ``pooling_kernel_size * patch_size`` so the
    pooled grid is exact.  The rounding is the reference processor's: scale by
    ``sqrt(target_px / actual_px)``, round each side down, then repair a side
    that collapsed to zero.  Any positive ``max_soft_tokens`` is executable.
    """

    patch_size = _positive_int("patch_size", patch_size)
    pooling_kernel_size = _positive_int("pooling_kernel_size", pooling_kernel_size)
    max_soft_tokens = _positive_int("max_soft_tokens", max_soft_tokens)
    height = _positive_int("height", height)
    width = _positive_int("width", width)

    max_patches = max_soft_tokens * pooling_kernel_size * pooling_kernel_size
    target_px = max_patches * patch_size * patch_size
    total_px = height * width
    factor = math.sqrt(target_px / total_px)
    side_multiple = pooling_kernel_size * patch_size

    target_height = int(math.floor(factor * height / side_multiple)) * side_multiple
    target_width = int(math.floor(factor * width / side_multiple)) * side_multiple
    if target_height == 0 and target_width == 0:
        raise ValueError(
            f"resizing {height}x{width} rounds both sides to zero; sides must be at least "
            f"pooling_kernel_size * patch_size = {side_multiple} pixels"
        )

    max_side_length = (max_patches // (pooling_kernel_size**2)) * side_multiple
    if target_height == 0:
        target_height = side_multiple
        target_width = min(
            int(math.floor(width / height)) * side_multiple,
            max_side_length,
        )
    elif target_width == 0:
        target_width = side_multiple
        target_height = min(
            int(math.floor(height / width)) * side_multiple,
            max_side_length,
        )

    if target_height <= 0 or target_width <= 0:
        raise ValueError(
            f"resizing {height}x{width} produced an empty target "
            f"{target_height}x{target_width}"
        )
    if target_height * target_width > target_px:
        raise ValueError(
            f"resizing {height}x{width} to {target_height}x{target_width} exceeds "
            f"{max_patches} patches at patch size {patch_size}"
        )
    return target_height, target_width


def gemma4_vision_patch_positions(
    grid_height: int,
    grid_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Row-major ``(x, y)`` patch positions: x cycles fastest, then y."""

    grid_height = _positive_int("grid_height", grid_height)
    grid_width = _positive_int("grid_width", grid_width)
    pos_x = np.tile(np.arange(grid_width, dtype=np.int32), grid_height)
    pos_y = np.repeat(np.arange(grid_height, dtype=np.int32), grid_width)
    return pos_x, pos_y


def gemma4_vision_patches(pixels: np.ndarray, patch_size: int) -> np.ndarray:
    """Cut a rescaled ``(height, width, 3)`` image into channel-last patches."""

    patch_size = _positive_int("patch_size", patch_size)
    array = np.asarray(pixels)
    if array.ndim != 3 or array.shape[2] != GEMMA4_VISION_CHANNELS:
        raise ValueError(
            f"pixels must have shape (height, width, {GEMMA4_VISION_CHANNELS}), "
            f"got {array.shape}"
        )
    height, width = array.shape[:2]
    if height % patch_size or width % patch_size:
        raise ValueError(
            f"image {height}x{width} is not divisible by patch size {patch_size}"
        )
    grid_height, grid_width = height // patch_size, width // patch_size
    blocks = array.reshape(grid_height, patch_size, grid_width, patch_size, 3)
    patches = blocks.transpose(0, 2, 1, 3, 4)
    return np.ascontiguousarray(
        patches.reshape(grid_height * grid_width, patch_size * patch_size * 3)
    )


def gemma4_vision_patches_as_chw(patches: np.ndarray, patch_size: int) -> np.ndarray:
    """Re-flatten channel-last patches as ``(channel, row, column)`` rows."""

    patch_size = _positive_int("patch_size", patch_size)
    array = np.asarray(patches)
    values = GEMMA4_VISION_CHANNELS * patch_size * patch_size
    if array.ndim != 2 or array.shape[1] != values:
        raise ValueError(
            f"patches must have shape (num_patches, {values}), got {array.shape}"
        )
    blocks = array.reshape(array.shape[0], patch_size, patch_size, GEMMA4_VISION_CHANNELS)
    return np.ascontiguousarray(blocks.transpose(0, 3, 1, 2).reshape(array.shape[0], values))


def gemma4_vision_image_array(image: Any) -> np.ndarray:
    """Return an image as a contiguous ``uint8`` RGB ``(height, width, 3)`` array.

    Accepts a Pillow image, a path to an image file, or an integer RGB array.
    Floating-point arrays are rejected rather than guessed at: the reference
    pipeline rescales ``[0, 255]`` itself, so a float input would be ambiguous.
    """

    if isinstance(image, np.ndarray):
        array = image
        if not np.issubdtype(array.dtype, np.integer):
            raise ValueError(
                "image arrays must be integer RGB values in [0, 255]; "
                f"got dtype {array.dtype}"
            )
        if array.ndim == 2:
            raise ValueError(
                "image arrays must be RGB with shape (height, width, 3); got a grayscale array"
            )
        if array.ndim != 3 or array.shape[2] != GEMMA4_VISION_CHANNELS:
            raise ValueError(
                f"image arrays must have shape (height, width, {GEMMA4_VISION_CHANNELS}), "
                f"got {array.shape}"
            )
        if array.size and (int(array.min()) < 0 or int(array.max()) > 255):
            raise ValueError("image array values must lie in [0, 255]")
        return np.ascontiguousarray(array.astype(np.uint8, copy=False))

    pil = _open_rgb(image)
    return np.ascontiguousarray(np.asarray(pil, dtype=np.uint8))


def _open_rgb(image: Any) -> Any:
    """Open an image source as an RGB Pillow image (Pillow imported lazily)."""

    from PIL import Image

    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            return opened.convert("RGB")
    if hasattr(image, "shape") or hasattr(image, "__array__"):
        return Image.fromarray(gemma4_vision_image_array(image)).convert("RGB")
    raise TypeError(
        "image must be a Pillow image, a path to an image file, or an integer RGB array; "
        f"got {type(image).__name__}"
    )


def _resize_bicubic(pixels: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Bicubic resize of a ``uint8`` RGB array (Pillow imported lazily)."""

    from PIL import Image

    height, width = target_size
    resized = Image.fromarray(pixels).resize((width, height), Image.BICUBIC)
    return np.ascontiguousarray(np.asarray(resized, dtype=np.uint8))


def preprocess_gemma4_image(
    image: Any,
    *,
    patch_size: int,
    max_soft_tokens: int = DEFAULT_MAX_SOFT_TOKENS,
    pooling_kernel_size: int = DEFAULT_POOLING_KERNEL_SIZE,
    resize: bool = True,
) -> Gemma4VisionImage:
    """Preprocess one image into patches, grid, and patch positions.

    ``resize=False`` keeps the image's own size, requiring only that it divide by
    ``patch_size`` and that the patch grid divide by the pooling kernel size.
    """

    patch_size = _positive_int("patch_size", patch_size)
    pooling_kernel_size = _positive_int("pooling_kernel_size", pooling_kernel_size)
    max_soft_tokens = _positive_int("max_soft_tokens", max_soft_tokens)

    pixels = gemma4_vision_image_array(image)
    height, width = pixels.shape[:2]
    if resize:
        target_height, target_width = gemma4_vision_resize_target(
            height,
            width,
            patch_size=patch_size,
            max_soft_tokens=max_soft_tokens,
            pooling_kernel_size=pooling_kernel_size,
        )
        if (target_height, target_width) != (height, width):
            pixels = _resize_bicubic(pixels, (target_height, target_width))
            height, width = pixels.shape[:2]
    elif height % patch_size or width % patch_size:
        raise ValueError(
            f"image {height}x{width} is not divisible by patch size {patch_size} and "
            "resize is disabled"
        )

    grid_height, grid_width = height // patch_size, width // patch_size
    if grid_height % pooling_kernel_size or grid_width % pooling_kernel_size:
        raise ValueError(
            f"patch grid {grid_height}x{grid_width} is not divisible by the pooling "
            f"kernel size {pooling_kernel_size}"
        )

    rescaled = pixels.astype(np.float32) / np.float32(255.0)
    patches = gemma4_vision_patches(rescaled, patch_size)
    pos_x, pos_y = gemma4_vision_patch_positions(grid_height, grid_width)
    return Gemma4VisionImage(
        patches=patches,
        pos_x=pos_x,
        pos_y=pos_y,
        grid_size=(grid_height, grid_width),
        target_size=(height, width),
        patch_size=patch_size,
        pooling_kernel_size=pooling_kernel_size,
    )


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


__all__ = [
    "GEMMA4_VISION_CHANNELS",
    "Gemma4VisionImage",
    "gemma4_vision_image_array",
    "gemma4_vision_patch_positions",
    "gemma4_vision_patches",
    "gemma4_vision_patches_as_chw",
    "gemma4_vision_resize_target",
    "preprocess_gemma4_image",
]
