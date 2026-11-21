"""Torch-free Inkling 2 vision encoder for one image."""

from __future__ import annotations

from math import sqrt
from pathlib import Path

import numpy as np

from hipengine.kernels.cpu_reference.muse_glimmer_vision import gelu_erf, rope_2d
from hipengine.loading.inkling2_mini import Inkling2TextWeights
from hipengine.runtime.inkling2_mini_reference import _softmax


def _rgb(image):
    if isinstance(image, (str, Path)):
        from PIL import Image

        with Image.open(image) as opened:
            return _rgb(opened)
    if not isinstance(image, np.ndarray):
        image = np.asarray(image.convert("RGBA" if "A" in image.getbands() or "transparency" in image.info else "RGB"))
    if image.ndim != 3 or image.shape[-1] not in (3, 4) or not all(image.shape[:2]):
        raise ValueError("Inkling 2 requires a nonempty RGB or RGBA image")
    if not np.issubdtype(image.dtype, np.integer) or image.min() < 0 or image.max() > 255:
        raise ValueError("Inkling 2 image pixels must be integers in [0, 255]")
    if image.shape[-1] == 4:
        h, w = image.shape[:2]
        y, x = np.ogrid[:h, :w]
        bg = np.where((y // 8 + x // 8) % 2 == 0, 204, 153)[..., None]
        rgba = image.astype(np.int32)
        alpha = rgba[..., 3:4]
        image = (rgba[..., :3] * alpha + bg * (255 - alpha) + 127) // 255
    return image


def prepare_vision_patches(image, *, patch, merge):
    rgb = _rgb(image)
    h, w = rgb.shape[:2]
    stride = merge * patch
    rows, cols = (h + stride - 1) // stride, (w + stride - 1) // stride
    pixels = np.full((rows * stride, cols * stride, 3), -1, dtype=np.float32)
    pixels[:h, :w] = rgb
    pixels = pixels * (2 / 255) - 1
    patches = pixels.reshape(rows, merge, patch, cols, merge, patch, 3).transpose(0, 3, 1, 4, 2, 5, 6)
    coords = np.stack(np.indices((rows * merge, cols * merge)), axis=-1)
    coords = coords.reshape(rows, merge, cols, merge, 2).transpose(0, 2, 1, 3, 4).reshape(-1, 2)
    return patches.reshape(-1, patch * patch * 3), coords, (rows, cols)


def interpolate_positions(table, coords, shape, side):
    scaled = (coords.astype(np.float32) + 0.5) * (side / np.asarray(shape, dtype=np.float32)) - 0.5
    lo = np.floor(scaled).astype(np.intp)
    mix = scaled - lo.astype(np.float32)
    grid = table.reshape(side, side, -1)
    out = np.zeros((len(coords), grid.shape[-1]), dtype=np.float32)
    for dy in range(2):
        for dx in range(2):
            y = np.clip(lo[:, 0] + dy, 0, side - 1)
            x = np.clip(lo[:, 1] + dx, 0, side - 1)
            wy = mix[:, 0] if dy else 1 - mix[:, 0]
            wx = mix[:, 1] if dx else 1 - mix[:, 1]
            out += (wy * wx)[:, None] * grid[y, x]
    return out


class Inkling2VisionReference:
    def __init__(self, index):
        self.cfg = index.config["vision_config"]
        self.weights = Inkling2TextWeights(index, namespace="model.visual")

    def project(self, name, rows):
        return rows @ self.weights.dense(name).T

    def attend(self, q, k, v, *, check=lambda: None):
        q, k, v = q.transpose(1, 0, 2), k.transpose(1, 2, 0), v.transpose(1, 0, 2)
        y = np.empty_like(q)
        for start in range(0, q.shape[1], 32):
            check()
            tile = slice(start, start + 32)
            y[:, tile] = _softmax((q[:, tile] @ k) / sqrt(q.shape[-1])) @ v
        return y.transpose(1, 0, 2)

    def encode(self, image, *, check=lambda: None):
        c, read = self.cfg, self.weights.dense
        patch = c.get("native_patch_size", c.get("patch_size", 20))
        merge = c.get("adapter_merge_size", c.get("spatial_merge_size", 2))
        patches, coords, (rows, cols) = prepare_vision_patches(image, patch=patch, merge=merge)
        emb = "backbone.embeddings."
        x = self.project(emb + "patch_embedding.weight", patches)
        x += interpolate_positions(read(emb + "position_embedding"), coords, (rows * merge, cols * merge), c["posemb_grid_size"])
        cls = read(emb + "class_embedding") + read(emb + "cls_position_embedding")
        x = np.concatenate((cls[None], x), axis=0)
        pos_y, pos_x = np.concatenate((np.zeros((1, 2), dtype=np.intp), coords + 1)).T
        eps = c["layer_norm_eps"]
        heads = c["num_attention_heads"]
        dim = c["hidden_size"] // heads
        theta = (c.get("rope_parameters") or {}).get("rope_theta", 10000.0)

        def norm(z, stem):
            z = z - z.mean(axis=-1, keepdims=True)
            return z / np.sqrt(np.mean(z * z, axis=-1, keepdims=True) + eps) * read(stem + ".weight") + read(stem + ".bias")

        def linear(z, stem, bias=True):
            result = self.project(stem + ".weight", z)
            return result + read(stem + ".bias") if bias else result

        x = norm(x, "backbone.ln_pre")
        for i in range(c["num_hidden_layers"]):
            check()
            stem = f"backbone.layers.{i}."
            h = norm(x, stem + "ln_1")
            q, k, v = (linear(h, stem + "self_attn." + name + "_proj").reshape(-1, heads, dim)
                       for name in ("q", "k", "v"))
            q = rope_2d(q, pos_x, pos_y, dim, theta)
            k = rope_2d(k, pos_x, pos_y, dim, theta)
            y = self.attend(q, k, v, check=check).reshape(x.shape)
            x += linear(y, stem + "self_attn.out_proj") * read(stem + "ls_1.gamma")
            h = gelu_erf(linear(norm(x, stem + "ln_2"), stem + "mlp.c_fc"))
            x += linear(h, stem + "mlp.c_proj") * read(stem + "ls_2.gamma")
        merged = x[1:].reshape(rows * cols, merge**2 * x.shape[-1])
        bias = c.get("projector_bias", False)
        return linear(gelu_erf(linear(merged, "proj_in", bias)), "proj_out", bias)
