"""Gemma 4 GGUF vision tower: metadata contract, preprocessing, and NumPy oracle.

Pinned on the CPU against a synthetic mmproj file, so no artifact is needed:

* the contract derives dimensions from ``clip.vision.*`` metadata and shapes
  from those dimensions, refusing an artifact it cannot execute with a named
  error rather than by model identity;
* the oracle matches a separately written loop reference through the real
  ``scan_gguf`` -> validate -> read -> forward path, and each stage matches a
  hand-computed value;
* preprocessing emits row-major channel-last patches with ``x`` fastest.

The arithmetic reference is the transformers ``Gemma4VisionModel`` forward:
RMSNorm with a learned scale where the checkpoint has one, non-causal attention
with **no** ``1/sqrt(head_dim)`` factor, per-head Q/K RMSNorm, a scale-free V
RMSNorm, two-axis NeoX RoPE, an average pool scaled by ``sqrt(hidden_size)``,
optional standardization, a scale-free RMSNorm, and the input projection.
"""

from __future__ import annotations

import math
import struct
from types import SimpleNamespace
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from hipengine.kernels.cpu_reference.gemma4_vision import (
    PATCH_ORDER_CHANNEL_FIRST,
    Gemma4VisionGeometry,
    gemma4_vision_apply_rope,
    gemma4_vision_attention,
    gemma4_vision_forward,
    gemma4_vision_forward_positions,
    gemma4_vision_gelu_tanh,
    gemma4_vision_geometry_from_config,
    gemma4_vision_patch_embed,
    gemma4_vision_pool,
    gemma4_vision_rms_norm,
    gemma4_vision_rope_inv_freq,
    gemma4_vision_standardize,
    gemma4_vision_weight_shapes,
    gemma4_vision_weights_from_slots,
)
from hipengine.loading.gemma4_image import (
    gemma4_vision_patch_positions,
    gemma4_vision_patches,
    gemma4_vision_patches_as_chw,
    gemma4_vision_resize_target,
    preprocess_gemma4_image,
)
from hipengine.loading.gemma4_vision_gguf import (
    Gemma4VisionGGUFError,
    build_gemma4_vision_gguf_map,
    gemma4_vision_config_from_metadata,
    gemma4_vision_expected_shapes,
    gemma4_vision_layer_slot,
    read_gemma4_vision_weights,
    validate_gemma4_vision_gguf,
)
from hipengine.loading.gguf import scan_gguf


def assert_allclose(actual: object, expected: object, *, rtol: float = 1e-9, atol: float = 1e-12) -> None:
    """Element-wise comparison that also accepts nested Python sequences."""

    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), rtol=rtol, atol=atol
    )

# ---------------------------------------------------------------------------
# Synthetic mmproj artifact
# ---------------------------------------------------------------------------

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
ALIGNMENT = 32
F32 = 0

FIXTURE_BLOCK_COUNT = 2
FIXTURE_HIDDEN = 8
FIXTURE_INTERMEDIATE = 6
FIXTURE_OUTPUT = 5
FIXTURE_HEADS = 2
FIXTURE_KV_HEADS = 2
FIXTURE_HEAD_DIM = FIXTURE_HIDDEN // FIXTURE_HEADS
FIXTURE_PATCH = 2
FIXTURE_POOLING = 2
FIXTURE_POSITION_TABLE = 160
FIXTURE_MAX_SOFT_TOKENS = 70
FIXTURE_IMAGE_SIZE = 8
CLAMP_FIELDS = ("input_min", "input_max", "output_min", "output_max")


def _gguf_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _gguf_scalar(value_type: int, value: object) -> bytes:
    formats = {4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q"}
    return struct.pack(formats[value_type], value)


def _gguf_value(value_type: int, value: object) -> bytes:
    if value_type == 8:
        return _gguf_string(str(value))
    if value_type == 9:
        item_type, items = value
        return struct.pack("<IQ", item_type, len(items)) + b"".join(
            _gguf_value(item_type, item) for item in items
        )
    return _gguf_scalar(value_type, value)


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def fixture_metadata(**overrides: object) -> list[tuple[str, int, object]]:
    metadata: list[tuple[str, int, object]] = [
        ("general.architecture", 8, "clip"),
        ("general.alignment", 4, ALIGNMENT),
        ("clip.has_vision_encoder", 7, True),
        ("clip.vision.projector_type", 8, "gemma4v"),
        ("clip.vision.block_count", 4, FIXTURE_BLOCK_COUNT),
        ("clip.vision.embedding_length", 4, FIXTURE_HIDDEN),
        ("clip.vision.feed_forward_length", 4, FIXTURE_INTERMEDIATE),
        ("clip.vision.projection_dim", 4, FIXTURE_OUTPUT),
        ("clip.vision.attention.head_count", 4, FIXTURE_HEADS),
        ("clip.vision.attention.head_count_kv", 4, FIXTURE_KV_HEADS),
        ("clip.vision.attention.layer_norm_epsilon", 6, 1.0e-6),
        ("clip.vision.patch_size", 4, FIXTURE_PATCH),
        ("clip.vision.image_size", 4, FIXTURE_IMAGE_SIZE),
        ("clip.vision.projector.scale_factor", 4, FIXTURE_POOLING),
        ("clip.vision.max_soft_tokens", 4, FIXTURE_MAX_SOFT_TOKENS),
        ("clip.vision.image_mean", 9, (6, [0.0, 0.0, 0.0])),
        ("clip.vision.image_std", 9, (6, [1.0, 1.0, 1.0])),
    ]
    for key, value in overrides.items():
        if value is None:
            metadata = [entry for entry in metadata if entry[0] != key]
            continue
        if isinstance(value, bool):
            entry = (key, 7, value)
        elif isinstance(value, str):
            entry = (key, 8, value)
        elif isinstance(value, float):
            entry = (key, 6, value)
        else:
            entry = (key, 4, int(value))
        metadata = [existing for existing in metadata if existing[0] != key]
        metadata.append(entry)
    return metadata


def fixture_tensor_shapes() -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {
        "v.patch_embd.weight": (
            FIXTURE_HIDDEN,
            3,
            FIXTURE_PATCH,
            FIXTURE_PATCH,
        ),
        "v.position_embd.weight": (2, FIXTURE_POSITION_TABLE, FIXTURE_HIDDEN),
        "mm.input_projection.weight": (FIXTURE_OUTPUT, FIXTURE_HIDDEN),
        "v.std_bias": (FIXTURE_HIDDEN,),
        "v.std_scale": (FIXTURE_HIDDEN,),
    }
    for layer_id in range(FIXTURE_BLOCK_COUNT):
        prefix = f"v.blk.{layer_id}."
        shapes.update(
            {
                prefix + "ln1.weight": (FIXTURE_HIDDEN,),
                prefix + "ln2.weight": (FIXTURE_HIDDEN,),
                prefix + "attn_q.weight": (
                    FIXTURE_HEADS * FIXTURE_HEAD_DIM,
                    FIXTURE_HIDDEN,
                ),
                prefix + "attn_k.weight": (
                    FIXTURE_KV_HEADS * FIXTURE_HEAD_DIM,
                    FIXTURE_HIDDEN,
                ),
                prefix + "attn_v.weight": (
                    FIXTURE_KV_HEADS * FIXTURE_HEAD_DIM,
                    FIXTURE_HIDDEN,
                ),
                prefix + "attn_out.weight": (
                    FIXTURE_HIDDEN,
                    FIXTURE_HEADS * FIXTURE_HEAD_DIM,
                ),
                prefix + "attn_q_norm.weight": (FIXTURE_HEAD_DIM,),
                prefix + "attn_k_norm.weight": (FIXTURE_HEAD_DIM,),
                prefix + "attn_post_norm.weight": (FIXTURE_HIDDEN,),
                prefix + "ffn_post_norm.weight": (FIXTURE_HIDDEN,),
                prefix + "ffn_gate.weight": (FIXTURE_INTERMEDIATE, FIXTURE_HIDDEN),
                prefix + "ffn_up.weight": (FIXTURE_INTERMEDIATE, FIXTURE_HIDDEN),
                prefix + "ffn_down.weight": (FIXTURE_HIDDEN, FIXTURE_INTERMEDIATE),
            }
        )
    return shapes


def fixture_payloads() -> dict[str, bytes]:
    """Deterministic F32 payloads, so the oracle sees exactly these weights."""

    rng = np.random.default_rng(20260226)
    payloads: dict[str, bytes] = {}
    for name, shape in fixture_tensor_shapes().items():
        if name.endswith("norm.weight") or name in ("v.std_bias",):
            values = 1.0 + 0.1 * rng.standard_normal(shape)
        elif name == "v.std_scale":
            values = np.exp(0.1 * rng.standard_normal(shape))
        else:
            values = 0.3 * rng.standard_normal(shape)
        payloads[name] = values.astype(np.float32).tobytes()
    return payloads


def write_fixture_gguf(
    path: Path,
    *,
    metadata: list[tuple[str, int, object]] | None = None,
    payloads: dict[str, bytes] | None = None,
    shapes: dict[str, tuple[int, ...]] | None = None,
    drop: tuple[str, ...] = (),
    extra: dict[str, tuple[tuple[int, ...], bytes]] | None = None,
) -> Path:
    """Write a structurally real mmproj GGUF with F32 tensors."""

    metadata = fixture_metadata() if metadata is None else metadata
    payloads = fixture_payloads() if payloads is None else payloads
    shapes = fixture_tensor_shapes() if shapes is None else shapes
    records = {
        name: (shape, payloads[name]) for name, shape in shapes.items() if name not in drop
    }
    for name, (shape, payload) in (extra or {}).items():
        records[name] = (shape, payload)

    tensor_blob = bytearray()
    entries: list[tuple[str, tuple[int, ...], int]] = []
    for name, (shape, payload) in records.items():
        offset = _align_up(len(tensor_blob), ALIGNMENT)
        tensor_blob += b"\x00" * (offset - len(tensor_blob))
        entries.append((name, tuple(reversed(shape)), offset))
        tensor_blob += payload

    header = bytearray()
    header += GGUF_MAGIC
    header += struct.pack("<IQQ", GGUF_VERSION, len(entries), len(metadata))
    for key, value_type, value in metadata:
        header += _gguf_string(key) + struct.pack("<I", value_type) + _gguf_value(
            value_type, value
        )
    for name, ggml_shape, offset in entries:
        header += _gguf_string(name)
        header += struct.pack("<I", len(ggml_shape))
        header += struct.pack(f"<{len(ggml_shape)}Q", *ggml_shape)
        header += struct.pack("<IQ", F32, offset)
    data_start = _align_up(len(header), ALIGNMENT)
    header += b"\x00" * (data_start - len(header))
    path.write_bytes(bytes(header) + bytes(tensor_blob))
    return path


@pytest.fixture()
def mmproj_path(tmp_path: Path) -> Path:
    return write_fixture_gguf(tmp_path / "mmproj-F16.gguf")


# ---------------------------------------------------------------------------
# Independent loop-based reference
# ---------------------------------------------------------------------------


def _ref_rms(x: np.ndarray, weight: np.ndarray | None, eps: float) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float64)
    flat = x.reshape(-1, x.shape[-1])
    target = out.reshape(-1, x.shape[-1])
    for row in range(flat.shape[0]):
        mean_square = sum(float(value) * float(value) for value in flat[row]) / flat.shape[1]
        scale = 1.0 / math.sqrt(mean_square + eps)
        for col in range(flat.shape[1]):
            value = float(flat[row][col]) * scale
            if weight is not None:
                value *= float(weight[col])
            target[row][col] = value
    return out


def _ref_rope(
    block: np.ndarray,
    positions: np.ndarray,
    inv_freq: np.ndarray,
) -> np.ndarray:
    """NeoX pair rotation of one spatial half, one patch and pair at a time."""

    out = np.array(block, dtype=np.float64, copy=True)
    pairs = inv_freq.size
    for patch in range(block.shape[0]):
        for pair in range(pairs):
            angle = float(positions[patch]) * float(inv_freq[pair])
            cos, sin = math.cos(angle), math.sin(angle)
            first = float(block[patch, pair])
            second = float(block[patch, pair + pairs])
            out[patch, pair] = first * cos - second * sin
            out[patch, pair + pairs] = first * sin + second * cos
    return out


def _ref_forward(
    slots: dict[str, np.ndarray],
    patches: np.ndarray,
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    geometry: Gemma4VisionGeometry,
) -> dict[str, np.ndarray]:
    """Loop-based transcription of the reference vision tower."""

    kernel = slots["patch_embd"].astype(np.float64)
    table = slots["position_embd"].astype(np.float64)
    num_patches = patches.shape[0]
    hidden_size = geometry.hidden_size
    patch_size = geometry.patch_size

    hidden = np.zeros((num_patches, hidden_size), dtype=np.float64)
    for patch in range(num_patches):
        values = patches[patch].astype(np.float64)
        for out_channel in range(hidden_size):
            total = 0.0
            index = 0
            for channel in range(3):
                for row in range(patch_size):
                    for col in range(patch_size):
                        # patches are channel-last: (row, col, channel)
                        pixel = values[(row * patch_size + col) * 3 + channel]
                        total += float(kernel[out_channel, channel, row, col]) * (
                            2.0 * (pixel - 0.5)
                        )
                        index += 1
            hidden[patch, out_channel] = total
    hidden += table[0][pos_x] + table[1][pos_y]

    inv_freq = gemma4_vision_rope_inv_freq(geometry.head_dim, geometry.rope_theta)
    half = geometry.head_dim // 2
    eps = geometry.rms_norm_eps

    for layer_id in range(geometry.block_count):
        slot = lambda name: slots[gemma4_vision_layer_slot(layer_id, name)]  # noqa: E731
        query_width = geometry.heads * geometry.head_dim

        normed = _ref_rms(hidden, slot("ln1"), eps)
        query = np.zeros((num_patches, geometry.heads, geometry.head_dim))
        key = np.zeros((num_patches, geometry.head_count_kv, geometry.head_dim))
        value = np.zeros((num_patches, geometry.head_count_kv, geometry.head_dim))
        for patch in range(num_patches):
            for head in range(geometry.heads):
                raw = np.array(
                    [
                        sum(
                            float(slot("attn_q")[head * geometry.head_dim + d, j])
                            * normed[patch, j]
                            for j in range(hidden_size)
                        )
                        for d in range(geometry.head_dim)
                    ]
                )
                raw = _ref_rms(raw, slot("attn_q_norm"), eps)
                rotated = np.concatenate(
                    (
                        _ref_rope(raw[None, :half], pos_x[patch : patch + 1], inv_freq)[0],
                        _ref_rope(raw[None, half:], pos_y[patch : patch + 1], inv_freq)[0],
                    )
                )
                query[patch, head] = rotated
            for head in range(geometry.head_count_kv):
                raw = np.array(
                    [
                        sum(
                            float(slot("attn_k")[head * geometry.head_dim + d, j])
                            * normed[patch, j]
                            for j in range(hidden_size)
                        )
                        for d in range(geometry.head_dim)
                    ]
                )
                raw = _ref_rms(raw, slot("attn_k_norm"), eps)
                key[patch, head] = np.concatenate(
                    (
                        _ref_rope(raw[None, :half], pos_x[patch : patch + 1], inv_freq)[0],
                        _ref_rope(raw[None, half:], pos_y[patch : patch + 1], inv_freq)[0],
                    )
                )
                raw_value = np.array(
                    [
                        sum(
                            float(slot("attn_v")[head * geometry.head_dim + d, j])
                            * normed[patch, j]
                            for j in range(hidden_size)
                        )
                        for d in range(geometry.head_dim)
                    ]
                )
                value[patch, head] = _ref_rms(raw_value, None, eps)

        repeats = geometry.heads // geometry.head_count_kv
        key = np.repeat(key, repeats, axis=1)
        value = np.repeat(value, repeats, axis=1)

        context = np.zeros((num_patches, geometry.heads, geometry.head_dim))
        for head in range(geometry.heads):
            scores = np.zeros((num_patches, num_patches))
            for i in range(num_patches):
                for j in range(num_patches):
                    # no 1/sqrt(head_dim): the reference vision graph uses 1.0
                    scores[i, j] = sum(
                        query[i, head, d] * key[j, head, d]
                        for d in range(geometry.head_dim)
                    )
            for i in range(num_patches):
                row = scores[i]
                weights = np.exp(row - row.max())
                weights = weights / weights.sum()
                for d in range(geometry.head_dim):
                    context[i, head, d] = sum(
                        weights[j] * value[j, head, d] for j in range(num_patches)
                    )
        flat = context.reshape(num_patches, query_width)
        projected = np.zeros((num_patches, hidden_size))
        for patch in range(num_patches):
            for row in range(hidden_size):
                projected[patch, row] = sum(
                    float(slot("attn_out")[row, col]) * flat[patch, col]
                    for col in range(query_width)
                )
        hidden = hidden + _ref_rms(projected, slot("attn_post_norm"), eps)

        normed = _ref_rms(hidden, slot("ln2"), eps)
        gate = np.zeros((num_patches, geometry.intermediate_size))
        up = np.zeros((num_patches, geometry.intermediate_size))
        for patch in range(num_patches):
            for row in range(geometry.intermediate_size):
                gate_value = sum(
                    float(slot("ffn_gate")[row, col]) * normed[patch, col]
                    for col in range(hidden_size)
                )
                up[patch, row] = sum(
                    float(slot("ffn_up")[row, col]) * normed[patch, col]
                    for col in range(hidden_size)
                )
                gate[patch, row] = (
                    0.5
                    * gate_value
                    * (
                        1.0
                        + math.tanh(
                            math.sqrt(2.0 / math.pi)
                            * (gate_value + 0.044715 * gate_value**3)
                        )
                    )
                )
        down = np.zeros((num_patches, hidden_size))
        for patch in range(num_patches):
            for row in range(hidden_size):
                down[patch, row] = sum(
                    float(slot("ffn_down")[row, col]) * gate[patch, col] * up[patch, col]
                    for col in range(geometry.intermediate_size)
                )
        hidden = hidden + _ref_rms(down, slot("ffn_post_norm"), eps)

    # Pool by grouping patches on their (x, y) block index, not by reshaping.
    # The reference processor's kernel index is (x // k) + (grid_width // k) *
    # (y // k), so pooled tokens are block-row-major: y outer, x inner.
    kernel_size = geometry.pooling_kernel_size
    blocks: dict[tuple[int, int], list[int]] = {}
    for patch in range(num_patches):
        block = (int(pos_x[patch]) // kernel_size, int(pos_y[patch]) // kernel_size)
        blocks.setdefault(block, []).append(patch)
    order = sorted(blocks, key=lambda block: (block[1], block[0]))
    pooled = np.zeros((len(order), hidden_size))
    for index, block in enumerate(order):
        members = blocks[block]
        for col in range(hidden_size):
            pooled[index, col] = sum(hidden[member, col] for member in members) / len(members)
    pooled = pooled * math.sqrt(hidden_size)

    features = (pooled - slots["std_bias"]) * slots["std_scale"]
    features = _ref_rms(features, None, eps)
    embeddings = np.zeros((features.shape[0], geometry.output_size))
    for token in range(features.shape[0]):
        for row in range(geometry.output_size):
            embeddings[token, row] = sum(
                float(slots["input_projection"][row, col]) * features[token, col]
                for col in range(hidden_size)
            )
    return {"embeddings": embeddings, "hidden_states": features, "pooled": pooled}


# ---------------------------------------------------------------------------
# Metadata and tensor contract
# ---------------------------------------------------------------------------


def test_config_is_derived_from_metadata(mmproj_path: Path) -> None:
    info = scan_gguf(mmproj_path)
    config = gemma4_vision_config_from_metadata(info)

    assert config.architecture == "clip"
    assert config.projector_type == "gemma4v"
    assert config.block_count == FIXTURE_BLOCK_COUNT
    assert config.hidden_size == FIXTURE_HIDDEN
    assert config.intermediate_size == FIXTURE_INTERMEDIATE
    assert config.output_size == FIXTURE_OUTPUT
    assert config.heads == FIXTURE_HEADS
    assert config.head_dim == FIXTURE_HEAD_DIM
    assert config.patch_size == FIXTURE_PATCH
    assert config.pooling_kernel_size == FIXTURE_POOLING
    assert config.position_table_size == FIXTURE_POSITION_TABLE
    assert config.max_soft_tokens == FIXTURE_MAX_SOFT_TOKENS
    assert config.standardize is True
    assert config.clipped_linears is False
    # No GGUF key carries the vision RoPE theta; the reference tower uses 100.0.
    assert config.rope_theta == pytest.approx(100.0)
    assert config.max_grid_side == FIXTURE_MAX_SOFT_TOKENS * FIXTURE_POOLING


def test_validation_accepts_the_declared_artifact(mmproj_path: Path) -> None:
    info = scan_gguf(mmproj_path)
    validation = validate_gemma4_vision_gguf(info)

    assert validation.passed, validation
    assert validation.missing == ()
    assert validation.unexpected == ()
    assert validation.shape_errors == ()
    assert validation.qtype_errors == ()
    model_map = build_gemma4_vision_gguf_map(info)
    assert model_map.tensor("patch_embd").name == "v.patch_embd.weight"
    assert model_map.tensor(gemma4_vision_layer_slot(1, "ffn_down")).name == (
        "v.blk.1.ffn_down.weight"
    )


def test_pooling_and_position_defaults_when_metadata_omits_them(tmp_path: Path) -> None:
    shapes = fixture_tensor_shapes()
    payloads = fixture_payloads()
    shapes["v.position_embd.weight"] = (2, 1024, FIXTURE_HIDDEN)
    payloads["v.position_embd.weight"] = np.zeros(
        (2, 1024, FIXTURE_HIDDEN), dtype=np.float32
    ).tobytes()
    path = write_fixture_gguf(
        tmp_path / "mmproj.gguf",
        metadata=fixture_metadata(
            **{
                "clip.vision.projector.scale_factor": None,
                "clip.vision.max_soft_tokens": None,
            }
        ),
        shapes=shapes,
        payloads=payloads,
    )
    config = gemma4_vision_config_from_metadata(scan_gguf(path))

    # The reference graph defaults to a 3x3 pool and 280 soft tokens.
    assert config.pooling_kernel_size == 3
    assert config.max_soft_tokens == 280
    assert config.position_table_size == 1024
    assert config.max_grid_side == 280 * 3


def test_small_position_table_is_admitted_but_refuses_a_real_grid(
    tmp_path: Path,
) -> None:
    # Admission does not gate on a hypothetical aspect ratio: the fixture's
    # 160-entry table is accepted even though 280 soft tokens and a 3x3 pool
    # could ask for a 840-patch side.  A real grid past the table is refused.
    path = write_fixture_gguf(
        tmp_path / "mmproj.gguf",
        metadata=fixture_metadata(
            **{
                "clip.vision.projector.scale_factor": None,
                "clip.vision.max_soft_tokens": None,
            }
        ),
    )
    validation = validate_gemma4_vision_gguf(scan_gguf(path))

    assert validation.passed, validation
    config = gemma4_vision_config_from_metadata(scan_gguf(path))
    assert config.position_table_size == FIXTURE_POSITION_TABLE
    assert config.max_grid_side == 280 * 3

    # A grid inside the table is fine; one pooling block past it is not.
    config.check_grid(159, 3)
    assert config.num_soft_tokens((159, 3)) == 53
    with pytest.raises(Gemma4VisionGGUFError, match="exceeds the position table"):
        config.check_grid(162, 3)
    with pytest.raises(Gemma4VisionGGUFError, match="exceeds the position table"):
        config.num_soft_tokens((3, 162))

    # The forward refuses the same grid from the table it was handed.
    positions = np.full((4, 2), FIXTURE_POSITION_TABLE + 1, dtype=np.int64)
    with pytest.raises(ValueError, match="exceed the position table size"):
        gemma4_vision_patch_embed(
            np.zeros((4, config.patch_values), dtype=np.float32),
            np.zeros((config.hidden_size, 3, config.patch_size, config.patch_size)),
            np.zeros((2, FIXTURE_POSITION_TABLE, config.hidden_size)),
            positions[:, 0],
            positions[:, 1],
        )


@pytest.mark.parametrize(
    ("override", "fragment"),
    [
        ({"clip.vision.projector_type": "gemma4uv"}, "gemma4uv"),
        ({"general.architecture": "llama"}, "mmproj artifact"),
        ({"clip.has_vision_encoder": False}, "has_vision_encoder"),
        ({"clip.vision.block_count": None}, "clip.vision.block_count"),
        ({"clip.vision.attention.head_count": 3}, "divisible by 3 attention heads"),
        (
            {
                "clip.vision.embedding_length": 12,
                "clip.vision.attention.head_count": 2,
                "clip.vision.attention.head_count_kv": 2,
            },
            "multiple of 4",
        ),
        ({"clip.vision.max_soft_tokens": 0}, "max_soft_tokens"),
        ({"clip.vision.projector.scale_factor": 0}, "must be positive"),
        ({"clip.vision.attention.layer_norm_epsilon": -1.0}, "must be positive"),
        ({"clip.vision.patch_size": 3}, "multiple of patch size"),
    ],
)
def test_bad_metadata_is_named(
    tmp_path: Path, override: dict[str, object], fragment: str
) -> None:
    path = write_fixture_gguf(
        tmp_path / "mmproj.gguf", metadata=fixture_metadata(**override)
    )
    validation = validate_gemma4_vision_gguf(scan_gguf(path))

    assert not validation.passed
    assert validation.config is None
    assert any(fragment in error for error in validation.metadata_errors), (
        validation.metadata_errors
    )
    with pytest.raises(Gemma4VisionGGUFError, match=fragment.split()[0]):
        gemma4_vision_config_from_metadata(scan_gguf(path))


def test_missing_and_unexpected_tensors_are_reported(tmp_path: Path) -> None:
    path = write_fixture_gguf(
        tmp_path / "mmproj.gguf",
        drop=("v.blk.1.attn_k_norm.weight",),
        extra={"v.blk.0.bogus.weight": ((FIXTURE_HIDDEN,), b"\x00" * (4 * FIXTURE_HIDDEN))},
    )
    validation = validate_gemma4_vision_gguf(scan_gguf(path))

    assert not validation.passed
    assert validation.missing == ("v.blk.1.attn_k_norm.weight",)
    assert validation.unexpected == ("v.blk.0.bogus.weight",)
    with pytest.raises(Gemma4VisionGGUFError):
        build_gemma4_vision_gguf_map(scan_gguf(path))


def test_bad_tensor_shapes_are_reported(tmp_path: Path) -> None:
    shapes = fixture_tensor_shapes()
    payloads = fixture_payloads()
    shapes["v.blk.0.attn_q.weight"] = (FIXTURE_HIDDEN, FIXTURE_HIDDEN - 2)
    payloads["v.blk.0.attn_q.weight"] = np.zeros(
        (FIXTURE_HIDDEN, FIXTURE_HIDDEN - 2), dtype=np.float32
    ).tobytes()
    shapes["v.position_embd.weight"] = (2, FIXTURE_POSITION_TABLE, FIXTURE_HIDDEN - 1)
    payloads["v.position_embd.weight"] = np.zeros(
        (2, FIXTURE_POSITION_TABLE, FIXTURE_HIDDEN - 1), dtype=np.float32
    ).tobytes()
    path = write_fixture_gguf(
        tmp_path / "mmproj.gguf", shapes=shapes, payloads=payloads
    )
    validation = validate_gemma4_vision_gguf(scan_gguf(path))

    assert not validation.passed
    assert any("v.blk.0.attn_q.weight" in error for error in validation.shape_errors)
    assert any("v.position_embd.weight" in error for error in validation.shape_errors)


def test_non_float_norm_tensor_is_a_type_error(tmp_path: Path) -> None:
    path = write_fixture_gguf(tmp_path / "mmproj.gguf")
    info = scan_gguf(path)
    tensors = tuple(
        tensor.__class__(**{**tensor.__dict__, "ggml_type_name": "Q4_0"})
        if tensor.name == "v.blk.0.ln1.weight"
        else tensor
        for tensor in info.tensors
    )
    patched = info.__class__(
        path=info.path,
        version=info.version,
        alignment=info.alignment,
        metadata=info.metadata,
        tensors=tensors,
        tensor_data_offset=info.tensor_data_offset,
    )
    validation = validate_gemma4_vision_gguf(patched)

    assert not validation.passed
    assert any("v.blk.0.ln1.weight" in error for error in validation.qtype_errors)


def test_standardize_and_clamp_pairs_must_be_complete(tmp_path: Path) -> None:
    path = write_fixture_gguf(
        tmp_path / "mmproj.gguf", drop=("v.std_scale",)
    )
    validation = validate_gemma4_vision_gguf(scan_gguf(path))
    assert not validation.passed
    assert any("present together" in error for error in validation.shape_errors)

    clamp_payload = np.array([0.5], dtype=np.float32).tobytes()
    path = write_fixture_gguf(
        tmp_path / "clamped.gguf",
        extra={"v.blk.0.attn_q.input_max": ((1,), clamp_payload)},
    )
    validation = validate_gemma4_vision_gguf(scan_gguf(path))
    assert not validation.passed
    assert any(
        name.endswith("v.blk.0.attn_q.input_min") for name in validation.missing
    ), validation.missing


def test_clamp_scalars_are_mapped_read_and_applied(
    tmp_path: Path, bundle: SimpleNamespace
) -> None:
    def scalars(weight: str, values: tuple[float, float, float, float]) -> dict[str, object]:
        return {
            f"{weight}.{field}": ((1,), np.array([value], dtype=np.float32).tobytes())
            for field, value in zip(CLAMP_FIELDS, values)
        }

    path = write_fixture_gguf(
        tmp_path / "mmproj.gguf",
        extra={
            **scalars("mm.input_projection", (0.0, 0.0, -1e4, 1e4)),
            **scalars("v.blk.0.attn_q", (-0.5, 0.5, -1.0, 1.0)),
        },
    )
    info = scan_gguf(path)
    model_map = build_gemma4_vision_gguf_map(info)

    # Clamp scalars are contract inventory: known, mapped, and read.
    assert model_map.validation.passed, model_map.validation
    assert model_map.tensor("input_projection.input_max").name == (
        "mm.input_projection.input_max"
    )
    assert model_map.tensor(gemma4_vision_layer_slot(0, "attn_q.output_max")).name == (
        "v.blk.0.attn_q.output_max"
    )
    slots = read_gemma4_vision_weights(info, model_map)
    assert slots["input_projection.input_min"].shape == (1,)
    assert slots["layers.0.attn_q.output_min"].dtype == np.float32

    geometry = gemma4_vision_geometry_from_config(model_map.config)
    weights = gemma4_vision_weights_from_slots(slots, geometry)
    assert weights.input_projection_clamp == (0.0, 0.0, -1e4, 1e4)
    assert weights.layers[0].clamps == {"attn_q": (-0.5, 0.5, -1.0, 1.0)}
    assert weights.layers[1].clamps == {}

    # A single-point projection clamp zeroes the soft tokens; the same tensors
    # without clamps do not.
    pixels = np.arange(4 * 4 * 3, dtype=np.float32).reshape(4, 4, 3) / 48.0
    patches = gemma4_vision_patches(pixels, model_map.config.patch_size)
    pos_x, pos_y = gemma4_vision_patch_positions(2, 2)
    clamped = gemma4_vision_forward(weights, patches, pos_x, pos_y, geometry)
    plain = gemma4_vision_forward(
        gemma4_vision_weights_from_slots(bundle.slots, geometry),
        patches,
        pos_x,
        pos_y,
        geometry,
    )
    assert clamped.num_soft_tokens == 1
    assert np.all(clamped.embeddings == 0.0)
    assert np.all(np.abs(plain.embeddings) > 0.0)

    # Per-layer clamps reach the attention: an identity clamp is a no-op and a
    # tight one changes the result.
    hidden = np.linspace(-2.0, 2.0, 4 * FIXTURE_HIDDEN).reshape(4, FIXTURE_HIDDEN)
    layer = replace(weights.layers[0], clamps={})
    base = gemma4_vision_attention(hidden, layer, geometry, pos_x, pos_y)
    identity = replace(layer, clamps={"attn_q": (-1e9, 1e9, -1e9, 1e9)})
    assert_allclose(
        gemma4_vision_attention(hidden, identity, geometry, pos_x, pos_y), base, rtol=1e-12
    )
    tight = replace(layer, clamps={"attn_q": (-0.25, 0.25, -1e9, 1e9)})
    assert not np.allclose(
        gemma4_vision_attention(hidden, tight, geometry, pos_x, pos_y), base
    )


def test_other_modalities_in_the_same_file_are_ignored(tmp_path: Path) -> None:
    # A vision+audio mmproj shares one file; the audio tensors are not this
    # contract's business and must not be reported as unexpected.
    clamp = np.array([0.5], dtype=np.float32).tobytes()
    path = write_fixture_gguf(
        tmp_path / "mmproj.gguf",
        extra={
            "conformer.linear.weight": ((4, 4), np.zeros((4, 4), dtype=np.float32).tobytes()),
            "conformer.linear.input_max": ((1,), clamp),
            "mm.a.input_projection.weight": ((4, 4), np.zeros((4, 4), dtype=np.float32).tobytes()),
            "mm.a.input_projection.input_max": ((1,), clamp),
        },
    )
    validation = validate_gemma4_vision_gguf(scan_gguf(path))

    assert validation.passed, validation
    assert set(validation.ignored) == {
        "conformer.linear.weight",
        "conformer.linear.input_max",
        "mm.a.input_projection.weight",
        "mm.a.input_projection.input_max",
    }
    # Audio clamp scalars are not vision clamp scalars.
    assert validation.config is not None
    assert validation.config.clipped_linears is False


def test_dynamic_image_size_is_accepted(tmp_path: Path) -> None:
    # Some converters write image_size 0 for a dynamic-size artifact; the resize
    # policy does not read it.
    path = write_fixture_gguf(
        tmp_path / "mmproj.gguf", metadata=fixture_metadata(**{"clip.vision.image_size": 0})
    )
    config = gemma4_vision_config_from_metadata(scan_gguf(path))

    assert config.image_size == 0


def test_loader_and_oracle_agree_on_weight_shapes(mmproj_path: Path) -> None:
    config = gemma4_vision_config_from_metadata(scan_gguf(mmproj_path))
    geometry = gemma4_vision_geometry_from_config(config)
    loader_shapes = dict(gemma4_vision_expected_shapes(config))
    oracle_shapes = dict(gemma4_vision_weight_shapes(geometry))

    assert set(loader_shapes) == set(oracle_shapes)
    for slot, shape in oracle_shapes.items():
        if slot == "position_embd":
            # The oracle does not assume a table capacity; the loader reads it.
            assert shape == (2, -1, FIXTURE_HIDDEN)
            continue
        assert loader_shapes[slot] == shape, slot


# ---------------------------------------------------------------------------
# Hand-computable stage checks
# ---------------------------------------------------------------------------


def test_rms_norm_matches_hand_computation() -> None:
    normed = gemma4_vision_rms_norm(np.array([3.0, 4.0]), eps=0.0)

    # rms = sqrt((9 + 16) / 2) = sqrt(12.5)
    rms = math.sqrt(12.5)
    assert normed == pytest.approx([3.0 / rms, 4.0 / rms])

    scaled = gemma4_vision_rms_norm(np.array([3.0, 4.0]), np.array([2.0, 1.0]), eps=0.0)
    assert scaled == pytest.approx([6.0 / rms, 4.0 / rms])

    with pytest.raises(ValueError, match="weight must have shape"):
        gemma4_vision_rms_norm(np.array([3.0, 4.0]), np.ones(3), eps=1e-6)
    with pytest.raises(ValueError, match="eps"):
        gemma4_vision_rms_norm(np.array([3.0, 4.0]), eps=0.0 - 1.0)


def test_epsilon_bounds_a_zero_vector() -> None:
    normed = gemma4_vision_rms_norm(np.zeros((2, 4)), eps=1e-6)

    assert normed == pytest.approx(np.zeros((2, 4)))


def test_gelu_tanh_matches_the_reference_formula() -> None:
    values = np.array([0.0, 1.0, -2.0], dtype=np.float64)
    expected = [
        0.5 * value * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (value + 0.044715 * value**3)))
        for value in values
    ]

    assert gemma4_vision_gelu_tanh(values) == pytest.approx(expected)


def test_float32_stages_stay_float32() -> None:
    # The GPU runner is all-FP32, so no stage may silently promote to float64.
    values = np.linspace(-2.0, 2.0, 16, dtype=np.float32).reshape(4, 4)
    positions = np.array([0, 1, 0, 1])
    assert gemma4_vision_gelu_tanh(values).dtype == np.float32
    assert gemma4_vision_rms_norm(values, np.ones(4, np.float32), eps=1e-6).dtype == np.float32
    assert gemma4_vision_pool(values, (2, 2), 1).dtype == np.float32
    assert (
        gemma4_vision_standardize(values, np.zeros(4, np.float32), np.ones(4, np.float32)).dtype
        == np.float32
    )
    assert gemma4_vision_apply_rope(
        values, pos_x=positions, pos_y=positions, theta=100.0
    ).dtype == np.float32


def test_rope_inv_freq_ladder() -> None:
    # head_dim 8 -> 4 rotated channels per axis -> 2 pairs, exponents 0 and 2/4.
    inv_freq = gemma4_vision_rope_inv_freq(8, 100.0)

    assert inv_freq == pytest.approx([1.0, 1.0 / 10.0])
    with pytest.raises(ValueError, match="multiple of 4"):
        gemma4_vision_rope_inv_freq(6, 100.0)


def test_rope_rotates_each_axis_half_by_its_own_position() -> None:
    # head_dim 4 -> one pair per axis with frequency 1.0 (exponent 0).
    vector = np.array([[1.0, 0.0, 0.0, 0.0]])
    rotated = gemma4_vision_apply_rope(
        vector, pos_x=np.array([1]), pos_y=np.array([2]), theta=100.0
    )

    assert_allclose(rotated, [[math.cos(1.0), math.sin(1.0), 0.0, 0.0]])

    # The y half rotates by pos_y only.
    vector = np.array([[0.0, 0.0, 1.0, 0.0]])
    rotated = gemma4_vision_apply_rope(
        vector, pos_x=np.array([1]), pos_y=np.array([2]), theta=100.0
    )
    assert_allclose(rotated, [[0.0, 0.0, math.cos(2.0), math.sin(2.0)]])

    # Position 0 is the identity.
    vector = np.array([[0.7, -0.3, 0.2, 0.9]])
    assert_allclose(
        gemma4_vision_apply_rope(
            vector, pos_x=np.array([0]), pos_y=np.array([0]), theta=100.0
        ),
        vector,
    )


def test_rope_preserves_pair_norms_and_rejects_bad_positions() -> None:
    rng = np.random.default_rng(7)
    vector = rng.standard_normal((3, 2, 8))
    rotated = gemma4_vision_apply_rope(
        vector, pos_x=np.array([0, 3, 7]), pos_y=np.array([1, 2, 5]), theta=100.0
    )
    # Two-axis NeoX pairs are (j, j + head_dim/4) inside each axis half.
    for axis_offset in (0, 4):
        first = rotated[:, :, axis_offset : axis_offset + 2]
        second = rotated[:, :, axis_offset + 2 : axis_offset + 4]
        original_first = vector[:, :, axis_offset : axis_offset + 2]
        original_second = vector[:, :, axis_offset + 2 : axis_offset + 4]
        assert np.allclose(
            np.hypot(first, second), np.hypot(original_first, original_second), atol=1e-12
        )

    with pytest.raises(ValueError, match="one position per patch"):
        gemma4_vision_apply_rope(vector, pos_x=np.array([0]), pos_y=np.array([0]), theta=100.0)
    with pytest.raises(ValueError, match="non-negative"):
        gemma4_vision_apply_rope(
            vector, pos_x=np.array([0, -1, 2]), pos_y=np.array([0, 1, 2]), theta=100.0
        )


def _single_head_geometry() -> Gemma4VisionGeometry:
    return Gemma4VisionGeometry(
        block_count=1,
        hidden_size=4,
        intermediate_size=4,
        output_size=4,
        heads=1,
        head_count_kv=1,
        head_dim=4,
        patch_size=1,
        pooling_kernel_size=1,
        rms_norm_eps=1e-6,
        rope_theta=100.0,
    )


def _single_head_layer(**overrides: np.ndarray):
    from hipengine.kernels.cpu_reference.gemma4_vision import Gemma4VisionLayerWeights

    base = {
        "ln1": np.ones(4),
        "ln2": np.ones(4),
        "attn_q": np.eye(4),
        "attn_k": np.eye(4),
        "attn_v": np.eye(4),
        "attn_out": np.eye(4),
        "attn_q_norm": np.ones(4),
        "attn_k_norm": np.ones(4),
        "attn_post_norm": np.ones(4),
        "ffn_post_norm": np.ones(4),
        "ffn_gate": np.zeros((4, 4)),
        "ffn_up": np.zeros((4, 4)),
        "ffn_down": np.zeros((4, 4)),
    }
    base.update(overrides)
    return Gemma4VisionLayerWeights(**base)


def test_attention_has_no_one_over_sqrt_head_dim_scaling() -> None:
    geometry = _single_head_geometry()
    hidden = np.array([[3.0, 1.0, 0.0, 0.0], [1.0, 3.0, 0.0, 0.0]])
    layer = _single_head_layer()
    positions = np.array([0, 0])

    out = gemma4_vision_attention(hidden, layer, geometry, positions, positions)

    # Position 0 is an identity rotation, so q = k = rms_norm(hidden) and v is the
    # same.  With no scaling, the scores are the plain dot products.
    normed = hidden / np.sqrt((hidden**2).mean(axis=1, keepdims=True) + 1e-6)
    scores = normed @ normed.T
    weights = np.exp(scores - scores.max(axis=1, keepdims=True))
    weights = weights / weights.sum(axis=1, keepdims=True)
    expected = weights @ normed
    assert out == pytest.approx(expected, abs=1e-9)

    # A 1/sqrt(head_dim) = 0.5 factor would produce a different result, so this
    # assertion is what pins the missing scale.
    scaled_scores = 0.5 * scores
    scaled = np.exp(scaled_scores - scaled_scores.max(axis=1, keepdims=True))
    scaled = scaled / scaled.sum(axis=1, keepdims=True)
    assert not np.allclose(out, scaled @ normed, atol=1e-3)


def test_attention_applies_qk_norms_per_head_and_v_norm_without_scale() -> None:
    geometry = _single_head_geometry()
    hidden = np.array([[3.0, 1.0, -2.0, 0.5], [0.25, 2.0, 1.0, -1.0]])
    q_norm = np.array([1.0, 2.0, 1.0, 0.5])
    k_norm = np.array([0.5, 1.0, 2.0, 1.0])
    layer = _single_head_layer(attn_q_norm=q_norm, attn_k_norm=k_norm)
    positions = np.array([0, 0])

    out = gemma4_vision_attention(hidden, layer, geometry, positions, positions)

    def norm_with(values: np.ndarray, weight: np.ndarray | None) -> np.ndarray:
        scale = 1.0 / np.sqrt((values**2).mean(axis=-1, keepdims=True) + 1e-6)
        normed = values * scale
        return normed if weight is None else normed * weight

    query = norm_with(hidden, q_norm)
    key = norm_with(hidden, k_norm)
    value = norm_with(hidden, None)
    scores = query @ key.T
    weights = np.exp(scores - scores.max(axis=1, keepdims=True))
    weights = weights / weights.sum(axis=1, keepdims=True)
    assert out == pytest.approx(weights @ value, abs=1e-9)


def test_patch_embed_scales_pixels_and_adds_both_axis_embeddings() -> None:
    # One 1x1 patch with three channels, one output channel.
    weight = np.zeros((1, 3, 1, 1))
    weight[0, 0, 0, 0] = 1.0
    weight[0, 1, 0, 0] = 2.0
    weight[0, 2, 0, 0] = -1.0
    patches = np.array([[0.25, 0.5, 1.0]])
    table = np.zeros((2, 4, 1))
    table[0, 2, 0] = 10.0
    table[1, 1, 0] = 100.0

    hidden = gemma4_vision_patch_embed(
        patches, weight, table, pos_x=np.array([2]), pos_y=np.array([1])
    )

    # 2 * (pixel - 0.5) = [-0.5, 0, 1] -> 1*-0.5 + 2*0 + -1*1 = -1.5, then +110.
    assert_allclose(hidden, [[108.5]])


def test_patch_embed_rejects_out_of_range_pixels_and_positions() -> None:
    weight = np.zeros((1, 3, 1, 1))
    table = np.zeros((2, 2, 1))
    with pytest.raises(ValueError, match=r"rescaled pixel values"):
        gemma4_vision_patch_embed(np.array([[2.0, 0.0, 0.0]]), weight)
    with pytest.raises(ValueError, match="position table size"):
        gemma4_vision_patch_embed(
            np.array([[0.5, 0.5, 0.5]]),
            weight,
            table,
            pos_x=np.array([2]),
            pos_y=np.array([0]),
        )


def test_pooling_averages_blocks_and_scales_by_sqrt_hidden() -> None:
    hidden = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ]
    )
    pooled = gemma4_vision_pool(hidden, (2, 2), 2)

    # Row-major patch order (0,0) (1,0) (0,1) (1,1) collapses to one block, then
    # the reference pooler scales by sqrt(hidden_size) = sqrt(2).
    assert_allclose(pooled, [[4.0 * math.sqrt(2.0), 5.0 * math.sqrt(2.0)]])

    pooled = gemma4_vision_pool(hidden, (2, 2), 1)
    assert pooled == pytest.approx(hidden * math.sqrt(2.0))

    with pytest.raises(ValueError, match="not divisible by the pooling kernel"):
        gemma4_vision_pool(hidden, (2, 2), 3)
    with pytest.raises(ValueError, match="does not cover"):
        gemma4_vision_pool(hidden, (3, 1), 1)


def test_standardization_and_projection_are_per_channel() -> None:
    features = np.array([[2.0, 4.0], [0.0, 8.0]])
    standardized = gemma4_vision_standardize(
        features, np.array([1.0, 2.0]), np.array([2.0, 0.5])
    )

    assert_allclose(standardized, [[2.0, 1.0], [-2.0, 3.0]])
    with pytest.raises(ValueError, match="std_bias expected shape"):
        gemma4_vision_standardize(features, np.ones(3), np.ones(2))


# ---------------------------------------------------------------------------
# End-to-end forward against the loop reference
# ---------------------------------------------------------------------------


@pytest.fixture
def bundle(mmproj_path: Path) -> SimpleNamespace:
    """The fixture artifact through the real validate -> read path."""

    info = scan_gguf(mmproj_path)
    model_map = build_gemma4_vision_gguf_map(info)
    geometry = gemma4_vision_geometry_from_config(model_map.config)
    return SimpleNamespace(
        config=model_map.config,
        geometry=geometry,
        slots=read_gemma4_vision_weights(info, model_map),
    )


@pytest.mark.parametrize(
    ("grid", "pixels", "soft_tokens"),
    [
        # 4x4 pixels at patch size 2 -> a 2x2 grid -> one soft token.
        ((2, 2), np.arange(4 * 4 * 3, dtype=np.float32).reshape(4, 4, 3) / 48.0, 1),
        # 8x16 pixels -> a 4x8 grid -> a 2x4 pooled grid, blocks row-major.
        ((4, 8), np.linspace(0.0, 1.0, 8 * 16 * 3, dtype=np.float32).reshape(8, 16, 3), 8),
    ],
)
def test_forward_matches_the_loop_reference(
    bundle: SimpleNamespace, grid: tuple[int, int], pixels: np.ndarray, soft_tokens: int
) -> None:
    geometry = bundle.geometry
    slots = bundle.slots
    patches = gemma4_vision_patches(pixels, bundle.config.patch_size)
    pos_x, pos_y = gemma4_vision_patch_positions(*grid)

    # Compare in float64 first: that isolates a formula difference from float32
    # accumulation noise.
    slots64 = {slot: value.astype(np.float64) for slot, value in slots.items()}
    patches64 = patches.astype(np.float64)
    exact = gemma4_vision_forward(
        gemma4_vision_weights_from_slots(slots64, geometry),
        patches64,
        pos_x,
        pos_y,
        geometry,
        grid_size=grid,
    )
    reference = _ref_forward(slots64, patches64, pos_x, pos_y, geometry)

    assert exact.grid_size == grid
    assert exact.num_soft_tokens == soft_tokens
    assert exact.pooled.shape == (soft_tokens, FIXTURE_HIDDEN)
    assert exact.embeddings.shape == (soft_tokens, FIXTURE_OUTPUT)
    assert_allclose(exact.pooled, reference["pooled"], rtol=1e-11, atol=1e-13)
    assert_allclose(exact.hidden_states, reference["hidden_states"], rtol=1e-11, atol=1e-13)
    assert_allclose(exact.embeddings, reference["embeddings"], rtol=1e-11, atol=1e-13)

    # The float32 path is the production dtype and must land in the same place.
    result = gemma4_vision_forward(
        gemma4_vision_weights_from_slots(slots, geometry),
        patches,
        pos_x,
        pos_y,
        geometry,
        grid_size=grid,
    )
    assert np.all(np.isfinite(result.embeddings))
    assert_allclose(result.embeddings, exact.embeddings, rtol=1e-4, atol=1e-5)


def test_zero_tower_yields_zero_embeddings() -> None:
    geometry = Gemma4VisionGeometry(
        block_count=1,
        hidden_size=4,
        intermediate_size=2,
        output_size=3,
        heads=1,
        head_count_kv=1,
        head_dim=4,
        patch_size=1,
        pooling_kernel_size=2,
        rms_norm_eps=1e-6,
        rope_theta=100.0,
    )
    zeros = np.zeros
    layer = _single_head_layer(
        ffn_gate=zeros((2, 4)),
        ffn_up=zeros((2, 4)),
        ffn_down=zeros((4, 2)),
    )
    from hipengine.kernels.cpu_reference.gemma4_vision import Gemma4VisionWeights

    weights = Gemma4VisionWeights(
        patch_embd=zeros((4, 3, 1, 1)),
        position_embd=zeros((2, 8, 4)),
        input_projection=zeros((3, 4)),
        layers=(layer,),
        std_bias=np.array([1.0, 2.0, 3.0, 4.0]),
        std_scale=np.array([1.0, 1.0, 1.0, 1.0]),
    )
    patches = zeros((4, 3))
    pos_x, pos_y = gemma4_vision_patch_positions(2, 2)

    result = gemma4_vision_forward(
        weights, patches, pos_x, pos_y, geometry, grid_size=(2, 2)
    )

    # Every stage is homogeneous, so an all-zero tower stays exactly zero: there
    # is no bias, no positional offset, and no normalization offset anywhere.
    assert result.num_soft_tokens == 1
    assert result.embeddings == pytest.approx(np.zeros((1, 3)))


def test_forward_rejects_inconsistent_inputs(bundle: SimpleNamespace) -> None:
    geometry = bundle.geometry
    slots = bundle.slots
    weights = gemma4_vision_weights_from_slots(slots, geometry)
    patches = np.zeros((4, geometry.patch_values), dtype=np.float32) + 0.5
    pos_x, pos_y = gemma4_vision_patch_positions(2, 2)

    with pytest.raises(ValueError, match="row-major grid"):
        gemma4_vision_forward(weights, patches, pos_y, pos_x, geometry)
    with pytest.raises(ValueError, match="does not cover"):
        gemma4_vision_forward(
            weights, patches, pos_x, pos_y, geometry, grid_size=(2, 3)
        )
    with pytest.raises(ValueError, match="one position per patch"):
        gemma4_vision_forward(weights, patches, pos_x[:3], pos_y[:3], geometry)
    with pytest.raises(ValueError, match="expected 2 encoder layers"):
        from hipengine.kernels.cpu_reference.gemma4_vision import Gemma4VisionWeights

        gemma4_vision_forward(
            Gemma4VisionWeights(
                patch_embd=weights.patch_embd,
                position_embd=weights.position_embd,
                input_projection=weights.input_projection,
                layers=weights.layers[:1],
            ),
            patches,
            pos_x,
            pos_y,
            geometry,
        )


def test_weights_from_slots_reports_missing_and_misshaped_slots(bundle: SimpleNamespace) -> None:
    geometry = bundle.geometry
    slots = bundle.slots

    without = dict(slots)
    without.pop(gemma4_vision_layer_slot(1, "ffn_up"))
    with pytest.raises(KeyError, match="ffn_up"):
        gemma4_vision_weights_from_slots(without, geometry)

    misshaped = dict(slots)
    misshaped["patch_embd"] = slots["patch_embd"][..., :1]
    with pytest.raises(ValueError, match="patch_embd"):
        gemma4_vision_weights_from_slots(misshaped, geometry)

    half = dict(slots)
    half.pop("std_scale")
    with pytest.raises(ValueError, match="std_bias and std_scale"):
        gemma4_vision_weights_from_slots(half, geometry)


def test_runtime_spellings_build_the_same_geometry(mmproj_path: Path) -> None:
    config = gemma4_vision_config_from_metadata(scan_gguf(mmproj_path))

    # The loader's canonical names and the runtime's provisional names are the
    # same geometry.
    assert config.spatial_merge_size == config.pooling_kernel_size
    assert config.norm_epsilon == config.rms_norm_eps
    canonical = gemma4_vision_geometry_from_config(config)

    runtime_style = type(
        "RuntimeVisionConfig",
        (),
        {
            "block_count": config.block_count,
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "output_size": config.output_size,
            "heads": config.heads,
            "head_dim": config.head_dim,
            "spatial_merge_size": config.pooling_kernel_size,
            "patch_size": config.patch_size,
            "norm_epsilon": config.rms_norm_eps,
            "rope_theta": config.rope_theta,
        },
    )()
    adapted = gemma4_vision_geometry_from_config(runtime_style)

    assert adapted == canonical
    assert adapted.head_count_kv == config.heads


def test_geometry_rejects_inconsistent_dimensions() -> None:
    with pytest.raises(ValueError, match="head_dim"):
        Gemma4VisionGeometry(
            block_count=1,
            hidden_size=8,
            intermediate_size=4,
            output_size=4,
            heads=2,
            head_count_kv=2,
            head_dim=2,
            patch_size=2,
            pooling_kernel_size=2,
            rms_norm_eps=1e-6,
            rope_theta=100.0,
        )
    with pytest.raises(ValueError, match="head_count_kv"):
        Gemma4VisionGeometry(
            block_count=1,
            hidden_size=8,
            intermediate_size=4,
            output_size=4,
            heads=2,
            head_count_kv=3,
            head_dim=4,
            patch_size=2,
            pooling_kernel_size=2,
            rms_norm_eps=1e-6,
            rope_theta=100.0,
        )
    with pytest.raises(ValueError, match="missing required attribute"):
        gemma4_vision_geometry_from_config(object())


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def _quadrant_image(size: int = 4) -> np.ndarray:
    """4x4 image whose four 2x2 quadrants are solid distinct colours."""

    image = np.zeros((size, size, 3), dtype=np.uint8)
    half = size // 2
    colors = {
        (0, 0): (10, 20, 30),
        (0, 1): (40, 50, 60),
        (1, 0): (70, 80, 90),
        (1, 1): (100, 110, 120),
    }
    for (row, col), color in colors.items():
        image[row * half : (row + 1) * half, col * half : (col + 1) * half] = color
    return image


def test_patch_positions_are_row_major_with_x_fastest() -> None:
    pos_x, pos_y = gemma4_vision_patch_positions(2, 3)

    assert pos_x.tolist() == [0, 1, 2, 0, 1, 2]
    assert pos_y.tolist() == [0, 0, 0, 1, 1, 1]


def test_patches_are_channel_last_in_patch_row_major_order() -> None:
    preprocessed = preprocess_gemma4_image(
        _quadrant_image(),
        patch_size=2,
        max_soft_tokens=70,
        pooling_kernel_size=2,
        resize=False,
    )

    assert preprocessed.grid_size == (2, 2)
    assert preprocessed.num_patches == 4
    assert preprocessed.num_soft_tokens == 1
    assert preprocessed.target_size == (4, 4)
    assert preprocessed.pos_x.tolist() == [0, 1, 0, 1]
    assert preprocessed.pos_y.tolist() == [0, 0, 1, 1]

    # Patch row 0 is the top-left quadrant, row 1 the top-right, and so on:
    # x advances fastest inside a row of patches.
    expected = np.array(
        [
            [10, 20, 30] * 4,
            [40, 50, 60] * 4,
            [70, 80, 90] * 4,
            [100, 110, 120] * 4,
        ],
        dtype=np.float32,
    ) / np.float32(255.0)
    assert preprocessed.patches == pytest.approx(expected)
    assert preprocessed.pixel_values is preprocessed.patches


def test_patch_values_follow_row_then_column_then_channel_order() -> None:
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    # (row, col) pixel values chosen so every ordering is distinguishable.
    image[0, 0] = (0, 0, 0)
    image[0, 1] = (3, 4, 5)
    image[1, 0] = (6, 7, 8)
    image[1, 1] = (9, 10, 11)

    preprocessed = preprocess_gemma4_image(
        image, patch_size=2, pooling_kernel_size=1, max_soft_tokens=70, resize=False
    )

    expected = np.array([[0, 0, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11]], dtype=np.float32) / 255.0
    assert preprocessed.patches == pytest.approx(expected)
    assert preprocessed.grid_size == (1, 1)


def test_resize_target_matches_the_reference_rule() -> None:
    # 280 soft tokens * 3x3 pool * 16x16 patch = 645120 target pixels.  A square
    # 224 image scales by sqrt(12.857) -> 803.2 -> 16 blocks of 48 = 768.
    assert gemma4_vision_resize_target(
        224, 224, patch_size=16, max_soft_tokens=280, pooling_kernel_size=3
    ) == (768, 768)

    # A 1x1000 image collapses its height to zero and is repaired to one block.
    height, width = gemma4_vision_resize_target(
        1, 1000, patch_size=16, max_soft_tokens=280, pooling_kernel_size=3
    )
    assert height == 48
    assert width == 280 * 48
    assert height * width <= 280 * 9 * 16 * 16


def test_resize_target_repairs_a_collapsed_side() -> None:
    # A very wide image rounds its height to zero; the reference repairs it to
    # one pooling block and clamps the width to the patch budget.
    height, width = gemma4_vision_resize_target(
        1, 100_000, patch_size=16, max_soft_tokens=280, pooling_kernel_size=3
    )
    assert height == 48
    assert width == 280 * 48
    assert height * width == 280 * 9 * 16 * 16

    # The transpose repairs the other side.
    height, width = gemma4_vision_resize_target(
        100_000, 1, patch_size=16, max_soft_tokens=280, pooling_kernel_size=3
    )
    assert width == 48
    assert height == 280 * 48


def test_resize_target_accepts_any_positive_budget() -> None:
    # 100 is not one of the reference presets but is a perfectly good budget:
    # 100 soft tokens * 3x3 pool * 16x16 patch = 230400 target pixels.
    height, width = gemma4_vision_resize_target(
        224, 224, patch_size=16, max_soft_tokens=100, pooling_kernel_size=3
    )
    assert (height, width) == (480, 480)
    assert height * width <= 100 * 9 * 16 * 16
    assert gemma4_vision_resize_target(
        224, 224, patch_size=16, max_soft_tokens=1, pooling_kernel_size=3
    ) == (48, 48)


def test_resize_target_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="max_soft_tokens must be a positive integer"):
        gemma4_vision_resize_target(
            224, 224, patch_size=16, max_soft_tokens=0, pooling_kernel_size=3
        )
    with pytest.raises(ValueError, match="must be a positive integer"):
        gemma4_vision_resize_target(
            0, 224, patch_size=16, max_soft_tokens=280, pooling_kernel_size=3
        )
    with pytest.raises(ValueError, match="must be a positive integer"):
        gemma4_vision_resize_target(
            224, 224, patch_size=16, max_soft_tokens=280, pooling_kernel_size=0
        )


def test_preprocess_rejects_bad_images_and_geometry() -> None:
    with pytest.raises(ValueError, match="integer RGB values"):
        preprocess_gemma4_image(np.zeros((4, 4, 3), dtype=np.float32), patch_size=2)
    with pytest.raises(ValueError, match="grayscale"):
        preprocess_gemma4_image(np.zeros((4, 4), dtype=np.uint8), patch_size=2)
    with pytest.raises(ValueError, match="height, width, 3"):
        preprocess_gemma4_image(np.zeros((4, 4, 4), dtype=np.uint8), patch_size=2)
    with pytest.raises(ValueError, match="values must lie in"):
        preprocess_gemma4_image(np.full((4, 4, 3), 300, dtype=np.int32), patch_size=2)
    with pytest.raises(ValueError, match="not divisible by patch size"):
        preprocess_gemma4_image(
            np.zeros((5, 4, 3), dtype=np.uint8), patch_size=2, resize=False
        )
    with pytest.raises(ValueError, match="not divisible by the pooling kernel"):
        preprocess_gemma4_image(
            np.zeros((4, 4, 3), dtype=np.uint8),
            patch_size=2,
            pooling_kernel_size=3,
            resize=False,
        )
    with pytest.raises(TypeError, match="Pillow image"):
        preprocess_gemma4_image(object(), patch_size=2)


def test_resize_path_uses_pillow_and_produces_the_reference_grid() -> None:
    pytest.importorskip("PIL")
    image = np.zeros((224, 224, 3), dtype=np.uint8)

    preprocessed = preprocess_gemma4_image(
        image, patch_size=16, max_soft_tokens=280, pooling_kernel_size=3
    )

    assert preprocessed.target_size == (768, 768)
    assert preprocessed.grid_size == (48, 48)
    assert preprocessed.num_soft_tokens == 256
    assert preprocessed.num_patches == 48 * 48
    assert int(preprocessed.pos_x.max()) == 47
    assert int(preprocessed.pos_y.max()) == 47
    assert preprocessed.patches.dtype == np.float32
    assert float(preprocessed.patches.min()) >= 0.0
    assert float(preprocessed.patches.max()) <= 1.0


def test_preprocessed_image_feeds_the_forward(bundle: SimpleNamespace) -> None:
    config = bundle.config
    geometry = bundle.geometry
    slots = bundle.slots
    weights = gemma4_vision_weights_from_slots(slots, geometry)

    preprocessed = preprocess_gemma4_image(
        _quadrant_image(),
        patch_size=config.patch_size,
        max_soft_tokens=config.max_soft_tokens,
        pooling_kernel_size=config.pooling_kernel_size,
        resize=False,
    )
    result = gemma4_vision_forward(
        weights,
        preprocessed.patches,
        preprocessed.pos_x,
        preprocessed.pos_y,
        geometry,
        grid_size=preprocessed.grid_size,
    )

    assert result.num_soft_tokens == preprocessed.num_soft_tokens
    assert result.embeddings.shape == (preprocessed.num_soft_tokens, config.output_size)


def test_positions_adapter_matches_the_separate_axis_api(bundle: SimpleNamespace) -> None:
    config = bundle.config
    geometry = bundle.geometry
    slots = bundle.slots
    weights = gemma4_vision_weights_from_slots(slots, geometry)

    preprocessed = preprocess_gemma4_image(
        _quadrant_image(),
        patch_size=config.patch_size,
        max_soft_tokens=config.max_soft_tokens,
        pooling_kernel_size=config.pooling_kernel_size,
        resize=False,
    )
    positions = np.stack((preprocessed.pos_x, preprocessed.pos_y), axis=1)

    adapter = gemma4_vision_forward_positions(
        weights, preprocessed.patches_chw, positions, geometry
    )
    direct = gemma4_vision_forward(
        weights,
        preprocessed.patches,
        preprocessed.pos_x,
        preprocessed.pos_y,
        geometry,
        grid_size=preprocessed.grid_size,
    )

    assert adapter.grid_size == preprocessed.grid_size
    assert_allclose(adapter.embeddings, direct.embeddings, rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError, match=r"shape \(num_patches, 2\)"):
        gemma4_vision_forward_positions(
            weights, preprocessed.patches_chw, positions[:, :1], geometry
        )


def test_channel_first_patches_match_channel_last_patches(bundle: SimpleNamespace) -> None:
    config = bundle.config
    geometry = bundle.geometry
    slots = bundle.slots
    weights = gemma4_vision_weights_from_slots(slots, geometry)

    preprocessed = preprocess_gemma4_image(
        _quadrant_image(),
        patch_size=config.patch_size,
        max_soft_tokens=config.max_soft_tokens,
        pooling_kernel_size=config.pooling_kernel_size,
        resize=False,
    )
    chw = preprocessed.patches_chw

    # Both orders describe the same pixels, so the tower output must be identical.
    assert chw.shape == preprocessed.patches.shape
    assert_allclose(
        gemma4_vision_patches_as_chw(preprocessed.patches, config.patch_size), chw
    )
    last = gemma4_vision_forward(
        weights,
        preprocessed.patches,
        preprocessed.pos_x,
        preprocessed.pos_y,
        geometry,
        grid_size=preprocessed.grid_size,
    )
    first = gemma4_vision_forward(
        weights,
        chw,
        preprocessed.pos_x,
        preprocessed.pos_y,
        geometry,
        grid_size=preprocessed.grid_size,
        patch_order=PATCH_ORDER_CHANNEL_FIRST,
    )
    # Permuting the kernel columns changes float32 summation order, so the two
    # orders agree to float32 noise rather than bit-exactly.
    assert_allclose(first.embeddings, last.embeddings, rtol=1e-5, atol=1e-6)

    # A channel-first row layout fed as channel-last is a real difference, so the
    # equality above is not vacuous.
    wrong = gemma4_vision_forward(
        weights,
        chw,
        preprocessed.pos_x,
        preprocessed.pos_y,
        geometry,
        grid_size=preprocessed.grid_size,
    )
    assert not np.allclose(wrong.embeddings, last.embeddings, rtol=1e-3, atol=1e-3)

    with pytest.raises(ValueError, match="patch_order must be one of"):
        gemma4_vision_patch_embed(
            preprocessed.patches, weights.patch_embd, patch_order="chw?"
        )
