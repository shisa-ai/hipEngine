"""Torch-free Gemma 4 mmproj (vision tower) GGUF contract.

A Gemma 4 ``mmproj`` is a ``clip`` artifact whose ``clip.vision.projector_type``
is ``gemma4v``.  Dimensions come from ``clip.vision.*`` metadata and the tensor
contract is derived from them, so an artifact that declares dimensions these
kernels can execute is accepted and one that does not fails with a named
capability miss.  Nothing keys on a model name, a path, or a hash.

Three numbers are not in the metadata:

* the pooling kernel size (``clip.vision.projector.scale_factor``, else the
  reference default 3);
* the vision RoPE theta (no GGUF key; the reference tower uses 100.0);
* the position table capacity, read from ``v.position_embd.weight``.  It is
  enforced when a patch grid is built, not at admission: a table that is small
  for one aspect ratio still serves another.

Optional ``Gemma4ClippableLinear`` scalars (``input_min``/``input_max``/
``output_min``/``output_max``) and a per-layer ``out_scale`` are contract
inventory when present: mapped, read, and applied by the reference forward.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from hipengine.loading.gguf import (
    GGUFModelInfo,
    GGUFTensorInfo,
    MissingGGUFTensorError,
)
from hipengine.quant.gguf import (
    dequantization_supported,
    dequantize_gguf_data,
    numpy_storage_dtype,
)

GEMMA4_VISION_ARCHITECTURE = "clip"
GEMMA4V_PROJECTOR_TYPE = "gemma4v"

# Reference vision tower defaults (transformers ``Gemma4VisionConfig``).
DEFAULT_POOLING_KERNEL_SIZE = 3
DEFAULT_ROPE_THETA = 100.0
DEFAULT_MAX_SOFT_TOKENS = 280
# The reference processor's published soft-token budgets.  Any positive budget
# is executable; these are the suggested presets.
REFERENCE_MAX_SOFT_TOKEN_PRESETS = (70, 140, 280, 560, 1120)
DEFAULT_IMAGE_SIZE = 224

# The patch embedder is a single ``patch x patch`` RGB convolution.
PATCH_CHANNELS = 3
# ``v.position_embd.weight`` holds one lookup table per spatial axis (x, y).
POSITION_TABLE_AXES = 2

_FLOAT_TYPE_NAMES = ("F32", "F16", "BF16", "F64")

_ROOT_SLOTS = {
    "patch_embd": "v.patch_embd.weight",
    "position_embd": "v.position_embd.weight",
    "input_projection": "mm.input_projection.weight",
}
_OPTIONAL_ROOT_SLOTS = {
    "std_bias": "v.std_bias",
    "std_scale": "v.std_scale",
}
# Per-layer slot names are exactly the GGUF tensor suffixes.
_LAYER_SLOTS = (
    "ln1",
    "ln2",
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_out",
    "attn_q_norm",
    "attn_k_norm",
    "attn_post_norm",
    "ffn_post_norm",
    "ffn_gate",
    "ffn_up",
    "ffn_down",
)
_OPTIONAL_LAYER_SLOTS = ("out_scale",)
_CLAMP_FIELDS = ("input_min", "input_max", "output_min", "output_max")
# Weights the reference graph projects through ``Gemma4ClippableLinear``.  The
# patch embedding is a convolution and the norms are elementwise, so neither
# carries clamp scalars.
_CLAMP_CAPABLE_ROOT_SLOTS = ("input_projection",)
_CLAMP_CAPABLE_LAYER_SLOTS = (
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_out",
    "ffn_gate",
    "ffn_up",
    "ffn_down",
)
# Tensors the vision graph never reads: audio encoders share the file.
_VISION_PREFIX = "v."
_AUDIO_PREFIX = "mm.a."


class Gemma4VisionGGUFError(ValueError):
    """A Gemma 4 mmproj artifact declares a layout this contract cannot execute."""


@dataclass(frozen=True)
class Gemma4VisionGGUFConfig:
    """Validated architecture dimensions for one Gemma 4 mmproj artifact."""

    architecture: str
    projector_type: str
    block_count: int
    hidden_size: int
    intermediate_size: int
    output_size: int
    heads: int
    head_count_kv: int
    patch_size: int
    pooling_kernel_size: int
    position_table_size: int
    rms_norm_eps: float
    rope_theta: float
    max_soft_tokens: int
    image_size: int
    standardize: bool
    clipped_linears: bool

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.heads

    @property
    def spatial_merge_size(self) -> int:
        """Alias for :attr:`pooling_kernel_size`.

        The reference vision tower and the runtime call the post-patch pooling
        factor ``spatial_merge_size``; the GGUF key that carries it is
        ``clip.vision.projector.scale_factor``.
        """

        return self.pooling_kernel_size

    @property
    def norm_epsilon(self) -> float:
        """Alias for :attr:`rms_norm_eps`."""

        return self.rms_norm_eps

    @property
    def query_width(self) -> int:
        return self.heads * self.head_dim

    @property
    def kv_width(self) -> int:
        return self.head_count_kv * self.head_dim

    @property
    def patch_values(self) -> int:
        """Flattened channel-last values per patch fed to the patch embedder."""

        return PATCH_CHANNELS * self.patch_size * self.patch_size

    @property
    def pooling_block_patches(self) -> int:
        return self.pooling_kernel_size * self.pooling_kernel_size

    @property
    def max_grid_side(self) -> int:
        """Longest patch-grid side the resize policy can produce.

        Informational: the bound a real image must satisfy is
        :attr:`position_table_size`, checked by :meth:`check_grid`.
        """

        return self.max_soft_tokens * self.pooling_kernel_size

    def num_soft_tokens(self, grid_size: Sequence[int]) -> int:
        grid_height, grid_width = (int(dim) for dim in grid_size)
        self.check_grid(grid_height, grid_width)
        return (grid_height // self.pooling_kernel_size) * (
            grid_width // self.pooling_kernel_size
        )

    def check_grid(self, grid_height: int, grid_width: int) -> None:
        """Raise when a patch grid cannot be pooled or cannot be indexed."""

        for label, value in (("height", grid_height), ("width", grid_width)):
            if int(value) <= 0:
                raise Gemma4VisionGGUFError(
                    f"Gemma 4 vision patch-grid {label} must be positive, got {value}"
                )
            if int(value) % self.pooling_kernel_size:
                raise Gemma4VisionGGUFError(
                    f"Gemma 4 vision patch-grid {label} {value} is not divisible by the "
                    f"pooling kernel size {self.pooling_kernel_size}"
                )
            if int(value) > self.position_table_size:
                raise Gemma4VisionGGUFError(
                    f"Gemma 4 vision patch-grid {label} {value} exceeds the position table "
                    f"size {self.position_table_size}"
                )


@dataclass(frozen=True)
class Gemma4VisionGGUFValidation:
    """Result of validating the vision tower's complete tensor inventory."""

    config: Gemma4VisionGGUFConfig | None
    metadata_errors: tuple[str, ...]
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]
    shape_errors: tuple[str, ...]
    qtype_errors: tuple[str, ...]
    ignored: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not (
            self.metadata_errors
            or self.missing
            or self.unexpected
            or self.shape_errors
            or self.qtype_errors
        )

    def raise_for_errors(self) -> None:
        if self.passed:
            return
        parts: list[str] = []
        for label, errors, limit in (
            ("metadata errors", self.metadata_errors, 4),
            ("missing tensors", self.missing, 8),
            ("unexpected tensors", self.unexpected, 8),
            ("shape errors", self.shape_errors, 4),
            ("type errors", self.qtype_errors, 4),
        ):
            if not errors:
                continue
            preview = "; ".join(errors[:limit])
            more = "" if len(errors) <= limit else f" (+{len(errors) - limit} more)"
            parts.append(f"{label}: {preview}{more}")
        raise Gemma4VisionGGUFError("; ".join(parts))


@dataclass(frozen=True)
class Gemma4VisionGGUFMap:
    """Canonical slot map for one validated Gemma 4 mmproj artifact."""

    config: Gemma4VisionGGUFConfig
    tensors: Mapping[str, GGUFTensorInfo]
    validation: Gemma4VisionGGUFValidation

    def tensor(self, slot: str) -> GGUFTensorInfo:
        try:
            return self.tensors[str(slot)]
        except KeyError as exc:
            raise MissingGGUFTensorError(
                f"Gemma 4 vision tower has no GGUF tensor slot {slot!r}"
            ) from exc

    def has(self, slot: str) -> bool:
        return str(slot) in self.tensors

    @property
    def slot_names(self) -> tuple[str, ...]:
        return tuple(self.tensors)

    @property
    def tensor_names(self) -> tuple[str, ...]:
        return tuple(tensor.name for tensor in self.tensors.values())

    @property
    def slot_tensor_names(self) -> Mapping[str, str]:
        """Canonical slot -> GGUF tensor name for every tensor present."""

        return MappingProxyType(
            {slot: tensor.name for slot, tensor in self.tensors.items()}
        )


# ---------------------------------------------------------------------------
# Slot naming
# ---------------------------------------------------------------------------


def gemma4_vision_layer_slot(layer_id: int, slot: str) -> str:
    """Return the canonical weight slot for one layer tensor."""

    return f"layers.{int(layer_id)}.{slot}"


def required_gemma4_vision_slot_names(config: Gemma4VisionGGUFConfig) -> tuple[str, ...]:
    slots: list[str] = list(_ROOT_SLOTS)
    for layer_id in range(config.block_count):
        slots.extend(gemma4_vision_layer_slot(layer_id, slot) for slot in _LAYER_SLOTS)
    return tuple(slots)


def required_gemma4_vision_tensor_names(
    config: Gemma4VisionGGUFConfig,
) -> tuple[str, ...]:
    """Return the GGUF tensor names the vision graph requires, in slot order."""

    names = list(_ROOT_SLOTS.values())
    for layer_id in range(config.block_count):
        names.extend(f"v.blk.{layer_id}.{suffix}.weight" for suffix in _LAYER_SLOTS)
    return tuple(names)


# ---------------------------------------------------------------------------
# Metadata decoding
# ---------------------------------------------------------------------------


def gemma4_vision_config_from_metadata(
    info: GGUFModelInfo,
) -> Gemma4VisionGGUFConfig:
    """Decode and validate vision-tower dimensions without reading weights.

    Raises :class:`Gemma4VisionGGUFError` on any metadata that this contract
    cannot execute.
    """

    config, errors = _config_from_metadata(info)
    if errors or config is None:
        raise Gemma4VisionGGUFError("; ".join(errors) or "Gemma 4 vision metadata is invalid")
    return config


def _config_from_metadata(
    info: GGUFModelInfo,
) -> tuple[Gemma4VisionGGUFConfig | None, tuple[str, ...]]:
    metadata = info.metadata
    errors: list[str] = []

    architecture = str(metadata.get("general.architecture", GEMMA4_VISION_ARCHITECTURE))
    if architecture != GEMMA4_VISION_ARCHITECTURE:
        errors.append(
            f"expected a {GEMMA4_VISION_ARCHITECTURE!r} mmproj artifact, got {architecture!r}"
        )

    if metadata.get("clip.has_vision_encoder") is False:
        errors.append("artifact declares clip.has_vision_encoder = false")

    projector_type = str(
        metadata.get("clip.vision.projector_type", metadata.get("clip.projector_type", ""))
    )
    if projector_type != GEMMA4V_PROJECTOR_TYPE:
        errors.append(
            f"vision projector type {projector_type!r} is not supported; this contract "
            f"executes {GEMMA4V_PROJECTOR_TYPE!r}"
        )

    block_count = _int_field(metadata, "clip.vision.block_count", errors)
    hidden_size = _int_field(metadata, "clip.vision.embedding_length", errors)
    intermediate_size = _int_field(metadata, "clip.vision.feed_forward_length", errors)
    heads = _int_field(metadata, "clip.vision.attention.head_count", errors)
    patch_size = _int_field(metadata, "clip.vision.patch_size", errors)

    head_count_kv = _int_field(
        metadata, "clip.vision.attention.head_count_kv", errors, default=heads or None
    )
    output_size = _int_field(
        metadata, "clip.vision.projection_dim", errors, default=_projection_dim(info)
    )
    pooling_kernel_size = _int_field(
        metadata, "clip.vision.projector.scale_factor", errors, default=DEFAULT_POOLING_KERNEL_SIZE
    )
    # ``image_size`` is informational for this contract (resize is driven by the
    # patch budget), and a converter may declare 0 for a dynamic-size artifact.
    image_size = _int_field(
        metadata, "clip.vision.image_size", errors, default=DEFAULT_IMAGE_SIZE, minimum=0
    )
    # A soft-token budget is a resource choice, not a format constraint.
    max_soft_tokens = _int_field(
        metadata, "clip.vision.max_soft_tokens", errors, default=DEFAULT_MAX_SOFT_TOKENS
    )
    rms_norm_eps = _float_field(
        metadata, "clip.vision.attention.layer_norm_epsilon", errors, default=1.0e-6
    )
    # No converter writes the vision RoPE theta; the reference tower uses 100.0.
    rope_theta_key = next(
        (
            key
            for key in ("clip.vision.rope_theta", "clip.vision.rope.theta")
            if metadata.get(key) is not None
        ),
        None,
    )
    rope_theta = (
        DEFAULT_ROPE_THETA
        if rope_theta_key is None
        else _float_field(metadata, rope_theta_key, errors)
    )

    head_dim = 0
    if heads and hidden_size:
        if hidden_size % heads:
            errors.append(
                f"vision hidden size {hidden_size} is not divisible by {heads} attention heads"
            )
        else:
            head_dim = hidden_size // heads
            if head_dim % 4:
                # Two-axis rotary embeddings rotate head_dim/2 per axis in pairs.
                errors.append(
                    f"vision head_dim {head_dim} must be a multiple of 4 for two-axis "
                    "rotary position embeddings"
                )
    if head_count_kv and heads and heads % head_count_kv:
        errors.append(
            f"vision head count {heads} is not divisible by head_count_kv {head_count_kv}"
        )

    if patch_size and image_size and image_size % patch_size:
        errors.append(
            f"vision image size {image_size} is not a multiple of patch size {patch_size}"
        )

    position_table_size = _position_table_size(info, errors)

    if errors:
        return None, tuple(errors)

    present = {tensor.name for tensor in info.tensors}
    standardize = "v.std_bias" in present or "v.std_scale" in present
    clamp_suffixes = tuple(f".{field}" for field in _CLAMP_FIELDS)
    clipped = any(
        _is_vision_name(name) and name.endswith(clamp_suffixes) for name in present
    )

    return (
        Gemma4VisionGGUFConfig(
            architecture=architecture,
            projector_type=projector_type,
            block_count=block_count,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            output_size=output_size,
            heads=heads,
            head_count_kv=head_count_kv,
            patch_size=patch_size,
            pooling_kernel_size=pooling_kernel_size,
            position_table_size=position_table_size,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            max_soft_tokens=max_soft_tokens,
            image_size=image_size,
            standardize=standardize,
            clipped_linears=clipped,
        ),
        (),
    )


def _projection_dim(info: GGUFModelInfo) -> int | None:
    """Read the projector width from the tensor table when metadata omits it."""

    tensor = next(
        (item for item in info.tensors if item.name == "mm.input_projection.weight"), None
    )
    if tensor is None or len(tensor.shape) != 2:
        return None
    return int(tensor.shape[0])


def _position_table_size(info: GGUFModelInfo, errors: list[str]) -> int:
    """Return the per-axis position table capacity from the tensor table."""

    tensor = next(
        (item for item in info.tensors if item.name == "v.position_embd.weight"), None
    )
    if tensor is None:
        # The missing tensor is reported by the tensor inventory, not here.
        return 0
    if len(tensor.shape) != 3 or tensor.shape[0] != POSITION_TABLE_AXES:
        errors.append(
            "v.position_embd.weight must have shape "
            f"({POSITION_TABLE_AXES}, position_table_size, hidden_size), got {tuple(tensor.shape)}"
        )
        return 0
    return int(tensor.shape[1])


# ---------------------------------------------------------------------------
# Tensor contract
# ---------------------------------------------------------------------------


def gemma4_vision_expected_shapes(
    config: Gemma4VisionGGUFConfig,
) -> Mapping[str, tuple[int, ...]]:
    """Return the shape each weight slot must carry, keyed by slot.

    Optional clamp scalars are absent here: they are ``(1,)`` scalars, validated
    by the clamp check, and :attr:`Gemma4VisionGGUFMap.tensors` is the
    authoritative inventory of what an artifact actually carries.
    """

    hidden = config.hidden_size
    head_dim = config.head_dim
    query_width = config.query_width
    kv_width = config.kv_width
    per_layer = {
        "ln1": (hidden,),
        "ln2": (hidden,),
        "attn_q": (query_width, hidden),
        "attn_k": (kv_width, hidden),
        "attn_v": (kv_width, hidden),
        "attn_out": (hidden, query_width),
        "attn_q_norm": (head_dim,),
        "attn_k_norm": (head_dim,),
        "attn_post_norm": (hidden,),
        "ffn_post_norm": (hidden,),
        "ffn_gate": (config.intermediate_size, hidden),
        "ffn_up": (config.intermediate_size, hidden),
        "ffn_down": (hidden, config.intermediate_size),
        "out_scale": (1,),
    }
    shapes: dict[str, tuple[int, ...]] = {
        "patch_embd": (hidden, PATCH_CHANNELS, config.patch_size, config.patch_size),
        "position_embd": (POSITION_TABLE_AXES, config.position_table_size, hidden),
        "input_projection": (config.output_size, hidden),
        "std_bias": (hidden,),
        "std_scale": (hidden,),
    }
    for layer_id in range(config.block_count):
        shapes.update(
            {gemma4_vision_layer_slot(layer_id, slot): shape for slot, shape in per_layer.items()}
        )
    return MappingProxyType(shapes)


def _slot_to_tensor_name(config: Gemma4VisionGGUFConfig) -> dict[str, str]:
    names: dict[str, str] = {**_ROOT_SLOTS, **_OPTIONAL_ROOT_SLOTS}
    for layer_id in range(config.block_count):
        for slot in (*_LAYER_SLOTS, *_OPTIONAL_LAYER_SLOTS):
            names[gemma4_vision_layer_slot(layer_id, slot)] = f"v.blk.{layer_id}.{slot}.weight"
    for slot in _CLAMP_CAPABLE_ROOT_SLOTS:
        for field in _CLAMP_FIELDS:
            names[f"{slot}.{field}"] = f"{_ROOT_SLOTS[slot].rsplit('.', 1)[0]}.{field}"
    for layer_id in range(config.block_count):
        for slot in _CLAMP_CAPABLE_LAYER_SLOTS:
            for field in _CLAMP_FIELDS:
                names[gemma4_vision_layer_slot(layer_id, f"{slot}.{field}")] = (
                    f"v.blk.{layer_id}.{slot}.{field}"
                )
    return names


def validate_gemma4_vision_gguf(info: GGUFModelInfo) -> Gemma4VisionGGUFValidation:
    """Validate one mmproj artifact's metadata and complete tensor inventory."""

    config, metadata_errors = _config_from_metadata(info)
    if config is None:
        return Gemma4VisionGGUFValidation(
            config=None,
            metadata_errors=metadata_errors,
            missing=(),
            unexpected=(),
            shape_errors=(),
            qtype_errors=(),
            ignored=tuple(sorted(_ignored_names(info))),
        )

    by_name = {tensor.name: tensor for tensor in info.tensors}
    slot_names = _slot_to_tensor_name(config)
    expected_shapes = gemma4_vision_expected_shapes(config)
    required = set(required_gemma4_vision_slot_names(config))

    missing = tuple(
        sorted(slot_names[slot] for slot in required if slot_names[slot] not in by_name)
    )
    unexpected: list[str] = []
    shape_errors: list[str] = []
    qtype_errors: list[str] = []
    for slot, name in sorted(slot_names.items()):
        tensor = by_name.get(name)
        if tensor is None:
            continue
        expected = expected_shapes.get(slot)
        if expected is not None and tuple(tensor.shape) != tuple(expected):
            shape_errors.append(f"{name} expected {expected} got {tuple(tensor.shape)}")
        if not dequantization_supported(tensor.ggml_type):
            qtype_errors.append(
                f"{name} uses GGUF type {tensor.ggml_type_name}, which has no CPU dequantizer"
            )
        elif len(tensor.shape) <= 1:
            # Every one-dimensional tensor here is a per-channel scale or a clamp
            # scalar; block-quantized storage of one is meaningless.
            if tensor.ggml_type_name not in _FLOAT_TYPE_NAMES:
                qtype_errors.append(
                    f"{name} must be stored as a float type, got {tensor.ggml_type_name}"
                )
    known_names = set(slot_names.values())
    unexpected.extend(
        sorted(name for name in by_name if _is_vision_name(name) and name not in known_names)
    )
    clamp_missing, clamp_shapes = _clamp_errors(config, by_name)
    shape_errors.extend(clamp_shapes)
    shape_errors.extend(_standardize_errors(by_name))

    return Gemma4VisionGGUFValidation(
        config=config,
        metadata_errors=(),
        missing=tuple(sorted((*missing, *clamp_missing))),
        unexpected=tuple(unexpected),
        shape_errors=tuple(shape_errors),
        qtype_errors=tuple(qtype_errors),
        ignored=tuple(sorted(_ignored_names(info))),
    )


def _standardize_errors(by_name: Mapping[str, GGUFTensorInfo]) -> list[str]:
    present = [name for name in ("v.std_bias", "v.std_scale") if name in by_name]
    if len(present) == 1:
        return [
            "v.std_bias and v.std_scale must be present together; found only "
            + present[0]
        ]
    return []


def _clamp_errors(
    config: Gemma4VisionGGUFConfig,
    by_name: Mapping[str, GGUFTensorInfo],
) -> tuple[list[str], list[str]]:
    """Check the optional Gemma4ClippableLinear scalars: complete, shape (1,)."""

    groups: dict[str, dict[str, str]] = {}
    for slot, name in _slot_to_tensor_name(config).items():
        head, _, field = slot.rpartition(".")
        if field in _CLAMP_FIELDS:
            groups.setdefault(head, {})[field] = name
    missing: list[str] = []
    shapes: list[str] = []
    for head, fields in sorted(groups.items()):
        present = {field: name for field, name in fields.items() if name in by_name}
        if not present:
            continue
        absent = [field for field in _CLAMP_FIELDS if field not in present]
        if absent:
            missing.extend(fields[field] for field in absent)
            continue
        for field, name in sorted(present.items()):
            if tuple(by_name[name].shape) != (1,):
                shapes.append(f"{name} expected (1,) got {tuple(by_name[name].shape)}")
    return missing, shapes


def build_gemma4_vision_gguf_map(
    info: GGUFModelInfo,
    *,
    strict: bool = True,
) -> Gemma4VisionGGUFMap:
    """Validate the artifact and return its slot map."""

    validation = validate_gemma4_vision_gguf(info)
    if strict:
        validation.raise_for_errors()
    if validation.config is None:
        raise Gemma4VisionGGUFError("; ".join(validation.metadata_errors))
    config = validation.config
    by_name = {tensor.name: tensor for tensor in info.tensors}
    tensors = MappingProxyType(
        {
            slot: by_name[name]
            for slot, name in _slot_to_tensor_name(config).items()
            if name in by_name
        }
    )
    return Gemma4VisionGGUFMap(config=config, tensors=tensors, validation=validation)


def _is_vision_name(name: str) -> bool:
    """True for a tensor this contract reads: the ``v.`` tower and its projector."""

    if name.startswith(_AUDIO_PREFIX):
        return False
    return name.startswith(_VISION_PREFIX) or name.startswith("mm.")


def _ignored_names(info: GGUFModelInfo) -> tuple[str, ...]:
    """Return tensor names that belong to another modality in the same file."""

    return tuple(tensor.name for tensor in info.tensors if not _is_vision_name(tensor.name))


# ---------------------------------------------------------------------------
# Weight reading
# ---------------------------------------------------------------------------


def read_gemma4_vision_tensor(info: GGUFModelInfo, tensor: GGUFTensorInfo) -> np.ndarray:
    """Return one vision tensor dequantized to float32 without rescanning the header."""

    storage = np.memmap(
        info.path,
        mode="r",
        dtype=numpy_storage_dtype(tensor.ggml_type),
        offset=tensor.data_offset,
        shape=tensor.byte_shape,
    )
    return np.asarray(
        dequantize_gguf_data(storage, tensor.ggml_type), dtype=np.float32
    ).reshape(tensor.shape)


def read_gemma4_vision_weights(
    info: GGUFModelInfo,
    model_map: Gemma4VisionGGUFMap,
    *,
    slots: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    """Dequantize the vision weights into a slot-keyed float32 mapping.

    The keys are the canonical slots (``patch_embd``, ``layers.0.attn_q``, …), so
    a caller can hand the result straight to the NumPy reference forward.
    """

    wanted = tuple(model_map.slot_names) if slots is None else tuple(slots)
    weights: dict[str, np.ndarray] = {}
    for slot in wanted:
        weights[slot] = read_gemma4_vision_tensor(info, model_map.tensor(slot))
    return weights


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


def _raw_number(
    metadata: Mapping[str, Any],
    key: str,
    errors: list[str],
    *,
    default: float | None,
    integral: bool,
) -> Any:
    """Metadata value coerced to int/float, or None after recording an error."""

    value = metadata.get(key)
    if value is None:
        if default is None:
            errors.append(f"GGUF metadata is missing required key {key!r}")
            return None
        value = default
    try:
        return int(value) if integral else float(value)
    except (TypeError, ValueError):
        kind = "an integer" if integral else "a float"
        errors.append(f"GGUF key {key!r} must be {kind}, got {value!r}")
        return None


def _int_field(
    metadata: Mapping[str, Any],
    key: str,
    errors: list[str],
    *,
    default: int | None = None,
    minimum: int = 1,
) -> int:
    number = _raw_number(metadata, key, errors, default=default, integral=True)
    if number is None:
        return 0
    if number < minimum:
        rule = "positive" if minimum else "non-negative"
        errors.append(f"GGUF key {key!r} must be {rule}, got {number}")
        return 0
    return number


def _float_field(
    metadata: Mapping[str, Any],
    key: str,
    errors: list[str],
    *,
    default: float | None = None,
) -> float:
    number = _raw_number(metadata, key, errors, default=default, integral=False)
    if number is None:
        return 0.0
    if not np.isfinite(number) or number <= 0.0:
        errors.append(f"GGUF key {key!r} must be positive and finite, got {number}")
        return 0.0
    return number


__all__ = [
    "DEFAULT_IMAGE_SIZE",
    "DEFAULT_MAX_SOFT_TOKENS",
    "DEFAULT_POOLING_KERNEL_SIZE",
    "DEFAULT_ROPE_THETA",
    "GEMMA4V_PROJECTOR_TYPE",
    "GEMMA4_VISION_ARCHITECTURE",
    "Gemma4VisionGGUFConfig",
    "Gemma4VisionGGUFError",
    "Gemma4VisionGGUFMap",
    "Gemma4VisionGGUFValidation",
    "REFERENCE_MAX_SOFT_TOKEN_PRESETS",
    "build_gemma4_vision_gguf_map",
    "gemma4_vision_config_from_metadata",
    "gemma4_vision_expected_shapes",
    "gemma4_vision_layer_slot",
    "read_gemma4_vision_tensor",
    "read_gemma4_vision_weights",
    "required_gemma4_vision_slot_names",
    "required_gemma4_vision_tensor_names",
    "validate_gemma4_vision_gguf",
]
