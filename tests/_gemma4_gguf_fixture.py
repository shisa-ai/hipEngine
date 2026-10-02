"""Tiny structurally real Gemma 4 GGUF writer for loader and materializer tests.

Writes a two-layer artifact — one sliding layer and one global layer — with the
metadata keys and tensor inventory llama.cpp emits for ``gemma4``. The point is
that the loader, the tensor contract, and (later) the materializer are exercised
through the real ``scan_gguf`` / ``build_gemma4_gguf_tensor_map`` entry points on
a CPU-only host, not through isolated helpers.

Two properties are load-bearing and are reproduced deliberately:

* ``rope_freqs.weight`` carries a finite prefix of ``1.0`` factors followed by
  the enormous unrotated sentinel, which is the only place the proportional
  ``partial_rotary_factor`` survives into the file.
* Global layers carry no ``attn_v.weight`` because ``attention_k_eq_v`` removes
  ``v_proj``; V is the raw K projection.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from hipengine.quant.gguf import GGMLQuantizationType, LlamaFileType
from tests._gguf_synthetic_weights import (
    make_q4_k_weight,
    make_q5_k_weight,
    make_q8_0_weight,
)

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
DEFAULT_ALIGNMENT = 32

# Chosen so the Q4_K/Q8_0 block constraints hold and the global rope_freqs
# tensor has both a rotated prefix and an unrotated tail.
FIXTURE_HIDDEN = 256
FIXTURE_VOCAB = 64
FIXTURE_FFN = 256
FIXTURE_EXPERTS = 4
FIXTURE_EXPERT_USED = 2
FIXTURE_EXPERT_FF = 64
FIXTURE_HEADS = 4
FIXTURE_KV_HEADS_SWA = 2
FIXTURE_KV_HEADS_GLOBAL = 1
FIXTURE_HEAD_DIM_SWA = 64
FIXTURE_HEAD_DIM_GLOBAL = 128
FIXTURE_SLIDING_WINDOW = 16
FIXTURE_LAYER_TYPES = ("sliding_attention", "full_attention")
FIXTURE_GLOBAL_ROTATED_PAIRS = 16  # partial_rotary_factor 0.25 of 64 pairs
FIXTURE_LAYER_COUNT = len(FIXTURE_LAYER_TYPES)

# A small but structurally faithful Gemma 4 SPM vocabulary. It carries the same
# three token classes the real artifact does, and the same deliberate quirk:
# ``<eos>`` is written with type NORMAL even though it is an added token.
FIXTURE_SPM_SPACE = "\u2581"
FIXTURE_TOKEN_STRINGS: tuple[tuple[str, int], ...] = (
    ("<pad>", 3),  # CONTROL
    ("<eos>", 1),  # NORMAL, as in the real artifact
    ("<bos>", 3),
    ("<unk>", 3),
    ("<mask>", 3),
    ("<|tool_response>", 4),  # USER_DEFINED: matched literally, rendered
    ("<turn|>", 3),  # CONTROL: matched literally, skipped when decoding
    (FIXTURE_SPM_SPACE, 1),
    ("a", 1),
    ("b", 1),
    ("c", 1),
    ("e", 1),
    ("h", 1),
    ("l", 1),
    ("o", 1),
    ("w", 1),
    ("r", 1),
    ("d", 1),
    ("\n", 1),
    (FIXTURE_SPM_SPACE + "a", 1),
    (FIXTURE_SPM_SPACE + "ab", 1),
    (FIXTURE_SPM_SPACE + "abc", 1),
    ("abc", 1),
    ("ab", 1),
    ("he", 1),
    ("hel", 1),
    ("hell", 1),
    ("hello", 1),
    (FIXTURE_SPM_SPACE + "h", 1),
    (FIXTURE_SPM_SPACE + "he", 1),
    (FIXTURE_SPM_SPACE + "hel", 1),
    (FIXTURE_SPM_SPACE + "hell", 1),
    (FIXTURE_SPM_SPACE + "hello", 1),
    (FIXTURE_SPM_SPACE + "w", 1),
    (FIXTURE_SPM_SPACE + "wo", 1),
    (FIXTURE_SPM_SPACE + "wor", 1),
    (FIXTURE_SPM_SPACE + "worl", 1),
    (FIXTURE_SPM_SPACE + "world", 1),
    *((f"<0x{byte:02X}>", 6) for byte in range(256)),  # UNUSED byte tokens
)
FIXTURE_TOKEN_IDS = {token: index for index, (token, _) in enumerate(FIXTURE_TOKEN_STRINGS)}
FIXTURE_MERGES: tuple[str, ...] = (
    f"{FIXTURE_SPM_SPACE} a",
    f"{FIXTURE_SPM_SPACE}a b",
    f"{FIXTURE_SPM_SPACE}ab c",
    "a b",
    "ab c",
    f"{FIXTURE_SPM_SPACE} h",
    f"{FIXTURE_SPM_SPACE}h e",
    f"{FIXTURE_SPM_SPACE}he l",
    f"{FIXTURE_SPM_SPACE}hel l",
    f"{FIXTURE_SPM_SPACE}hell o",
    "h e",
    "he l",
    "hel l",
    "hell o",
    f"{FIXTURE_SPM_SPACE} w",
    f"{FIXTURE_SPM_SPACE}w o",
    f"{FIXTURE_SPM_SPACE}wo r",
    f"{FIXTURE_SPM_SPACE}wor l",
    f"{FIXTURE_SPM_SPACE}worl d",
)
FIXTURE_BOS_TOKEN_ID = FIXTURE_TOKEN_IDS["<bos>"]
FIXTURE_EOS_TOKEN_ID = FIXTURE_TOKEN_IDS["<turn|>"]
FIXTURE_CHAT_TEMPLATE = (
    "{{ bos_token }}{% for m in messages %}{{ '<|turn>' + m['role'] + '\\n' }}"
    "{{ m['content'] }}{{ '<turn|>' + '\\n' }}{% endfor %}"
)

_UNROTATED_FREQ_FACTOR = np.float32(1.0e30)


def _gguf_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _gguf_scalar(value_type: int, value: object) -> bytes:
    if value_type == 4:  # UINT32
        return struct.pack("<I", int(value))
    if value_type == 5:  # INT32
        return struct.pack("<i", int(value))
    if value_type == 6:  # FLOAT32
        return struct.pack("<f", float(value))
    if value_type == 7:  # BOOL
        return struct.pack("<?", bool(value))
    if value_type == 10:  # UINT64
        return struct.pack("<Q", int(value))
    raise ValueError(f"unsupported GGUF scalar type {value_type}")


def _gguf_value(value_type: int, value: object) -> bytes:
    if value_type == 8:  # STRING
        return _gguf_string(str(value))
    if value_type == 9:  # ARRAY: (element type, elements)
        item_type, items = value
        return struct.pack("<IQ", item_type, len(items)) + b"".join(
            _gguf_value(item_type, item) for item in items
        )
    return _gguf_scalar(value_type, value)


def _align_up(value: int, alignment: int) -> int:
    return (int(value) + int(alignment) - 1) // int(alignment) * int(alignment)


def rope_freqs_values(
    head_dim: int = FIXTURE_HEAD_DIM_GLOBAL,
    rotated_pairs: int = FIXTURE_GLOBAL_ROTATED_PAIRS,
) -> np.ndarray:
    """Return a ``rope_freqs`` tensor with a rotated prefix and an unrotated tail."""

    half = head_dim // 2
    if not 0 <= rotated_pairs <= half:
        raise ValueError("rotated_pairs must be within the head half-width")
    values = np.empty(half, dtype=np.float32)
    values[:rotated_pairs] = np.float32(1.0)
    values[rotated_pairs:] = _UNROTATED_FREQ_FACTOR
    return values


def payload_bytes(shape: Sequence[int], qtype: GGMLQuantizationType) -> bytes:
    """Return a valid GGUF payload for one fixture tensor."""

    shape = tuple(int(dim) for dim in shape)
    if qtype in (GGMLQuantizationType.F32, GGMLQuantizationType.F16):
        dtype = np.float32 if qtype == GGMLQuantizationType.F32 else np.float16
        elements = 1
        for dim in shape:
            elements *= dim
        return (np.arange(elements, dtype=np.float32) * 0.125 - 1.0).astype(dtype).tobytes()
    if qtype == GGMLQuantizationType.BF16:
        elements = 1
        for dim in shape:
            elements *= dim
        return np.zeros(elements, dtype=np.uint16).tobytes()
    out_features = int(np.prod(shape[:-1])) if len(shape) > 1 else 1
    in_features = shape[-1]
    if qtype == GGMLQuantizationType.Q8_0:
        return make_q8_0_weight(out_features, in_features).tobytes()
    if qtype == GGMLQuantizationType.Q4_K:
        return make_q4_k_weight(out_features, in_features).tobytes()
    if qtype == GGMLQuantizationType.Q5_K:
        return make_q5_k_weight(out_features, in_features).tobytes()
    raise ValueError(f"no fixture payload builder for {qtype.name}")


def _layer_tensors(
    layer_id: int,
    *,
    layer_type: str,
    projection_type: GGMLQuantizationType,
    expert_type: GGMLQuantizationType,
) -> list[tuple[str, tuple[int, ...], GGMLQuantizationType]]:
    prefix = f"blk.{layer_id}"
    if layer_type == "sliding_attention":
        head_dim = FIXTURE_HEAD_DIM_SWA
        kv_heads = FIXTURE_KV_HEADS_SWA
    else:
        head_dim = FIXTURE_HEAD_DIM_GLOBAL
        kv_heads = FIXTURE_KV_HEADS_GLOBAL
    query_width = FIXTURE_HEADS * head_dim
    kv_width = kv_heads * head_dim

    tensors: list[tuple[str, tuple[int, ...], GGMLQuantizationType]] = [
        (f"{prefix}.attn_norm.weight", (FIXTURE_HIDDEN,), GGMLQuantizationType.F32),
        (f"{prefix}.attn_q.weight", (query_width, FIXTURE_HIDDEN), projection_type),
        (f"{prefix}.attn_k.weight", (kv_width, FIXTURE_HIDDEN), projection_type),
        (f"{prefix}.attn_q_norm.weight", (head_dim,), GGMLQuantizationType.F32),
        (f"{prefix}.attn_k_norm.weight", (head_dim,), GGMLQuantizationType.F32),
        (f"{prefix}.attn_output.weight", (FIXTURE_HIDDEN, query_width), projection_type),
        (f"{prefix}.post_attention_norm.weight", (FIXTURE_HIDDEN,), GGMLQuantizationType.F32),
        (f"{prefix}.ffn_norm.weight", (FIXTURE_HIDDEN,), GGMLQuantizationType.F32),
        (f"{prefix}.ffn_gate.weight", (FIXTURE_FFN, FIXTURE_HIDDEN), projection_type),
        (f"{prefix}.ffn_up.weight", (FIXTURE_FFN, FIXTURE_HIDDEN), projection_type),
        (f"{prefix}.ffn_down.weight", (FIXTURE_HIDDEN, FIXTURE_FFN), projection_type),
        (f"{prefix}.ffn_gate_inp.weight", (FIXTURE_EXPERTS, FIXTURE_HIDDEN), GGMLQuantizationType.F32),
        (f"{prefix}.ffn_gate_inp.scale", (FIXTURE_HIDDEN,), GGMLQuantizationType.F32),
        (f"{prefix}.pre_ffw_norm_2.weight", (FIXTURE_HIDDEN,), GGMLQuantizationType.F32),
        (f"{prefix}.post_ffw_norm.weight", (FIXTURE_HIDDEN,), GGMLQuantizationType.F32),
        (f"{prefix}.post_ffw_norm_1.weight", (FIXTURE_HIDDEN,), GGMLQuantizationType.F32),
        (f"{prefix}.post_ffw_norm_2.weight", (FIXTURE_HIDDEN,), GGMLQuantizationType.F32),
        (
            f"{prefix}.ffn_gate_up_exps.weight",
            (FIXTURE_EXPERTS, 2 * FIXTURE_EXPERT_FF, FIXTURE_HIDDEN),
            expert_type,
        ),
        (
            f"{prefix}.ffn_down_exps.weight",
            (FIXTURE_EXPERTS, FIXTURE_HIDDEN, FIXTURE_EXPERT_FF),
            projection_type,
        ),
        (f"{prefix}.ffn_down_exps.scale", (FIXTURE_EXPERTS,), GGMLQuantizationType.F32),
        (f"{prefix}.layer_output_scale.weight", (1,), GGMLQuantizationType.F32),
    ]
    if layer_type == "sliding_attention":
        tensors.insert(
            3,
            (f"{prefix}.attn_v.weight", (kv_width, FIXTURE_HIDDEN), projection_type),
        )
    return tensors


def default_fixture_tensors(
    *,
    layer_types: Sequence[str] = FIXTURE_LAYER_TYPES,
    projection_type: GGMLQuantizationType = GGMLQuantizationType.Q8_0,
    expert_type: GGMLQuantizationType = GGMLQuantizationType.Q4_K,
    embedding_type: GGMLQuantizationType = GGMLQuantizationType.Q8_0,
    vocab: int = FIXTURE_VOCAB,
) -> list[tuple[str, tuple[int, ...], GGMLQuantizationType]]:
    """Return the fixture tensor list.

    ``vocab`` sizes the embedding and must match whatever the metadata declares,
    since the loader checks the two against each other. ``tokenizer_fixture_metadata``
    carries a wider token list than ``FIXTURE_VOCAB``, so an end-to-end generation
    test has to widen this to match or the load fails with a shape error.
    """

    tensors: list[tuple[str, tuple[int, ...], GGMLQuantizationType]] = [
        ("token_embd.weight", (vocab, FIXTURE_HIDDEN), embedding_type),
        ("output_norm.weight", (FIXTURE_HIDDEN,), GGMLQuantizationType.F32),
        ("rope_freqs.weight", (FIXTURE_HEAD_DIM_GLOBAL // 2,), GGMLQuantizationType.F32),
    ]
    for layer_id, layer_type in enumerate(layer_types):
        tensors.extend(
            _layer_tensors(
                layer_id,
                layer_type=layer_type,
                projection_type=projection_type,
                expert_type=expert_type,
            )
        )
    return tensors


def fixture_metadata(
    *,
    layer_types: Sequence[str] = FIXTURE_LAYER_TYPES,
    file_type: int = int(LlamaFileType.MOSTLY_Q4_K_M),
    extra: Mapping[str, object] | None = None,
    tokens: Sequence[str] | None = None,
    token_types: Sequence[int] | None = None,
    merges: Sequence[str] | None = None,
    tokenizer_model: str = "gemma4",
) -> list[tuple[str, int, object]]:
    layer_count = len(layer_types)
    kv_heads = [
        FIXTURE_KV_HEADS_SWA if layer_type == "sliding_attention" else FIXTURE_KV_HEADS_GLOBAL
        for layer_type in layer_types
    ]
    metadata: list[tuple[str, int, object]] = [
        ("general.architecture", 8, "gemma4"),
        ("general.alignment", 4, DEFAULT_ALIGNMENT),
        ("general.file_type", 4, int(file_type)),
        ("gemma4.block_count", 4, layer_count),
        ("gemma4.embedding_length", 4, FIXTURE_HIDDEN),
        ("gemma4.context_length", 4, 128),
        ("gemma4.attention.head_count", 4, FIXTURE_HEADS),
        ("gemma4.attention.head_count_kv", 9, (4, list(kv_heads))),
        ("gemma4.attention.key_length", 4, FIXTURE_HEAD_DIM_GLOBAL),
        ("gemma4.attention.key_length_swa", 4, FIXTURE_HEAD_DIM_SWA),
        ("gemma4.attention.value_length", 4, FIXTURE_HEAD_DIM_GLOBAL),
        ("gemma4.attention.value_length_swa", 4, FIXTURE_HEAD_DIM_SWA),
        ("gemma4.attention.layer_norm_rms_epsilon", 6, 1.0e-6),
        ("gemma4.attention.sliding_window", 4, FIXTURE_SLIDING_WINDOW),
        (
            "gemma4.attention.sliding_window_pattern",
            9,
            (7, [layer_type == "sliding_attention" for layer_type in layer_types]),
        ),
        ("gemma4.attention.shared_kv_layers", 4, 0),
        ("gemma4.embedding_length_per_layer_input", 4, 0),
        ("gemma4.expert_count", 4, FIXTURE_EXPERTS),
        ("gemma4.expert_used_count", 4, FIXTURE_EXPERT_USED),
        ("gemma4.expert_feed_forward_length", 4, FIXTURE_EXPERT_FF),
        ("gemma4.feed_forward_length", 4, FIXTURE_FFN),
        ("gemma4.final_logit_softcapping", 6, 30.0),
        ("gemma4.rope.dimension_count", 4, FIXTURE_HEAD_DIM_GLOBAL),
        ("gemma4.rope.dimension_count_swa", 4, FIXTURE_HEAD_DIM_SWA),
        ("gemma4.rope.freq_base", 6, 1_000_000.0),
        ("gemma4.rope.freq_base_swa", 6, 10_000.0),
        (
            "tokenizer.ggml.tokens",
            9,
            (
                8,
                list(
                    tokens
                    if tokens is not None
                    else [f"<tok{index}>" for index in range(FIXTURE_VOCAB)]
                ),
            ),
        ),
        ("tokenizer.ggml.model", 8, tokenizer_model),
        ("tokenizer.ggml.bos_token_id", 4, 2),
        ("tokenizer.ggml.eos_token_id", 4, 1),
    ]
    if token_types is not None:
        metadata.append(("tokenizer.ggml.token_type", 9, (5, list(token_types))))
    if merges is not None:
        metadata.append(("tokenizer.ggml.merges", 9, (8, list(merges))))
    if extra:
        metadata.extend(
            (key, 8 if isinstance(value, str) else 4, value) for key, value in extra.items()
        )
    return metadata


def tokenizer_fixture_metadata(
    *,
    extra: Mapping[str, object] | None = None,
    drop: Sequence[str] = (),
    **overrides: object,
) -> list[tuple[str, int, object]]:
    """Return loader metadata carrying the full Gemma 4 tokenizer section."""

    defaults: dict[str, object] = {
        "tokens": [token for token, _ in FIXTURE_TOKEN_STRINGS],
        "token_types": [kind for _, kind in FIXTURE_TOKEN_STRINGS],
        "merges": FIXTURE_MERGES,
    }
    defaults.update(overrides)
    metadata = fixture_metadata(**defaults)  # type: ignore[arg-type]
    merged_extra: dict[str, object] = {
        "tokenizer.ggml.bos_token_id": FIXTURE_BOS_TOKEN_ID,
        "tokenizer.ggml.eos_token_id": FIXTURE_EOS_TOKEN_ID,
        "tokenizer.ggml.add_bos_token": True,
        "tokenizer.chat_template": FIXTURE_CHAT_TEMPLATE,
    }
    if extra:
        merged_extra.update(extra)
    for key, value in merged_extra.items():
        metadata = [entry for entry in metadata if entry[0] != key]
        metadata.append((key, 8 if isinstance(value, str) else (7 if isinstance(value, bool) else 4), value))
    if drop:
        metadata = [entry for entry in metadata if entry[0] not in set(drop)]
    return metadata


def write_gemma4_gguf(
    path: str | Path,
    tensors: Sequence[tuple[str, Sequence[int], GGMLQuantizationType]],
    metadata: Sequence[tuple[str, int, object]],
    *,
    alignment: int = DEFAULT_ALIGNMENT,
    payload_overrides: Mapping[str, bytes] | None = None,
) -> Path:
    """Write one structurally real Gemma 4 GGUF file and return its path.

    ``payload_overrides`` replaces the synthetic payload for named tensors. It
    exists for ``rope_freqs.weight``, whose content is the only record of the
    proportional rotary span and therefore must not be a filler pattern.
    """

    overrides = dict(payload_overrides or {})
    tensor_blob = bytearray()
    records: list[tuple[str, tuple[int, ...], GGMLQuantizationType, int, int]] = []
    for name, shape, qtype in tensors:
        payload = overrides.get(name)
        if payload is None:
            payload = payload_bytes(shape, qtype)
        offset = _align_up(len(tensor_blob), alignment)
        tensor_blob += b"\x00" * (offset - len(tensor_blob))
        ggml_shape = tuple(reversed(tuple(int(dim) for dim in shape)))
        records.append((name, ggml_shape, qtype, offset, len(payload)))
        tensor_blob += payload

    header = bytearray()
    header += GGUF_MAGIC
    header += struct.pack("<IQQ", GGUF_VERSION, len(records), len(metadata))
    for key, value_type, value in metadata:
        header += _gguf_string(key)
        header += struct.pack("<I", int(value_type))
        header += _gguf_value(value_type, value)
    for name, ggml_shape, qtype, offset, _nbytes in records:
        header += _gguf_string(name)
        header += struct.pack("<I", len(ggml_shape))
        header += struct.pack(f"<{len(ggml_shape)}Q", *ggml_shape)
        header += struct.pack("<IQ", int(qtype), offset)
    data_start = _align_up(len(header), alignment)
    header += b"\x00" * (data_start - len(header))
    Path(path).write_bytes(bytes(header) + bytes(tensor_blob))
    return Path(path)


def default_payload_overrides() -> dict[str, bytes]:
    """Payload overrides every Gemma 4 fixture artifact needs.

    ``rope_freqs.weight`` is the only record of the proportional rotary span, so
    a filler pattern there would silently change the decoded architecture.
    """

    return {"rope_freqs.weight": rope_freqs_values().tobytes()}


def write_fixture_gguf(
    path: str | Path,
    tensors: Sequence[tuple[str, Sequence[int], GGMLQuantizationType]],
    metadata: Sequence[tuple[str, int, object]],
) -> Path:
    """Write a Gemma 4 fixture artifact with the required payload overrides."""

    return write_gemma4_gguf(
        path,
        tensors,
        metadata,
        payload_overrides=default_payload_overrides(),
    )


def write_default_gemma4_gguf(path: str | Path, **kwargs) -> Path:
    """Write the standard two-layer fixture artifact."""

    return write_fixture_gguf(path, default_fixture_tensors(**kwargs), fixture_metadata())


__all__ = [
    "FIXTURE_EXPERTS",
    "FIXTURE_EXPERT_FF",
    "FIXTURE_EXPERT_USED",
    "FIXTURE_FFN",
    "FIXTURE_GLOBAL_ROTATED_PAIRS",
    "FIXTURE_HEAD_DIM_GLOBAL",
    "FIXTURE_HEAD_DIM_SWA",
    "FIXTURE_HEADS",
    "FIXTURE_HIDDEN",
    "FIXTURE_KV_HEADS_GLOBAL",
    "FIXTURE_KV_HEADS_SWA",
    "FIXTURE_LAYER_COUNT",
    "FIXTURE_LAYER_TYPES",
    "FIXTURE_SLIDING_WINDOW",
    "FIXTURE_VOCAB",
    "default_fixture_tensors",
    "default_payload_overrides",
    "fixture_metadata",
    "payload_bytes",
    "rope_freqs_values",
    "write_default_gemma4_gguf",
    "write_fixture_gguf",
    "write_gemma4_gguf",
]
