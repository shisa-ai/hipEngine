"""Capability admission and shared-KV mapping for Gemma 4 MTP assistant sidecars.

An assistant sidecar is admitted on what the kernels can execute, never on
which file it is: the ``gemma4-assistant`` architecture, an output width equal
to the target's hidden size, an internally consistent layer table, and a
shared-KV mapping under which every draft layer's key/value geometry matches
the target layer whose cache it reads. No artifact name, path or digest takes
part in the decision.

The assistant keeps no KV cache of its own. Draft layer ``d`` reads the target
cache at ``shared_kv_target_layers(target_pattern)[d]``: sliding draft layers
read the target's last sliding layer and global draft layers read the target's
last global layer, so a 4-layer assistant whose pattern is ``[T, T, T, F]``
against a 30-layer target reads target layers 28 and 29.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from hipengine.loading.gguf import GGUFModelInfo, scan_gguf

_ASSISTANT_ARCH = "gemma4-assistant"
_TARGET_ARCH = "gemma4"


class Gemma4MTPGGUFError(ValueError):
    """Raised when an assistant sidecar fails capability admission."""


def _md_int(metadata: Mapping[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(metadata.get(key, default))
    except (TypeError, ValueError):
        return default


def _md_bools(metadata: Mapping[str, Any], key: str) -> tuple[bool, ...]:
    value = metadata.get(key)
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(bool(item) for item in value)


def _md_ints(metadata: Mapping[str, Any], key: str, block_count: int) -> tuple[int, ...]:
    value = metadata.get(key)
    if isinstance(value, (list, tuple)):
        return tuple(int(item) for item in value)
    if value is None:
        return ()
    return (int(value),) * block_count


@dataclass(frozen=True)
class Gemma4MTPConfig:
    """Assistant metadata needed for admission and for reading the target cache."""

    architecture: str
    block_count: int
    hidden_size: int
    output_width: int
    feed_forward_length: int
    head_count: int
    head_count_kv: tuple[int, ...]
    key_length: int
    key_length_swa: int
    value_length: int
    value_length_swa: int
    sliding_window_pattern: tuple[bool, ...]
    shared_kv_layers: int
    nextn_predict_layers: int
    vocab_size: int


@dataclass(frozen=True)
class Gemma4MTPValidation:
    config: Gemma4MTPConfig
    admission_errors: tuple[str, ...]
    missing_tensor_names: tuple[str, ...]
    shape_errors: tuple[str, ...]
    shared_kv_layers: tuple[int, ...]

    @property
    def passed(self) -> bool:
        return not (
            self.admission_errors or self.missing_tensor_names or self.shape_errors
        )


def parse_gemma4_mtp_config(info: GGUFModelInfo) -> Gemma4MTPConfig:
    """Decode assistant metadata without reading any weight payload."""

    metadata = info.metadata
    prefix = _ASSISTANT_ARCH
    block_count = _md_int(metadata, f"{prefix}.block_count")
    head_count_kv = _md_ints(metadata, f"{prefix}.attention.head_count_kv", block_count)
    pattern = _md_bools(metadata, f"{prefix}.attention.sliding_window_pattern")
    vocab = 0
    for tensor in info.tensors:
        if tensor.name == "token_embd.weight" and len(tensor.shape) >= 1:
            vocab = int(tensor.shape[0])
            break
    return Gemma4MTPConfig(
        architecture=str(metadata.get("general.architecture", "")),
        block_count=block_count,
        hidden_size=_md_int(metadata, f"{prefix}.embedding_length"),
        output_width=_md_int(metadata, f"{prefix}.embedding_length_out"),
        feed_forward_length=_md_int(metadata, f"{prefix}.feed_forward_length"),
        head_count=_md_int(metadata, f"{prefix}.attention.head_count"),
        head_count_kv=head_count_kv,
        key_length=_md_int(metadata, f"{prefix}.attention.key_length"),
        key_length_swa=_md_int(metadata, f"{prefix}.attention.key_length_swa"),
        value_length=_md_int(metadata, f"{prefix}.attention.value_length"),
        value_length_swa=_md_int(metadata, f"{prefix}.attention.value_length_swa"),
        sliding_window_pattern=pattern,
        shared_kv_layers=_md_int(metadata, f"{prefix}.attention.shared_kv_layers"),
        nextn_predict_layers=_md_int(metadata, f"{prefix}.nextn_predict_layers"),
        vocab_size=vocab,
    )


def _target_metadata(info: GGUFModelInfo) -> dict[str, Any]:
    """Pull the target fields the assistant must be compatible with."""

    metadata = info.metadata
    block_count = _md_int(metadata, "gemma4.block_count")
    return {
        "architecture": str(metadata.get("general.architecture", "")),
        "hidden_size": _md_int(metadata, "gemma4.embedding_length"),
        "block_count": block_count,
        "head_count_kv": _md_ints(metadata, "gemma4.attention.head_count_kv", block_count),
        "key_length": _md_int(metadata, "gemma4.attention.key_length"),
        "key_length_swa": _md_int(metadata, "gemma4.attention.key_length_swa"),
        "value_length": _md_int(metadata, "gemma4.attention.value_length"),
        "value_length_swa": _md_int(metadata, "gemma4.attention.value_length_swa"),
        "sliding_window_pattern": _md_bools(
            metadata, "gemma4.attention.sliding_window_pattern"
        ),
        "vocab_size": _token_embd_rows(info),
    }


def _token_embd_rows(info: GGUFModelInfo) -> int:
    """Vocabulary width as rows of the tied embedding, 0 when absent."""

    for tensor in info.tensors:
        if tensor.name == "token_embd.weight" and len(tensor.shape) >= 1:
            return int(tensor.shape[0])
    return 0


def shared_kv_target_layers(
    assistant_pattern: Sequence[bool], target_pattern: Sequence[bool]
) -> tuple[int, ...]:
    """Map each assistant layer to the target layer whose KV cache it reads.

    Sliding draft layers read the target's last sliding layer; global draft
    layers read the target's last global layer. Either target class must
    exist for a mapping to be well defined.
    """

    if not target_pattern:
        raise Gemma4MTPGGUFError("target declares no sliding_window_pattern")
    sliding_indices = [i for i, is_sliding in enumerate(target_pattern) if is_sliding]
    global_indices = [i for i, is_sliding in enumerate(target_pattern) if not is_sliding]
    if not sliding_indices or not global_indices:
        raise Gemma4MTPGGUFError(
            "target sliding_window_pattern must contain both sliding and global layers"
        )
    last_sliding, last_global = max(sliding_indices), max(global_indices)
    return tuple(
        last_sliding if is_sliding else last_global for is_sliding in assistant_pattern
    )


def _required_tensor_names(config: Gemma4MTPConfig) -> set[str]:
    names = {"rope_freqs.weight", "token_embd.weight", "output_norm.weight"}
    if config.block_count:
        names.update({"nextn.pre_projection.weight", "nextn.post_projection.weight"})
    layer_suffixes = (
        "attn_norm.weight",
        "layer_output_scale.weight",
        "ffn_down.weight",
        "ffn_gate.weight",
        "ffn_up.weight",
        "post_attention_norm.weight",
        "post_ffw_norm.weight",
        "ffn_norm.weight",
        "attn_output.weight",
        "attn_q_norm.weight",
        "attn_q.weight",
    )
    for layer in range(config.block_count):
        names.update(f"blk.{layer}.{suffix}" for suffix in layer_suffixes)
    return names


def _kv_width(heads: int, key_length: int, value_length: int) -> int:
    return int(heads) * max(int(key_length), int(value_length))


def validate_gemma4_mtp_gguf(
    target: GGUFModelInfo, assistant: GGUFModelInfo
) -> Gemma4MTPValidation:
    """Admit ``assistant`` for use with ``target`` on capability alone."""

    config = parse_gemma4_mtp_config(assistant)
    tgt = _target_metadata(target)
    errors: list[str] = []

    if config.architecture != _ASSISTANT_ARCH:
        errors.append(
            f"assistant architecture {config.architecture!r}, expected {_ASSISTANT_ARCH!r}"
        )
    if tgt["architecture"] != _TARGET_ARCH:
        errors.append(f"target architecture {tgt['architecture']!r}, expected {_TARGET_ARCH!r}")

    # The core capability gate: the draft's output width must be the target's
    # hidden width, because nextn.post_projection writes target-width states
    # and nextn.pre_projection consumes target-width embeddings plus h.
    if config.output_width and tgt["hidden_size"] and config.output_width != tgt["hidden_size"]:
        errors.append(
            f"assistant embedding_length_out {config.output_width} does not match target "
            f"hidden size {tgt['hidden_size']}"
        )
    # Drafted token ids must exist in the target's vocabulary, since the target
    # re-embeds every token the draft proposes.
    if (
        config.vocab_size
        and tgt["vocab_size"]
        and config.vocab_size != tgt["vocab_size"]
    ):
        errors.append(
            f"assistant vocabulary {config.vocab_size} does not match target vocabulary "
            f"{tgt['vocab_size']}"
        )

    if config.block_count <= 0:
        errors.append(f"assistant block_count {config.block_count} must be positive")
    if len(config.head_count_kv) != config.block_count:
        errors.append(
            f"head_count_kv has {len(config.head_count_kv)} entries for "
            f"{config.block_count} layers"
        )
    if config.sliding_window_pattern and (
        len(config.sliding_window_pattern) != config.block_count
    ):
        errors.append(
            f"sliding_window_pattern has {len(config.sliding_window_pattern)} entries for "
            f"{config.block_count} layers"
        )

    mapping: tuple[int, ...] = ()
    if config.sliding_window_pattern and tgt["sliding_window_pattern"]:
        try:
            mapping = shared_kv_target_layers(
                config.sliding_window_pattern, tgt["sliding_window_pattern"]
            )
        except Gemma4MTPGGUFError as exc:
            errors.append(str(exc))
    elif config.sliding_window_pattern:
        errors.append("target declares no sliding_window_pattern to map shared KV onto")

    # Every draft layer reads a target cache slot, so its key/value geometry
    # must equal that slot's. A mismatch would silently misread the cache.
    for draft_layer, target_layer in enumerate(mapping):
        if target_layer >= len(tgt["head_count_kv"]):
            errors.append(
                f"layer {draft_layer} maps to target layer {target_layer}, beyond the "
                f"target's {len(tgt['head_count_kv'])} layers"
            )
            continue
        if draft_layer >= len(config.head_count_kv):
            continue
        draft_sliding = config.sliding_window_pattern[draft_layer]
        target_sliding = tgt["sliding_window_pattern"][target_layer]
        if draft_sliding != target_sliding:
            errors.append(
                f"layer {draft_layer} is sliding={draft_sliding} but maps to target layer "
                f"{target_layer} which is sliding={target_sliding}"
            )
            continue
        draft_kv = _kv_width(
            config.head_count_kv[draft_layer],
            config.key_length_swa if draft_sliding else config.key_length,
            config.value_length_swa if draft_sliding else config.value_length,
        )
        target_kv = _kv_width(
            tgt["head_count_kv"][target_layer],
            tgt["key_length_swa"] if target_sliding else tgt["key_length"],
            tgt["value_length_swa"] if target_sliding else tgt["value_length"],
        )
        if draft_kv != target_kv:
            errors.append(
                f"layer {draft_layer} shared-KV width {draft_kv} does not match target layer "
                f"{target_layer} width {target_kv}"
            )

    actual = {tensor.name for tensor in assistant.tensors}
    required = _required_tensor_names(config)
    missing = tuple(sorted(required - actual))

    shapes: list[str] = []
    by_name = {tensor.name: tensor for tensor in assistant.tensors}
    hidden = config.hidden_size or 0
    if hidden:
        expectations: dict[str, tuple[int, ...]] = {
            "token_embd.weight": (config.vocab_size or 0, hidden),
            "output_norm.weight": (hidden,),
        }
        if config.block_count and tgt["hidden_size"]:
            expectations["nextn.pre_projection.weight"] = (
                hidden,
                2 * tgt["hidden_size"],
            )
            expectations["nextn.post_projection.weight"] = (
                tgt["hidden_size"],
                hidden,
            )
        for layer in range(config.block_count):
            sliding = (
                config.sliding_window_pattern[layer]
                if layer < len(config.sliding_window_pattern)
                else False
            )
            key_len = config.key_length_swa if sliding else config.key_length
            value_len = config.value_length_swa if sliding else config.value_length
            expectations[f"blk.{layer}.attn_q.weight"] = (
                config.head_count * key_len,
                hidden,
            )
            expectations[f"blk.{layer}.attn_output.weight"] = (
                hidden,
                config.head_count * value_len,
            )
            expectations[f"blk.{layer}.ffn_gate.weight"] = (
                config.feed_forward_length,
                hidden,
            )
            expectations[f"blk.{layer}.ffn_up.weight"] = (
                config.feed_forward_length,
                hidden,
            )
            expectations[f"blk.{layer}.ffn_down.weight"] = (
                hidden,
                config.feed_forward_length,
            )
        for name, expected in expectations.items():
            tensor = by_name.get(name)
            if tensor is None:
                continue
            actual_shape = tuple(int(v) for v in tensor.shape)
            wanted = tuple(v for v in expected if v)
            if wanted and actual_shape != wanted:
                shapes.append(f"{name} shape={actual_shape}, expected {wanted}")

    return Gemma4MTPValidation(
        config=config,
        admission_errors=tuple(errors),
        missing_tensor_names=missing,
        shape_errors=tuple(shapes),
        shared_kv_layers=mapping,
    )


def discover_gemma4_mtp_artifacts(target_path: str | Path) -> tuple[Path, ...]:
    """Find assistant sidecars in ``MTP/`` beside the target artifact.

    Returns an empty tuple when no sidecar exists: the assistant is optional,
    so its absence is not an error. Parse failures of individual files are not
    swallowed either -- discovery only lists, admission validates.
    """

    target = Path(target_path).expanduser().resolve()
    base = target.parent if target.is_file() else target
    mtp_dir = base / "MTP"
    if not mtp_dir.is_dir():
        return ()
    return tuple(sorted(p.resolve() for p in mtp_dir.glob("*.gguf")))


def load_gemma4_mtp_validation(
    target_path: str | Path, assistant_path: str | Path
) -> Gemma4MTPValidation:
    """Scan both artifacts and run capability admission."""

    return validate_gemma4_mtp_gguf(scan_gguf(target_path), scan_gguf(assistant_path))


def require_gemma4_mtp_valid(
    validation: Gemma4MTPValidation,
) -> Gemma4MTPValidation:
    """Raise ``Gemma4MTPGGUFError`` naming the first failed capability."""

    if not validation.passed:
        problems = (
            *validation.admission_errors,
            *(f"missing {name}" for name in validation.missing_tensor_names[:6]),
            *validation.shape_errors[:6],
        )
        raise Gemma4MTPGGUFError(
            "assistant sidecar fails capability admission: " + "; ".join(problems)
        )
    return validation


def require_gemma4_mtp(
    target_path: str | Path, assistant_path: str | Path
) -> Gemma4MTPValidation:
    """Scan both artifacts and admit, raising on the first failed capability."""

    return require_gemma4_mtp_valid(load_gemma4_mtp_validation(target_path, assistant_path))


__all__ = [
    "Gemma4MTPConfig",
    "Gemma4MTPGGUFError",
    "Gemma4MTPValidation",
    "discover_gemma4_mtp_artifacts",
    "load_gemma4_mtp_validation",
    "parse_gemma4_mtp_config",
    "require_gemma4_mtp",
    "require_gemma4_mtp_valid",
    "shared_kv_target_layers",
    "validate_gemma4_mtp_gguf",
]