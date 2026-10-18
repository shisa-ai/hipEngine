"""Evaluator-only DMS selection adapters and sealed decision injection.

This module is intentionally independent of model identity and production routing.
Artifacts bind the complete geometry/prompt contract so an unavailable or mismatched
oracle fails closed rather than silently becoming a different experiment.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_SCHEMA = 1
_KINDS = {"injected_mask", "scores", "recency", "random", "oracle_mass"}


def _require_sha256(value: str, field: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{field} must be a 64-character hexadecimal SHA-256")
    return digest


def _mask(value: Any, shape: tuple[int, int, int]) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != shape:
        raise ValueError(f"diagnostic eviction mask must have shape {shape}")
    if raw.dtype != np.bool_:
        if not np.issubdtype(raw.dtype, np.number):
            raise ValueError("diagnostic eviction mask must contain booleans or 0/1 values")
        numeric = np.asarray(raw, dtype=np.float64)
        if not np.isfinite(numeric).all() or not np.logical_or(numeric == 0, numeric == 1).all():
            raise ValueError("diagnostic eviction mask must contain only finite 0/1 values")
    return np.asarray(raw, dtype=bool)


def token_hash(tokens: Any) -> str:
    return hashlib.sha256(np.asarray(tokens, dtype=np.int64).tobytes()).hexdigest()


def _positions(value: Any, tokens: int) -> np.ndarray:
    p = np.asarray(value, dtype=np.int64)
    if p.shape != (tokens,) or not np.array_equal(p, np.arange(tokens, dtype=np.int64)):
        raise ValueError("diagnostic injection requires canonical contiguous positions")
    return p


def _shape(scores: Any, tokens: int, layers: int, heads: int) -> np.ndarray:
    a = np.asarray(scores, dtype=np.float32)
    if a.shape != (tokens, layers, heads):
        raise ValueError(f"diagnostic scores must have shape {(tokens, layers, heads)}")
    if not np.isfinite(a).all():
        raise ValueError("diagnostic scores must be finite")
    return a


def exact_eviction(scores: Any, *, positions: Any, window: int, ratio: int,
                   current_position: int) -> np.ndarray:
    """Return [tokens,layers,heads] evictions; high score means evict."""
    a = np.asarray(scores, dtype=np.float32)
    if a.ndim != 3 or not np.isfinite(a).all():
        raise ValueError("diagnostic scores must be finite [tokens,layers,heads]")
    n, layers, heads = a.shape
    p = _positions(positions, n)
    if window < 0 or ratio <= 0 or current_position < int(p[-1]):
        raise ValueError("invalid diagnostic budget geometry")
    eligible = np.flatnonzero((int(current_position) - p) > int(window))
    evict_count = max(0, int(eligible.size) - int(np.ceil(eligible.size / ratio)))
    out = np.zeros_like(a, dtype=bool)
    # lexsort's final key is primary: score descending, oldest position first.
    for layer in range(layers):
        for head in range(heads):
            order = np.lexsort((p[eligible], -a[eligible, layer, head]))
            out[eligible[order[:evict_count]], layer, head] = True
    if np.any(out[(int(current_position) - p) <= int(window)]):
        raise ValueError("diagnostic selector evicted a protected token")
    return out


def adapter(kind: str, *, tokens: int, layers: int, heads: int, positions: Any,
            window: int, ratio: int, current_position: int, seed: int = 0,
            mass: Any | None = None) -> np.ndarray:
    """Build deterministic non-learned controls in eviction-score polarity."""
    kind = str(kind).lower().replace("-", "_")
    p = _positions(positions, tokens)
    if kind == "recency":
        # Eviction polarity: oldest eligible positions receive the highest score.
        scores = np.broadcast_to(-p[:, None, None].astype(np.float32), (tokens, layers, heads))
    elif kind == "random":
        rng = np.random.default_rng(int(seed))
        scores = rng.random((tokens, layers, heads), dtype=np.float32)
    elif kind == "oracle_mass":
        if mass is None:
            raise ValueError("oracle_mass requires within-sequence mass scores")
        # Importance is retained, therefore negate it for eviction polarity.
        scores = -_shape(mass, tokens, layers, heads)
    else:
        raise ValueError(f"unsupported diagnostic adapter {kind!r}")
    return exact_eviction(scores, positions=p, window=window, ratio=ratio,
                          current_position=current_position)


def frozen_decode_mask(prefill_mask: Any, *, steps: int, layers: int, heads: int) -> list[np.ndarray]:
    """Freeze prompt decisions and retain every newly decoded token."""
    mask = np.asarray(prefill_mask, dtype=bool)
    if mask.ndim != 3 or mask.shape[1:] != (layers, heads):
        raise ValueError("prefill mask geometry does not match decode geometry")
    if steps < 0:
        raise ValueError("diagnostic decode steps must be non-negative")
    return [mask.copy() for _ in range(int(steps))]


@dataclass(frozen=True)
class DiagnosticInjection:
    prompt_tokens: int
    prompt_token_sha256: str
    physical_layer_ids: tuple[int, ...]
    num_kv_heads: int
    positions: tuple[int, ...]
    selector_kind: str
    seed: int
    source_sha256: str
    window_size: int
    target_compression_ratio: int
    eviction_mask: np.ndarray
    decode_retention_steps: int = 0

    def __post_init__(self) -> None:
        if int(self.prompt_tokens) <= 0:
            raise ValueError("diagnostic injection prompt_tokens must be positive")
        if not self.physical_layer_ids or len(set(self.physical_layer_ids)) != len(self.physical_layer_ids):
            raise ValueError("diagnostic injection physical-layer IDs must be non-empty and unique")
        if int(self.num_kv_heads) <= 0:
            raise ValueError("diagnostic injection head count must be positive")
        if self.selector_kind not in _KINDS:
            raise ValueError("unsupported diagnostic selector kind")
        _require_sha256(self.prompt_token_sha256, "prompt_token_sha256")
        _require_sha256(self.source_sha256, "source_sha256")
        if int(self.window_size) < 0 or int(self.target_compression_ratio) <= 0:
            raise ValueError("diagnostic injection budget geometry is invalid")
        if int(self.decode_retention_steps) < 0:
            raise ValueError("decode_retention_steps must be non-negative")
        _positions(self.positions, int(self.prompt_tokens))
        expected = (
            int(self.prompt_tokens),
            len(self.physical_layer_ids),
            int(self.num_kv_heads),
        )
        mask = np.ascontiguousarray(_mask(self.eviction_mask, expected))
        mask.setflags(write=False)
        object.__setattr__(self, "eviction_mask", mask)

    def validate(self, *, prompt: Any, physical_layer_ids: Any, num_kv_heads: int,
                 window_size: int, target_compression_ratio: int,
                 decode_steps: int | None = None) -> None:
        tokens = tuple(int(x) for x in prompt)
        if len(tokens) != self.prompt_tokens or token_hash(tokens) != self.prompt_token_sha256:
            raise ValueError("diagnostic injection prompt hash/token count mismatch")
        if tuple(int(x) for x in physical_layer_ids) != self.physical_layer_ids:
            raise ValueError("diagnostic injection physical-layer map mismatch")
        if int(num_kv_heads) != self.num_kv_heads:
            raise ValueError("diagnostic injection head geometry mismatch")
        if (int(window_size), int(target_compression_ratio)) != (self.window_size, self.target_compression_ratio):
            raise ValueError("diagnostic injection budget mismatch")
        if decode_steps is not None and int(decode_steps) != int(self.decode_retention_steps):
            raise ValueError("diagnostic injection decode-retention schedule mismatch")
        _positions(self.positions, self.prompt_tokens)
        expected = (self.prompt_tokens, len(self.physical_layer_ids), self.num_kv_heads)
        if self.eviction_mask.shape != expected:
            raise ValueError(f"diagnostic injection mask must have shape {expected}")
        if np.any(self.eviction_mask[(int(self.prompt_tokens - 1) - np.asarray(self.positions)) <= self.window_size]):
            raise ValueError("diagnostic injection evicts protected tokens")
        eligible = (int(self.prompt_tokens - 1) - np.asarray(self.positions)) > self.window_size
        expected_evict = int(eligible.sum() - np.ceil(eligible.sum() / self.target_compression_ratio))
        if any(int(self.eviction_mask[:, layer, head][eligible].sum()) != expected_evict
               for layer in range(len(self.physical_layer_ids)) for head in range(self.num_kv_heads)):
            raise ValueError("diagnostic injection violates exact per-head budgets")

    @property
    def digest(self) -> str:
        h = hashlib.sha256()
        h.update(self.eviction_mask.astype(np.uint8, copy=False).tobytes())
        return h.hexdigest()


def build_injection(
    *,
    prompt: Any,
    physical_layer_ids: Any,
    num_kv_heads: int,
    selector_kind: str,
    seed: int,
    source_sha256: str,
    window_size: int,
    target_compression_ratio: int,
    decode_retention_steps: int,
    scores: Any | None = None,
    eviction_mask: Any | None = None,
) -> DiagnosticInjection:
    """Seal either finite eviction scores or an exact mask into one injection."""
    tokens = tuple(int(value) for value in prompt)
    layer_ids = tuple(int(value) for value in physical_layer_ids)
    shape = (len(tokens), len(layer_ids), int(num_kv_heads))
    if (scores is None) == (eviction_mask is None):
        raise ValueError("provide exactly one of scores or eviction_mask")
    if scores is not None:
        values = _shape(scores, *shape)
        mask = exact_eviction(
            values,
            positions=np.arange(len(tokens), dtype=np.int64),
            window=int(window_size),
            ratio=int(target_compression_ratio),
            current_position=len(tokens) - 1,
        )
    else:
        mask = _mask(eviction_mask, shape)
    injection = DiagnosticInjection(
        prompt_tokens=len(tokens),
        prompt_token_sha256=token_hash(tokens),
        physical_layer_ids=layer_ids,
        num_kv_heads=int(num_kv_heads),
        positions=tuple(range(len(tokens))),
        selector_kind=str(selector_kind),
        seed=int(seed),
        source_sha256=_require_sha256(source_sha256, "source_sha256"),
        window_size=int(window_size),
        target_compression_ratio=int(target_compression_ratio),
        eviction_mask=mask,
        decode_retention_steps=int(decode_retention_steps),
    )
    injection.validate(
        prompt=tokens,
        physical_layer_ids=layer_ids,
        num_kv_heads=int(num_kv_heads),
        window_size=int(window_size),
        target_compression_ratio=int(target_compression_ratio),
        decode_steps=int(decode_retention_steps),
    )
    return injection


def write_injection(path: str | Path, injection: DiagnosticInjection) -> None:
    """Write one immutable, atomic diagnostic-injection artifact."""
    output = Path(path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": _SCHEMA,
        "prompt_tokens": injection.prompt_tokens,
        "prompt_token_sha256": injection.prompt_token_sha256,
        "physical_layer_ids": list(injection.physical_layer_ids),
        "num_kv_heads": injection.num_kv_heads,
        "positions": list(injection.positions),
        "selector_kind": injection.selector_kind,
        "seed": injection.seed,
        "source_sha256": injection.source_sha256,
        "window_size": injection.window_size,
        "target_compression_ratio": injection.target_compression_ratio,
        "decode_retention_steps": injection.decode_retention_steps,
        "eviction_mask": injection.eviction_mask.tolist(),
        "eviction_mask_sha256": injection.digest,
    }
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"), sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def load_injection(path: str | Path) -> DiagnosticInjection:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != _SCHEMA:
        raise ValueError("unsupported diagnostic injection schema")
    kind = str(payload.get("selector_kind", ""))
    if kind not in _KINDS:
        raise ValueError("unsupported diagnostic selector kind")
    raw = payload.get("eviction_mask")
    if not isinstance(raw, list):
        raise ValueError("sealed diagnostic injection must contain an explicit eviction_mask")
    mask = np.asarray(raw)
    injection = DiagnosticInjection(
        prompt_tokens=int(payload["prompt_tokens"]),
        prompt_token_sha256=str(payload["prompt_token_sha256"]),
        physical_layer_ids=tuple(int(x) for x in payload["physical_layer_ids"]),
        num_kv_heads=int(payload["num_kv_heads"]),
        positions=tuple(int(x) for x in payload["positions"]),
        selector_kind=kind, seed=int(payload.get("seed", 0)),
        source_sha256=str(payload["source_sha256"]),
        window_size=int(payload["window_size"]),
        target_compression_ratio=int(payload["target_compression_ratio"]),
        eviction_mask=mask,
        decode_retention_steps=int(payload.get("decode_retention_steps", 0)),
    )
    expected_digest = _require_sha256(
        str(payload.get("eviction_mask_sha256", "")),
        "eviction_mask_sha256",
    )
    if injection.digest != expected_digest:
        raise ValueError("diagnostic injection eviction-mask digest mismatch")
    return injection


def overlap(a: Any, b: Any) -> float:
    x, y = np.asarray(a, bool), np.asarray(b, bool)
    if x.shape != y.shape:
        raise ValueError("mask geometry mismatch")
    kept_x, kept_y = ~x, ~y
    denom = int(np.count_nonzero(kept_x | kept_y))
    return 1.0 if denom == 0 else float(np.count_nonzero(kept_x & kept_y) / denom)


def discarded_mass(mass: Any, mask: Any) -> float:
    m, e = np.asarray(mass, dtype=np.float64), np.asarray(mask, bool)
    if m.shape != e.shape or not np.isfinite(m).all():
        raise ValueError("mass/mask geometry or finiteness mismatch")
    total = float(m.sum())
    return 0.0 if total == 0 else float(m[e].sum() / total)


def concentration(values: Any, *, axis: tuple[int, ...] = (0,)) -> np.ndarray:
    a = np.asarray(values, dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError("concentration input must be finite")
    return a.sum(axis=axis)


def causal_last_query_scores(attention_mass: Any) -> np.ndarray:
    """Collapse causal query-row mass to the last available query row.

    Input is ``[query_rows, tokens, layers, heads]`` or ``[query_rows, tokens, heads]``;
    only the final query row is legal for a causal diagnostic.
    """
    values = np.asarray(attention_mass, dtype=np.float32)
    if values.ndim not in (3, 4) or values.shape[0] <= 0 or not np.isfinite(values).all():
        raise ValueError("causal last-query mass must be finite [queries,tokens,...]")
    return np.ascontiguousarray(values[-1])


def continuation_scores(attention_mass: Any) -> np.ndarray:
    """Collapse continuation-query mass into one noncausal oracle score."""
    values = np.asarray(attention_mass, dtype=np.float32)
    if values.ndim not in (3, 4) or values.shape[0] <= 0 or not np.isfinite(values).all():
        raise ValueError("continuation mass must be finite [queries,tokens,...]")
    return np.ascontiguousarray(values.sum(axis=0))
