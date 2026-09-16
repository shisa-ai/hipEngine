"""Torch-free YuE2 AR generation: staged ``plan`` and ``generate_semantic`` loops.

The reference pipeline separates symbolic planning from semantic generation and
resets its request-local random stream at each phase. This module keeps that
structure over :class:`~hipengine.runtime.yue2_ar.Yue2ArRuntime`: the sampler,
the masks, the penalties and the CFG arithmetic are the pure functions in
:mod:`hipengine.generation.yue2`, and every emitted token is one runtime decode
step per branch.

Nothing here imports torch. Seeded identity is promised only for
:class:`~hipengine.generation.yue2.YuE2Random` (``numpy-pcg64-v1``); the
reference draws from PyTorch's generator, so cross-implementation seeded
equality is never claimed - temperature-zero requests are the ones that can be
compared token-for-token.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from hipengine.generation.yue2 import (
    ABC_END,
    ABC_START,
    CODEC_OFFSET,
    CODEC_SIZE,
    CONTEXT,
    EOD,
    GenerationConfig,
    Sampling,
    SongRequest,
    YuE2Random,
    combine_cfg,
    distribution,
    MUSIC_START,
    VOCAB_SIZE,
    negative_prefix,
    phase_end_token,
    resolve_sampling,
    softmax_f32,
    token_prefixes,
)
from hipengine.runtime.yue2_ar import Yue2ArRuntime, bf16_bits_to_f32

Cancelled = Callable[[], bool] | None
TokenCallback = Callable[[str, int], None] | None


@dataclass(frozen=True)
class SymbolicPlan:
    """The ABC stage's output: exact IDs plus the assembled semantic prefix."""

    request: SongRequest
    abc: str | None
    abc_ids: tuple[int, ...]
    prefix: tuple[int, ...]
    timing: dict = field(default_factory=dict)
    truncated: bool = False

    def to_dict(self) -> dict:
        return {
            "protocol": "yue2-native-v1",
            "request": self.request.to_dict(),
            "abc": self.abc,
            "abc_ids": list(self.abc_ids),
            "prefix": list(self.prefix),
            "timing": self.timing,
            "truncated": self.truncated,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SymbolicPlan":
        plan = cls(
            request=SongRequest.from_dict(data["request"]),
            abc=data["abc"],
            abc_ids=tuple(int(token) for token in data["abc_ids"]),
            prefix=tuple(int(token) for token in data["prefix"]),
            timing=dict(data.get("timing") or {}),
            truncated=bool(data.get("truncated", False)),
        )
        plan.validate()
        return plan

    def validate(self) -> None:
        """Domain and structural checks applied to every plan, loaded or fresh."""
        _validate_ids(self.abc_ids, EOD, "ABC IDs must be ordinary text tokens")
        _validate_ids(self.prefix, VOCAB_SIZE, "prefix IDs must be vocabulary tokens")
        if self.request.cot == "off":
            if self.abc is not None or self.abc_ids:
                raise ValueError("The off mode has no symbolic stage")
            if self.prefix[-3:] != (ABC_START, ABC_END, MUSIC_START):
                raise ValueError("Off-mode prefix does not end with the empty ABC block")
            return
        if not self.abc_ids:
            raise ValueError("A symbolic plan must carry ABC IDs")
        if self.abc is None or not self.abc.strip():
            raise ValueError("A symbolic plan must carry ABC text")
        tail = self.prefix[-len(self.abc_ids) - 2 :]
        if tail[:-2] != self.abc_ids or tail[-2:] != (ABC_END, MUSIC_START):
            raise ValueError("Prefix does not embed the plan's exact ABC IDs")

    def save(self, directory: str | Path) -> Path:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        path = target / "plan.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        return path

    @classmethod
    def load(cls, directory: str | Path) -> "SymbolicPlan":
        return cls.from_dict(json.loads((Path(directory) / "plan.json").read_text()))


@dataclass(frozen=True)
class SemanticResult:
    """The semantic stage's output: raw codec IDs (offset removed)."""

    plan: SymbolicPlan
    tokens: tuple[int, ...]
    timing: dict = field(default_factory=dict)
    truncated: bool = False

    def to_dict(self) -> dict:
        return {
            "protocol": "yue2-native-v1",
            "tokens": list(self.tokens),
            "timing": self.timing,
            "truncated": self.truncated,
        }

    @classmethod
    def from_dict(cls, data: dict, plan: SymbolicPlan) -> "SemanticResult":
        result = cls(
            plan=plan,
            tokens=tuple(int(token) for token in data["tokens"]),
            timing=dict(data.get("timing") or {}),
            truncated=bool(data.get("truncated", False)),
        )
        _validate_ids(result.tokens, CODEC_SIZE, "semantic codes must be raw codec IDs")
        return result

    def save(self, directory: str | Path) -> Path:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        self.plan.save(target)
        path = target / "semantic.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        return path

    @classmethod
    def load(cls, directory: str | Path) -> "SemanticResult":
        target = Path(directory)
        plan = SymbolicPlan.load(target)
        return cls.from_dict(json.loads((target / "semantic.json").read_text()), plan)


def _validate_ids(ids: Sequence[int], limit: int, message: str) -> None:
    for token in ids:
        if not 0 <= int(token) < limit:
            raise ValueError(f"{message}; got {token}")


def generate_tokens(
    runtime: Yue2ArRuntime,
    prefix: Sequence[int],
    sampling: Sampling,
    seed: int,
    phase: str,
    *,
    negative: Sequence[int] | None = None,
    cfg_scale: float = 1.0,
    legacy_off: bool = False,
    cancelled: Cancelled = None,
    on_token: TokenCallback = None,
) -> tuple[list[int], dict, bool]:
    """Reference AR loop over the HIP runtime.

    Returns ``(content_ids, timing, truncated)``. ``content_ids`` excludes the
    phase's end token; ``truncated`` is true when the budget ran out before the
    end token arrived, which is the reference's distinction between an EOS exit
    and a budget exit.
    """

    prefix = [int(token) for token in prefix]
    if not prefix:
        raise ValueError("empty prefix")
    if len(prefix) + sampling.max_tokens > CONTEXT:
        raise ValueError("Prefix + requested generation budget exceeds 24576; no implicit truncation")
    if cfg_scale != 1 and negative is None:
        raise ValueError("CFG requires a negative prefix")
    if negative is not None:
        negative = [int(token) for token in negative]
        if len(negative) + sampling.max_tokens > CONTEXT:
            raise ValueError("Negative prefix + generation budget exceeds context")
        if runtime.branches < 2:
            raise ValueError("this runtime was built with branches=1; CFG needs two branches")
    if cancelled is not None and cancelled():
        raise InterruptedError("Cancelled before prefill")

    if phase not in ("abc", "semantic"):
        raise ValueError("phase must be abc or semantic")
    end = phase_end_token(phase)

    generator = YuE2Random(seed)
    runtime.reset()
    start = time.perf_counter()
    runtime.prefill_host_rows([runtime.embed_row(token) for token in prefix], branch=0, start_pos=0)
    if negative is not None:
        runtime.prefill_host_rows([runtime.embed_row(token) for token in negative], branch=1, start_pos=0)
    prefill_seconds = time.perf_counter() - start

    history: list[int] = []
    first: float | None = None
    eos = False
    for step in range(sampling.max_tokens):
        if cancelled is not None and cancelled():
            raise InterruptedError(f"Cancelled during {phase}")
        conditional = bf16_bits_to_f32(runtime.logits(0))
        if negative is None:
            logits = conditional
        else:
            unconditional = bf16_bits_to_f32(runtime.logits(1))
            logits = combine_cfg(conditional, unconditional, cfg_scale)
        scores = distribution(logits, sampling, history, step, phase, legacy_off=legacy_off)
        if sampling.temperature == 0:
            token = int(np.argmax(scores))
        else:
            token = generator.sample_categorical(softmax_f32(scores))
        if first is None:
            first = time.perf_counter() - start
        if on_token is not None:
            on_token(phase, token)
        if token == end:
            eos = True
            break
        history.append(token)
        if step + 1 < sampling.max_tokens:
            row = runtime.embed_row(token)
            position = runtime.context_length(0)
            runtime.push_token(row, position, branch=0)
            runtime.forward_layers(position, branch=0)
            if negative is not None:
                negative_position = runtime.context_length(1)
                runtime.push_token(row, negative_position, branch=1)
                runtime.forward_layers(negative_position, branch=1)
    seconds = time.perf_counter() - start
    count = len(history) + int(eos)
    timing = {
        "seconds": seconds,
        "prefill_seconds": prefill_seconds,
        "ttft_seconds": first,
        "output_tokens": count,
        "content_tokens": len(history),
        "output_tps": (count / seconds) if seconds > 0 else None,
        "prefix_tokens": len(prefix),
        "negative_prefix_tokens": len(negative) if negative is not None else 0,
        "cfg_branches": 1 if negative is None else 2,
        "execution": "eager",
        "attention": "spans",
        "random": generator.state(),
    }
    return history, timing, not eos


class Yue2ArSession:
    """Staged YuE2 AR generation over one resident runtime.

    ``plan`` and ``generate_semantic`` mirror the reference's staged API. Each
    call resets the runtime's per-branch contexts first, so a cancelled or
    failed request cannot leak into the next one, and the two phases draw from
    independent request-local random streams seeded with the request's seed.
    """

    def __init__(
        self,
        runtime: Yue2ArRuntime,
        *,
        encode: Callable[[str], list[int]],
        decode: Callable[[Iterable[int]], str],
        config: GenerationConfig | None = None,
    ) -> None:
        self.runtime = runtime
        self.encode = encode
        self.decode = decode
        self.config = config or GenerationConfig()
        self.last_fallback_reason: str | None = None

    # -- staged API ----------------------------------------------------
    def plan(
        self,
        request: SongRequest,
        *,
        sampling: Sampling | dict | None = None,
        cancelled: Cancelled = None,
        on_token: TokenCallback = None,
    ) -> SymbolicPlan:
        if request.cot == "off":
            return SymbolicPlan(
                request=request,
                abc=None,
                abc_ids=(),
                prefix=tuple(token_prefixes(request, self.encode)),
                timing={"seconds": 0.0, "output_tokens": 0},
                truncated=False,
            )
        if request.abc is not None:
            ids = self.encode(request.abc)
            return SymbolicPlan(
                request=request,
                abc=request.abc,
                abc_ids=tuple(ids),
                prefix=tuple(token_prefixes(request, self.encode, ids)),
                timing={"seconds": 0.0, "output_tokens": 0, "external_prefix_tokens": len(ids)},
                truncated=False,
            )
        resolved = resolve_sampling(sampling, self.config.abc)
        prefix = token_prefixes(request, self.encode)
        ids, timing, truncated = generate_tokens(
            self.runtime,
            prefix,
            resolved,
            request.seed,
            "abc",
            cancelled=cancelled,
            on_token=on_token,
        )
        self.last_fallback_reason = self.runtime.prefill_fallback_reason
        return SymbolicPlan(
            request=request,
            abc=self.decode(ids),
            abc_ids=tuple(ids),
            prefix=tuple(token_prefixes(request, self.encode, ids)),
            timing=timing,
            truncated=truncated,
        )

    def generate_semantic(
        self,
        plan: SymbolicPlan,
        *,
        sampling: Sampling | dict | None = None,
        cancelled: Cancelled = None,
        on_token: TokenCallback = None,
    ) -> SemanticResult:
        if not isinstance(plan, SymbolicPlan):
            raise TypeError("Pass the SymbolicPlan returned by plan()")
        request = plan.request
        expected = tuple(token_prefixes(request, self.encode, plan.abc_ids))
        if expected != tuple(plan.prefix):
            raise ValueError("Plan prefix disagrees with request/exact ABC IDs")
        resolved = resolve_sampling(sampling, self.config.semantic)
        negative = (
            negative_prefix(request, self.encode, plan.abc_ids)
            if request.needs_negative_branch
            else None
        )
        ids, timing, truncated = generate_tokens(
            self.runtime,
            plan.prefix,
            resolved,
            request.seed,
            "semantic",
            negative=negative,
            cfg_scale=request.guidance,
            legacy_off=request.cot == "off",
            cancelled=cancelled,
            on_token=on_token,
        )
        self.last_fallback_reason = self.runtime.prefill_fallback_reason
        return SemanticResult(
            plan=plan,
            tokens=tuple(int(token) - CODEC_OFFSET for token in ids),
            timing=timing,
            truncated=truncated,
        )

    def run(
        self,
        request: SongRequest,
        *,
        abc_sampling: Sampling | dict | None = None,
        semantic_sampling: Sampling | dict | None = None,
        cancelled: Cancelled = None,
        on_token: TokenCallback = None,
    ) -> SemanticResult:
        plan = self.plan(request, sampling=abc_sampling, cancelled=cancelled, on_token=on_token)
        return self.generate_semantic(
            plan, sampling=semantic_sampling, cancelled=cancelled, on_token=on_token
        )

    def reset(self) -> None:
        self.runtime.reset()

    def close(self) -> None:
        self.runtime.close()
