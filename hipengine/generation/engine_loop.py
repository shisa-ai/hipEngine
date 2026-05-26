"""Long-lived scheduler-owned generation loop scaffolding.

This module is intentionally host-only and torch-free.  It wires the existing
``ResidentBatchScheduler`` to a small runner protocol so tests and early server
adapters can exercise a persistent ``submit``/``poll``/``cancel`` lifecycle
before native c>N sessions become correctness-green.
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence

from hipengine.dispatch import WorkItem, WorkKind
from hipengine.generation.batch_scheduler import CompletedRequest, GeneratedToken, ResidentBatchScheduler

PREFILL_DECODE_POLICIES = ("protect_decode", "protect_ttft", "fair")
DEFAULT_KV_POOL_INITIAL_PAGES = 128
DEFAULT_KV_POOL_LOW_WATER_PAGES = 128
DEFAULT_KV_POOL_CHUNK_PAGES = 128
DEFAULT_KV_POOL_IDLE_GRACE_SECONDS = 30.0


@dataclass(frozen=True, slots=True)
class EngineLoopConfig:
    """CLI/env-resolved knobs for the C4 scheduler-owned engine loop."""

    prefill_decode_policy: str = "protect_decode"
    kv_pool_initial_pages: int = DEFAULT_KV_POOL_INITIAL_PAGES
    kv_pool_low_water_pages: int = DEFAULT_KV_POOL_LOW_WATER_PAGES
    kv_pool_high_water_pages: int | None = None
    kv_pool_chunk_pages: int = DEFAULT_KV_POOL_CHUNK_PAGES
    kv_pool_idle_grace_seconds: float = DEFAULT_KV_POOL_IDLE_GRACE_SECONDS

    def __post_init__(self) -> None:
        if self.prefill_decode_policy not in PREFILL_DECODE_POLICIES:
            raise ValueError(f"prefill_decode_policy must be one of {PREFILL_DECODE_POLICIES!r}")
        if self.kv_pool_initial_pages <= 0:
            raise ValueError("kv_pool_initial_pages must be positive")
        if self.kv_pool_low_water_pages <= 0:
            raise ValueError("kv_pool_low_water_pages must be positive")
        if self.kv_pool_low_water_pages > self.kv_pool_initial_pages:
            raise ValueError("kv_pool_low_water_pages cannot exceed kv_pool_initial_pages")
        if self.kv_pool_high_water_pages is not None and self.kv_pool_high_water_pages < self.kv_pool_initial_pages:
            raise ValueError("kv_pool_high_water_pages cannot be below kv_pool_initial_pages")
        if self.kv_pool_chunk_pages <= 0:
            raise ValueError("kv_pool_chunk_pages must be positive")
        if self.kv_pool_idle_grace_seconds < 0:
            raise ValueError("kv_pool_idle_grace_seconds must be non-negative")


@dataclass(frozen=True, slots=True)
class EngineLoopEvent:
    """One externally visible event produced by ``ResidentEngineLoop.poll``."""

    kind: str
    request_id: int | None = None
    request_ids: tuple[int, ...] = ()
    work_kind: WorkKind | None = None
    token_id: int | None = None
    completed: CompletedRequest | None = None


class EngineLoopRunner(Protocol):
    """Minimal serial-bridge/fake-runner hooks consumed by the engine loop."""

    def prefill(self, work: WorkItem) -> None:
        """Run or record one prefill work item."""

    def decode(self, work: WorkItem) -> Sequence[GeneratedToken]:
        """Return one generated token per decoded request row."""


def add_engine_loop_config_args(
    parser: argparse.ArgumentParser,
    *,
    environ: Mapping[str, str] | None = None,
) -> None:
    """Add C4 engine-loop CLI knobs with env-backed defaults."""

    env = os.environ if environ is None else environ
    parser.add_argument(
        "--prefill-decode-policy",
        choices=PREFILL_DECODE_POLICIES,
        default=_env_prefill_decode_policy(env),
        help="Prefill/decode scheduler policy (env HIPENGINE_PREFILL_DECODE_POLICY; default: protect_decode)",
    )
    parser.add_argument(
        "--kv-pool-initial-pages",
        type=_positive_int_arg,
        default=_env_positive_int(env, "HIPENGINE_KV_POOL_INITIAL_PAGES", DEFAULT_KV_POOL_INITIAL_PAGES),
        help="Initial dynamic KV pool pages (env HIPENGINE_KV_POOL_INITIAL_PAGES; default: 128)",
    )
    parser.add_argument(
        "--kv-pool-low-water-pages",
        type=_positive_int_arg,
        default=_env_positive_int(env, "HIPENGINE_KV_POOL_LOW_WATER_PAGES", DEFAULT_KV_POOL_LOW_WATER_PAGES),
        help="KV pool idle-shrink low-water pages (env HIPENGINE_KV_POOL_LOW_WATER_PAGES; default: 128)",
    )
    parser.add_argument(
        "--kv-pool-high-water-pages",
        type=_positive_int_arg,
        default=_env_optional_positive_int(env, "HIPENGINE_KV_POOL_HIGH_WATER_PAGES"),
        help="Optional KV pool high-water page cap (env HIPENGINE_KV_POOL_HIGH_WATER_PAGES; default: unset)",
    )
    parser.add_argument(
        "--kv-pool-chunk-pages",
        type=_positive_int_arg,
        default=_env_positive_int(env, "HIPENGINE_KV_POOL_CHUNK_PAGES", DEFAULT_KV_POOL_CHUNK_PAGES),
        help="KV pool grow/shrink chunk size in pages (env HIPENGINE_KV_POOL_CHUNK_PAGES; default: 128)",
    )
    parser.add_argument(
        "--kv-pool-idle-grace-seconds",
        type=_nonnegative_float_arg,
        default=_env_nonnegative_float(
            env,
            "HIPENGINE_KV_POOL_IDLE_GRACE_SECONDS",
            DEFAULT_KV_POOL_IDLE_GRACE_SECONDS,
        ),
        help="Seconds before idle tail chunks can shrink (env HIPENGINE_KV_POOL_IDLE_GRACE_SECONDS; default: 30.0)",
    )


def engine_loop_config_from_args(args: object) -> EngineLoopConfig:
    """Build an ``EngineLoopConfig`` from an argparse namespace-like object."""

    return EngineLoopConfig(
        prefill_decode_policy=str(getattr(args, "prefill_decode_policy")),
        kv_pool_initial_pages=int(getattr(args, "kv_pool_initial_pages")),
        kv_pool_low_water_pages=int(getattr(args, "kv_pool_low_water_pages")),
        kv_pool_high_water_pages=(
            None
            if getattr(args, "kv_pool_high_water_pages") is None
            else int(getattr(args, "kv_pool_high_water_pages"))
        ),
        kv_pool_chunk_pages=int(getattr(args, "kv_pool_chunk_pages")),
        kv_pool_idle_grace_seconds=float(getattr(args, "kv_pool_idle_grace_seconds")),
    )


def engine_loop_config_from_env(environ: Mapping[str, str] | None = None) -> EngineLoopConfig:
    """Resolve C4 engine-loop knobs directly from environment values."""

    parser = argparse.ArgumentParser(add_help=False)
    add_engine_loop_config_args(parser, environ=environ)
    return engine_loop_config_from_args(parser.parse_args([]))


def _env_prefill_decode_policy(environ: Mapping[str, str]) -> str:
    raw = environ.get("HIPENGINE_PREFILL_DECODE_POLICY")
    value = "protect_decode" if raw is None or raw == "" else raw.strip()
    if value not in PREFILL_DECODE_POLICIES:
        raise ValueError(f"HIPENGINE_PREFILL_DECODE_POLICY must be one of {PREFILL_DECODE_POLICIES!r}")
    return value


def _env_positive_int(environ: Mapping[str, str], name: str, default: int) -> int:
    raw = environ.get(name)
    return int(default) if raw is None or raw == "" else _positive_int_arg(raw)


def _env_optional_positive_int(environ: Mapping[str, str], name: str) -> int | None:
    raw = environ.get(name)
    return None if raw is None or raw == "" else _positive_int_arg(raw)


def _env_nonnegative_float(environ: Mapping[str, str], name: str, default: float) -> float:
    raw = environ.get(name)
    return float(default) if raw is None or raw == "" else _nonnegative_float_arg(raw)


def _positive_int_arg(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def _nonnegative_float_arg(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


class ResidentEngineLoop:
    """Persistent ``submit``/``poll``/``cancel`` driver for resident batches.

    The loop currently executes at most one scheduler work item per tick.  It is
    deliberately conservative: requests stay resident across polls, admission
    fills free slots, the prefill/decode choice is explicit, and completion
    reclaim is delegated to ``ResidentBatchScheduler``.
    """

    def __init__(
        self,
        runner: EngineLoopRunner,
        *,
        capacity: int,
        prefill_chunk_size: int = 256,
        context_bucket_size: int = 256,
        prefill_decode_policy: str = "protect_decode",
        config: EngineLoopConfig | None = None,
    ) -> None:
        if prefill_chunk_size <= 0:
            raise ValueError("prefill_chunk_size must be positive")
        if config is not None and prefill_decode_policy != "protect_decode":
            raise ValueError("pass either config or prefill_decode_policy override, not both")
        resolved_config = config or EngineLoopConfig(prefill_decode_policy=prefill_decode_policy)
        self.runner = runner
        self.prefill_chunk_size = int(prefill_chunk_size)
        self.config = resolved_config
        self.prefill_decode_policy = resolved_config.prefill_decode_policy
        self._last_work_kind: WorkKind | None = None
        self.scheduler = ResidentBatchScheduler(capacity=capacity, context_bucket_size=context_bucket_size)

    @property
    def pending_count(self) -> int:
        return self.scheduler.pending_count

    @property
    def active_count(self) -> int:
        return self.scheduler.active_count

    @property
    def completed(self) -> dict[int, CompletedRequest]:
        return dict(self.scheduler.completed)

    def submit(self, prompt_tokens: Iterable[int], *, max_new_tokens: int, request_id: int | None = None) -> int:
        return self.scheduler.submit(prompt_tokens, max_new_tokens=max_new_tokens, request_id=request_id)

    def cancel(self, request_id: int, *, reason: str = "cancel") -> bool:
        """Cancel a pending or active request and reclaim active scheduler state."""

        return self.scheduler.cancel(request_id, reason=reason) is not None

    def disconnect(self, request_id: int) -> bool:
        """Reclaim a disconnected request through the unified cancel path."""

        return self.cancel(request_id, reason="disconnect")

    def timeout(self, request_id: int) -> bool:
        """Reclaim a timed-out request through the unified cancel path."""

        return self.cancel(request_id, reason="timeout")

    def poll(self, *, max_ticks: int = 1) -> tuple[EngineLoopEvent, ...]:
        """Advance the loop by up to ``max_ticks`` scheduler ticks."""

        if max_ticks <= 0:
            raise ValueError("max_ticks must be positive")
        events: list[EngineLoopEvent] = []
        for _ in range(int(max_ticks)):
            tick_events = self.tick()
            if not tick_events:
                break
            events.extend(tick_events)
        return tuple(events)

    def tick(self) -> tuple[EngineLoopEvent, ...]:
        """Run one admission/prefill/decode tick."""

        events: list[EngineLoopEvent] = []
        admitted = self.scheduler.admit_pending()
        events.extend(
            EngineLoopEvent(kind="admitted", request_id=request_id, request_ids=(request_id,))
            for request_id in admitted
        )

        decode = self.scheduler.next_decode_work()
        prefill_available = self.scheduler.has_prefill_work()
        if self._choose_decode_first(decode_available=decode is not None, prefill_available=prefill_available):
            assert decode is not None
            events.extend(self._run_decode(decode))
            return tuple(events)

        if prefill_available:
            prefill = self.scheduler.next_prefill_work(chunk_size=self.prefill_chunk_size)
            assert prefill is not None
            events.extend(self._run_prefill(prefill))
            return tuple(events)

        if decode is None:
            return tuple(events)
        events.extend(self._run_decode(decode))
        return tuple(events)

    def _choose_decode_first(self, *, decode_available: bool, prefill_available: bool) -> bool:
        if not decode_available:
            return False
        if not prefill_available:
            return True
        if self.prefill_decode_policy == "protect_decode":
            return True
        if self.prefill_decode_policy == "protect_ttft":
            return False
        return self._last_work_kind is WorkKind.PREFILL

    def _run_prefill(self, work: WorkItem) -> tuple[EngineLoopEvent, ...]:
        start = time.perf_counter()
        self.runner.prefill(work)
        self.scheduler.record_work_duration(work, time.perf_counter() - start)
        self._last_work_kind = work.kind
        return (EngineLoopEvent(kind="work", request_ids=work.request_ids, work_kind=work.kind),)

    def _run_decode(self, work: WorkItem) -> tuple[EngineLoopEvent, ...]:
        start = time.perf_counter()
        generated = tuple(self.runner.decode(work))
        self.scheduler.record_work_duration(work, time.perf_counter() - start)
        completed = self.scheduler.record_generated(generated)
        self._last_work_kind = work.kind
        events = [EngineLoopEvent(kind="work", request_ids=work.request_ids, work_kind=work.kind)]
        for token in generated:
            events.append(
                EngineLoopEvent(
                    kind="token",
                    request_id=token.request_id,
                    request_ids=(token.request_id,),
                    token_id=token.token_id,
                )
            )
        for item in completed:
            events.append(
                EngineLoopEvent(
                    kind="completed",
                    request_id=item.request_id,
                    request_ids=(item.request_id,),
                    completed=item,
                )
            )
        return tuple(events)


__all__ = [
    "DEFAULT_KV_POOL_CHUNK_PAGES",
    "DEFAULT_KV_POOL_IDLE_GRACE_SECONDS",
    "DEFAULT_KV_POOL_INITIAL_PAGES",
    "DEFAULT_KV_POOL_LOW_WATER_PAGES",
    "EngineLoopConfig",
    "EngineLoopEvent",
    "EngineLoopRunner",
    "PREFILL_DECODE_POLICIES",
    "ResidentEngineLoop",
    "add_engine_loop_config_args",
    "engine_loop_config_from_args",
    "engine_loop_config_from_env",
]
