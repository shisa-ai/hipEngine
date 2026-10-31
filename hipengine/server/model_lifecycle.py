"""Loaded-model handle lifecycle for the HTTP server.

This owner constructs the one ``LLM`` handle, attaches the existing MTP circuit
breaker, marks readiness, and tears the handle down after HTTP work has drained.
It does not own device scheduling, request admission, startup warmup stages,
readiness payload composition, or application state.

The engine is looked up after the generation batcher drains. A drain can admit
or prepare work that creates the engine, so reading the handle before the await
would close the wrong object or leak the new one.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from hipengine.kvcache.dms_config import DMSConfig


class LifecycleReadiness(Protocol):
    """The readiness fields this owner transitions."""

    ready: bool
    status: str
    model_loaded: bool


class MTPAttachment(Protocol):
    """The existing restart-scoped MTP breaker attachment surface."""

    def attach(self, engine: Any) -> None: ...


class ShutdownBatcher(Protocol):
    """The existing HTTP generation-service drain surface."""

    async def shutdown(self, *, grace_seconds: float) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ModelBuildSpec:
    """The already-normalized server values used to construct one engine.

    This is a typed translation of the ``LLM`` constructor arguments, not a
    second configuration owner. ``ServerConfig`` resolves defaults before this
    spec is built.
    """

    model: str
    backend: str
    quant: str
    execution_profile: str | None
    max_active_requests: int | None
    max_sequence_length: int | None
    dms: DMSConfig | None
    prefix_cache: str
    speculative_mtp_serving: str
    speculative_provider: str | None
    draft_model: str | None
    speculative_candidate_budget: int | None
    kv_storage: str
    dms_metadata_path: str | None
    kv_scale_dtype: str
    kv_scale_granularity: str
    vision_model: str | None
    vae_model: str | None


class ModelLifecycle:
    """Construct, attach, and close the single loaded-model handle."""

    def __init__(
        self,
        *,
        mtp_circuit_breaker: MTPAttachment,
        log_effective_mtp: Callable[[Any], None],
        shutdown_grace_seconds: float,
    ) -> None:
        self._mtp_circuit_breaker = mtp_circuit_breaker
        self._log_effective_mtp = log_effective_mtp
        self._shutdown_grace_seconds = float(shutdown_grace_seconds)

    def build_engine(
        self,
        spec: ModelBuildSpec,
        llm_factory: Callable[..., Any],
    ) -> Any:
        """Construct the engine without publishing or attaching it."""

        kwargs: dict[str, Any] = {
            "backend": spec.backend,
            "quant": spec.quant,
            "execution_profile": spec.execution_profile,
            "max_active_requests": spec.max_active_requests,
            "max_sequence_length": spec.max_sequence_length,
            "prefix_cache": spec.prefix_cache,
            "speculative_mtp_serving": spec.speculative_mtp_serving,
            "speculative_provider": spec.speculative_provider,
            "draft_model": spec.draft_model,
            "speculative_candidate_budget": spec.speculative_candidate_budget,
            "kv_storage": spec.kv_storage,
            "dms_metadata_path": spec.dms_metadata_path,
            "kv_scale_dtype": spec.kv_scale_dtype,
            "kv_scale_granularity": spec.kv_scale_granularity,
            "vision_model": spec.vision_model,
            "vae_model": spec.vae_model,
        }
        if spec.dms is not None:
            kwargs["dms"] = spec.dms
        return llm_factory(spec.model, **kwargs)

    def attach_engine(
        self,
        engine: Any,
        *,
        newly_built: bool,
        readiness: LifecycleReadiness,
    ) -> None:
        """Attach the breaker and mark readiness in the existing order.

        The first effective-MTP log belongs to the construction path. Later
        calls attach the breaker again and reassert readiness, but do not repeat
        that log.
        """

        if newly_built:
            self._log_effective_mtp(engine)
        self._mtp_circuit_breaker.attach(engine)
        readiness.model_loaded = True

    async def shutdown(
        self,
        *,
        readiness: LifecycleReadiness,
        generation_batcher: ShutdownBatcher,
        engine_getter: Callable[[], Any | None],
        close_runner: Callable[[Callable[[], None]], Awaitable[Any]],
    ) -> dict[str, Any]:
        """Drain HTTP work, then read and close the engine that actually exists.

        ``engine_getter`` is deliberately called only after the batcher drain.
        The batcher may create the engine while draining, and the old shutdown
        path closed whichever handle existed after that await.
        """

        readiness.ready = False
        readiness.status = "draining"
        shutdown_result = await generation_batcher.shutdown(
            grace_seconds=self._shutdown_grace_seconds
        )
        engine = engine_getter()
        closer = None if engine is None else getattr(engine, "close", None)
        if callable(closer):
            await close_runner(closer)
        return shutdown_result

    def mark_stopped(self, readiness: LifecycleReadiness) -> None:
        """Record a completed shutdown after the caller stores its result."""

        readiness.status = "stopped"
