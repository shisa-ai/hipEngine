"""Focused contracts for the loaded-model lifecycle owner.

The owner constructs the one engine handle, attaches the existing MTP breaker,
marks readiness, and tears the handle down after HTTP work drains. These tests
pin the ordering that the application wrapper depends on, especially the
after-drain engine lookup: a batcher may create the engine while draining.
"""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from hipengine.kvcache.dms_config import DMSConfig
from hipengine.server import ServerConfig, create_app
from hipengine.server.model_lifecycle import ModelBuildSpec, ModelLifecycle
from tests.support.server_api_fakes import FakeLLM

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "hipengine" / "server" / "model_lifecycle.py"
API_PATH = REPO_ROOT / "hipengine" / "server" / "api.py"


class _Readiness:
    def __init__(self, events: list[tuple[Any, ...]] | None = None) -> None:
        self.ready = True
        self.status = "ready"
        self._model_loaded = False
        self.events = [] if events is None else events

    @property
    def model_loaded(self) -> bool:
        return self._model_loaded

    @model_loaded.setter
    def model_loaded(self, value: bool) -> None:
        self.events.append(("model_loaded", bool(value)))
        self._model_loaded = bool(value)


class _Engine:
    def __init__(
        self,
        events: list[tuple[Any, ...]] | None = None,
        *,
        name: str = "engine",
        close_error: BaseException | None = None,
    ) -> None:
        self.events = [] if events is None else events
        self.name = name
        self.close_error = close_error
        self.closed = False

    def close(self) -> None:
        self.events.append(("close", self.name))
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


class _Breaker:
    def __init__(
        self,
        events: list[tuple[Any, ...]] | None = None,
        *,
        error: BaseException | None = None,
    ) -> None:
        self.events = [] if events is None else events
        self.error = error

    def attach(self, engine: Any) -> None:
        self.events.append(("attach", engine))
        if self.error is not None:
            raise self.error


class _Batcher:
    def __init__(
        self,
        events: list[tuple[Any, ...]],
        *,
        result: dict[str, Any] | None = None,
        error: BaseException | None = None,
        on_shutdown: Any = None,
    ) -> None:
        self.events = events
        self.result = (
            {
                "forced": False,
                "grace_seconds": 0.05,
                "active_requests": 0,
                "queued_requests": 0,
            }
            if result is None
            else result
        )
        self.error = error
        self.on_shutdown = on_shutdown

    async def shutdown(self, *, grace_seconds: float) -> dict[str, Any]:
        self.events.append(("drain", grace_seconds))
        if self.on_shutdown is not None:
            self.on_shutdown()
        if self.error is not None:
            raise self.error
        return dict(self.result)


def _lifecycle(
    events: list[tuple[Any, ...]],
    *,
    log_error: BaseException | None = None,
    attach_error: BaseException | None = None,
    grace_seconds: float = 5.0,
) -> ModelLifecycle:
    def log_effective_mtp(engine: Any) -> None:
        events.append(("log", engine))
        if log_error is not None:
            raise log_error

    return ModelLifecycle(
        mtp_circuit_breaker=_Breaker(events, error=attach_error),
        log_effective_mtp=log_effective_mtp,
        shutdown_grace_seconds=grace_seconds,
    )


def _spec(**overrides: Any) -> ModelBuildSpec:
    values: dict[str, Any] = {
        "model": "fake-path",
        "backend": "auto",
        "quant": "auto",
        "execution_profile": None,
        "max_active_requests": 8,
        "max_sequence_length": 768,
        "dms": None,
        "prefix_cache": "radix",
        "speculative_mtp_serving": "auto",
        "speculative_provider": None,
        "draft_model": None,
        "speculative_candidate_budget": None,
        "kv_storage": "int8_per_token_head",
        "dms_metadata_path": None,
        "kv_scale_dtype": "fp32",
        "kv_scale_granularity": "per_token_head",
        "vision_model": None,
        "vae_model": None,
    }
    values.update(overrides)
    return ModelBuildSpec(**values)


def _expected_kwargs() -> dict[str, Any]:
    return {
        "backend": "auto",
        "quant": "auto",
        "execution_profile": None,
        "max_active_requests": 8,
        "max_sequence_length": 768,
        "prefix_cache": "radix",
        "speculative_mtp_serving": "auto",
        "speculative_provider": None,
        "draft_model": None,
        "speculative_candidate_budget": None,
        "kv_storage": "int8_per_token_head",
        "dms_metadata_path": None,
        "kv_scale_dtype": "fp32",
        "kv_scale_granularity": "per_token_head",
        "vision_model": None,
        "vae_model": None,
    }


def test_build_engine_passes_exact_factory_kwargs_and_omits_absent_dms() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def factory(model: str, **kwargs: Any) -> object:
        calls.append((model, kwargs))
        return object()

    engine = _lifecycle([]).build_engine(_spec(), factory)

    assert engine is not None
    assert calls == [("fake-path", _expected_kwargs())]
    assert "dms" not in calls[0][1]


def test_build_engine_forwards_an_explicit_dms_sidecar() -> None:
    dms = DMSConfig(metadata_path="sidecar.json")
    calls: list[tuple[str, dict[str, Any]]] = []

    def factory(model: str, **kwargs: Any) -> object:
        calls.append((model, kwargs))
        return object()

    _lifecycle([]).build_engine(_spec(dms=dms), factory)

    assert calls[0][1]["dms"] is dms


def test_attach_engine_logs_then_attaches_then_marks_readiness() -> None:
    events: list[tuple[Any, ...]] = []
    readiness = _Readiness(events)
    engine = _Engine(events)

    _lifecycle(events).attach_engine(
        engine,
        newly_built=True,
        readiness=readiness,
    )

    assert events == [("log", engine), ("attach", engine), ("model_loaded", True)]
    assert readiness.model_loaded is True


def test_attach_engine_does_not_repeat_the_first_log() -> None:
    events: list[tuple[Any, ...]] = []
    readiness = _Readiness(events)
    engine = _Engine(events)

    _lifecycle(events).attach_engine(
        engine,
        newly_built=False,
        readiness=readiness,
    )

    assert events == [("attach", engine), ("model_loaded", True)]


def test_attach_engine_log_failure_does_not_attach_or_mark_readiness() -> None:
    events: list[tuple[Any, ...]] = []
    readiness = _Readiness(events)
    engine = _Engine(events)

    with pytest.raises(RuntimeError, match="log failed"):
        _lifecycle(events, log_error=RuntimeError("log failed")).attach_engine(
            engine,
            newly_built=True,
            readiness=readiness,
        )

    assert events == [("log", engine)]
    assert readiness.model_loaded is False


def test_attach_engine_breaker_failure_does_not_mark_readiness() -> None:
    events: list[tuple[Any, ...]] = []
    readiness = _Readiness(events)
    engine = _Engine(events)

    with pytest.raises(RuntimeError, match="attach failed"):
        _lifecycle(events, attach_error=RuntimeError("attach failed")).attach_engine(
            engine,
            newly_built=True,
            readiness=readiness,
        )

    assert events == [("log", engine), ("attach", engine)]
    assert readiness.model_loaded is False


def test_shutdown_reads_engine_after_batcher_drain_and_closes_replacement() -> None:
    events: list[tuple[Any, ...]] = []
    old_engine = _Engine(events, name="old")
    new_engine = _Engine(events, name="new")
    holder = {"engine": old_engine}

    def replace_engine() -> None:
        holder["engine"] = new_engine

    batcher = _Batcher(events, on_shutdown=replace_engine)
    readiness = _Readiness(events)

    def engine_getter() -> _Engine:
        assert ("drain", 5.0) in events, "engine was read before the drain completed"
        events.append(("get_engine",))
        return holder["engine"]

    async def close_runner(closer: Any) -> None:
        events.append(("close_runner",))
        await asyncio.sleep(0)
        closer()

    async def run() -> dict[str, Any]:
        return await _lifecycle(events).shutdown(
            readiness=readiness,
            generation_batcher=batcher,
            engine_getter=engine_getter,
            close_runner=close_runner,
        )

    result = asyncio.run(run())

    assert result == {
        "forced": False,
        "grace_seconds": 0.05,
        "active_requests": 0,
        "queued_requests": 0,
    }
    assert events == [
        ("drain", 5.0),
        ("get_engine",),
        ("close_runner",),
        ("close", "new"),
    ]
    assert old_engine.closed is False
    assert new_engine.closed is True
    assert readiness.ready is False
    assert readiness.status == "draining"


def test_shutdown_without_an_engine_does_not_call_the_close_runner() -> None:
    events: list[tuple[Any, ...]] = []
    readiness = _Readiness(events)
    close_runner_called = False

    async def close_runner(closer: Any) -> None:
        nonlocal close_runner_called
        close_runner_called = True
        closer()

    async def run() -> dict[str, Any]:
        return await _lifecycle(events).shutdown(
            readiness=readiness,
            generation_batcher=_Batcher(events),
            engine_getter=lambda: None,
            close_runner=close_runner,
        )

    asyncio.run(run())

    assert close_runner_called is False
    assert events == [("drain", 5.0)]
    assert readiness.status == "draining"


def test_shutdown_batcher_failure_does_not_read_or_close_the_engine() -> None:
    events: list[tuple[Any, ...]] = []
    readiness = _Readiness(events)
    engine = _Engine(events)

    def engine_getter() -> _Engine:
        events.append(("get_engine",))
        return engine

    async def close_runner(closer: Any) -> None:
        events.append(("close_runner",))
        closer()

    async def run() -> None:
        with pytest.raises(RuntimeError, match="drain failed"):
            await _lifecycle(events).shutdown(
                readiness=readiness,
                generation_batcher=_Batcher(events, error=RuntimeError("drain failed")),
                engine_getter=engine_getter,
                close_runner=close_runner,
            )

    asyncio.run(run())

    assert events == [("drain", 5.0)]
    assert readiness.ready is False
    assert readiness.status == "draining"


def test_shutdown_close_failure_leaves_readiness_draining() -> None:
    events: list[tuple[Any, ...]] = []
    readiness = _Readiness(events)
    engine = _Engine(events, close_error=RuntimeError("close failed"))

    async def close_runner(closer: Any) -> None:
        events.append(("close_runner",))
        await asyncio.sleep(0)
        closer()

    async def run() -> None:
        with pytest.raises(RuntimeError, match="close failed"):
            await _lifecycle(events).shutdown(
                readiness=readiness,
                generation_batcher=_Batcher(events),
                engine_getter=lambda: engine,
                close_runner=close_runner,
            )

    asyncio.run(run())

    assert events == [
        ("drain", 5.0),
        ("close_runner",),
        ("close", "engine"),
    ]
    assert readiness.ready is False
    assert readiness.status == "draining"


def test_shutdown_awaits_the_close_runner_before_returning() -> None:
    events: list[tuple[Any, ...]] = []
    readiness = _Readiness(events)
    engine = _Engine(events)

    async def close_runner(closer: Any) -> None:
        events.append(("runner_enter",))
        await asyncio.sleep(0)
        events.append(("runner_resume",))
        closer()

    async def run() -> None:
        await _lifecycle(events).shutdown(
            readiness=readiness,
            generation_batcher=_Batcher(events),
            engine_getter=lambda: engine,
            close_runner=close_runner,
        )

    asyncio.run(run())

    assert events == [
        ("drain", 5.0),
        ("runner_enter",),
        ("runner_resume",),
        ("close", "engine"),
    ]


def test_lazy_request_reuses_an_injected_engine_without_constructing_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeLLM(outputs=["ok"])

    def fail_factory(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("an injected engine must not be replaced")

    monkeypatch.setattr("hipengine.server.api.LLM", fail_factory)
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
        ),
        llm=fake,
    )

    with TestClient(app) as client:
        first = client.post(
            "/v1/completions",
            json={"model": "fake-model", "prompt": "one", "max_tokens": 1},
        )
        second = client.post(
            "/v1/completions",
            json={"model": "fake-model", "prompt": "two", "max_tokens": 1},
        )

    assert first.status_code == second.status_code == 200
    assert app.state.hipengine_llm is fake
    assert len(fake.calls) == 2


def test_lazy_engine_is_published_before_breaker_attach_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = FakeLLM()

    class FailingBreaker:
        def __init__(self) -> None:
            pass

        def attach(self, engine: Any) -> None:
            raise RuntimeError("attach failed")

    monkeypatch.setattr("hipengine.server.api._MTPCircuitBreaker", FailingBreaker)
    monkeypatch.setattr("hipengine.server.api.LLM", lambda *args, **kwargs: engine)
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
        )
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/completions",
            json={"model": "fake-model", "prompt": "one", "max_tokens": 1},
        )

    assert response.status_code >= 500
    assert app.state.hipengine_llm is engine
    assert app.state.hipengine_readiness.model_loaded is False


def test_server_shutdown_closes_an_engine_created_during_batcher_drain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ClosableLLM(FakeLLM):
        def __init__(self) -> None:
            super().__init__()
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class ReplacingBatcher:
        def __init__(self) -> None:
            self.app: Any = None
            self.replacement: ClosableLLM | None = None
            self.drained = False

        async def shutdown(self, *, grace_seconds: float) -> dict[str, Any]:
            self.drained = True
            assert self.replacement is not None
            self.app.state.hipengine_llm = self.replacement
            return {
                "forced": False,
                "grace_seconds": grace_seconds,
                "active_requests": 0,
                "queued_requests": 0,
            }

    old_engine = ClosableLLM()
    new_engine = ClosableLLM()
    batcher = ReplacingBatcher()
    monkeypatch.setattr(
        "hipengine.server.api._GenerationBatcher",
        lambda **kwargs: batcher,
    )
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
        ),
        llm=old_engine,
    )
    batcher.app = app
    batcher.replacement = new_engine

    with TestClient(app):
        pass

    assert batcher.drained is True
    assert old_engine.closed is False
    assert new_engine.closed is True
    assert app.state.hipengine_shutdown == {
        "forced": False,
        "grace_seconds": 5.0,
        "active_requests": 0,
        "queued_requests": 0,
    }


def test_server_shutdown_close_failure_leaves_no_shutdown_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingCloseLLM(FakeLLM):
        def close(self) -> None:
            raise RuntimeError("close failed")

    batcher = _Batcher([])
    monkeypatch.setattr(
        "hipengine.server.api._GenerationBatcher",
        lambda **kwargs: batcher,
    )
    app = create_app(
        ServerConfig(
            model="fake-path",
            served_model_name="fake-model",
            eager_load=False,
        ),
        llm=FailingCloseLLM(),
    )

    with pytest.raises(RuntimeError, match="close failed"):
        with TestClient(app):
            pass

    assert not hasattr(app.state, "hipengine_shutdown")
    assert app.state.hipengine_readiness.ready is False
    assert app.state.hipengine_readiness.status == "draining"


def test_model_lifecycle_has_no_framework_application_or_engine_imports() -> None:
    text = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(text)
    forbidden_modules = {
        "fastapi",
        "starlette",
        "starlette.concurrency",
        "hipengine.server.api",
        "hipengine.llm",
        "hipengine.generation.engine_service",
    }
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in forbidden_modules:
            violations.append(f"line {node.lineno}: from {node.module}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in forbidden_modules:
                    violations.append(f"line {node.lineno}: import {alias.name}")
    assert not violations, "\n".join(violations)
    assert "app.state" not in text
    assert "_GenerationBatcher" not in text
    assert "_MTPCircuitBreaker" not in text


def test_api_imports_and_uses_the_model_lifecycle_owner() -> None:
    tree = ast.parse(API_PATH.read_text(encoding="utf-8"))
    imported: set[str] = set()
    loaded = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "hipengine.server.model_lifecycle":
            imported.update(alias.name for alias in node.names)
    assert imported == {"ModelBuildSpec", "ModelLifecycle"}
    assert imported <= loaded
