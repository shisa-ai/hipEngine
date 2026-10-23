"""A failed generator construction reclaims every owner it already built.

``LLM._get_text_generator`` builds up to five owners in sequence: the
registered generator, the speculative provider attached to it, the resident
runner the generator's factory returns, the resident driver that wraps both,
and the engine service that wraps the driver.  Only the last of those is
published on ``LLM._text_generator``.

Fallible handoffs include capacity/context/DMS configuration, runner preparation,
provider attachment and service startup. The temporary construction owner must
release unpublished resources without replacing the setup exception.

The contract these tests pin, for every window:

* the caller sees the error the refusing stage raised -- the early
  ``ResidentPreparationError`` wrap for generator configuration refusals, the
  raw error for every later stage -- and never a cleanup error instead;
* nothing is published on ``LLM._text_generator``;
* each owner that was actually constructed is released exactly once, whether
  the unwind closes it directly or through the driver-shaped ``close()`` that
  closes the runner and then the generator;
* a retry starts from a fresh generator rather than reusing the failed one.

Release is asserted on the underlying resource, not on ``close()`` call count:
production already relies on generator close being idempotent where a runner
closed it first (``engine_loop.py:1624-1630``), so counting calls would forbid
the unwind from being written either way.

The resident runner below is built on the production compatibility bridge, so a
successful ``generate`` runs through the real scheduler, driver and service;
only the ownership hooks are recorded.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from hipengine import LLM, ResidentPreparationError, SamplingParams
from hipengine.generation import EngineService, GenerationOutput, register_text_generator
from hipengine.generation.engine_loop import _SubmitPollTextRunner

_BACKEND = "fake_backend"
_QUANT = "fake_quant"
_AUTO_CAPACITY = 2
_PROVIDER = "fake_provider"

# One distinguishable refusal per hook the seam can call, so a window that
# catches the wrong stage cannot pass on a message match alone.
_REFUSALS: dict[str, tuple[type, str]] = {
    "capacity": (ValueError, "serving capacity refusal"),
    "context": (ValueError, "serving context refusal"),
    "speculative_capabilities": (RuntimeError, "capability probe refusal"),
    "attach": (ValueError, "provider contract refusal"),
    "configure_engine_loop": (RuntimeError, "engine-loop reconfiguration refusal"),
    "prepare": (RuntimeError, "resident preparation refusal"),
    "runner_close": (RuntimeError, "resident runner release refusal"),
}


def _refusals() -> dict[str, BaseException]:
    return {token: kind(message) for token, (kind, message) in _REFUSALS.items()}


class _Ledger:
    """Ordered record of construction, configuration and release events."""

    def __init__(self) -> None:
        self.events: list[tuple[object, ...]] = []

    def record(self, *event: object) -> None:
        self.events.append(event)

    def count(self, owner: str, action: str) -> int:
        return sum(
            1
            for event in self.events
            if len(event) >= 2 and event[0] == owner and event[1] == action
        )

    def position(self, *event: object) -> int:
        """Return where ``event`` was recorded; fail if it never was."""

        return self.events.index(event)

    def released_owners(self) -> dict[str, int]:
        """Release count per owner that was constructed at all."""

        return {
            owner: self.count(owner, "release")
            for owner in ("generator", "runner", "provider")
            if self.count(owner, "construct")
        }


class _FakeResidentRunner(_SubmitPollTextRunner):
    """Resident runner whose request plumbing is the production bridge.

    Ownership follows the Qwen GGUF runner: the runner releases its own resident
    state and leaves the generator's weights to the driver, so closing the
    runner alone never releases the generator.  Release is idempotent, which is
    what lets the driver close the runner and the generator in either order.
    """

    def __init__(
        self,
        generator: "_FakeGenerator",
        *,
        capacity: int | None,
        ledger: _Ledger,
        refuse: frozenset[str],
    ) -> None:
        super().__init__(
            generator,
            capacity=_AUTO_CAPACITY if capacity is None else int(capacity),
        )
        self._ledger = ledger
        self._refuse = refuse
        self.refusals = _refusals()
        self.release_count = 0
        self.close_call_count = 0
        ledger.record("runner", "construct", capacity)

    def configure_engine_loop(self, config: object) -> None:
        self._ledger.record("runner", "configure_engine_loop")
        if "configure_engine_loop" in self._refuse:
            raise self.refusals["configure_engine_loop"]

    def prepare(self) -> None:
        self._ledger.record("runner", "prepare")
        if "prepare" in self._refuse:
            raise self.refusals["prepare"]

    def close(self) -> None:
        self._ledger.record("runner", "close_call")
        self.close_call_count += 1
        self._rows.clear()
        self._outputs.clear()
        if self.release_count:
            return
        self.release_count += 1
        self._ledger.record("runner", "release")
        if "runner_close" in self._refuse:
            raise self.refusals["runner_close"]


class _FakeGenerator:
    """Registered resident generator whose serving hooks can refuse on demand."""

    def __init__(
        self,
        ledger: _Ledger,
        *,
        refuse: frozenset[str] = frozenset(),
    ) -> None:
        self._ledger = ledger
        self._refuse = refuse
        self.refusals = _refusals()
        self._closed = False
        self._speculative_provider = None
        self._resident_model_runner: _FakeResidentRunner | None = None
        self.release_count = 0
        self.close_call_count = 0

    # -- construction hooks, in the order the seam calls them ----------------

    def declare_serving_capacity(self, capacity: int | None) -> None:
        self._ledger.record("generator", "declare_capacity", capacity)
        if "capacity" in self._refuse:
            raise self.refusals["capacity"]

    def configure_dms(self, source: object) -> None:
        self._ledger.record("generator", "configure_dms")

    def declare_max_sequence_length(self, context: int) -> None:
        self._ledger.record("generator", "declare_context", context)
        if "context" in self._refuse:
            raise self.refusals["context"]

    # -- resident ownership -------------------------------------------------

    def create_resident_model_runner(self, *, capacity: int | None) -> _FakeResidentRunner:
        # The family stores the runner it built, the same way Qwen GGUF does;
        # the runner is not reachable from anything else until the driver owns it.
        runner = _FakeResidentRunner(
            self,
            capacity=capacity,
            ledger=self._ledger,
            refuse=self._refuse,
        )
        self._resident_model_runner = runner
        return runner

    def attach_speculative_provider(self, provider: object) -> None:
        """Validate before taking ownership, as the shipped families do.

        Gemma 4 and Laguna check the provider contract and their own state first
        and assign last (``gemma4_gguf.py:809-844``, ``laguna_gguf.py:146-163``),
        so a refused attach leaves the provider unowned and the caller has to
        close it.
        """

        if provider is None:
            raise TypeError("speculative provider must not be None")
        for name in ("generate_detailed", "stream_detailed", "capabilities", "close"):
            if not callable(getattr(provider, name, None)):
                raise self.refusals["attach"]
        if self._closed:
            raise RuntimeError("fake generator is closed")
        if self._speculative_provider is not None:
            raise RuntimeError("a speculative provider is already attached")
        self._speculative_provider = provider
        self._ledger.record("generator", "attach")

    def speculative_capabilities(self) -> dict[str, object]:
        if "speculative_capabilities" in self._refuse:
            raise self.refusals["speculative_capabilities"]
        provider = self._speculative_provider
        return {} if provider is None else dict(provider.capabilities())

    def close(self) -> None:
        self._ledger.record("generator", "close_call")
        self.close_call_count += 1
        if self._closed:
            return
        self._closed = True
        self.release_count += 1
        self._ledger.record("generator", "release")

    # -- generation ---------------------------------------------------------

    def generate(self, request: object) -> list[str]:
        return [f"{prompt}!" for prompt in request.prompts]

    def generate_detailed(self, request: object) -> list[GenerationOutput]:
        return [GenerationOutput(text=f"{prompt}!") for prompt in request.prompts]


class _NoDmsHookGenerator(_FakeGenerator):
    """A family with no DMS sidecar adapter."""

    configure_dms = None


class _NoAttachHookGenerator(_FakeGenerator):
    """A family whose generator cannot attach a speculative provider."""

    attach_speculative_provider = None


class _FakeProvider:
    """Speculative provider with the contract the shipped attach hooks check."""

    def __init__(self, ledger: _Ledger) -> None:
        self._ledger = ledger
        self.release_count = 0
        ledger.record("provider", "construct")

    def generate_detailed(self, request: object) -> list[GenerationOutput]:
        raise NotImplementedError("the provider is never executed at construction")

    def stream_detailed(self, request: object) -> object:
        raise NotImplementedError("the provider is never executed at construction")

    def capabilities(self) -> dict[str, object]:
        return {"max_verifier_rows": 1}

    def close(self) -> None:
        self._ledger.record("provider", "close_call")
        if self.release_count:
            return
        self.release_count += 1
        self._ledger.record("provider", "release")


class _ProviderWithoutCapabilities(_FakeProvider):
    """A provider that fails the attach hook's contract check."""

    capabilities = None


def _install_generator(monkeypatch, name: str, factory) -> None:
    import hipengine.generation as generation
    import hipengine.loading as loading
    import hipengine.models as models

    monkeypatch.setattr(generation, "register_builtin_generators", lambda: None)
    monkeypatch.setattr(
        loading,
        "load_weight_index",
        lambda model: SimpleNamespace(
            config={"architectures": [f"{name}ForCausalLM"]},
            model_path="/tmp/unused-model",
        ),
    )
    monkeypatch.setattr(models, "resolve_model", lambda architecture: SimpleNamespace(name=name))
    register_text_generator(
        model=name,
        backend=_BACKEND,
        quant=_QUANT,
        factory=factory,
        replace=True,
    )


def _install_provider(monkeypatch, target_model: str, ledger: _Ledger, provider_cls) -> None:
    import hipengine.speculative.registry as speculative_registry

    def provider_factory(*, target_generator, config):
        return provider_cls(ledger)

    monkeypatch.setattr(
        speculative_registry,
        "register_builtin_speculative_providers",
        lambda: None,
    )
    speculative_registry.register_speculative_provider(
        speculative_registry.SpeculativeProviderKey(
            _PROVIDER,
            target_model,
            _BACKEND,
            _QUANT,
        ),
        provider_factory,
        replace=True,
    )


def _build_llm(name: str, **kwargs) -> LLM:
    return LLM("/tmp/unused-model", backend=_BACKEND, quant=_QUANT, **kwargs)


@dataclass(frozen=True)
class _Window:
    """One construction stage that can refuse, and what it leaves behind."""

    id: str
    refuse: frozenset[str]
    llm_kwargs: dict[str, object]
    error_type: type
    message: str
    wrapped: bool
    origin: str
    hook: str | None = None
    generator_cls: type = _FakeGenerator
    provider_cls: type | None = None
    prepare_called: bool = False

    @property
    def expects_runner(self) -> bool:
        return self.origin == "runner" or "speculative_capabilities" in self.refuse


_WINDOWS = (
    pytest.param(
        _Window(
            id="capacity-declaration",
            refuse=frozenset({"capacity"}),
            llm_kwargs={"max_active_requests": 2},
            error_type=ValueError,
            message="serving capacity refusal",
            wrapped=True,
            origin="generator",
            hook="capacity",
        ),
        id="capacity-declaration",
    ),
    pytest.param(
        _Window(
            id="context-declaration",
            refuse=frozenset({"context"}),
            llm_kwargs={"max_active_requests": 2, "max_sequence_length": 8192},
            error_type=ValueError,
            message="serving context refusal",
            wrapped=True,
            origin="generator",
            hook="context",
        ),
        id="context-declaration",
    ),
    pytest.param(
        _Window(
            id="dms-sidecar-missing",
            refuse=frozenset(),
            llm_kwargs={"dms_metadata_path": "sidecar.json"},
            error_type=NotImplementedError,
            message="the selected model generator has no DMS sidecar implementation",
            wrapped=True,
            origin="llm",
            generator_cls=_NoDmsHookGenerator,
        ),
        id="dms-sidecar-missing",
    ),
    pytest.param(
        _Window(
            id="engine-loop-configuration",
            refuse=frozenset({"configure_engine_loop"}),
            llm_kwargs={"max_active_requests": 2},
            error_type=RuntimeError,
            message="engine-loop reconfiguration refusal",
            wrapped=False,
            origin="runner",
            hook="configure_engine_loop",
        ),
        id="engine-loop-configuration",
    ),
    pytest.param(
        _Window(
            id="runner-prepare",
            refuse=frozenset({"prepare"}),
            llm_kwargs={"max_active_requests": 2},
            error_type=RuntimeError,
            message="resident preparation refusal",
            wrapped=False,
            origin="runner",
            hook="prepare",
            prepare_called=True,
        ),
        id="runner-prepare",
    ),
    pytest.param(
        _Window(
            id="service-capability-probe",
            refuse=frozenset({"speculative_capabilities"}),
            llm_kwargs={"max_active_requests": 2},
            error_type=RuntimeError,
            message="capability probe refusal",
            wrapped=False,
            origin="generator",
            hook="speculative_capabilities",
            prepare_called=True,
        ),
        id="service-capability-probe",
    ),
    pytest.param(
        _Window(
            id="provider-attach-missing",
            refuse=frozenset(),
            llm_kwargs={
                "max_active_requests": 2,
                "speculative_provider": _PROVIDER,
                "draft_model": "draft",
            },
            error_type=TypeError,
            message="registered target generator cannot attach a speculative provider",
            wrapped=False,
            origin="llm",
            generator_cls=_NoAttachHookGenerator,
            provider_cls=_FakeProvider,
        ),
        id="provider-attach-missing",
    ),
    pytest.param(
        _Window(
            id="provider-attach-refused",
            refuse=frozenset(),
            llm_kwargs={
                "max_active_requests": 2,
                "speculative_provider": _PROVIDER,
                "draft_model": "draft",
            },
            error_type=ValueError,
            message="provider contract refusal",
            wrapped=False,
            origin="generator",
            hook="attach",
            provider_cls=_ProviderWithoutCapabilities,
        ),
        id="provider-attach-refused",
    ),
)


@pytest.mark.parametrize("window", _WINDOWS)
def test_a_failed_construction_reclaims_every_owner_it_built(monkeypatch, window: _Window) -> None:
    """Each window surfaces its own error and releases what it constructed."""

    ledger = _Ledger()
    model = f"construction_lifetime_{window.id.replace('-', '_')}"
    constructed: list[_FakeGenerator] = []

    def factory(**kwargs):
        generator = window.generator_cls(ledger, refuse=window.refuse)
        constructed.append(generator)
        ledger.record("generator", "construct")
        return generator

    _install_generator(monkeypatch, model, factory)
    if window.provider_cls is not None:
        _install_provider(monkeypatch, model, ledger, window.provider_cls)

    llm = _build_llm(model, **window.llm_kwargs)
    raised = ResidentPreparationError if window.wrapped else window.error_type
    with pytest.raises(raised) as caught:
        llm.generate(["hello"], SamplingParams(max_tokens=4))

    error = caught.value
    assert str(error) == window.message
    assert len(constructed) == 1
    generator = constructed[0]
    refusing_owner = (
        generator._resident_model_runner if window.origin == "runner" else generator
    )

    if window.wrapped:
        assert isinstance(error, ResidentPreparationError)
        cause = error.__cause__
        assert isinstance(cause, window.error_type)
        assert str(cause) == window.message
        if window.hook is not None:
            assert cause is refusing_owner.refusals[window.hook]
    else:
        assert not isinstance(error, ResidentPreparationError)
        if window.hook is not None:
            # The refusing owner's own exception instance reaches the caller:
            # no wrapper, and no cleanup error substituted for it.
            assert error is refusing_owner.refusals[window.hook]

    assert llm._text_generator is None

    # Exactly one release per owner that was actually constructed.
    assert ledger.count("runner", "construct") == (1 if window.expects_runner else 0)
    assert ledger.count("provider", "construct") == (1 if window.provider_cls else 0)
    expected = {"generator": 1}
    if window.expects_runner:
        expected["runner"] = 1
    if window.provider_cls is not None:
        expected["provider"] = 1
    assert ledger.released_owners() == expected
    assert ledger.count("runner", "prepare") == (1 if window.prepare_called else 0)


def test_a_failing_cleanup_does_not_replace_the_construction_error(monkeypatch) -> None:
    """The refusal that stopped construction is the error the caller sees.

    Unwinding a half-built generator can fail on its own: a resident runner that
    refuses ``prepare()`` may also refuse to release, and a generator closing
    pooled sessions can raise.  That cleanup failure is diagnostic material, not
    the answer to "why did my generate() call fail", so it may be attached as a
    note but never substituted.  The unwind also has to keep going: the
    generator must be released even though the runner's release raised.
    """

    ledger = _Ledger()
    model = "construction_lifetime_cleanup_failure"
    constructed: list[_FakeGenerator] = []

    def factory(**kwargs):
        generator = _FakeGenerator(
            ledger,
            refuse=frozenset({"prepare", "runner_close"}),
        )
        constructed.append(generator)
        ledger.record("generator", "construct")
        return generator

    _install_generator(monkeypatch, model, factory)

    llm = _build_llm(model, max_active_requests=2)
    with pytest.raises(RuntimeError) as caught:
        llm.generate(["hello"], SamplingParams(max_tokens=4))

    error = caught.value
    runner = constructed[0]._resident_model_runner
    assert runner is not None
    assert error is runner.refusals["prepare"]
    assert str(error) == "resident preparation refusal"
    assert not isinstance(error, ResidentPreparationError)

    # The cleanup failure is reachable from the original error, if it was kept.
    notes = " ".join(getattr(error, "__notes__", ()))
    if notes:
        assert "resident runner release refusal" in notes

    # The runner released before it raised, and the generator was released after.
    assert ledger.count("runner", "release") == 1
    assert ledger.count("generator", "release") == 1
    assert ledger.position("generator", "release") > ledger.position("runner", "release")
    assert llm._text_generator is None


def test_each_failed_attempt_reclaims_its_own_generator(monkeypatch) -> None:
    """A retry after a failed construction builds and reclaims a fresh generator.

    ``_get_text_generator`` only caches on success, so every ``generate()`` call
    after a failed construction re-enters construction from the top.  Each
    attempt therefore has to reclaim the generator it built: one attempt's
    cleanup must not be satisfied by another attempt's object, and a failure
    must not leave the first generator cached for the next call to reuse.
    """

    ledger = _Ledger()
    model = "construction_lifetime_retry"
    constructed: list[_FakeGenerator] = []

    def factory(**kwargs):
        generator = _FakeGenerator(ledger, refuse=frozenset({"capacity"}))
        constructed.append(generator)
        ledger.record("generator", "construct")
        return generator

    _install_generator(monkeypatch, model, factory)

    llm = _build_llm(model, max_active_requests=2)
    for _attempt in range(2):
        with pytest.raises(ResidentPreparationError):
            llm.generate(["hello"], SamplingParams(max_tokens=4))
        assert llm._text_generator is None

    assert len(constructed) == 2
    assert constructed[0] is not constructed[1]
    assert [generator.release_count for generator in constructed] == [1, 1]
    assert ledger.count("generator", "construct") == 2
    assert ledger.count("generator", "release") == 2


def test_a_successful_construction_publishes_one_owner_that_close_releases_once(monkeypatch) -> None:
    """The success path owns both the runner and the generator, once each.

    This is the control for the windows above: whatever the seam does on
    failure, a construction that completes must still publish exactly one
    service, prepare the runner once, and release the runner and then the
    generator exactly once when ``LLM.close()`` runs.  Release is asserted on
    the resource, not on ``close()`` call count, because production already
    relies on generator close being idempotent where the runner closed first.
    """

    ledger = _Ledger()
    model = "construction_lifetime_success"
    constructed: list[_FakeGenerator] = []

    def factory(**kwargs):
        generator = _FakeGenerator(ledger)
        constructed.append(generator)
        ledger.record("generator", "construct")
        return generator

    _install_generator(monkeypatch, model, factory)

    llm = _build_llm(model, max_active_requests=2)
    assert llm.generate(["hello"], SamplingParams(max_tokens=4)) == ["hello!"]

    driver = llm._text_generator
    assert isinstance(driver, EngineService)
    assert constructed[0]._resident_model_runner is not None
    assert ledger.count("runner", "prepare") == 1
    assert ledger.count("runner", "release") == 0
    assert ledger.count("generator", "release") == 0

    llm.close()
    assert ledger.count("runner", "release") == 1
    assert ledger.count("generator", "release") == 1
    releases = [event[0] for event in ledger.events if len(event) >= 2 and event[1] == "release"]
    assert releases == ["runner", "generator"]

    # On the success path one close per owner is the contract; only the failure
    # paths have to tolerate a generator closed by the runner first.
    assert constructed[0].close_call_count == 1

    # Closing again must not release either owner a second time.
    llm.close()
    assert ledger.count("runner", "release") == 1
    assert ledger.count("generator", "release") == 1


def test_llm_forwards_its_serving_context_and_width_to_the_generator(monkeypatch) -> None:
    """``LLM(max_sequence_length=...)`` reaches the generator before it sizes.

    ``_declare_serving_context`` is what stops an automatic context selection
    from silently replacing the context the caller asked for.  It runs inside
    the construction seam, so a seam that moves the call can drop it while every
    other test stays green.  The width declaration has the same shape and the
    same ordering requirement: both are declarations the resident session is
    sized from, so both must land before the runner is built.
    """

    ledger = _Ledger()
    model = "construction_lifetime_declarations"

    def factory(**kwargs):
        generator = _FakeGenerator(ledger)
        ledger.record("generator", "construct")
        return generator

    _install_generator(monkeypatch, model, factory)

    llm = _build_llm(model, max_active_requests=2, max_sequence_length=4096)
    assert llm.generate(["hello"], SamplingParams(max_tokens=4)) == ["hello!"]

    assert ledger.count("generator", "declare_capacity") == 1
    assert ledger.count("generator", "declare_context") == 1
    assert ("generator", "declare_capacity", 2) in ledger.events
    assert ledger.position("generator", "declare_context", 4096) < ledger.position(
        "runner", "construct", 2
    )
    llm.close()


def test_an_unset_serving_context_is_not_declared(monkeypatch) -> None:
    """An unpinned context stays the generator's choice, not a declared bound."""

    ledger = _Ledger()
    model = "construction_lifetime_unset_context"

    def factory(**kwargs):
        generator = _FakeGenerator(ledger)
        ledger.record("generator", "construct")
        return generator

    _install_generator(monkeypatch, model, factory)

    llm = _build_llm(model, max_active_requests=2)
    assert llm.generate(["hello"], SamplingParams(max_tokens=4)) == ["hello!"]

    assert ledger.count("generator", "declare_context") == 0
    assert ("generator", "declare_capacity", 2) in ledger.events
    llm.close()


def test_startup_timeout_joins_unpublished_service_and_closes_driver_once(monkeypatch):
    import threading

    ledger = _Ledger()
    model = "construction_lifetime_startup_timeout"
    constructed = []

    def factory(**kwargs):
        generator = _FakeGenerator(ledger)
        constructed.append(generator)
        return generator

    _install_generator(monkeypatch, model, factory)
    original_wait = threading.Event.wait

    def missed_ready(event, timeout=None):
        if timeout == 0.123:
            return False
        return original_wait(event, timeout)

    monkeypatch.setattr(threading.Event, "wait", missed_ready)
    before = set(threading.enumerate())
    llm = _build_llm(model, engine_command_timeout_seconds=0.123)
    with pytest.raises(RuntimeError, match="driver thread did not start"):
        llm.generate("hello")
    assert llm._text_generator is None
    assert set(threading.enumerate()) == before
    assert ledger.count("runner", "release") == 1
    assert ledger.count("generator", "release") == 1
    assert constructed[0].close_call_count == 1

