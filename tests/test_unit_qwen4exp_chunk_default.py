"""The generator constructor owns the prefill chunk; the binder only reads it.

R0115: the chunk, its admission and the staging ring are all sized in
``Qwen4ExpGGUFTextGenerator.__init__``, which passes the same value to the
resident runner. The profile binder used to repeat that value through
``_bind_default_chunk`` - and, when no admission had run, fabricated a fixed
1024/512 over a caller's own runner. These tests pin the constructor as the
single owner for the explicit and auto rungs, the caller-supplied-runner case,
and the absence of any binder write.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from hipengine.generation import qwen4_exp_gguf as generation
from hipengine.generation import qwen4_exp_profiles as profiles
from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator


@dataclass
class _Admission:
    """Stand-in for the context-admission record (``asdict`` needs a dataclass)."""

    context_tokens: int
    mode: str = "auto"


class _Tokenizer:
    eos_token_id = 9

    def encode(self, text):
        return [1]

    def decode(self, ids, *, skip_special=False):
        del skip_special
        return ":".join(str(value) for value in ids)


class _Runner:
    """Caller-owned session: the constructor must not touch its chunk."""

    max_sequence_length = 4352

    def __init__(self, prefill_chunk_size=777):
        self.prefill_chunk_size = prefill_chunk_size
        self.closed = False

    def close(self):
        self.closed = True


class _FrozenChunkRunner:
    """A runner that fails loudly if anything assigns ``prefill_chunk_size``."""

    max_sequence_length = 4352

    def __init__(self, prefill_chunk_size):
        self._chunk = int(prefill_chunk_size)

    @property
    def prefill_chunk_size(self):
        return self._chunk

    def close(self):
        pass


def _install_construction(monkeypatch, *, admitted_context=262144):
    """Stub every device-touching step of ``__init__``; record what it asked for."""

    seen = {"plan": [], "admission": [], "materialize": [], "runner": []}

    def plan_fn(model_map, *, staging_token_capacity):
        seen["plan"].append(int(staging_token_capacity))
        return SimpleNamespace(staging_token_capacity=int(staging_token_capacity))

    def admit_fn(plan, **kwargs):
        seen["admission"].append(kwargs.get("prefill_chunk_size"))
        return _Admission(context_tokens=int(admitted_context))

    def materialize(readers, *, plan, **kwargs):
        seen["materialize"].append(plan)
        return SimpleNamespace(close=lambda: None)

    def resident_runner(resident, *, max_sequence_length, prefill_chunk_size, **kwargs):
        seen["runner"].append(
            {"chunk": int(prefill_chunk_size), "context": int(max_sequence_length)})
        return _Runner(prefill_chunk_size)

    monkeypatch.setattr(generation, "discover_gguf_files", lambda path: (path,))
    monkeypatch.setattr(
        generation, "GGUFReader", lambda path: SimpleNamespace(info=SimpleNamespace()))
    monkeypatch.setattr(
        generation, "build_qwen4_exp_gguf_tensor_map",
        lambda infos: SimpleNamespace(
            config=SimpleNamespace(context_length=262144)))
    monkeypatch.setattr(generation, "plan_qwen4_exp_residency", plan_fn)
    monkeypatch.setattr(
        "hipengine.core.hip.get_hip_runtime",
        lambda: SimpleNamespace(mem_get_info=lambda: (64 << 30, 64 << 30)))
    monkeypatch.setattr(
        "hipengine.loading.qwen4_exp_context.resolve_qwen4_exp_context", admit_fn)
    monkeypatch.setattr(generation, "materialize_qwen4_exp_weights", materialize)
    monkeypatch.setattr(generation, "Qwen4ExpGGUFResidentModelRunner", resident_runner)
    monkeypatch.setattr(
        generation.Qwen4ExpGGUFTokenizer, "from_gguf_info",
        classmethod(lambda cls, info: _Tokenizer()))
    return seen


def _construct(**kwargs):
    return Qwen4ExpGGUFTextGenerator(
        model_path="unused.gguf", weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace(), tokenizer=_Tokenizer(), **kwargs)


@pytest.mark.parametrize("chunk", [512, 1024, 2048, 4096])
def test_explicit_chunk_sizes_plan_admission_and_runner_identically(monkeypatch, chunk):
    seen = _install_construction(monkeypatch)
    generator = _construct(prefill_chunk_size=chunk)
    # The explicit rung is honoured verbatim everywhere, with no ladder walk.
    assert seen["plan"] == [chunk]
    assert seen["admission"] == [chunk]
    assert seen["runner"] == [{"chunk": chunk, "context": 262144}]
    assert generator.admitted_prefill_chunk_size == chunk
    assert generator.runner.prefill_chunk_size == chunk


def test_auto_chunk_admits_the_largest_rung_and_plans_for_it(monkeypatch):
    seen = _install_construction(monkeypatch)
    generator = _construct(prefill_chunk_size=None)
    # Every rung preserves the requested context, so the largest is admitted,
    # and the residency plan is built for exactly that rung.
    assert seen["plan"] == [4096]
    assert seen["admission"] == [4096]
    assert seen["runner"] == [{"chunk": 4096, "context": 262144}]
    assert generator.admitted_prefill_chunk_size == 4096
    assert generator.runner.prefill_chunk_size == 4096
    assert seen["materialize"] == [SimpleNamespace(staging_token_capacity=4096)] or (
        [plan.staging_token_capacity for plan in seen["materialize"]] == [4096])


def test_auto_chunk_fallback_sizes_the_plan_for_the_admitted_rung(monkeypatch):
    seen = _install_construction(monkeypatch)
    # 4096 cannot hold the native context here; 2048 can.
    original = generation._select_prefill_chunk

    def admit_only_2048(plan, **kwargs):
        seen["admission"].append(kwargs.get("prefill_chunk_size"))
        return _Admission(
            context_tokens=262144 if kwargs.get("prefill_chunk_size") == 2048 else 131072)

    monkeypatch.setattr(
        "hipengine.loading.qwen4_exp_context.resolve_qwen4_exp_context", admit_only_2048)
    generator = _construct(prefill_chunk_size=None)
    assert generation._AUTO_PREFILL_CHUNK_LADDER == (4096, 2048, 1024)
    assert seen["plan"] == [4096, 2048]
    assert seen["runner"] == [{"chunk": 2048, "context": 262144}]
    assert generator.admitted_prefill_chunk_size == 2048
    assert generator.runner.prefill_chunk_size == 2048
    assert [plan.staging_token_capacity for plan in seen["materialize"]] == [2048]
    assert original is generation._select_prefill_chunk


def test_caller_supplied_runner_keeps_its_own_chunk(monkeypatch):
    seen = _install_construction(monkeypatch)
    runner = _Runner(prefill_chunk_size=777)
    generator = _construct(runner=runner)
    # No admission ran, so nothing records an admitted chunk and nothing
    # overrides the caller's session.
    assert generator.admitted_prefill_chunk_size is None
    assert generator.runner is runner
    assert runner.prefill_chunk_size == 777
    assert seen == {"plan": [], "admission": [], "materialize": [], "runner": []}


@pytest.mark.parametrize("production", [True, False])
def test_binder_does_not_write_the_chunk(monkeypatch, production):
    from hipengine.execution_profiles import ExecutionProfile
    from hipengine.generation.qwen4_exp_profiles import (
        _bind, _production_selections, _strict_selections,
    )

    monkeypatch.setattr(profiles.os, "environ", dict(profiles.os.environ))
    profile = _profile(production)
    runner = _FrozenChunkRunner(777)
    generator = SimpleNamespace(runner=runner)
    # A binder write would raise AttributeError on the read-only property.
    _bind(generator, profile, production=production)
    assert runner.prefill_chunk_size == 777
    assert not hasattr(profiles, "_bind_default_chunk")


@pytest.mark.parametrize("production", [True, False])
def test_a_stale_implicit_chunk_marker_is_inert(monkeypatch, production):
    """The retired binder's marker must not resurrect a chunk write."""

    from hipengine.generation.qwen4_exp_profiles import _bind

    monkeypatch.setattr(profiles.os, "environ", dict(profiles.os.environ))
    runner = _FrozenChunkRunner(4096)
    generator = SimpleNamespace(
        runner=runner, admitted_prefill_chunk_size=4096,
        _implicit_profile_chunk_size=True)
    _bind(generator, _profile(production), production=production)
    assert runner.prefill_chunk_size == 4096
    assert generator.admitted_prefill_chunk_size == 4096


def test_ud_factory_passes_the_chunk_through_without_marking_implicitness(monkeypatch):
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            runner=SimpleNamespace(prefill_chunk_size=kwargs["prefill_chunk_size"]))

    monkeypatch.setattr(generation, "Qwen4ExpGGUFTextGenerator", factory)
    for explicit in (None, 512, 1024, 4096):
        calls.clear()
        result = generation.make_qwen4_exp_ud_q4_k_xl_generator_gfx1151(
            model_path="unused", weight_index=object(), model_plugin=object(),
            max_sequence_length=8192, resident_capacity=2,
            prefill_chunk_size=explicit)
        # An unset chunk reaches the constructor as None so the capacity guard
        # picks the rung; an explicit chunk is honoured verbatim.
        assert calls[0]["prefill_chunk_size"] == explicit
        assert calls[0]["max_sequence_length"] == 8192
        assert calls[0]["resident_capacity"] == 2
        assert not hasattr(result, "_implicit_profile_chunk_size")


def _profile(production):
    """A resolved-profile stand-in carrying the plan's own selection rows."""

    from hipengine.execution_profiles import ExecutionProfile
    from hipengine.generation.qwen4_exp_profiles import (
        _production_selections, _strict_selections,
    )

    rows = (_production_selections(recovery=True) if production
            else _strict_selections(recovery=True))
    return SimpleNamespace(
        manifest_sha256="test", profile=ExecutionProfile.PRODUCTION,
        fell_back_to_strict=False, manifest={
            "quant": "gguf_ud_q4_k_xl", "selections": [
                {"layer": r.layer, "scope": r.scope,
                 "selected_variant": r.selected_variant}
                for r in rows
            ],
        },
    )


@pytest.mark.parametrize("production", [True, False])
def test_public_auto_path_admits_the_chunk_and_the_binder_keeps_it(
    monkeypatch, production,
):
    """The shipped factory's admitted rung survives profile binding unchanged."""

    from hipengine.generation.qwen4_exp_profiles import _bind

    monkeypatch.setattr(profiles.os, "environ", dict(profiles.os.environ))
    _install_construction(monkeypatch)
    generator = generation.make_qwen4_exp_ud_q4_k_xl_generator_gfx1151(
        model_path="unused.gguf", weight_index=SimpleNamespace(),
        model_plugin=SimpleNamespace())
    assert generator.runner.prefill_chunk_size == 4096
    assert generator.admitted_prefill_chunk_size == 4096
    _bind(generator, _profile(production), production=production)
    assert generator.runner.prefill_chunk_size == 4096


def test_binder_keeps_a_directly_constructed_callers_runner_chunk(monkeypatch):
    """A caller-owned session keeps its chunk through binding, with no default."""

    from hipengine.generation.qwen4_exp_profiles import _bind

    monkeypatch.setattr(profiles.os, "environ", dict(profiles.os.environ))
    seen = _install_construction(monkeypatch)
    runner = _Runner(prefill_chunk_size=2048)
    generator = _construct(runner=runner)
    assert generator.admitted_prefill_chunk_size is None
    _bind(generator, _profile(True), production=True)
    # Neither the constructor nor the binder may replace the caller's chunk.
    assert runner.prefill_chunk_size == 2048
    assert seen == {"plan": [], "admission": [], "materialize": [], "runner": []}


def _plan(monkeypatch, *, context_tokens, staging_token_capacity):
    return SimpleNamespace(
        context_tokens=context_tokens, staging_token_capacity=staging_token_capacity)


def test_capacity_guard_takes_the_largest_admissible_chunk(monkeypatch):
    """Every rung preserves the requested context, so the largest is taken."""
    seen = []
    def plan_fn(model_map, *, staging_token_capacity):
        seen.append(staging_token_capacity)
        return _plan(monkeypatch,context_tokens=262144,
                     staging_token_capacity=staging_token_capacity)
    def admit_fn(plan, **kwargs):
        return plan
    chunk, plan, admission = generation._select_prefill_chunk(
        object(),ladder=(4096,2048,1024),available_device_bytes=1<<40,
        requested_context=None,native_context_length=262144,resident_capacity=1,
        plan_fn=plan_fn,admit_fn=admit_fn)
    assert chunk == 4096
    # The ladder short-circuits: only the winning rung is evaluated.
    assert seen == [4096]


def test_capacity_guard_falls_back_when_the_larger_chunk_costs_context(monkeypatch):
    """4096 cannot hold 262144 tokens here; 2048 can, so 2048 is taken."""
    seen = []
    def plan_fn(model_map, *, staging_token_capacity):
        seen.append(staging_token_capacity)
        return _plan(monkeypatch,context_tokens=0,
                     staging_token_capacity=staging_token_capacity)
    def admit_fn(plan, **kwargs):
        # Only the 2048 rung admits the full native context.
        return _plan(monkeypatch,context_tokens=262144 if kwargs["prefill_chunk_size"]==2048 else 131072,
                     staging_token_capacity=plan.staging_token_capacity)
    chunk, plan, admission = generation._select_prefill_chunk(
        object(),ladder=(4096,2048,1024),available_device_bytes=1<<40,
        requested_context=None,native_context_length=262144,resident_capacity=1,
        plan_fn=plan_fn,admit_fn=admit_fn)
    assert chunk == 2048
    assert seen == [4096,2048]


def test_capacity_guard_falls_back_on_memory_error(monkeypatch):
    def plan_fn(model_map, *, staging_token_capacity):
        return _plan(monkeypatch,context_tokens=262144,
                     staging_token_capacity=staging_token_capacity)
    def admit_fn(plan, **kwargs):
        if kwargs["prefill_chunk_size"] == 4096:
            raise MemoryError("weights, resident scratch and reserve do not fit")
        return plan
    chunk, _, _ = generation._select_prefill_chunk(
        object(),ladder=(4096,2048,1024),available_device_bytes=1<<40,
        requested_context=None,native_context_length=262144,resident_capacity=1,
        plan_fn=plan_fn,admit_fn=admit_fn)
    assert chunk == 2048


def test_capacity_guard_reraises_when_no_rung_fits(monkeypatch):
    def plan_fn(model_map, *, staging_token_capacity):
        return _plan(monkeypatch,context_tokens=262144,
                     staging_token_capacity=staging_token_capacity)
    def admit_fn(plan, **kwargs):
        raise MemoryError("nothing fits")
    with pytest.raises(MemoryError,match="nothing fits"):
        generation._select_prefill_chunk(
            object(),ladder=(4096,2048,1024),available_device_bytes=1,
            requested_context=None,native_context_length=262144,resident_capacity=1,
            plan_fn=plan_fn,admit_fn=admit_fn)


def test_capacity_guard_takes_the_smallest_when_context_is_unreachable(monkeypatch):
    """A caller asking for more context than any rung admits still gets a runner."""
    def plan_fn(model_map, *, staging_token_capacity):
        return _plan(monkeypatch,context_tokens=262144,
                     staging_token_capacity=staging_token_capacity)
    def admit_fn(plan, **kwargs):
        return _plan(monkeypatch,context_tokens=131072,
                     staging_token_capacity=plan.staging_token_capacity)
    chunk, _, admission = generation._select_prefill_chunk(
        object(),ladder=(4096,2048,1024),available_device_bytes=1<<40,
        requested_context=262144,native_context_length=262144,resident_capacity=1,
        plan_fn=plan_fn,admit_fn=admit_fn)
    assert chunk == 1024
    assert admission.context_tokens == 131072


def test_factories_are_quant_scoped_and_capacity_forwardable():
    from hipengine.generation.registry import resolve_text_generator
    from hipengine.llm import _factory_capacity_kwargs
    factory = resolve_text_generator(model="qwen4_exp_gguf",backend="hip_gfx1151",
                                     quant="gguf_ud_q4_k_xl")
    assert factory is generation.make_qwen4_exp_ud_q4_k_xl_generator_gfx1151
    assert resolve_text_generator(model="qwen4_exp_gguf",backend="hip_gfx1151",
                                  quant="gguf_q4_k_m") is generation.make_qwen4_exp_gguf_generator_gfx1151
    assert _factory_capacity_kwargs(factory,max_sequence_length=8192,resident_capacity=2) == {
        "max_sequence_length":8192,"resident_capacity":2}
