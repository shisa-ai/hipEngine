"""Public staged-F32 dense Q8_0 evaluation interface.

The layer-scoped gate exposes exact F32 parents by standing down rowtile and
DPP in both arms. The tests use that real binding through
``LLM.generate_detailed`` and inspect final registered dispatches, not inferred
output drift. Two and 73 rows exercise genuine staged launches; one row is a
decode-shaped no-op control. The selector records weight-layer ownership and
cached repeats still require final-dispatch engagement. Serving can run saved
request contexts, so the test observes the final hook directly rather than
relying on an outer ContextVar recorder crossing that boundary.
"""
from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path

import pytest

MODEL = Path('/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL')
CONTEXT = 4608
DPP_ENV = 'HIPENGINE_QWEN4_EXP_Q8_DPP_REDUCTION'
_MISSING = object()


def _hip_available():
    try:
        ctypes.CDLL('libamdhip64.so')
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not (_hip_available() and MODEL.is_dir()),
    reason='requires ROCm and the Flash-Next UD-Q4_K_XL artifact',
)


def _unwrap(generator):
    while getattr(generator, 'inner', None) is not None:
        generator = generator.inner
    return generator


def _scope_of(owner):
    return getattr(owner, 'q8_wmma_layer_scope', _MISSING)


def _install_logit_capture(monkeypatch, captures):
    """Copy every prefill and decode logits row before its buffer is reused."""
    import numpy as np

    from hipengine.core.hip import HipMemcpyKind
    from hipengine.runtime import qwen4_exp_runner as module
    from hipengine.runtime.qwen4_exp_batch_runner import Qwen4ExpGGUFBatchRunner

    def resident_logits(self, result):
        logits = result.logits
        if logits is None:
            self.runtime.device_synchronize()
            logits = np.empty((1, self.config.vocab_size), dtype=np.float32)
            self.runtime.memcpy(logits.ctypes.data, self.logits_buffer.ptr,
                                logits.nbytes, HipMemcpyKind.DEVICE_TO_HOST)
        return np.array(logits, copy=True)

    def batch_logits(self, result):
        logits = result.logits
        if logits is None:
            assert self.logits is not None, 'batch logits buffer is not live'
            self.runtime.device_synchronize()
            rows = len(result.slot_ids)
            logits = np.empty((rows, self.config.vocab_size), dtype=np.float32)
            self.runtime.memcpy(logits.ctypes.data, self.logits.ptr,
                                logits.nbytes, HipMemcpyKind.DEVICE_TO_HOST)
        return np.array(logits, copy=True)

    real_prefill = module.Qwen4ExpGGUFResidentModelRunner.prefill
    real_step = module.Qwen4ExpGGUFResidentModelRunner.step
    real_batch_step = Qwen4ExpGGUFBatchRunner.step

    def prefill(self, *args, **kwargs):
        result = real_prefill(self, *args, **kwargs)
        captures.append(resident_logits(self, result))
        return result

    def step(self, *args, **kwargs):
        result = real_step(self, *args, **kwargs)
        captures.append(resident_logits(self, result))
        return result

    def batch_step(self, *args, **kwargs):
        result = real_batch_step(self, *args, **kwargs)
        captures.append(batch_logits(self, result))
        return result

    monkeypatch.setattr(module.Qwen4ExpGGUFResidentModelRunner,
                        'prefill', prefill)
    monkeypatch.setattr(module.Qwen4ExpGGUFResidentModelRunner, 'step', step)
    monkeypatch.setattr(Qwen4ExpGGUFBatchRunner, 'step', batch_step)


def _install_selector_capture(monkeypatch, seen):
    """Record the weight layer of every selector decision that stages."""
    from hipengine.runtime import gguf_linear

    real = gguf_linear._raw_k_staged_f32_dispatch

    def staged(dispatch, *, weight=None):
        result = real(dispatch, weight=weight)
        if (result is not dispatch
                and result.key.variant == gguf_linear._Q8_STAGED_F32_VARIANT):
            spec = getattr(weight, 'spec', None)
            seen.append((getattr(spec, 'slot_path', None),
                         dispatch.key.variant))
        return result

    monkeypatch.setattr(gguf_linear, '_raw_k_staged_f32_dispatch', staged)


@dataclass
class _ArmResult:
    output: list
    captures: list
    records: list
    selector_seen: list
    finals: list

class _Harness:
    """Run one public arm with the staged interface's preemptors stood down."""

    def __init__(self, *, llm, generator, runner, route, gate, module, linear,
                 captures, selector_seen):
        self.llm = llm
        self.generator = generator
        self.runner = runner
        self.route = route
        self.gate = gate
        self.module = module
        self.linear = linear
        self.captures = captures
        self.selector_seen = selector_seen
        self.prior_wmma_env = os.environ.get(gate.WMMA_LAYERS_ENV)
        self.prior_generator_scope = _scope_of(generator)
        self.prior_runner_scope = _scope_of(runner)
        self.prior_rowtile = linear._resolve_use_q4k_rowtile(None)
        self.prior_mmq_generator = module.resolve_q8_mmq_prefill(generator)
        self.prior_mmq_runner = module.resolve_q8_mmq_prefill(runner)

    def arm(self, *, candidate, layers, prompt):
        from hipengine import SamplingParams

        self.captures.clear()
        self.selector_seen.clear()
        finals = []
        real_record = self.linear._record_gguf_linear_dispatch
        def record(**kwargs):
            finals.append(dict(kwargs))
            return real_record(**kwargs)
        self.linear._record_gguf_linear_dispatch = record
        records: list = []
        prior_env = {key: os.environ.get(key) for key in self.route.env_keys}
        prior_dpp = os.environ.get(DPP_ENV)
        try:
            with (
                self.module.q8_mmq_prefill_enabled(self.generator, False),
                self.gate._apply_route_env(
                    self.route, candidate=candidate, layers=layers,
                    generator=self.generator,
                ),
            ):
                expected_scope = (
                    frozenset({int(layers)}) if candidate else None
                )
                assert self.linear._q8_staged_f32_layers() == expected_scope
                assert self.linear.resolve_q8_wmma_layers(
                    self.generator).is_empty
                assert self.linear.resolve_q8_wmma_layers(self.runner).is_empty
                assert os.environ[self.gate.DENSE_WIDE_ENV] == '0'
                assert os.environ[DPP_ENV] == '0'
                assert self.linear._resolve_use_q4k_rowtile(None) is False
                assert self.module.resolve_q8_mmq_prefill(self.runner) is False
                with self.linear.wmma_prefill_dispatch_recorder(records):
                    output = self.llm.generate_detailed(
                        [prompt],
                        SamplingParams(max_tokens=3, temperature=0,
                                       ignore_eos=True, speculative_mtp=False),
                    )
                captured = [row.copy() for row in self.captures]
                seen = list(self.selector_seen)
        finally:
            self.linear._record_gguf_linear_dispatch = real_record
            self.gate._restore_env(prior_env)
            if prior_dpp is None:
                os.environ.pop(DPP_ENV, None)
            else:
                os.environ[DPP_ENV] = prior_dpp
        # The diagnostic stand-downs are request-scoped, not process state.
        assert {key: os.environ.get(key) for key in self.route.env_keys} == prior_env
        assert os.environ.get(DPP_ENV) == prior_dpp
        assert os.environ.get(self.gate.WMMA_LAYERS_ENV) == self.prior_wmma_env
        assert _scope_of(self.generator) is self.prior_generator_scope
        assert _scope_of(self.runner) is self.prior_runner_scope
        assert (self.linear._resolve_use_q4k_rowtile(None)
                is self.prior_rowtile)
        assert (self.module.resolve_q8_mmq_prefill(self.generator)
                is self.prior_mmq_generator)
        assert (self.module.resolve_q8_mmq_prefill(self.runner)
                is self.prior_mmq_runner)
        return _ArmResult(output=output, captures=captured,
                          records=records, selector_seen=seen, finals=finals)


def _staged_records(records):
    return [dict(selector='staged_f32', rows=row['rows'],
                 engaged=row['variant'] == 'staged_f32_bm128_bn64_bk32',
                 detail=row['variant']) for row in records]


def _assert_valid_output(result):
    import numpy as np

    assert len(result.output) == 1
    assert len(result.output[0].generated_token_ids) == 3
    assert result.captures
    assert all(np.isfinite(row).all() for row in result.captures)


def _assert_exact(reference, candidate):
    import numpy as np

    assert (tuple(reference.output[0].generated_token_ids)
            == tuple(candidate.output[0].generated_token_ids))
    assert len(reference.captures) == len(candidate.captures)
    for expected, actual in zip(reference.captures, candidate.captures,
                                strict=True):
        np.testing.assert_array_equal(expected, actual)


def _assert_engaged(result, *, length, scope, variant, require_selector=True):
    import numpy as np

    engaged = [row for row in _staged_records(result.finals)
               if row['engaged']]
    assert engaged, (
        'the staged route never reached the final registered dispatch; '
        f'records={result.records[:8]!r}; selector={result.selector_seen!r}; '
        f'finals={result.finals[:4]!r}'
    )
    assert all(row['rows'] == length for row in engaged)
    assert all(row['detail'] == variant for row in engaged)
    if require_selector:
        assert result.selector_seen, (
            'no staged selector decision was observed, so the final records '
            'cannot be attributed to a weight layer'
        )
    assert all(
        slot is not None and slot.startswith(f'layers.{scope}.')
        for slot, _ in result.selector_seen
    ), result.selector_seen
    exact_parents = {
        'prefill_f32_f32_out',
        'coltile8_rowbatch4_f32_f32_out',
        'coltile8_rowbatch4_wave_scale_f32_f32_out',
        'coltile8_rowbatch4_wave_scale_pipelined_f32_f32_out',
    }
    assert all(
        incoming in exact_parents for _, incoming in result.selector_seen
    ), result.selector_seen
    assert all(np.isfinite(row).all() for row in result.captures)


def _assert_not_engaged(result):
    assert result.finals, 'no final launch was observed'
    assert not any(row['engaged'] for row in _staged_records(result.finals))
    assert not result.selector_seen


def _final_dispatch_sequence(result):
    return [(row['rows'], row['detail'])
            for row in _staged_records(result.finals)]


@pytest.fixture
def dispatch_cache():
    from hipengine.runtime import gguf_linear

    saved = dict(gguf_linear._DISPATCH_RESOLVE_CACHE)
    gguf_linear._DISPATCH_RESOLVE_CACHE.clear()
    try:
        yield gguf_linear._DISPATCH_RESOLVE_CACHE
    finally:
        gguf_linear._DISPATCH_RESOLVE_CACHE.clear()
        gguf_linear._DISPATCH_RESOLVE_CACHE.update(saved)


@pytest.mark.parametrize('length', [2, 73])
def test_public_staged_scope_engages_named_layer_and_restores(
        monkeypatch, dispatch_cache, length):
    from hipengine import LLM
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import memory_stats
    from hipengine.kernels.backends import backend_package_capability
    from hipengine.kernels.registry import KernelKey, is_registered
    from hipengine.runtime import gguf_linear as linear
    from hipengine.runtime import qwen4_exp_runner as module
    from scripts import execution_profile_q8_wmma_prefill_layers_gate as gate

    route = gate.SELECTORS['staged_f32']
    variant = backend_package_capability('hip_gfx1151', 'STAGED_F32_VARIANT')
    assert variant == linear._Q8_STAGED_F32_VARIANT
    assert is_registered(
        KernelKey('hip_gfx1151', 'linear', 'gguf_q8_0', variant)
    )
    monkeypatch.setenv(linear._Q8_STAGED_F32_LAYERS_ENV, '')

    runtime = get_hip_runtime()
    runtime.device_synchronize()
    before = memory_stats()
    llm = LLM(str(MODEL), backend='hip_gfx1151',
              execution_profile='production',
              max_sequence_length=CONTEXT, max_active_requests=1,
              prefix_cache='off')
    try:
        generator = _unwrap(llm._get_text_generator())
        serving = generator.create_resident_model_runner(capacity=1)
        serving.prepare(max_sequence_length=CONTEXT)
        owners = serving._all_runners
        assert len(owners) == 1
        runner = owners[0]
        assert runner is generator.runner
        # The UD production recovery already closes the Q8 MMQ prefill owner.
        assert module.resolve_q8_mmq_prefill(generator) is False
        # Production publishes the empty WMMA owner scope and leaves the
        # staged selector unset; the diagnostic stand-downs below are scoped
        # to each arm, not inherited from the process.
        assert linear._q8_staged_f32_layers() is None
        assert linear.resolve_q8_wmma_layers(generator).is_empty
        prior_dense_wide = os.environ.get(gate.DENSE_WIDE_ENV)
        prior_wmma_layers = os.environ.get(gate.WMMA_LAYERS_ENV)

        captures: list = []
        selector_seen: list = []
        _install_logit_capture(monkeypatch, captures)
        _install_selector_capture(monkeypatch, selector_seen)
        harness = _Harness(
            llm=llm, generator=generator, runner=runner, route=route,
            gate=gate, module=module, linear=linear,
            captures=captures, selector_seen=selector_seen,
        )
        prompt = [1] * length
        teacher1 = harness.arm(candidate=False, layers='', prompt=prompt)
        candidate0 = harness.arm(candidate=True, layers='0', prompt=prompt)
        candidate3 = harness.arm(candidate=True, layers='3', prompt=prompt)
        # Re-run layer 0 without clearing the dispatch memo, so the repeat
        # exercises the cached arm and still has to record staged launches.
        candidate0_repeat = harness.arm(
            candidate=True, layers='0', prompt=prompt)
        teacher2 = harness.arm(candidate=False, layers='', prompt=prompt)

        for result in (teacher1, candidate0, candidate3,
                       candidate0_repeat, teacher2):
            _assert_valid_output(result)

        _assert_not_engaged(teacher1)
        _assert_engaged(candidate0, length=length, scope=0, variant=variant)
        _assert_engaged(candidate3, length=length, scope=3, variant=variant)
        # The repeat may be fully served from the dispatch memo; the final
        # recorder still has to report the registered staged launches.
        _assert_engaged(candidate0_repeat, length=length, scope=0,
                        variant=variant, require_selector=False)
        _assert_not_engaged(teacher2)
        assert os.environ.get(gate.DENSE_WIDE_ENV) == prior_dense_wide
        assert os.environ.get(gate.WMMA_LAYERS_ENV) == prior_wmma_layers

        # The control repeats are exact, and the candidate is deterministic
        # within its own changed-arithmetic arm. Candidate/teacher parity is
        # deliberately not required: this is an evaluation interface.
        _assert_exact(teacher1, teacher2)
        _assert_exact(candidate0, candidate0_repeat)
    finally:
        llm.close()
    runtime.device_synchronize()
    after = memory_stats()
    for key in ('current_allocated_bytes', 'active_allocations'):
        assert after[key] == before[key]


@pytest.mark.parametrize('scope', [0, 3])
def test_public_one_row_prefill_is_a_no_op_control(
        monkeypatch, dispatch_cache, scope):
    """A one-token prefill is decode-shaped, so the selector declines.

    This is an explicit no-op control, not engagement coverage. The genuine
    engagement cases are the two- and 73-row arms above. The assertion is that
    the candidate binding changes neither the final dispatch sequence nor the
    logits, even with the rowtile and DPP preemptors stood down.
    """
    from hipengine import LLM
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import memory_stats
    from hipengine.runtime import gguf_linear as linear
    from hipengine.runtime import qwen4_exp_runner as module
    from scripts import execution_profile_q8_wmma_prefill_layers_gate as gate

    route = gate.SELECTORS['staged_f32']
    monkeypatch.setenv(linear._Q8_STAGED_F32_LAYERS_ENV, '')

    runtime = get_hip_runtime()
    runtime.device_synchronize()
    before = memory_stats()
    llm = LLM(str(MODEL), backend='hip_gfx1151',
              execution_profile='production',
              max_sequence_length=CONTEXT, max_active_requests=1,
              prefix_cache='off')
    try:
        generator = _unwrap(llm._get_text_generator())
        serving = generator.create_resident_model_runner(capacity=1)
        serving.prepare(max_sequence_length=CONTEXT)
        owners = serving._all_runners
        assert len(owners) == 1
        runner = owners[0]
        assert runner is generator.runner
        assert module.resolve_q8_mmq_prefill(generator) is False
        assert linear._q8_staged_f32_layers() is None

        captures: list = []
        selector_seen: list = []
        _install_logit_capture(monkeypatch, captures)
        _install_selector_capture(monkeypatch, selector_seen)
        harness = _Harness(
            llm=llm, generator=generator, runner=runner, route=route,
            gate=gate, module=module, linear=linear,
            captures=captures, selector_seen=selector_seen,
        )
        # Exclude first-use graph construction from the launch comparison.
        # Numerical and route checks below cover the three measured arms.
        harness.arm(candidate=False, layers='', prompt=[1])
        teacher1 = harness.arm(candidate=False, layers='', prompt=[1])
        candidate = harness.arm(
            candidate=True, layers=str(scope), prompt=[1])
        teacher2 = harness.arm(candidate=False, layers='', prompt=[1])

        for result in (teacher1, candidate, teacher2):
            _assert_valid_output(result)
            _assert_not_engaged(result)

        _assert_exact(teacher1, candidate)
        _assert_exact(teacher1, teacher2)
        assert (_final_dispatch_sequence(candidate)
                == _final_dispatch_sequence(teacher1))
    finally:
        llm.close()
    runtime.device_synchronize()
    after = memory_stats()
    for key in ('current_allocated_bytes', 'active_allocations'):
        assert after[key] == before[key]
