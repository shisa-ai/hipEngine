"""Public dense QSA prefill keeps the promoted leaf and ignores the retired key.

``HIPENGINE_QWEN4_EXP_FULLATTN_CONTEXT_LEGACY`` used to divert the QSA prefill
token mixer's dense dispatch to the pre-promotion fixed256 leaf. This test drives
that dispatch through the public surface (``LLM.generate_detailed``), proves the
promoted leaf runs for two or more dense rows and the adaptive batch leaf for one,
and proves the retired key changes neither the route nor the output.

The production F16-route plan binds QSA WMMA attention for this artifact, so the
dense dispatch owns a chunk only with that runner-local owner stood down. The
stand-down is ``qwen4_exp_runner.qsa_wmma_attention`` - the same diagnostic lever
the other live route tests use - and it is restored before every assertion.
"""
from __future__ import annotations

import ctypes
from contextlib import ExitStack
from pathlib import Path

import pytest

MODEL = Path('/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL')
CONTEXT = 4608
LEGACY_KEY = 'HIPENGINE_QWEN4_EXP_FULLATTN_CONTEXT_LEGACY'
PROMOTED = 'qwen35_paged_full_attn_decode_context_bf16_batch_fixed256_promoted_spans'
SHIPPED = 'qwen35_paged_full_attn_decode_context_bf16_batch_fixed256_spans'
ADAPTIVE = 'qwen35_paged_full_attn_decode_context_bf16_batch_spans'


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


@pytest.mark.parametrize('length', [1, 73, 257])
def test_public_dense_route_ignores_retired_legacy_key(monkeypatch, length):
    import numpy as np

    from hipengine import LLM, SamplingParams
    from hipengine.core.hip import get_hip_runtime, HipMemcpyKind
    from hipengine.core.memory import memory_stats
    from hipengine.runtime import qwen4_exp_runner as module
    from hipengine.runtime.qwen4_exp_batch_runner import Qwen4ExpGGUFBatchRunner

    real_step = Qwen4ExpGGUFBatchRunner.step
    real_prefill = module.Qwen4ExpGGUFResidentModelRunner._prefill_chunk_steps
    leaves = {name: getattr(module, name) for name in (PROMOTED, ADAPTIVE)}
    launches, captures, prefill_rows = [], [], []

    def wrap(name):
        fn = leaves[name]

        def launch(*args, **kwargs):
            launches.append((name, args[5], args[12] if len(args) > 12 else None))
            return fn(*args, **kwargs)
        return launch

    def step(self, *args, **kwargs):
        result = real_step(self, *args, **kwargs)
        logits = result.logits
        if logits is None:
            self.runtime.device_synchronize()
            logits = np.empty((len(result.slot_ids), self.config.vocab_size),
                              dtype=np.float32)
            self.runtime.memcpy(logits.ctypes.data, self.logits.ptr,
                                logits.nbytes, HipMemcpyKind.DEVICE_TO_HOST)
        captures.append(np.array(logits, copy=True))
        return result

    def prefill(self, *args, **kwargs):
        token_ids = args[0] if args else kwargs['token_ids']
        prefill_rows.append(len(token_ids))
        return real_prefill(self, *args, **kwargs)

    monkeypatch.setattr(Qwen4ExpGGUFBatchRunner, 'step', step)
    monkeypatch.setattr(module.Qwen4ExpGGUFResidentModelRunner,
                        '_prefill_chunk_steps', prefill)
    for name in leaves:
        monkeypatch.setattr(module, name, wrap(name))

    runtime = get_hip_runtime()
    runtime.device_synchronize()
    before = memory_stats()
    llm = LLM(str(MODEL), backend='hip_gfx1151', execution_profile='production',
              max_sequence_length=CONTEXT, max_active_requests=1,
              prefix_cache='off')
    try:
        generator = llm._get_text_generator()
        while getattr(generator, 'inner', None) is not None:
            generator = generator.inner
        serving = generator.create_resident_model_runner(capacity=1)
        serving.prepare(max_sequence_length=CONTEXT)
        owners = serving._all_runners
        assert len(owners) == 1
        # Stand the WMMA owner down on every runner the request can reach.
        measured_runners = list(owners)
        own = getattr(generator, 'runner', None)
        if own is not None and all(own is not runner for runner in owners):
            measured_runners.append(own)
        results = []
        for enabled in (False, True):
            launches.clear()
            captures.clear()
            prefill_rows.clear()
            monkeypatch.setenv(LEGACY_KEY, '1' if enabled else '0')
            priors = [runner.qsa_wmma_enabled for runner in measured_runners]
            with ExitStack() as scopes:
                for runner in measured_runners:
                    scopes.enter_context(
                        module.qsa_wmma_attention(runner, False))
                # Without this stand-down the published WMMA owner takes the
                # chunk and the dense dispatch under test never runs.
                assert all(runner.qsa_wmma_enabled is False
                           for runner in measured_runners)
                output = llm.generate_detailed(
                    [[1] * length],
                    SamplingParams(max_tokens=3, temperature=0, ignore_eos=True,
                                   speculative_mtp=False))
            # The diagnostic stand-down is request-scoped, not process state.
            assert [runner.qsa_wmma_enabled
                    for runner in measured_runners] == priors
            assert prefill_rows and sum(prefill_rows) == length
            assert len(output) == 1
            assert len(output[0].generated_token_ids) == 3
            # Every dense chunk takes the promoted leaf at two or more rows and
            # the adaptive leaf at one, and carries the KV capacity only there.
            assert launches, 'the dense fixed256 dispatch never launched'
            assert all(
                (name == PROMOTED) if rows >= 2 else (name == ADAPTIVE)
                for name, rows, _ in launches
            ), (
                f'{LEGACY_KEY}={"1" if enabled else "0"} changed the dense route: '
                f'{sorted(set(launches))!r}'
            )
            assert all(
                (capacity is None) if rows < 2
                else (isinstance(capacity, int) and capacity >= rows)
                for _, rows, capacity in launches
            ), launches
            assert sum(rows for _, rows, _ in launches) >= length
            assert captures and all(row.size for row in captures)
            assert all(np.isfinite(row).all() for row in captures)
            results.append((tuple(output[0].generated_token_ids),
                            [row.copy() for row in captures]))
        assert results[0][0] == results[1][0]
        assert len(results[0][1]) == len(results[1][1])
        for reference, candidate in zip(results[0][1], results[1][1],
                                        strict=True):
            np.testing.assert_array_equal(reference, candidate)
    finally:
        llm.close()
    runtime.device_synchronize()
    after = memory_stats()
    for key in ('current_allocated_bytes', 'active_allocations'):
        assert after[key] == before[key]
