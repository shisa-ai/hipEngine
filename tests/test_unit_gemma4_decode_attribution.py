"""CPU phase-accounting contracts for the fixed-context decode diagnostic."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scripts.gemma4_decode_attribution import summarize_samples, summarize_trace_windows, kernel_family, decode_orders, main


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_logits_cannot_pass_raw_bit_repeatability(value):
    from scripts.gemma4_decode_attribution import finite_logit_bits
    with pytest.raises(RuntimeError, match='non-finite'):
        finite_logit_bits(np.array([value, 1], dtype=np.float32))


def test_profile_labelling_requires_clean_declaration_and_respects_injection():
    from scripts.gemma4_decode_attribution import instrumentation_state
    assert instrumentation_state(None, env={})['status'] == 'unknown'
    assert instrumentation_state('clean', env={})['status'] == 'declared_clean'
    for env in ({'ROCPROFILER_TOOL_LIBRARIES': 'tool.so'},
                {'LD_PRELOAD': '/opt/librocprofiler-sdk.so'},
                {'HSA_TOOLS_LIB': 'tool.so'}):
        assert instrumentation_state('clean', env=env)['status'] == 'instrumented'
    assert instrumentation_state('clean', local=True, env={})['status'] == 'instrumented'
    assert instrumentation_state('profiled', env={})['status'] == 'instrumented'


@pytest.mark.parametrize('endpoint', [float('nan'), float('inf'), -float('inf')])
def test_trace_rejects_nonfinite_timestamps(endpoint):
    with pytest.raises(ValueError):
        summarize_trace_windows([dict(message='a', start=10, end=endpoint)],
                                [dict(name='k', start=11, end=12)])
    with pytest.raises(ValueError):
        summarize_trace_windows([dict(message='a', start=10, end=20)],
                                [dict(name='k', start=11, end=endpoint)])


def test_trace_rejects_distinct_overlapping_windows():
    with pytest.raises(ValueError, match='overlap'):
        summarize_trace_windows([dict(message='a', start=10, end=20),
                                 dict(message='b', start=10, end=20)],
                                [dict(name='k', start=11, end=12)])


@pytest.mark.parametrize('failure', ['provenance', 'generator', 'forward', 'profile_dump'])
def test_main_closes_loaded_llm_on_exception(tmp_path, monkeypatch, failure):
    from scripts import gemma4_campaign_bench as campaign
    from hipengine.benchmark import provenance
    from hipengine.core import hip
    closes = []
    def fail(*args, **kwargs):
        raise RuntimeError('injected failure')
    generator = SimpleNamespace(tokenize=lambda text: [11])
    runner = SimpleNamespace(reset=lambda: None, rewind=lambda n: None,
        forward=fail if failure == 'forward' else lambda ids: np.array([0, 1], dtype=np.float32),
        next_token=lambda logits: 1)
    llm = SimpleNamespace(close=lambda: closes.append(True),
        _get_text_generator=fail if failure == 'generator' else lambda: generator)
    monkeypatch.setattr(campaign, '_resolve_generator', lambda *args: (llm, runner, {}))
    monkeypatch.setattr(campaign, 'exact_prompt_ids', lambda tokenize, n: [11] * n)
    monkeypatch.setattr(provenance, 'collect_artifact_provenance',
                        fail if failure == 'provenance' else lambda **kwargs: {})
    monkeypatch.setattr(hip, 'get_hip_runtime', lambda: SimpleNamespace(device_synchronize=lambda: None))
    argv = ['--contexts', '3', '--repeats', '1', '--out', str(tmp_path/'out.json')]
    if failure == 'profile_dump':
        argv += ['--python-profile', str(tmp_path)]  # dumping onto a directory fails
    with pytest.raises((RuntimeError, IsADirectoryError)):
        main(argv)
    assert closes == [True]


def test_instrumentation_unwinds_after_forward_failure_even_if_disable_fails():
    from scripts.gemma4_decode_attribution import decode_instrumentation
    calls = []
    def disable():
        calls.append('disable')
        raise RuntimeError('disable failure')
    marker = SimpleNamespace(roctxRangePushA=lambda name: calls.append('push'),
                             roctxRangePop=lambda: calls.append('pop'))
    profiler = SimpleNamespace(enable=lambda: calls.append('enable'), disable=disable)
    with pytest.raises(RuntimeError):
        with decode_instrumentation(marker, profiler, 'decode'):
            raise RuntimeError('forward failure')
    assert calls == ['push', 'enable', 'disable', 'pop']


def test_build_key_observer_excludes_launches_and_restores_after_failure(monkeypatch):
    from scripts.gemma4_decode_attribution import observe_build_keys
    from hipengine.core import build
    def original(**kwargs):
        if kwargs.get('fail'):
            raise RuntimeError('key failure')
        return ('key',)
    monkeypatch.setattr(build, '_build_fast_key', original)
    with pytest.raises(RuntimeError):
        with observe_build_keys(True) as stats:
            assert build._build_fast_key() == ('key',)
            build._build_fast_key(fail=True)
    assert build._build_fast_key is original
    assert stats['calls'] == 2 and stats['wall_ns'] > 0
    with observe_build_keys(False) as stats:
        assert build._build_fast_key is original
    assert stats == {'calls': 0, 'wall_ns': 0}


def test_host_launch_observer_excludes_geometry_and_restores_bindings(monkeypatch):
    import ctypes
    import sys
    from hipengine.core.ctypes_cache import signed_kernel_fn
    from scripts.gemma4_decode_attribution import observe_host_launches
    def launch(*args):
        raise RuntimeError('launch failure')
    def geometry(*args):
        return 32
    module = SimpleNamespace(signed_kernel_fn=signed_kernel_fn)
    lib = SimpleNamespace(launch=launch, geometry=geometry)
    monkeypatch.setitem(sys.modules, 'hipengine.kernels.fake_launch_timing', module)
    with pytest.raises(RuntimeError):
        with observe_host_launches(True) as stats:
            assert module.signed_kernel_fn(lib, 'geometry', (), ctypes.c_size_t)() == 32
            module.signed_kernel_fn(lib, 'launch', (), ctypes.c_int)()
    assert stats['calls'] == 1 and stats['wall_ns'] > 0
    assert module.signed_kernel_fn is signed_kernel_fn


def test_decode_accounting_uses_medians_not_trace_wall_or_sum_of_medians():
    rows = [dict(forward_s=a, sample_s=b, synchronize_s=c, wall_s=a+b+c)
            for a, b, c in ((1, 4, 0), (3, 1, 1), (2, 2, 2))]
    result = summarize_samples(rows)
    assert result == dict(forward_s=2, sample_s=2, synchronize_s=1, wall_s=5)


@pytest.mark.parametrize('rows', [[], [dict(forward_s=-1, sample_s=0, synchronize_s=0, wall_s=-1)],
    [dict(forward_s=1, sample_s=2, synchronize_s=0, wall_s=4)],
    [dict(forward_s=float('nan'), sample_s=0, synchronize_s=0, wall_s=0)]])
def test_decode_accounting_rejects_missing_invalid_or_forged_phases(rows):
    with pytest.raises(ValueError):
        summarize_samples(rows)


def test_decode_orders_alternate_pairs_without_dropping_either_arm():
    assert decode_orders(3, paired=True) == [('baseline', 'candidate'), ('candidate', 'baseline'), ('baseline', 'candidate')]
    assert decode_orders(3, paired=False) == [('baseline',)] * 3
    with pytest.raises(ValueError):
        decode_orders(0, paired=True)


@pytest.mark.parametrize('bad_candidate', [False, True])
def test_paired_main_keeps_one_teacher_prefix_and_restores_library(tmp_path, monkeypatch, bad_candidate):
    from scripts import gemma4_campaign_bench as campaign
    from hipengine.benchmark import provenance
    from hipengine.core import hip
    from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_attention_staged as staged
    import scripts.gemma4_decode_attribution as diagnostic

    baseline, candidate, output = [tmp_path / name for name in ('baseline', 'candidate', 'report.json')]
    baseline.write_bytes(b'baseline')
    candidate.write_bytes(b'candidate')
    original = staged.build_gemma4_attention_staged
    calls, rewinds, closes = [], [], []

    class Runner:
        def reset(self):
            pass

        def forward(self, ids):
            arm = staged.build_gemma4_attention_staged().arm
            calls.append((len(ids), arm))
            value = 1.0 if bad_candidate and arm == 'candidate' else 0.0
            return np.array([value, 2.0], dtype=np.float32)

        def next_token(self, logits):
            return 31

        def rewind(self, context):
            rewinds.append(context)

    runner = Runner()
    generator = SimpleNamespace(tokenize=lambda text: [11])
    llm = SimpleNamespace(_get_text_generator=lambda: generator, close=lambda: closes.append(True))
    monkeypatch.setattr(campaign, '_resolve_generator', lambda artifact, capacity: (llm, runner, {}))
    monkeypatch.setattr(campaign, 'exact_prompt_ids', lambda tokenize, length: [11] * length)
    monkeypatch.setattr(provenance, 'collect_artifact_provenance', lambda **kwargs: {})
    monkeypatch.setattr(hip, 'get_hip_runtime', lambda: SimpleNamespace(device_synchronize=lambda: None))
    monkeypatch.setattr(diagnostic.ctypes, 'CDLL', lambda path: SimpleNamespace(arm=Path(path).name))
    argv = ['--contexts', '513', '--repeats', '2', '--out', str(output),
            '--attention-library', str(baseline), '--candidate-library', str(candidate)]
    if bad_candidate:
        with pytest.raises(RuntimeError, match='candidate warmup changed baseline logits'):
            main(argv)
        assert not output.exists()
    else:
        assert main(argv) == 0
        case = json.loads(output.read_text())['cases'][0]
        assert list(case['median_by_arm']) == ['baseline', 'candidate']
        assert len({r['logits_sha256'] for r in case['samples']}) == 1
        assert [r['arm'] for r in case['samples']] == ['baseline', 'candidate', 'candidate', 'baseline']
        assert calls == [(513, 'baseline'), (1, 'baseline'), (1, 'candidate'),
                         (1, 'baseline'), (1, 'candidate'), (1, 'candidate'), (1, 'baseline')]
        assert rewinds == [513] * 5
    assert closes == [True]
    assert staged.build_gemma4_attention_staged is original


def test_trace_contains_only_decode_launches_and_preserves_family_counts():
    windows = [dict(message='decode', start=10, end=100)]
    kernels = [dict(name=name, start=a, end=b) for name, a, b in (
        ('prefill', 1, 9), ('q8_0_t16_gemv', 11, 20),
        ('gguf_q4_k_selected', 22, 40), ('gemma4_attention_staged_pv', 41, 70),
        ('gguf_k_pack8_prefill_out', 71, 90), ('rmsnorm', 91, 99))]
    row = summarize_trace_windows(windows, kernels)[0]
    assert row['launches'] == 5 and row['kernel_sum_ns'] == 83
    assert row['families']['attention'] == {'ns': 29, 'launches': 1}
    assert kernel_family('q5_1_selected') == 'expert_projection'


@pytest.mark.parametrize('windows,kernels', [([], []),
    ([dict(message='decode', start=10, end=20)], []),
    ([dict(message='decode', start=10, end=20)], [dict(name='attention', start=9, end=11)]),
    ([dict(message='decode', start=10, end=20)], [dict(name='attention', start=11, end=11)]),
    ([dict(message='decode', start=10, end=20)]*2, [dict(name='attention', start=11, end=12)])])
def test_trace_rejects_empty_duplicate_crossing_or_invalid_launches(windows, kernels):
    with pytest.raises(ValueError):
        summarize_trace_windows(windows, kernels)
