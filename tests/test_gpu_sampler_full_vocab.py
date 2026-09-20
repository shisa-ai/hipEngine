"""Full-vocabulary fast sampler gates; no model or torch dependency."""
from __future__ import annotations

import numpy as np
import pytest

NEG = np.finfo(np.float32).min


def test_fast_sampler_scratch_contract():
    from hipengine.kernels.hip_gfx1100.sampling.sampler import fast_sampler_scratch_bytes, sample_sorted_f32_rows_i32
    assert fast_sampler_scratch_bytes(1, 248320) >= 248320 * 24
    for rows, vocab in [(0, 8), (1, 0), (1, 2**31), (65536, 8)]:
        with pytest.raises(ValueError):
            fast_sampler_scratch_bytes(rows, vocab)
    for kwargs in ({"scratch_ptr": 8, "scratch_bytes": 0},
                   {"scratch_ptr": 9, "scratch_bytes": 10000},
                   {"scratch_ptr": 8, "scratch_bytes": 10000, "step_index": -1},
                   {"scratch_ptr": 8, "scratch_bytes": 10000, "top_logprobs": 65},
                   {"scratch_ptr": 8, "scratch_bytes": 10000, "out_top_indices_i32_ptr": 8}):
        with pytest.raises(ValueError):
            sample_sorted_f32_rows_i32(0, 0, 0, 0, 0, 0, None, None, 1, 8, **kwargs)


@pytest.fixture
def gpu():
    # Explicit guard: importing this module does not load HIP on CPU-only CI.
    try:
        from hipengine.core.hip import get_hip_runtime
        runtime = get_hip_runtime()
        pointer = runtime.malloc(8)
        runtime.free(pointer)
    except (OSError, RuntimeError) as exc:
        pytest.skip(f"HIP device unavailable: {exc}")
    from scripts.sampler_full_vocab_bench import SamplerCase
    return SamplerCase


def uniform(seed, step, row):
    mask = (1 << 64) - 1
    value = int(seed) ^ (((row + 1) * 0xBF58476D1CE4E5B9) & mask) ^ (((step + 1) * 0x9E3779B97F4A7C15) & mask)
    z = (value + 0x9E3779B97F4A7C15) & mask
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
    return np.float32(((z ^ (z >> 31)) >> 11) / 2**53)


def reference(logits, temperatures, ps, mins, seeds, top, step=13):
    """Independent FP64 softmax/filter oracle over the full finite vocabulary.

    Center FP64 logits before division; do not mirror a GPU overflow fallback or
    its FP32 exponentials. Only the documented native draw ABI is shared.
    """
    rows, vocab = logits.shape
    result = (np.full(rows, -1, np.int32), np.full(rows, NEG, np.float32),
              np.zeros(rows, np.int32), np.full((rows, top), -1, np.int32),
              np.full((rows, top), NEG, np.float32), np.full(rows, -1, np.int64),
              np.full(rows, NEG, np.float32))
    for row in range(rows):
        ids = np.flatnonzero(np.isfinite(logits[row]))
        ids = ids[np.lexsort((ids, -logits[row, ids]))]
        if not len(ids):
            continue
        temp = np.float32(temperatures[row])
        if not (temp > 0 and np.isfinite(temp)):
            ids, weights, log_weights = ids[:1], np.ones(1), np.zeros(1)
        else:
            values = logits[row, ids].astype(np.float64)
            log_weights = (values - values[0]) / float(temp)
            weights = np.exp(log_weights)
            assert np.isfinite(weights).all() and weights.sum() >= 1.
            p, m = np.float32(ps[row]), np.float32(mins[row])
            if p <= 0:
                count = 1
            elif p < 1:
                count = np.searchsorted(np.cumsum(weights), p * weights.sum(), side="left") + 1
            else:
                count = len(ids)
            if m > 0:
                count = max(1, np.count_nonzero(weights[:count] >= m))
            ids, weights = ids[:count], weights[:count]
        probs = weights / weights.sum()
        chosen = min(len(ids) - 1, np.searchsorted(np.cumsum(weights), float(uniform(seeds[row], step, row)) * weights.sum(), side="left"))
        result[0][row] = result[5][row] = ids[chosen]
        result[1][row] = np.log(probs[chosen])
        result[2][row] = len(ids)
        result[6][row] = logits[row, ids[chosen]]
        width = min(top, len(ids))
        result[3][row, :width] = ids[:width]
        with np.errstate(over="ignore"):
            result[4][row, :width] = log_weights[:width] - np.log(weights.sum())
    return result


def assert_result(actual, expected):
    for i in (0, 2, 3, 5, 6):
        np.testing.assert_array_equal(actual[i], expected[i])
    for i in (1, 4):
        np.testing.assert_allclose(actual[i], expected[i], rtol=0, atol=2e-5)


@pytest.mark.parametrize("vocab", [1, 17, 255, 256, 257, 511, 1023, 4097, 248320])
def test_full_vocabulary_reference_repeat_and_poison(gpu, vocab):
    logits = np.random.default_rng(2026).normal(size=(4, vocab)).astype(np.float32)
    logits[1] = 0  # ties, includes a large flat retained set with no hidden cap
    logits[2] = np.round(logits[2], 1)
    logits[3] *= 8
    temperatures = np.array([.7, 1., 1.7, .4], np.float32)
    ps, mins = np.array([.95, 1., .5, .99], np.float32), np.array([0, 0, .3, 1.], np.float32)
    seeds = np.array([0, 17, 2**64 - 1, 100], np.uint64)
    expected = reference(logits, temperatures, ps, mins, seeds, 8)
    with gpu(logits, temperatures, ps, mins, seeds) as case:
        for _ in range(3):
            case.runtime.memset(case.scratch.ptr, 0xff, case.scratch.nbytes)
            for out in case.outputs:
                case.runtime.memset(out.ptr, 0xa5, out.nbytes)
            case.launch()
            actual = case.result()
            assert_result(actual, expected)
        assert actual[2][1] == vocab
        assert actual[0][1] > 64 if vocab == 248320 else True
        # Read-only inputs and exact optional model-facing commit outputs.
        np.testing.assert_array_equal(case.read(case.inputs[0], np.float32, logits.shape), logits)
        np.testing.assert_array_equal(actual[0], actual[5])


@pytest.mark.parametrize("p,m", [(0., 0.), (1e-8, 0.), (.5, 0.), (1., 0.),
                                  (1., 1.), (.95, .5), (.01, 1.), (.95, np.nextafter(np.float32(1), np.float32(0)))])
def test_parameter_nonfinite_and_extreme_boundaries(gpu, p, m):
    logits = np.array([[0., -0., 1., 1., np.nan, np.inf, -np.inf, -100.],
                       [np.nan, np.inf, -np.inf, np.nan, np.inf, -np.inf, np.nan, np.inf],
                       [NEG, NEG, NEG, NEG, NEG, NEG, NEG, NEG],
                       [1e30, -1e30, 1., 1., -1., 0., 0., 0.],
                       [0., 1., 2., 3., 4., 5., 6., 7.]], np.float32)
    temps = np.array([1., 1., 1., 1e-30, 0.], np.float32)
    ps, mins, seeds = np.full(5, p), np.full(5, m), np.arange(5, dtype=np.uint64)
    with gpu(logits, temps, ps, mins, seeds, top_logprobs=64) as case:
        case.launch()
        actual = case.result()
        assert_result(actual, reference(logits, temps, ps, mins, seeds, 64))
        # The original selector is not an oracle at extreme inputs: its
        # uncentered FP32 division can overflow and collapse to argmax. The
        # independent FP64 distribution above is the binding correctness gate.


@pytest.mark.parametrize("temp", [np.nan, np.inf, -1., np.float32(1e-38), 1e30])
def test_temperature_boundary(gpu, temp):
    logits = np.array([[1, 2, 2, -3]], np.float32)
    with gpu(logits, [temp], [.95], [0], [17]) as case:
        case.launch()
        assert_result(case.result(), reference(logits, [temp], [.95], [0], [17], 8))


@pytest.mark.parametrize("maximum", [1e30, -1e30, np.finfo(np.float32).max, -np.finfo(np.float32).max])
def test_extreme_tied_maxima_have_half_half_law(gpu, maximum):
    # Symmetry establishes the expected law independently of softmax arithmetic.
    # A finite positive temperature cannot turn two equal finite logits into an
    # argmax distribution, even if dividing either logit by temperature overflows.
    logits = np.full((1, 2), maximum, np.float32)
    with gpu(logits, [1e-30], [1.], [0.], [17], top_logprobs=2) as case:
        selected = set()
        for step in range(16):
            case.launch(step=step)
            result = case.result()
            np.testing.assert_array_equal(result[2], [2])
            np.testing.assert_array_equal(result[3], [[0, 1]])
            np.testing.assert_allclose(result[1], [-np.log(2.)], rtol=0, atol=2e-7)
            np.testing.assert_allclose(result[4], [[-np.log(2.), -np.log(2.)]], rtol=0, atol=2e-7)
            weights = case.sorted_weights()[0]
            np.testing.assert_allclose(weights / weights.sum(), [.5, .5], rtol=0, atol=0)
            expected_id = int(uniform(17, step, 0) > .5)
            assert result[0][0] == result[5][0] == expected_id
            assert result[6][0] == logits[0, expected_id]
            selected.add(int(result[0][0]))
        assert selected == {0, 1}


def test_floatmax_range_is_centered_before_division(gpu):
    maximum = np.finfo(np.float32).max
    logits = np.array([[maximum, -maximum]], np.float32)
    # Temperature=max makes the centered log weights exactly [0,-2]. Subtracting
    # in FP32 overflows, while separately scaling loses ties at tiny temperature.
    expected = np.array([1., np.exp(-2.)]); expected /= expected.sum()
    with gpu(logits, [maximum], [1.], [0.], [17], top_logprobs=2) as case:
        case.launch()
        result = case.result()
        np.testing.assert_array_equal(result[2], [2])
        np.testing.assert_allclose(np.exp(result[4][0]), expected, rtol=2e-7, atol=0)
        weights = case.sorted_weights()[0]
        np.testing.assert_allclose(weights / weights.sum(), expected, rtol=2e-7, atol=0)
        np.testing.assert_allclose(result[1][0], np.log(expected[result[0][0]]), rtol=0, atol=2e-7)


@pytest.mark.parametrize("invalid_row", [0, 1])
@pytest.mark.parametrize("surface", ["state", "outputs"])
@pytest.mark.parametrize("algorithm", ["sorted", "strict"])
def test_invalid_batch_is_transactional_and_retryable(gpu, invalid_row, surface, algorithm):
    from copy import deepcopy
    from types import SimpleNamespace
    from hipengine.core.memory import copy_host_to_device, host_array_ptr
    from hipengine.generation.sampling import RowSamplingState
    from hipengine.runtime.native_sampler import NativeSamplerWorkspace
    good = np.array([[3., 2., 1., 0.], [0., 1., 2., 3.]], np.float32)
    bad = good.copy(); bad[invalid_row] = [np.nan, np.inf, -np.inf, np.nan]
    params = SimpleNamespace(temperature=.7, top_k=0, top_p=.95, min_p=0.,
        logprobs=True, top_logprobs=2, repetition_penalty=1., presence_penalty=0.,
        frequency_penalty=0., logit_bias=(), suppress_token_ids=(), min_tokens=0,
        eos_token_id=None, stop_token_ids=(), stop_token_sequences=())
    def states():
        return tuple(RowSamplingState(seed=17 + row, step_index=3, prompt_tokens=(2,),
            generated_tokens=(1,), stop_token_sequences=((1, 2),)) for row in range(2))
    def snapshot(items):
        return [(s.step_index, tuple(s.generated_tokens), s.stop_suffix_state,
                 s.forced_tokens, deepcopy(s._rng.bit_generator.state)) for s in items]
    with gpu(bad, [.7]*2, [.95]*2, [0.]*2, [17, 18], top_logprobs=2) as case:
        workspace = NativeSamplerWorkspace(runtime=case.runtime, vocab_size=4,
            sampler_library=case.library, full_vocab_algorithm=algorithm)
        stream = case.runtime.stream_create(nonblocking=True)
        row_states = states()
        before = snapshot(row_states)
        # Enqueue sentinels on the same non-default stream before the sampler;
        # neither output may be overwritten by an invalid-row failure.
        index_sentinel = np.full(2, -1234567, np.int64)
        value_sentinel = np.full(2, 12345.5, np.float32)
        index_source, value_source = case.upload(index_sentinel), case.upload(value_sentinel)
        from hipengine.core.hip import MemcpyKind
        for dst, src in zip(case.outputs[5:7], (index_source, value_source)):
            case.runtime.memcpy_async(dst.ptr, src.ptr, src.nbytes, MemcpyKind.DEVICE_TO_DEVICE, stream)
        def sample():
            return workspace.sample_rows(case.inputs[0].ptr, (params, params), row_states,
                out_indices_i64_ptr=case.outputs[5].ptr, out_values_f32_ptr=case.outputs[6].ptr,
                stream=stream)
        try:
            with pytest.raises(ValueError, match="outside vocab"):
                sample()
            if surface == "state":
                assert snapshot(row_states) == before
            else:
                case.runtime.stream_synchronize(stream)
                np.testing.assert_array_equal(case.read(case.outputs[5], np.int64, (2,)), index_sentinel)
                np.testing.assert_array_equal(case.read(case.outputs[6], np.float32, (2,)), value_sentinel)
            copy_host_to_device(case.inputs[0], host_array_ptr(good), good.nbytes, runtime=case.runtime)
            retry = sample()
            fresh = NativeSamplerWorkspace(runtime=case.runtime, vocab_size=4,
                sampler_library=case.library, full_vocab_algorithm=algorithm)
            try:
                fresh_states = states()
                expected = fresh.sample_rows(case.inputs[0].ptr, (params, params), fresh_states, stream=stream)
            finally:
                fresh.close()
            assert retry == expected
            assert snapshot(row_states) == snapshot(fresh_states)
            assert all(s.step_index == 4 for s in row_states)
            np.testing.assert_array_equal(case.read(case.outputs[5], np.int64, (2,)), [s.token_id for s in retry])
            np.testing.assert_array_equal(case.read(case.outputs[6], np.float32, (2,)), [s.logit for s in retry])
            oracle = reference(good, [.7]*2, [.95]*2, [0.]*2, [17, 18], 2, step=3)
            assert [s.token_id for s in retry] == oracle[0].tolist()
            np.testing.assert_allclose([s.logprob for s in retry], oracle[1], rtol=0, atol=2e-5)
        finally:
            workspace.close()
            case.runtime.stream_destroy(stream)


def test_multirow_isolation_permutation_and_streams(gpu):
    logits = np.random.default_rng(99).normal(size=(4, 4097)).astype(np.float32)
    temps, ps, mins, seeds = [0.7] * 4, [.95] * 4, [0.] * 4, [11, 22, 33, 44]
    with gpu(logits, temps, ps, mins, seeds) as case:
        stream = case.runtime.stream_create()
        try:
            case.launch(stream=stream)
            case.runtime.stream_synchronize(stream)
            original = case.result()
            changed = logits.copy()
            changed[1] = np.nan
            changed[3] *= 10
            from hipengine.core.memory import copy_host_to_device, host_array_ptr
            copy_host_to_device(case.inputs[0], host_array_ptr(changed), changed.nbytes, runtime=case.runtime)
            case.launch(stream=stream)
            case.runtime.stream_synchronize(stream)
            mutated = case.result()
            for a, b in zip(original, mutated):
                np.testing.assert_array_equal(a[[0, 2]], b[[0, 2]])
            case.launch(step=14, stream=stream)
            case.runtime.stream_synchronize(stream)
            assert_result(case.result(), reference(changed, temps, ps, mins, seeds, 8, step=14))
        finally:
            case.runtime.stream_destroy(stream)
    # Existing native ABI hashes physical row into the draw. Compensate seeds
    # explicitly to test permutation equivalence, not claim c1/cN RNG invariance.
    order = [3, 0, 2, 1]
    mask, mul = (1 << 64) - 1, 0xBF58476D1CE4E5B9
    permuted_seeds = [seeds[old] ^ (((old + 1) * mul) & mask) ^ (((new + 1) * mul) & mask)
                      for new, old in enumerate(order)]
    with gpu(logits[order], temps, ps, mins, permuted_seeds) as case:
        case.launch()
        for a, b in zip(case.result(), original):
            np.testing.assert_array_equal(a, b[order])


@pytest.mark.parametrize("p", [np.nextafter(np.float32(.5), np.float32(0)), np.float32(.5),
                              np.nextafter(np.float32(.5), np.float32(1))])
def test_nucleus_adjacent_float_boundary(gpu, p):
    logits = np.zeros((1, 1024), np.float32)
    with gpu(logits, [1.], [p], [0.], [17]) as case:
        case.launch()
        actual = case.result()
        assert actual[2][0] == (513 if p > .5 else 512)
        assert_result(actual, reference(logits, [1.], [p], [0.], [17], 8))


def test_workspace_growth_shrink_commit_and_reuse(gpu):
    from types import SimpleNamespace
    from hipengine.generation.sampling import RowSamplingState
    from hipengine.runtime.native_sampler import NativeSamplerWorkspace
    logits = np.random.default_rng(76).normal(size=(8, 4097)).astype(np.float32)
    params = SimpleNamespace(temperature=.7, top_k=0, top_p=1., min_p=0.,
        logprobs=True, top_logprobs=8, repetition_penalty=1., presence_penalty=0.,
        frequency_penalty=0., logit_bias=(), suppress_token_ids=(), min_tokens=0,
        eos_token_id=None, stop_token_ids=(), stop_token_sequences=())
    with gpu(logits, [.7]*8, [1.]*8, [0.]*8, list(range(8))) as case:
        workspace = NativeSamplerWorkspace(runtime=case.runtime, vocab_size=4097, sampler_library=case.library)
        stream = case.runtime.stream_create()
        try:
            for rows in (1, 2, 4, 8, 2, 1):
                states = tuple(RowSamplingState(seed=row, step_index=13) for row in range(rows))
                result = workspace.sample_rows(case.inputs[0].ptr, (params,)*rows, states,
                    out_indices_i64_ptr=case.outputs[5].ptr, out_values_f32_ptr=case.outputs[6].ptr,
                    stream=stream)
                expected = reference(logits[:rows], [.7]*rows, [1.]*rows, [0.]*rows, list(range(rows)), 8)
                executed = workspace.provenance["last_selection"]
                assert executed == {
                    "scope": "full_vocab_top_logprobs", "selected_variant": "sorted_rows_i32",
                    "strict_fallback_variant": "temperature_top_logprobs_rows_i32", "rows": rows,
                }
                assert [r.token_id for r in result] == expected[0].tolist()
                assert [r.candidate_count for r in result] == [4097]*rows
                np.testing.assert_allclose([r.logprob for r in result], expected[1], rtol=0, atol=2e-5)
                assert all(state.step_index == 14 for state in states)
                assert all(tuple(state.generated_tokens) == (r.token_id,) for state, r in zip(states, result))
                np.testing.assert_array_equal(case.read(case.outputs[5], np.int64, (8,))[:rows], expected[5])
                np.testing.assert_array_equal(case.read(case.outputs[6], np.float32, (8,))[:rows], expected[6])
                for row, sample in enumerate(result):
                    assert [i for i, _ in sample.top_logprobs] == expected[3][row].tolist()
                    np.testing.assert_allclose([v for _, v in sample.top_logprobs], expected[4][row], rtol=0, atol=2e-5)
                # Existing buffers may be larger than the active shape. Poison
                # all bytes and prove the next smaller call reads no stale data.
                scratch = workspace._named_buffers["sorted_sampler_scratch"]
                case.runtime.memset(scratch.ptr, 0xff, scratch.nbytes)
            assert workspace.full_vocab_algorithm == "sorted"
        finally:
            workspace.close()
            case.runtime.stream_destroy(stream)
        assert workspace.closed and not workspace._buffers
        with NativeWorkspaceStrict(case) as strict:
            assert strict.full_vocab_algorithm == "strict"
            sampled = strict.sample(case.inputs[0].ptr, params, RowSamplingState(seed=0, step_index=13))
            assert 0 <= sampled.token_id < 4097
            assert "sorted_sampler_scratch" not in strict._named_buffers
            assert strict.provenance["last_selection"]["selected_variant"] == "temperature_top_logprobs_rows_i32"


class NativeWorkspaceStrict:
    def __init__(self, case):
        from hipengine.runtime.native_sampler import NativeSamplerWorkspace
        self.workspace = NativeSamplerWorkspace(runtime=case.runtime, vocab_size=case.vocab,
                                                sampler_library=case.library, full_vocab_algorithm="strict")
    def __enter__(self):
        return self.workspace
    def __exit__(self, *_):
        self.workspace.close()


@pytest.mark.parametrize("filtered", [False, True])
def test_distribution_numerical_envelope(gpu, filtered):
    # Complete distribution comparison, not just the selected token. All finite
    # vocabulary entries participate; broad, tied, spiky and long-tail rows.
    logits = np.random.default_rng(18).normal(size=(8, 248320)).astype(np.float32)
    for row, scale in enumerate([0, .01, .1, 1, 2, 4, 8, 16]):
        logits[row] *= scale
    temps = np.full(8, .7, np.float32)
    ps = np.full(8, .95 if filtered else 1., np.float32)
    mins = np.full(8, .01 if filtered else 0., np.float32)
    with gpu(logits, temps, ps, mins, np.arange(8, dtype=np.uint64)) as case:
        case.launch()
        result = case.result()
        weights = case.sorted_weights()
    kls, agreements = [], []
    for row in range(8):
        ids = np.lexsort((np.arange(logits.shape[1]), -logits[row]))
        values = logits[row, ids].astype(np.float64)
        ideal = np.exp((values - values[0]) / float(temps[row]))
        count = len(ideal)
        if filtered:
            count = int(np.searchsorted(np.cumsum(ideal), float(ps[row]) * ideal.sum(), side="left")) + 1
            count = max(1, int(np.count_nonzero(ideal[:count] >= float(mins[row]))))
        assert result[2][row] == count
        ideal = ideal[:count]; ideal /= ideal.sum()
        observed = weights[row, :count] / weights[row, :count].sum()
        positive = observed > 0
        kls.append(float(np.sum(observed[positive] * np.log(observed[positive] / ideal[positive]))))
        agreements.append(int(np.argmax(observed)) == int(np.argmax(ideal)))
    assert max(kls) <= 1e-3
    assert np.mean(agreements) == 1.
    print(f"sampler full-distribution KL mean/p95/p99/max={np.mean(kls):.9g}/"
          f"{np.percentile(kls,95):.9g}/{np.percentile(kls,99):.9g}/{max(kls):.9g}; top1=100%")
