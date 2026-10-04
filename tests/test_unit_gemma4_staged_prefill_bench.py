"""CPU contracts for the staged attention performance harness's workload."""
import numpy as np
import pytest
from scripts import gemma4_staged_prefill_bench as bench
from scripts.gemma4_staged_prefill_bench import inputs


@pytest.mark.parametrize('duration', [0, -1, float('nan'), float('inf')])
def test_event_timing_rejects_invalid_samples(duration):
    with pytest.raises(ValueError, match='event duration'):
        bench.valid_event_duration(duration)


def test_event_timing_accepts_positive_finite_samples():
    assert bench.valid_event_duration(0.02) == 0.02


def test_prefill_block_is_causal_with_nonzero_absolute_offset():
    q, k, v, mask, offset = inputs(3, 7, 4, 2, 8, 0)
    assert offset == 4
    assert q.shape == (3, 4, 8) and k.shape == v.shape == (7, 2, 8)
    assert q.dtype == k.dtype == v.dtype == np.uint16
    assert mask.sum(axis=1).tolist() == [5, 6, 7]
    assert not mask[0, 5] and mask[2, 6]


def test_sliding_block_masks_old_keys_and_future_keys():
    *_, mask, offset = inputs(3, 7, 4, 2, 8, 2)
    assert offset == 4
    assert np.flatnonzero(mask[0]).tolist() == [3, 4]
    assert np.flatnonzero(mask[2]).tolist() == [5, 6]


def test_inputs_repeat_with_seed_not_variant():
    a = inputs(3, 7, 4, 2, 8, 2)
    b = inputs(3, 7, 4, 2, 8, 2)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


@pytest.mark.parametrize('omitted', [slice(None), slice(1, 2)])
def test_capture_rejects_missing_stores_even_after_correct_baseline(omitted):
    # Fake device memory contains the preceding arm's correct result.
    device = np.full(4, 0x3f80, dtype=np.uint16)
    def poison():
        device.fill(0x7fc1)  # BF16 quiet NaN
    def incomplete_launch():
        active = np.ones(device.shape, dtype=bool)
        active[omitted] = False
        device[active] = 0x3f80
    with pytest.raises(RuntimeError, match='non-finite'):
        bench.capture_bf16(incomplete_launch, poison, lambda: device.copy())


def test_capture_rejects_missing_workspace_writes():
    workspace = np.full(3, 0x3f80, dtype=np.uint16)
    device = workspace.copy()
    def poison():
        workspace.fill(0x7fc1)
        device.fill(0x7fc1)
    # An incomplete score/softmax stage cannot inherit the previous arm's scratch.
    def launch():
        device[:] = workspace
    with pytest.raises(RuntimeError, match='non-finite'):
        bench.capture_bf16(launch, poison, lambda: device.copy())


def test_capture_and_parity_reject_different_finite_output():
    device = np.empty(3, dtype=np.uint16)
    def poison():
        device.fill(0x7fc1)
    actual = bench.capture_bf16(lambda: device.fill(0x4000), poison, lambda: device.copy())
    with pytest.raises(RuntimeError, match='bitwise'):
        bench.require_bitwise_equal(np.full(3, 0x3f80, dtype=np.uint16), actual)
    assert bench.require_bitwise_equal(actual, actual.copy()) is True


def test_gates_still_reject_under_optimized_python():
    import subprocess
    import sys
    program = '''
import numpy as np
from scripts.gemma4_staged_prefill_bench import capture_bf16, require_bitwise_equal
rejected = 0
try:
    capture_bf16(lambda: None, lambda: None, lambda: np.array([0x7fc1], dtype=np.uint16))
except RuntimeError:
    rejected += 1
try:
    require_bitwise_equal(np.array([1], dtype=np.uint16), np.array([2], dtype=np.uint16))
except RuntimeError:
    rejected += 1
raise SystemExit(0 if rejected == 2 else 1)
'''
    result = subprocess.run([sys.executable, '-O', '-c', program], cwd=bench.ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
