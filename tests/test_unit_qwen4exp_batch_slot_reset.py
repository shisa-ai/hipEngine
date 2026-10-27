"""Each batch-state owner exposes slot reset instead of runner-owned memset logic."""
import pytest
from hipengine.runtime.qwen4_exp_batch_state import Qwen4ExpBatchAttentionState
from hipengine.runtime.qwen4_exp_batch_gdn import Qwen4ExpBatchGdnState
from hipengine.runtime.qwen4_exp_batch_ple import Qwen4ExpBatchPleState
from hipengine.runtime.qwen4_exp_batch_qsa_index import Qwen4ExpBatchQsaIndexState


@pytest.mark.parametrize('owner', [Qwen4ExpBatchAttentionState, Qwen4ExpBatchGdnState,
                                  Qwen4ExpBatchPleState, Qwen4ExpBatchQsaIndexState])
def test_owner_has_slot_reset_protocol(owner):
    assert callable(getattr(owner, 'reset_slot', None))


def test_reset_rejects_all_invalid_blocks_before_any_write():
    from hipengine.runtime.batch_slot_reset import zero_block_runs
    from hipengine.core.memory import DeviceBuffer
    from tests.test_unit_qwen4exp_batch_runner import _FakeRuntime

    runtime = _FakeRuntime()
    for blocks in ([0, 2], [0, 0], [-1]):
        with pytest.raises(ValueError):
            zero_block_runs(runtime, (DeviceBuffer(100, 16),), blocks, 8)
        assert runtime.memset_count == 0


def test_index_reset_failure_preserves_generation_and_cursors():
    from tests.test_unit_qwen4exp_batch_runner import _FakeRuntime, _FakeIndex

    runtime = _FakeRuntime()
    state = _FakeIndex(2, runtime)
    state.cursor_host[0] = 7
    state.pooled_count_host[0] = 2
    runtime.fail_memset_at = 1
    with pytest.raises(RuntimeError):
        state.reset_slot(0, blocks=(0,))
    assert state.cursor_host[0] == 7
    assert state.pooled_count_host[0] == 2
    assert state.reset_generation[0] == 0
    runtime.fail_memset_at = None
    state.reset_slot(0, blocks=(0,))
    assert state.cursor_host[0] == state.pooled_count_host[0] == 0
    assert state.reset_generation[0] == 1
