"""CPU ownership regressions for the Q6 full-GEMM prefill scratch."""
from types import SimpleNamespace

import pytest

from hipengine.core import hipblaslt, memory
from hipengine.kernels.hip_gfx1100.quant import gguf_q6_k_f16_rocblas_prefill as q6


@pytest.fixture
def rig(monkeypatch):
    live = {}
    events = []
    next_ptr = iter(range(100, 10000))

    class Runtime:
        device = 0

        def current_device(self):
            return self.device

        def stream_synchronize(self, stream):
            events.append(('sync', stream))

    def malloc(nbytes, *, runtime):
        buf = SimpleNamespace(ptr=next(next_ptr), nbytes=nbytes)
        live[buf.ptr] = (runtime, buf)
        return buf

    def free(buf, *, runtime):
        assert live.pop(buf.ptr)[0] is runtime
        events.append(('free', buf.ptr))

    class Problem:
        def __init__(self, owner, *args, **kwargs):
            self.owner = owner
            self.closed = False

        def algorithms(self, maximum):
            return [object()]

        def launch(self, algo, x, weight, out, workspace, *, stream):
            events.append(('launch', weight, out, workspace, stream))

        def close(self):
            assert not self.closed
            self.closed = True
            events.append(('problem-close',))

    class Lt:
        def close(self):
            events.append(('handle-close',))

    monkeypatch.setattr(memory, 'malloc', malloc)
    monkeypatch.setattr(memory, 'free', free)
    monkeypatch.setattr(hipblaslt, 'HipblasLt', Lt)
    monkeypatch.setattr(hipblaslt, 'HipblasLtProblem', Problem)
    monkeypatch.setattr(q6, 'build_gguf_q6_k_f16_rocblas_prefill', lambda **kw: object())
    monkeypatch.setattr(q6, '_launch_t16_dequantize_f16_tile', lambda *a, **kw: None)
    monkeypatch.setattr(q6, 'f32_to_bf16', lambda *a, **kw: None)
    return SimpleNamespace(runtime=Runtime(), Runtime=Runtime, live=live, events=events)


def launch(rig, owner=None, stream=0, runtime=None):
    q6.gguf_q6_k_t16_qmicro_planar_fullgemm_fp16_in_bf16_out(
        1, 2, 3, 8, 256, 16, runtime=runtime or rig.runtime,
        stream=stream, workspace_owner=owner,
    )


def test_algorithm_choice_does_not_depend_on_wall_clock(rig, monkeypatch):
    import time

    algorithms = [object(), object()]
    selected = []
    monkeypatch.setattr(
        hipblaslt.HipblasLtProblem, 'algorithms', lambda self, maximum: algorithms,
    )
    monkeypatch.setattr(
        hipblaslt.HipblasLtProblem, 'launch',
        lambda self, algo, *args, **kwargs: selected.append(algo),
    )
    def no_timing():
        raise AssertionError('algorithm selection must not benchmark wall time')
    monkeypatch.setattr(time, 'perf_counter', no_timing)
    for _ in range(2):
        owner = q6.Q6FullGemmWorkspace()
        launch(rig, owner)
        owner.close()
    assert selected and all(algo is algorithms[0] for algo in selected)


def test_algorithm_choice_skips_rejected_candidates_in_order(rig, monkeypatch):
    algorithms = [object(), object()]
    selected = []
    monkeypatch.setattr(
        hipblaslt.HipblasLtProblem, 'algorithms', lambda self, maximum: algorithms,
    )
    def run(self, algo, *args, **kwargs):
        selected.append(algo)
        if algo is algorithms[0]:
            raise RuntimeError('unsupported algorithm')
    monkeypatch.setattr(hipblaslt.HipblasLtProblem, 'launch', run)
    owner = q6.Q6FullGemmWorkspace()
    launch(rig, owner)
    launch(rig, owner)
    owner.close()
    assert selected == [algorithms[0], algorithms[1], algorithms[1], algorithms[1]]


def test_direct_launch_releases_all_temporary_allocations(rig):
    launch(rig)
    assert not rig.live
    assert ('problem-close',) in rig.events
    assert ('handle-close',) in rig.events


def test_owner_reuses_shape_but_isolates_sessions_streams_and_runtimes(rig):
    a, b = q6.Q6FullGemmWorkspace(), q6.Q6FullGemmWorkspace()
    launch(rig, a)
    original = set(rig.live)
    launch(rig, a)
    assert set(rig.live) == original
    launch(rig, a, stream=7)
    launch(rig, a, runtime=rig.Runtime())
    launch(rig, b)
    assert len(rig.live) == 12
    a.close()
    assert len(rig.live) == 3
    launch(rig, b)
    assert len(rig.live) == 3
    b.close()
    b.close()
    assert not rig.live
    with pytest.raises(RuntimeError, match='closed'):
        launch(rig, b)


def test_owner_refuses_wrong_device_close_without_freeing(rig):
    owner = q6.Q6FullGemmWorkspace()
    launch(rig, owner)
    rig.runtime.device = 1
    with pytest.raises(RuntimeError, match='device'):
        launch(rig, owner)
    with pytest.raises(RuntimeError, match='device'):
        owner.close()
    assert len(rig.live) == 3
    rig.runtime.device = 0
    owner.close()
    assert not rig.live


def test_failed_plan_creation_does_not_publish_or_leak(rig, monkeypatch):
    owner = q6.Q6FullGemmWorkspace()
    def fail(*args, **kwargs):
        raise RuntimeError('no algorithms')
    monkeypatch.setattr(hipblaslt.HipblasLtProblem, 'algorithms', fail)
    with pytest.raises(RuntimeError, match='no algorithms'):
        launch(rig, owner)
    assert not rig.live
    owner.close()


def test_failed_synchronization_keeps_live_storage_for_retry(rig, monkeypatch):
    owner = q6.Q6FullGemmWorkspace()
    launch(rig, owner)
    original = rig.runtime.stream_synchronize
    def fail(stream):
        raise RuntimeError('sync failed')
    monkeypatch.setattr(rig.runtime, 'stream_synchronize', fail)
    with pytest.raises(RuntimeError, match='sync failed'):
        owner.close()
    assert len(rig.live) == 3
    monkeypatch.setattr(rig.runtime, 'stream_synchronize', original)
    owner.close()
    assert not rig.live


@pytest.mark.parametrize('allocation', [1, 2, 3])
def test_each_allocation_failure_releases_prior_resources(rig, monkeypatch, allocation):
    owner = q6.Q6FullGemmWorkspace()
    original = memory.malloc
    calls = 0
    def allocate(nbytes, *, runtime):
        nonlocal calls
        calls += 1
        if calls == allocation:
            raise RuntimeError('allocation failed')
        return original(nbytes, runtime=runtime)
    monkeypatch.setattr(memory, 'malloc', allocate)
    with pytest.raises(RuntimeError, match='allocation failed'):
        launch(rig, owner)
    assert not rig.live
    owner.close()


def test_descriptor_failure_keeps_other_release_actions(rig, monkeypatch):
    owner = q6.Q6FullGemmWorkspace()
    launch(rig, owner)
    def fail(self):
        raise RuntimeError('descriptor close failed')
    monkeypatch.setattr(hipblaslt.HipblasLtProblem, 'close', fail)
    with pytest.raises(RuntimeError, match='descriptor close failed'):
        owner.close()
    assert not rig.live
    assert ('handle-close',) in rig.events
    owner.close()


def test_staging_passes_owner_only_to_declaring_kernel(rig, monkeypatch):
    from hipengine.runtime import gguf_linear as gl
    from hipengine.kernels.hip_gfx1100.fused import gguf_ops
    owner = q6.Q6FullGemmWorkspace()
    calls = []
    def sibling(*args, stream=0, runtime=None):
        calls.append(None)
    monkeypatch.setattr(gl, 'is_registered', lambda key: True)
    monkeypatch.setattr(gl, 'resolve', lambda **kw: sibling)
    monkeypatch.setattr(gl, 'linear_weight_pointers', lambda abi, weight: (2,))
    monkeypatch.setattr(gguf_ops, 'gguf_cast_bf16_to_f16', lambda *a, **kw: None)
    with gl.prefill_f16_staging_session(
        workspace_ptr=400, workspace_nbytes=4096, workspace_owner=owner,
    ):
        assert gl._launch_prefill_f16_staged(
            None, object(), 1, 3, 8, 256, 16, {'runtime': rig.runtime},
            backend='test', quant='test', layer='linear', runtime=rig.runtime,
        )
        def owned_sibling(*args, workspace_owner=None, **kw):
            calls.append(workspace_owner)
        owned_sibling.workspace_owner_type = q6.Q6FullGemmWorkspace
        sibling = owned_sibling
        assert gl._launch_prefill_f16_staged(
            None, object(), 1, 3, 8, 256, 16, {'runtime': rig.runtime},
            backend='test', quant='test', layer='linear', runtime=rig.runtime,
        )
    assert calls == [None, owner]


def test_failed_plan_sync_retains_unpublished_resources_until_close(rig, monkeypatch):
    owner = q6.Q6FullGemmWorkspace()
    original = rig.runtime.stream_synchronize
    def fail(stream):
        raise RuntimeError('sync failed')
    monkeypatch.setattr(rig.runtime, 'stream_synchronize', fail)
    with pytest.raises(RuntimeError, match='sync failed'):
        launch(rig, owner)
    assert len(rig.live) == 3
    assert not owner.plans
    monkeypatch.setattr(rig.runtime, 'stream_synchronize', original)
    owner.close()
    assert not rig.live
