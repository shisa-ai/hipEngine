"""Real stream-capture classification and eager timeout containment."""
import ctypes

import pytest

try:
    ctypes.CDLL('libamdhip64.so')
    HAVE_HIP = True
except OSError:
    HAVE_HIP = False

pytestmark = pytest.mark.skipif(not HAVE_HIP, reason='HIP runtime unavailable')


def test_capture_then_unbumped_eager_enqueue_cannot_erase_timeout():
    from hipengine.core.device import scoped_current_device
    from hipengine.core.hip import get_hip_runtime
    from hipengine.distributed.device_exchange_compiled import CompiledDeviceExchange
    from hipengine.distributed.transport import TransportStateError

    rt = get_hip_runtime()
    if rt.device_count() < 2:
        pytest.skip('requires two visible devices')
    streams, buffers, graphs, executions = {}, {}, {}, {}
    exchange = None
    try:
        for device in (0, 1):
            with scoped_current_device(rt, device):
                streams[device] = rt.stream_create()
                buffers[device] = []
                for _ in range(2):
                    ptr = rt.malloc(16)
                    buffers[device].append(ptr)
                    rt.memset(ptr, 0, 16)
        exchange = CompiledDeviceExchange(rt, devices=(0, 1), streams=streams,
            num_layers=2, hidden=8, max_spins=20_000)
        for rank in (0, 1):
            with scoped_current_device(rt, rank):
                rt.stream_begin_capture(streams[rank])
                exchange.enqueue_rank(rank, buffers[rank][0], 0, buffers[rank][1])
                graphs[rank] = rt.stream_end_capture(streams[rank])
                executions[rank] = rt.graph_instantiate(graphs[rank])
        # Recording both graphs submitted no work; clearing remains legal.
        exchange.reset_timeouts()
        exchange.step_begin()
        for rank in (0, 1):
            with scoped_current_device(rt, rank):
                rt.graph_launch(executions[rank], streams[rank])
        exchange.wait()
        for rank in (0, 1):
            with scoped_current_device(rt, rank):
                out = (ctypes.c_ubyte * 16)()
                rt.memcpy(ctypes.addressof(out), buffers[rank][1], 16, 2)
                assert bytes(out) == bytes(16)
        # No bump after wait. The second slot's peer flag is still zero, so
        # this accepted eager enqueue times out and must remain observable.
        exchange.enqueue_rank(0, buffers[0][0], 1, buffers[0][1])
        for clear in (exchange.reset_timeouts, exchange.step_begin):
            with pytest.raises(TransportStateError, match='not waited'):
                clear()
        with pytest.raises(TransportStateError, match='spin timeout'):
            exchange.wait()
        assert exchange.poisoned
    finally:
        for device, stream in streams.items():
            with scoped_current_device(rt, device):
                rt.stream_synchronize(stream)
                if device in executions:
                    rt.graph_exec_destroy(executions[device])
                if device in graphs:
                    rt.graph_destroy(graphs[device])
                for ptr in buffers.get(device, []):
                    rt.free(ptr)
        if exchange is not None:
            exchange.close()
        for device, stream in streams.items():
            with scoped_current_device(rt, device):
                rt.stream_destroy(stream)
