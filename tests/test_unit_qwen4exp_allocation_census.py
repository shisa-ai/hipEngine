from types import SimpleNamespace

import pytest

from hipengine.core.memory import DeviceBuffer, memory_stats
from scripts.qwen4exp_allocation_census import CountingRuntime, attribute_allocations, census


def test_owner_census_counts_roots_once_and_ignores_nonowning_views():
    runtime = CountingRuntime()
    ptr = runtime.malloc(32)
    runner = SimpleNamespace(views=[DeviceBuffer(ptr + 8, 8), DeviceBuffer(ptr, 8)],
                             roots=[DeviceBuffer(ptr, 32), DeviceBuffer(ptr, 32)])
    records, totals = attribute_allocations(runner, runtime)
    assert records == [{"owner": "roots.0", "nbytes": 32}]
    assert totals == {"roots": 32}
    runtime.malloc(16)
    with pytest.raises(ValueError, match="missing"):
        attribute_allocations(runner, runtime)


def test_real_recipes_close_without_loading_gpu_and_preserve_outer_stats():
    from hipengine.loading.qwen4_exp_gguf import qwen4_exp_gguf_config_from_metadata
    from tests.test_live_qwen4_exp_gguf_config import _info

    config = qwen4_exp_gguf_config_from_metadata(_info())
    before = memory_stats()
    small = census(config, context=256, chunk=16)
    large = census(config, context=256, chunk=32)
    assert memory_stats() == before
    assert small["teardown_bytes"] == large["teardown_bytes"] == 0
    assert large["prepared_bytes"] > small["prepared_bytes"]
    assert large["owner_bytes"]["state"] == small["owner_bytes"]["state"]
    assert large["owner_bytes"]["gdn_prefill_scratch"] > small["owner_bytes"]["gdn_prefill_scratch"]
    assert large["owner_bytes"]["attention_states"] == small["owner_bytes"]["attention_states"]


def test_counting_runtime_rejects_out_of_bounds_writes():
    from hipengine.core.runtime import MemcpyKind

    runtime = CountingRuntime()
    ptr = runtime.malloc(16)
    runtime.memset(ptr + 8, 0, 8)
    with pytest.raises(ValueError, match="exceeds"):
        runtime.memset(ptr + 8, 0, 9)
    with pytest.raises(ValueError, match="must not read"):
        runtime.memcpy(1, ptr, 8, MemcpyKind.DEVICE_TO_HOST)
