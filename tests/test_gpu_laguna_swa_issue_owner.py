"""Device owner/reference equality for saturated and wrapped SWA comparisons."""

from types import SimpleNamespace

import numpy as np
import pytest

from tests.test_gpu_laguna_kv_attention import _gfx1151_available
from scripts.laguna_swa_issue_diagnostic import swa_issue_arm


@pytest.mark.skipif(not _gfx1151_available(), reason="requires gfx1151 HIP runtime")
@pytest.mark.parametrize("name", ["assume_exp", "mixed32", "mixed32_exp4", "mixed32_exp8", "mixed32_exp16", "mixed32_exp32"])
@pytest.mark.parametrize("position", [382, 510, 511, 777])
def test_swa_issue_owner_matches_registered_primitive(name, position):
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
    from hipengine.loading.laguna_gguf import SLIDING_ATTENTION
    from hipengine.runtime.laguna_kv import allocate_laguna_kv_cache

    runtime = get_hip_runtime()
    config = SimpleNamespace(block_count=1, layer_types=(SLIDING_ATTENTION,),
                             head_counts=(72,), head_count_kv=8, key_length=128,
                             value_length=128, sliding_window=512)
    cache = allocate_laguna_kv_cache(config, context_length=4096, backend="hip_gfx1151", runtime=runtime)
    allocations = []

    def allocate(nbytes):
        result = malloc(nbytes, runtime=runtime)
        allocations.append(result)
        return result

    def upload(array):
        result = allocate(array.nbytes)
        copy_host_to_device(result, host_array_ptr(array), array.nbytes, runtime=runtime)
        return result

    def download(buffer, dtype):
        result = np.empty((72, 128), dtype=dtype)
        copy_device_to_host(host_array_ptr(result), buffer, result.nbytes, runtime=runtime)
        return result

    try:
        rng = np.random.default_rng(343)
        key = upload(rng.normal(0, .12, (8, 128)).astype(np.float32))
        value = upload(rng.normal(0, .12, (8, 128)).astype(np.float32))
        query = upload(rng.normal(0, .12, (72, 128)).astype(np.float32))
        gate = upload(rng.normal(0, .4, 72).astype(np.float32))
        out, primitive_out = allocate(72 * 128 * 4), allocate(72 * 128 * 4)
        gated, primitive_gated = allocate(72 * 128 * 2), allocate(72 * 128 * 2)
        for step in range(position + 1):
            cache.prepare_position(step)
            cache.append(0, key.ptr, value.ptr)
        before = vars(cache).copy()
        original_resolve = cache._resolve
        calls = []

        def resolve(layer, variant):
            fn = original_resolve(layer, variant)

            def launch(*args, **kwargs):
                calls.append((variant, fn, args, kwargs))
                return fn(*args, **kwargs)
            return launch

        cache._resolve = resolve
        try:
            for candidate in (False, True):
                with swa_issue_arm(SimpleNamespace(kv_cache=cache), SimpleNamespace(**{
                    "compare_swa_" + name: True,
                }), candidate=candidate):
                    assert cache.attend(0, query.ptr, out.ptr, gate_ptr=gate.ptr, gated_out_ptr=gated.ptr)
                    variant, fn, args, kwargs = calls[-1]
                    if position >= 511:
                        prefixes = ["gqa3_vstage64_vec16_direct", "gqa3_vstage64_vec16_direct_assume_exp",
                                    "mixed32_vstage64_vec16_direct_assume_exp", "mixed32_exp4_vstage64_vec16_direct_assume_exp",
                                    "mixed32_exp8_vstage64_vec16_direct_assume_exp", "mixed32_exp16_vstage64_vec16_direct_assume_exp",
                                    "mixed32_exp32_vstage64_vec16_direct_assume_exp"]
                        index = ["assume_exp", "mixed32", "mixed32_exp4", "mixed32_exp8", "mixed32_exp16", "mixed32_exp32"].index(name)
                        assert variant == "swa_context_fused_exact_gated_" + prefixes[index + candidate] + "_fixed512_spans"
                    else:
                        assert "assume_exp" not in variant
                    primitive_args = list(args)
                    primitive_args[3], primitive_args[5] = primitive_out.ptr, primitive_gated.ptr
                    fn(*primitive_args, **kwargs)
                    runtime.device_synchronize()
                    np.testing.assert_array_equal(download(out, np.float32), download(primitive_out, np.float32))
                    np.testing.assert_array_equal(download(gated, np.uint16), download(primitive_gated, np.uint16))
        finally:
            del cache._resolve
        assert vars(cache) == before
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)
        cache.free()
