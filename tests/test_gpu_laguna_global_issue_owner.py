"""Device contracts for global issue reference ownership at routing boundaries."""

from types import SimpleNamespace

import numpy as np
import pytest

from tests.test_gpu_laguna_kv_attention import _gfx1151_available
from scripts.laguna_global_issue_diagnostic import global_issue_arm


@pytest.mark.skipif(not _gfx1151_available(), reason="requires gfx1151 HIP runtime")
@pytest.mark.parametrize("name", ["assume_exp", "exp32", "mixed32"])
@pytest.mark.parametrize("capacity,live", [
    (4096, 126), (4096, 127), (4096, 576),
    (4096, 3999), (4096, 4000), (4096, 4001),
    (4095, 576), (4097, 576),
])
def test_global_issue_owner_matches_selected_primitive(name, capacity, live):
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import (
        copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc,
    )
    from hipengine.loading.laguna_gguf import FULL_ATTENTION
    from hipengine.runtime.laguna_kv import allocate_laguna_kv_cache

    runtime = get_hip_runtime()
    config = SimpleNamespace(
        block_count=1, layer_types=(FULL_ATTENTION,), head_counts=(48,),
        head_count_kv=8, key_length=128, value_length=128, sliding_window=512,
    )
    cache = allocate_laguna_kv_cache(config, context_length=capacity,
                                   backend="hip_gfx1151", runtime=runtime)
    allocations = []

    def allocate(nbytes):
        buffer = malloc(nbytes, runtime=runtime)
        allocations.append(buffer)
        return buffer

    def upload(array):
        buffer = allocate(array.nbytes)
        copy_host_to_device(buffer, host_array_ptr(array), array.nbytes, runtime=runtime)
        return buffer

    def download(buffer, dtype, shape):
        array = np.empty(shape, dtype=dtype)
        copy_device_to_host(host_array_ptr(array), buffer, array.nbytes, runtime=runtime)
        return array

    try:
        rng = np.random.default_rng(349)
        keys = upload(rng.normal(0, .12, (live, 8, 128)).astype(np.float32))
        values = upload(rng.normal(0, .12, (live, 8, 128)).astype(np.float32))
        query = upload(rng.normal(0, .12, (48, 128)).astype(np.float32))
        gate = upload(rng.normal(0, .4, 48).astype(np.float32))
        out, primitive_out = allocate(48 * 128 * 4), allocate(48 * 128 * 4)
        gated, primitive_gated = allocate(48 * 128 * 2), allocate(48 * 128 * 2)
        cache.prepare_rows(tuple(range(live)))
        cache.append_rows(0, keys.ptr, values.ptr, live)
        cache.commit_rows()
        owner = SimpleNamespace(kv_cache=cache)
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
                with global_issue_arm(owner, SimpleNamespace(**{
                    "compare_global_" + name: True,
                }), candidate=candidate):
                    fused = cache.attend(0, query.ptr, out.ptr,
                                         gate_ptr=gate.ptr, gated_out_ptr=gated.ptr)
                    variant, fn, args, kwargs = calls[-1]
                    if live < 127:
                        assert not fused
                        assert "fused" not in variant
                    elif capacity != 4096:
                        assert variant == "global_context_split_exact_gated_spans"
                    elif live > 4000:
                        assert variant == "global_context_fused_exact_gated_gqa1_fixedshape_spans"
                    else:
                        pairs = {
                            "assume_exp": ("gqa2_vstage64_vec16_direct", "gqa2_vstage64_vec16_direct_assume_exp"),
                            "exp32": ("gqa2_vstage64_vec16_direct_assume_exp", "gqa2_exp32_vstage64_vec16_direct_assume_exp"),
                            "mixed32": ("gqa2_exp32_vstage64_vec16_direct_assume_exp", "mixed32_exp32_vstage64_vec16_direct_assume_exp"),
                        }
                        assert variant == "global_context_fused_exact_gated_" + pairs[name][candidate] + "_fixedshape_spans"
                    primitive_args = list(args)
                    primitive_args[3] = primitive_out.ptr
                    if fused:
                        primitive_args[5] = primitive_gated.ptr
                    fn(*primitive_args, **kwargs)
                    runtime.device_synchronize()
                    np.testing.assert_array_equal(download(out, np.float32, (48, 128)),
                                                  download(primitive_out, np.float32, (48, 128)))
                    if fused:
                        np.testing.assert_array_equal(download(gated, np.uint16, (48, 128)),
                                                      download(primitive_gated, np.uint16, (48, 128)))
        finally:
            del cache._resolve
        assert vars(cache) == before
    finally:
        for allocation in reversed(allocations):
            free(allocation, runtime=runtime)
        cache.free()
