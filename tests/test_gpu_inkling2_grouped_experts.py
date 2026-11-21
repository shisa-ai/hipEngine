import ctypes
from contextlib import closing
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.loading.inkling2_device_experts import Inkling2DeviceExperts
from hipengine.loading.inkling2_mini import Inkling2ExpertBank


@pytest.mark.parametrize("wave_rows,paired", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("nvfp4", [False, True])
def test_grouped_projection_matches_singletons(nvfp4, wave_rows, paired, monkeypatch):
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError as exc:
        pytest.skip(f"HIP unavailable: {exc}")
    from hipengine.core.hip import HipMemcpyKind, get_hip_runtime
    from hipengine.kernels.backends import resolve_backend
    from hipengine.kernels.hip_gfx1100.quant.inkling2_experts import (
        launch_expert_pointwise, launch_grouped_experts, launch_inkling2_expert, launch_wave_grouped_experts,
    )
    from hipengine.kernels.hip_gfx1100.quant.inkling2_short_conv import launch_short_conv
    from hipengine.runtime.inkling2_device_convolution import Inkling2DeviceConvolution

    if resolve_backend("auto", execution_device=True) != "hip_gfx1100":
        pytest.skip("test requires gfx1100")
    rng = np.random.default_rng(42)
    experts, rows, cols = 13, 64, 1152
    values = rng.integers(0, 127, (experts, rows, cols // 2 if nvfp4 else cols), dtype=np.uint8)
    scales = rng.integers(20 if nvfp4 else 110, 100 if nvfp4 else 125,
                         (experts, rows, cols // (16 if nvfp4 else 32)), dtype=np.uint8)
    amax = rng.uniform(0.1, 1.0, experts).astype(np.float32) if nvfp4 else None
    bank = Inkling2ExpertBank(values, scales, amax, 1.3 if nvfp4 else None)
    down_values = rng.integers(0, 127, (experts, cols, rows // 4 if nvfp4 else rows // 2), dtype=np.uint8)
    down_scales = np.full((experts, cols, rows // (32 if nvfp4 else 64)), 48 if nvfp4 else 116, dtype=np.uint8)
    down = Inkling2ExpertBank(down_values, down_scales, amax, bank.denominator)
    source = SimpleNamespace(expert_bank=lambda layer, stem: bank if stem == "w13" else down)
    ids = np.array([12, 5, 3, 9, 1, 0, 7, 6])
    runtime = get_hip_runtime()
    copies = []
    memcpy = type(runtime).memcpy

    def record_copy(self, dst, src, size, kind):
        copies.append(kind)
        return memcpy(self, dst, src, size, kind)

    monkeypatch.setattr(type(runtime), "memcpy", record_copy)
    memcpy_async = type(runtime).memcpy_async
    synchronize = type(runtime).stream_synchronize
    exchanges = []

    def record_async(self, dst, src, size, kind, stream):
        exchanges.append(kind)
        return memcpy_async(self, dst, src, size, kind, stream)

    def record_wait(self, stream):
        exchanges.append("wait")
        return synchronize(self, stream)

    monkeypatch.setattr(type(runtime), "memcpy_async", record_async)
    monkeypatch.setattr(type(runtime), "stream_synchronize", record_wait)
    expected_exchange = [HipMemcpyKind.HOST_TO_DEVICE, HipMemcpyKind.DEVICE_TO_HOST, "wait"]
    grouped = launch_wave_grouped_experts if wave_rows else launch_grouped_experts
    pair_launches = []

    def launch_pairs(*args, **kwargs):
        from hipengine.kernels.hip_gfx1100.quant.inkling2_experts import launch_paired_experts
        pair_launches.append(kwargs["pairs"])
        launch_paired_experts(*args, **kwargs)

    options = {"paired": launch_pairs, "expand_nvfp4_scales": nvfp4} if paired else {}
    with Inkling2DeviceExperts(source, [0], runtime=runtime, launch=launch_inkling2_expert,
                              grouped=grouped, max_selected=len(ids),
                              pointwise=launch_expert_pointwise, **options) as owner:
        for selected in (ids, ids[::2]):
            x = rng.standard_normal((len(selected), cols), dtype=np.float32)
            for shared in (True, False):
                reference = np.stack([owner.project(0, "w13", int(e), x[0] if shared else x[i])
                                      for i, e in enumerate(selected)])
                copies.clear()
                actual = owner.project_group(0, "w13", selected, x[0] if shared else x)
                assert copies == [HipMemcpyKind.HOST_TO_DEVICE, HipMemcpyKind.DEVICE_TO_HOST]
                np.testing.assert_array_equal(actual.view(np.uint32), reference.view(np.uint32))
            factors = np.full(len(selected), 1 / len(selected), dtype=np.float32)
            projection = owner.project_group(0, "w13", selected, x[0])
            gate, up = projection[:, 0::2], projection[:, 1::2]
            activated = 4 * np.tanh(gate / 4) * np.exp(-np.logaddexp(0, -gate)) * 6 * np.tanh(up / 6)
            expected = np.zeros(cols, dtype=np.float32)
            for factor, value in zip(factors, owner.project_group(0, "w2", selected, activated)):
                expected += factor * value
            exchanges.clear()
            actual = owner.feed_forward(0, selected, factors, x[0], 4, 6)
            assert exchanges == expected_exchange
            np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)
            for width in ((2, 5, 17) if paired else (2, 5, 3)):
                block = rng.standard_normal((width, cols), dtype=np.float32)
                choices = np.stack([np.roll(selected, i) for i in range(width)])
                if paired and width == 17:
                    choices[-1, 0] = 11  # Include an expert used by just one row.
                mixes = rng.uniform(0.05, 0.8, choices.shape).astype(np.float32)
                independent = np.stack([owner.feed_forward(0, ids, weights, row, 4, 6)
                                        for ids, weights, row in zip(choices, mixes, block, strict=True)])
                exchanges.clear()
                packed = owner.feed_forward(0, choices, mixes, block, 4, 6)
                assert exchanges == expected_exchange
                np.testing.assert_array_equal(packed.view(np.uint32), independent.view(np.uint32))
            if paired:
                assert pair_launches[-2:] == [9 * len(selected)] * 2

            taps = rng.standard_normal((cols, 3), dtype=np.float32)

            def convolution():
                return closing(Inkling2DeviceConvolution(runtime=runtime, launch=launch_short_conv,
                                                         channels=cols, width=3))

            with convolution() as separate, convolution() as chained:
                expected = np.stack([separate.step(0, "mlp", row, taps) for row in packed])

                def convolve(ptr, shape):
                    chained.apply_static(0, "mlp", taps, ptr, shape)

                exchanges.clear()
                actual = owner.feed_forward(0, choices, mixes, block, 4, 6, transform_output=convolve)
                assert exchanges == expected_exchange
                np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
