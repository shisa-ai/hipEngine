import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.loading.inkling2_device_experts import Inkling2DeviceExperts
from hipengine.loading.inkling2_mini import Inkling2ExpertBank


@pytest.mark.parametrize("wave_rows", [False, True])
@pytest.mark.parametrize("nvfp4", [False, True])
def test_grouped_projection_matches_singletons(nvfp4, wave_rows, monkeypatch):
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError as exc:
        pytest.skip(f"HIP unavailable: {exc}")
    from hipengine.core.hip import HipMemcpyKind, get_hip_runtime
    from hipengine.kernels.backends import resolve_backend
    from hipengine.kernels.hip_gfx1100.quant.inkling2_experts import (
        launch_expert_pointwise, launch_grouped_experts, launch_inkling2_expert, launch_wave_grouped_experts,
    )

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
    grouped = launch_wave_grouped_experts if wave_rows else launch_grouped_experts
    with Inkling2DeviceExperts(source, [0], runtime=runtime, launch=launch_inkling2_expert,
                              grouped=grouped, max_selected=len(ids),
                              pointwise=launch_expert_pointwise) as owner:
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
            copies.clear()
            actual = owner.feed_forward(0, selected, factors, x[0], 4, 6)
            assert copies == [HipMemcpyKind.HOST_TO_DEVICE, HipMemcpyKind.DEVICE_TO_HOST]
            np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)
