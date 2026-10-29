"""GR scale-only views borrow full-capacity fallback storage, never own it."""
import pytest

from hipengine.core.memory import DeviceBuffer
from hipengine.runtime import qwen4_exp_runner as runner


@pytest.mark.parametrize("rows", [1, 256, 257, 566, 4096])
def test_scale_views_preserve_fallback_capacity_and_single_ownership(monkeypatch, rows):
    allocated, freed = [], []

    def malloc(nbytes, **kwargs):
        buffer = DeviceBuffer((len(allocated) + 1) << 32, nbytes)
        allocated.append(buffer)
        return buffer

    monkeypatch.setattr(runner, "malloc", malloc)
    monkeypatch.setattr(runner, "free", lambda buffer, **kwargs: freed.append(buffer))
    scratch = runner.Qwen4ExpGRScratch.allocate(
        rows=rows, branches=4, hidden=2560, low_rank=320, runtime=object())
    scale = scratch.inv_rms_view(rows=rows, branches=4)
    mirror = scratch.mixed_f16_view(rows=rows, hidden=2560)
    assert scale == DeviceBuffer(scratch.gate.ptr, rows * 4 * 4)
    assert mirror == DeviceBuffer(scratch.normalized.ptr, rows * 2560 * 2)
    assert scratch.gate.nbytes == scratch.normalized.nbytes == rows * 4 * 2560 * 4
    assert len(allocated) == 5
    for kwargs in ({"rows": 0, "branches": 4}, {"rows": rows, "branches": -1},
                   {"rows": rows * 2560 + 1, "branches": 4}):
        with pytest.raises(ValueError):
            scratch.inv_rms_view(**kwargs)
    with pytest.raises(ValueError):
        scratch.mixed_f16_view(rows=rows, hidden=8 * 2560 + 1)
    scratch.close()
    scratch.close()
    assert freed == list(reversed(allocated))
    with pytest.raises(RuntimeError, match="closed"):
        scratch.inv_rms_view(rows=1, branches=4)
    with pytest.raises(RuntimeError, match="closed"):
        scratch.mixed_f16_view(rows=1, hidden=2560)
