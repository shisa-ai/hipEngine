import pytest

from hipengine.kernels.hip_gfx1100.linear.f16_mirror import (
    declare_f16_mirror,
    invalidate_f16_mirror,
    lookup_f16_mirror,
)


def test_lookup_requires_the_exact_buffer_and_extent():
    invalidate_f16_mirror()
    assert lookup_f16_mirror(4096, 10) == 0
    declare_f16_mirror(4096, 10, 8192)
    assert lookup_f16_mirror(4096, 10) == 8192
    assert lookup_f16_mirror(4096, 9) == 0
    assert lookup_f16_mirror(4100, 10) == 0
    declare_f16_mirror(1 << 20, 4, 1 << 21)
    assert lookup_f16_mirror(4096, 10) == 0
    assert lookup_f16_mirror(1 << 20, 4) == 1 << 21
    invalidate_f16_mirror()
    assert lookup_f16_mirror(1 << 20, 4) == 0


@pytest.mark.parametrize("args", [(0, 4, 8), (8, 0, 16), (8, 4, 0)])
def test_declare_rejects_null_or_empty(args):
    with pytest.raises(ValueError):
        declare_f16_mirror(*args)
