import ctypes
import pytest
from scripts.qwen4exp_hip_wait_policy import configure


class Function:
    def __init__(self,fn):
        self.fn=fn
    def __call__(self,*args):
        return self.fn(*args)


class Library:
    def __init__(self,ignore=False):
        self.flags=8
        def get(ptr):
            ctypes.cast(ptr,ctypes.POINTER(ctypes.c_uint))[0]=self.flags
            return 0
        def put(value):
            if not ignore:
                self.flags=value
            return 0
        self.hipGetDeviceFlags=Function(get)
        self.hipSetDeviceFlags=Function(put)


def test_preserves_nonscheduling_flags():
    result=configure("spin",Library())
    assert result["before_flags"]==8
    assert result["observed_flags"]==9


def test_readback_failure():
    with pytest.raises(RuntimeError,match="readback"):
        configure("spin",Library(ignore=True))


def test_invalid_mode():
    with pytest.raises(ValueError):
        configure("invalid",Library())
