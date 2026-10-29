"""TimesFM HIP registrations shared by the two gfx11 peer backends."""

from functools import wraps

from hipengine.dispatch.timesfm import (
    TIMESFM_DUAL, TIMESFM_F16, TIMESFM3_DUAL, TIMESFM3_F16, TIMESFM3_F32,
    timesfm_key,
)
from hipengine.kernels.backends import hip_target_arch_environment, hip_target_arch_for_backend
from hipengine.kernels.registry import is_registered, register
from . import timesfm
from ..timesfm3 import timesfm3


def _launcher(fn, backend, precision, *, has_dtype):
    arch = hip_target_arch_for_backend(backend)
    dtype = {"fp16": "f16", "fp32": "f32"}[precision]

    @wraps(fn)
    def launch(*args, **kwargs):
        if has_dtype:
            requested = kwargs.pop("dtype", dtype)
            if requested != dtype:
                raise ValueError(f"{fn.__name__} registered for {precision}, got dtype={requested!r}")
            kwargs["dtype"] = dtype
        with hip_target_arch_environment(arch):
            return fn(*args, **kwargs)
    return launch


def register_timesfm_kernels(backend, *, replace=False):
    keys = set()
    for module, names, precisions, has_dtype in (
        (timesfm, TIMESFM_DUAL, ("fp16", "fp32"), True),
        (timesfm, TIMESFM_F16, ("fp16",), False),
        (timesfm3, TIMESFM3_DUAL, ("fp16", "fp32"), True),
        (timesfm3, TIMESFM3_F16, ("fp16",), False),
        (timesfm3, TIMESFM3_F32, ("fp32",), False),
    ):
        for name in names:
            for precision in precisions:
                key = timesfm_key(backend, name, precision)
                keys.add(key)
                if replace or not is_registered(key):
                    register(key, _launcher(getattr(module, name), backend, precision,
                                           has_dtype=has_dtype), replace=replace)
    return keys
