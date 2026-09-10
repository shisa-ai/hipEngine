"""Process-local HIP scheduling diagnostic; no system policy writes."""
import ctypes

MODES={"auto":0,"spin":1,"yield":2,"blocking":4}


def configure(mode, library=None):
    if mode not in MODES:
        raise ValueError("unknown HIP wait mode")
    lib=library or ctypes.CDLL("libamdhip64.so")
    lib.hipGetDeviceFlags.argtypes=[ctypes.POINTER(ctypes.c_uint)]
    lib.hipGetDeviceFlags.restype=ctypes.c_int
    lib.hipSetDeviceFlags.argtypes=[ctypes.c_uint]
    lib.hipSetDeviceFlags.restype=ctypes.c_int
    before=ctypes.c_uint()
    error=lib.hipGetDeviceFlags(ctypes.byref(before))
    if error:
        raise RuntimeError(f"hipGetDeviceFlags failed: {error}")
    requested=(before.value & ~7)|MODES[mode]
    error=lib.hipSetDeviceFlags(requested)
    if error:
        raise RuntimeError(f"hipSetDeviceFlags failed: {error}")
    after=ctypes.c_uint()
    error=lib.hipGetDeviceFlags(ctypes.byref(after))
    if error or after.value!=requested:
        raise RuntimeError(f"HIP wait mode readback failed: error={error}, flags={after.value}")
    return dict(mode=mode,before_flags=before.value,requested_flags=requested,
                observed_flags=after.value,
                scope="current device in disposable process; before model allocation",
                limit="Readback proves flags accepted,not that every HIP wait path uses them")
