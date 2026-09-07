"""Single-device HIP fixture for the UD numerical qualification tests."""
import ctypes

import pytest

from hipengine.kernels.backends import detect_hip_target_arches, resolve_backend


@pytest.fixture(scope="module")
def ud_hip_backend():
    try:
        library = ctypes.CDLL("libamdhip64.so")
    except OSError:
        pytest.skip("HIP runtime unavailable")
    count = ctypes.c_int()
    get_count = library.hipGetDeviceCount
    get_count.argtypes = [ctypes.POINTER(ctypes.c_int)]
    get_count.restype = ctypes.c_int
    status = int(get_count(ctypes.byref(count)))
    # Not initialized, insufficient driver, or no accessible device.
    if status in (3, 35, 100):
        pytest.skip(f"HIP device unavailable (hipGetDeviceCount status {status})")
    if status:
        raise RuntimeError(f"hipGetDeviceCount failed with status {status}")
    if count.value != 1:
        pytest.skip(f"UD numerical gate requires one visible HIP device; found {count.value}")
    arches = tuple(dict.fromkeys(detect_hip_target_arches()))
    if len(arches) != 1:
        pytest.skip(f"UD numerical gate requires one detected physical architecture; found {arches}")
    backend = resolve_backend(env={}, detected_arches=arches, warn=False)
    targets = {"hip_gfx1100": "gfx1100", "hip_gfx1151": "gfx1151"}
    if backend not in targets:
        pytest.skip("UD numerical gate requires gfx1100 or gfx1151")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("HIPENGINE_HIP_ARCH", targets[backend])
        yield backend
