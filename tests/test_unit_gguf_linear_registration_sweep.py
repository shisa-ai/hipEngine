"""Gate for the ``_ensure_linear_kernel_registered`` repair-sweep memo.

The full registration sweep behind that helper is a *repair* path for a registry
that a test cleared, not a per-call requirement: production registers every
family at import. A key that is missing at dispatch time is normally an
unregistered *optional* candidate - the Gemma 4 grouped route probes a variant
preference list, so an unbuilt variant name misses on purpose and then falls
through.

Before the memo, each such miss re-ran ~650 registrations through
``load_backend_kernel_package``. On a 29-layer MoE prefill the Q5_1 preference
probe missed once per layer, so the forward paid 29 full sweeps. The memo keys
the sweep on the registry generation, which keeps the repair semantics (any
mutation re-arms it) while making a missing key cost one dict lookup.
"""

from __future__ import annotations

import pytest

import hipengine.runtime.gguf_linear as gl
from hipengine.kernels.registry import KernelKey, clear_registry_for_tests, register

_SWEEP_MARKER = KernelKey("hip_gfx1100", "linear", "gguf_q4_k", "sweep_marker_gemv")
_MISSING_KEY = KernelKey(
    "hip_gfx1100", "moe_linear", "gguf_q5_1", "selected_grouped_prefill_never_registered"
)


@pytest.fixture()
def sweeps(monkeypatch) -> list[int]:
    """Count full repair sweeps and start each test with the memo un-armed."""

    counter = [0]

    def _counting_sweep(*args: object, **kwargs: object) -> None:
        counter[0] += 1

    monkeypatch.setattr(gl, "_REGISTRATION_SWEEP_GENERATION", -1)
    monkeypatch.setattr(gl, "register_dense_gemv_kernels", _counting_sweep)
    # The remaining families in the sweep are irrelevant to the memo and would
    # each touch the real registry; the sweep is entered through the first call.
    def _noop(*args: object, **kwargs: object) -> None:
        return None

    for name in (
        "register_gguf_k_gemv_kernels",
        "register_gguf_k_t16_selected_prefill_kernels",
        "register_gguf_k_mmq_prefill_kernels",
        "register_gguf_q4_k_gemv_kernels",
        "register_gguf_q4_k_prefill_kernels",
        "register_gguf_q4_k_pack8_gemv_kernels",
    ):
        monkeypatch.setattr(gl, name, _noop)
    monkeypatch.setattr(gl, "load_backend_kernel_package", _noop)
    return counter


def test_registered_key_never_sweeps(sweeps: list[int]) -> None:
    register(_SWEEP_MARKER, lambda: None, replace=True)
    gl._ensure_linear_kernel_registered(_SWEEP_MARKER)
    assert sweeps[0] == 0


def test_missing_key_sweeps_once_per_generation(sweeps: list[int]) -> None:
    for _ in range(5):
        gl._ensure_linear_kernel_registered(_MISSING_KEY)
    assert sweeps[0] == 1


def test_unrepairable_key_still_sweeps_only_once(sweeps: list[int]) -> None:
    """The pathological case: the sweep cannot register the key, so it stays missing.

    This is what the Gemma 4 route hits - the sweep is not able to satisfy a
    deliberately absent variant - so the memo is what bounds the cost.
    """

    for _ in range(29):
        gl._ensure_linear_kernel_registered(_MISSING_KEY)
    assert gl.is_registered(_MISSING_KEY) is False
    assert sweeps[0] == 1


def test_registry_mutation_rearms_the_sweep(sweeps: list[int]) -> None:
    gl._ensure_linear_kernel_registered(_MISSING_KEY)
    assert sweeps[0] == 1
    clear_registry_for_tests()
    gl._ensure_linear_kernel_registered(_MISSING_KEY)
    assert sweeps[0] == 2


def test_register_mutation_rearms_the_sweep(sweeps: list[int]) -> None:
    gl._ensure_linear_kernel_registered(_MISSING_KEY)
    assert sweeps[0] == 1
    register(KernelKey("hip_gfx1100", "linear", "gguf_q4_k", "rearm_marker"), lambda: None, replace=True)
    gl._ensure_linear_kernel_registered(_MISSING_KEY)
    assert sweeps[0] == 2
