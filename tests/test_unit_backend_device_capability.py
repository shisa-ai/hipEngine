"""Execution-selection refusal for an explicit HIP backend the device cannot run.

The live INT8 MTP prefix-reuse module pinned the ``hip_gfx1151`` backend on a
gfx1100 host; every JIT kernel was compiled for ``gfx1151`` and HIP aborted the
process with a segmentation fault at the first launch. These tests pin the
replacement contract:

- The probe result is a *per-device* architecture list in device order, with
  duplicates preserved, so an index names the device it actually names.
- ``ROCR_VISIBLE_DEVICES`` filters the physical list and ``HIP_VISIBLE_DEVICES``
  then indexes the result. The two compose; neither overrides the other.
- The check validates the *current logical device*, not "the target arch appears
  somewhere in the host": a gfx1151 on device 1 does not make gfx1151 kernels
  runnable on the default device 0.
- Automatic selection and the compatibility check share one device identity, so
  auto never picks a backend for a device other than the one it will validate.
- Only producers that enumerate installed hardware are hardware evidence: an
  offline ``ROCM_TARGET_LST``/``target.lst`` cannot force an admission or a
  refusal.
- An unresolved current device is unknown, not device zero.
- An explicit HIP backend the current device cannot execute is refused with a
  named error before any generator, weight load, or device work.
- Compatible execution, automatic selection, and offline cross-compilation are
  untouched.

``resolve_backend`` without ``execution_device`` stays a pure key resolver,
because registry helpers such as ``resolve_vibevoice_kernels`` resolve a backend
for a dispatch table without touching a device. The capability check runs at the
execution seam that the public ``LLM`` and the server both use.

Every case runs on CPU: the arch probes and the ``hipGetDevice`` query are
monkeypatched, so no ROCm runtime or device is required.
"""

from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from hipengine.core.build import plan_hip_build
from hipengine.kernels.backends import (
    CPU_BACKEND,
    BackendDeviceCapabilityError,
    _current_hip_device_ordinal,
    detect_hip_hardware_target_arches,
    detect_hip_physical_device_arches,
    detect_hip_target_arches,
    hip_target_arch_environment,
    require_backend_device_compatibility,
    resolve_backend,
    selected_device_arches,
)

MIXED_HOST = ("gfx1100", "gfx1151")
TWO_CARD_GFX1100 = ("gfx1100", "gfx1100")


# --- per-device indexing ----------------------------------------------------


def test_duplicate_arch_devices_keep_their_own_index() -> None:
    """``(gfx1100, gfx1100, gfx1151)`` index 1 is a gfx1100, not a gfx1151.

    Deduplicating the host list first collapses the two gfx1100 cards, so index 1
    lands on the gfx1151 and the request is wrongly admitted.
    """

    host = ("gfx1100", "gfx1100", "gfx1151")

    with pytest.raises(BackendDeviceCapabilityError):
        require_backend_device_compatibility(
            "hip_gfx1151",
            detected_arches=host,
            env={"ROCR_VISIBLE_DEVICES": "1"},
            current_device=0,
        )

    # Index 2 really is the gfx1151 card.
    require_backend_device_compatibility(
        "hip_gfx1151",
        detected_arches=host,
        env={"ROCR_VISIBLE_DEVICES": "2"},
        current_device=0,
    )


def test_same_arch_devices_stay_resolvable() -> None:
    """A second card of the same arch must not become an out-of-range index."""

    assert selected_device_arches(
        detected_arches=TWO_CARD_GFX1100,
        env={"ROCR_VISIBLE_DEVICES": "1"},
        current_device=0,
    ) == ("gfx1100",)

    with pytest.raises(BackendDeviceCapabilityError):
        require_backend_device_compatibility(
            "hip_gfx1151",
            detected_arches=TWO_CARD_GFX1100,
            env={"ROCR_VISIBLE_DEVICES": "1"},
            current_device=0,
        )


# --- current device, not host membership ------------------------------------


def test_mixed_host_current_device_decides() -> None:
    """A gfx1151 elsewhere on the host does not excuse kernels on device 0."""

    with pytest.raises(BackendDeviceCapabilityError):
        require_backend_device_compatibility(
            "hip_gfx1151", detected_arches=MIXED_HOST, env={}, current_device=0
        )
    require_backend_device_compatibility(
        "hip_gfx1151", detected_arches=MIXED_HOST, env={}, current_device=1
    )

    require_backend_device_compatibility(
        "hip_gfx1100", detected_arches=MIXED_HOST, env={}, current_device=0
    )
    with pytest.raises(BackendDeviceCapabilityError):
        require_backend_device_compatibility(
            "hip_gfx1100", detected_arches=MIXED_HOST, env={}, current_device=1
        )


def test_unresolved_current_device_is_not_device_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed ``hipGetDevice`` query is unknown, not a definite device zero.

    Treating the failure as ordinal 0 would refuse a compatible explicit
    ``hip_gfx1151`` on a mixed host, contradicting "unknown is not refusal".
    """

    monkeypatch.setattr(
        "hipengine.kernels.backends._current_hip_device_ordinal", lambda: None
    )
    assert selected_device_arches(detected_arches=MIXED_HOST, env={}) == ()

    require_backend_device_compatibility("hip_gfx1151", detected_arches=MIXED_HOST, env={})
    require_backend_device_compatibility("hip_gfx1100", detected_arches=MIXED_HOST, env={})


def test_known_default_device_still_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    """A resolved ordinal 0 on a mixed host refuses the incompatible backend."""

    monkeypatch.setattr(
        "hipengine.kernels.backends._current_hip_device_ordinal", lambda: 0
    )
    with pytest.raises(BackendDeviceCapabilityError):
        require_backend_device_compatibility(
            "hip_gfx1151", detected_arches=MIXED_HOST, env={}
        )
    require_backend_device_compatibility("hip_gfx1100", detected_arches=MIXED_HOST, env={})


# --- visibility composition -------------------------------------------------


def test_visibility_filters_compose_instead_of_overriding() -> None:
    """ROCR=1 then HIP=0 selects physical device 1, so logical 0 is the gfx1151.

    Treating ``HIP_VISIBLE_DEVICES`` as taking precedence over
    ``ROCR_VISIBLE_DEVICES`` would report gfx1100 here and wrongly refuse
    gfx1151.
    """

    env = {"ROCR_VISIBLE_DEVICES": "1", "HIP_VISIBLE_DEVICES": "0"}

    assert selected_device_arches(
        detected_arches=MIXED_HOST, env=env, current_device=0
    ) == ("gfx1151",)
    require_backend_device_compatibility(
        "hip_gfx1151", detected_arches=MIXED_HOST, env=env, current_device=0
    )
    with pytest.raises(BackendDeviceCapabilityError):
        require_backend_device_compatibility(
            "hip_gfx1100", detected_arches=MIXED_HOST, env=env, current_device=0
        )


def test_each_visibility_filter_remaps_on_its_own() -> None:
    assert selected_device_arches(
        detected_arches=MIXED_HOST, env={"ROCR_VISIBLE_DEVICES": "1"}, current_device=0
    ) == ("gfx1151",)
    assert selected_device_arches(
        detected_arches=MIXED_HOST, env={"HIP_VISIBLE_DEVICES": "1"}, current_device=0
    ) == ("gfx1151",)
    # HIP indexes the ROCR-visible list, not the physical list.
    assert selected_device_arches(
        detected_arches=("gfx1100", "gfx1151", "gfx1100"),
        env={"ROCR_VISIBLE_DEVICES": "2,1", "HIP_VISIBLE_DEVICES": "1"},
        current_device=0,
    ) == ("gfx1151",)


# --- automatic selection shares the device identity -------------------------


def test_auto_selection_uses_the_visible_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto must not take the first ambient arch on a filtered mixed host.

    Physical ``(gfx1100, gfx1151)`` with ``ROCR_VISIBLE_DEVICES=1`` selects a
    gfx1151. Ambient-first selection would resolve ``hip_gfx1100`` and then the
    compatibility check would refuse a compatible device.
    """

    llm = _llm(monkeypatch, "auto", MIXED_HOST, env={"ROCR_VISIBLE_DEVICES": "1"})

    assert llm.resolved_backend == "hip_gfx1151"


def test_auto_selection_uses_the_current_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unfiltered mixed host, current device 1: auto must resolve gfx1151."""

    llm = _llm(monkeypatch, "auto", MIXED_HOST, device_ordinal=1)

    assert llm.resolved_backend == "hip_gfx1151"


def test_auto_selection_matches_validation_on_device_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = _llm(monkeypatch, "auto", MIXED_HOST, device_ordinal=0)

    assert llm.resolved_backend == "hip_gfx1100"


def test_auto_selection_falls_back_when_the_device_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unresolved device does not force a specific HIP backend."""

    llm = _llm(monkeypatch, "auto", MIXED_HOST, device_ordinal=None)

    # Ambient hardware order still offers gfx1100, and validation is permissive.
    assert llm.resolved_backend == "hip_gfx1100"


def test_execution_device_selection_is_opt_in() -> None:
    """Without the flag, automatic selection stays ambient for registry use."""

    assert (
        resolve_backend("auto", detected_arches=MIXED_HOST, env={}) == "hip_gfx1100"
    )


# --- hardware evidence only -------------------------------------------------


def _fake_probes(monkeypatch: pytest.MonkeyPatch, *, amdgpu_arch: str | None) -> None:
    """Fake the arch commands: amdgpu-arch unavailable, enumerator offline list."""

    def _run(command, **kwargs):
        name = " ".join(str(part) for part in command)
        if "amdgpu-arch" in name:
            if amdgpu_arch is None:
                raise FileNotFoundError(name)
            return subprocess.CompletedProcess(command, 0, stdout=amdgpu_arch, stderr="")
        if "rocm_agent_enumerator" in name:
            # An offline ROCM_TARGET_LST / adjacent target.lst result.
            return subprocess.CompletedProcess(command, 0, stdout="gfx1151\n", stderr="")
        raise FileNotFoundError(name)

    monkeypatch.setattr("hipengine.kernels.backends.subprocess.run", _run)


def test_offline_target_list_is_not_hardware_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fallback enumerator's offline target list must not force a decision.

    ``rocm_agent_enumerator`` reads ``ROCM_TARGET_LST`` or an adjacent
    ``target.lst`` before querying hardware, so its output cannot be treated as
    installed devices. With ``amdgpu-arch`` unavailable the hardware evidence is
    empty, and neither admission nor refusal may be derived from the list.
    """

    _fake_probes(monkeypatch, amdgpu_arch=None)

    assert detect_hip_physical_device_arches() == ()
    assert detect_hip_hardware_target_arches() == ()
    # The ambient probe still reports it, for diagnostics only.
    assert detect_hip_target_arches() == ("gfx1151",)

    # No admission: execution-time auto does not adopt the offline target.
    with pytest.warns(RuntimeWarning, match="could not detect"):
        assert (
            resolve_backend("auto", execution_device=True, physical_arches=(), env={})
            == CPU_BACKEND
        )
    # No refusal: the incompatible and compatible requests both stay permissive.
    require_backend_device_compatibility("hip_gfx1151", detected_arches=(), env={})
    require_backend_device_compatibility("hip_gfx1100", detected_arches=(), env={})


def test_hardware_probe_ignores_the_offline_target_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When hardware is present, the offline list cannot override it."""

    _fake_probes(monkeypatch, amdgpu_arch="gfx1100\n")

    assert detect_hip_physical_device_arches() == ("gfx1100",)
    assert detect_hip_hardware_target_arches() == ("gfx1100",)
    require_backend_device_compatibility(
        "hip_gfx1100", detected_arches=("gfx1100",), env={}
    )
    with pytest.raises(BackendDeviceCapabilityError):
        require_backend_device_compatibility(
            "hip_gfx1151", detected_arches=("gfx1100",), env={}
        )


# --- permissive only when genuinely unavailable -----------------------------


def test_unavailable_detection_is_permissive() -> None:
    # No probe result at all: incompatibility is not proven.
    require_backend_device_compatibility("hip_gfx1151", detected_arches=[], env={})
    assert selected_device_arches(detected_arches=[], env={}) == ()

    # An opaque device handle cannot be mapped to an architecture.
    require_backend_device_compatibility(
        "hip_gfx1151", detected_arches=MIXED_HOST, env={"ROCR_VISIBLE_DEVICES": "GPU-abc"}
    )
    # An empty visibility list cannot be mapped either.
    require_backend_device_compatibility(
        "hip_gfx1151", detected_arches=MIXED_HOST, env={"ROCR_VISIBLE_DEVICES": ""}
    )
    # An ordinal beyond the visible set is inconsistent, so it is not a refusal.
    require_backend_device_compatibility(
        "hip_gfx1151", detected_arches=MIXED_HOST, env={}, current_device=7
    )


class _FakeHipFunction:
    def __init__(self, rc: int, ordinal: int | None) -> None:
        self.rc = rc
        self.ordinal = ordinal
        self.argtypes = None
        self.restype = None

    def __call__(self, ptr):
        if self.ordinal is not None:
            ctypes.cast(ptr, ctypes.POINTER(ctypes.c_int))[0] = self.ordinal
        return self.rc


class _FakeHipLibrary:
    def __init__(self, rc: int, ordinal: int | None) -> None:
        self.hipGetDevice = _FakeHipFunction(rc, ordinal)


def test_current_device_query_reports_success_and_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ctypes.CDLL", lambda *a, **k: _FakeHipLibrary(rc=0, ordinal=1)
    )
    assert _current_hip_device_ordinal() == 1

    # A nonzero hipGetDevice result is a failure, not ordinal zero.
    monkeypatch.setattr(
        "ctypes.CDLL", lambda *a, **k: _FakeHipLibrary(rc=1, ordinal=None)
    )
    assert _current_hip_device_ordinal() is None


def test_missing_rocm_does_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    def _no_runtime(*args, **kwargs):
        raise OSError("libamdhip64.so not found")

    monkeypatch.setattr("ctypes.CDLL", _no_runtime)
    assert _current_hip_device_ordinal() is None


def test_non_hip_and_unregistered_backends_are_not_gated() -> None:
    require_backend_device_compatibility("cpu_reference", detected_arches=MIXED_HOST, env={})
    require_backend_device_compatibility("custom_backend", detected_arches=MIXED_HOST, env={})

    assert resolve_backend("cpu_reference", detected_arches=MIXED_HOST, env={}) == CPU_BACKEND
    assert (
        resolve_backend("custom_backend", detected_arches=MIXED_HOST, env={})
        == "custom_backend"
    )


def test_hipengine_hip_arch_is_not_hardware_evidence() -> None:
    # HIPENGINE_HIP_ARCH is a build-time cross-compile override, not a device.
    with pytest.raises(BackendDeviceCapabilityError):
        require_backend_device_compatibility(
            "hip_gfx1151",
            detected_arches=("gfx1100",),
            env={"HIPENGINE_HIP_ARCH": "gfx1151"},
        )


# --- compatible execution and offline compilation ---------------------------


def test_compatible_explicit_backend_is_not_refused() -> None:
    require_backend_device_compatibility(
        "hip_gfx1151", detected_arches=("gfx1151",), env={}, current_device=0
    )
    require_backend_device_compatibility(
        "hip_gfx1100", detected_arches=("gfx1100",), env={}, current_device=0
    )
    assert resolve_backend("hip_gfx1151", detected_arches=("gfx1151",), env={}) == "hip_gfx1151"


def test_auto_selection_is_preserved() -> None:
    assert resolve_backend("auto", detected_arches=("gfx1100",), env={}) == "hip_gfx1100"
    assert resolve_backend("auto", detected_arches=("gfx1151",), env={}) == "hip_gfx1151"

    with pytest.warns(RuntimeWarning, match="gfx1102"):
        assert resolve_backend("auto", detected_arches=("gfx1102",), env={}) == CPU_BACKEND


def test_offline_cross_compilation_is_not_gated() -> None:
    source = Path("hipengine/kernels/hip_gfx1100/quant/gguf_q6_k_embedding.hip")
    with hip_target_arch_environment("gfx1151"):
        artifact = plan_hip_build(
            sources=[source],
            family="gguf_q6_k_embedding",
            profile="decode",
            compiler_version="test-compiler",
        )

    assert artifact.target_arch == "gfx1151"
    assert "--offload-arch=gfx1151" in artifact.flags


# --- public execution seam --------------------------------------------------


def _llm(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    host: tuple[str, ...],
    *,
    device_ordinal: int | None = None,
    env: dict[str, str] | None = None,
):
    from hipengine import LLM
    from hipengine.kernels import backends

    monkeypatch.setattr(backends, "detect_hip_target_arches", lambda: tuple(dict.fromkeys(host)))
    monkeypatch.setattr(
        backends, "detect_hip_hardware_target_arches", lambda: tuple(dict.fromkeys(host))
    )
    monkeypatch.setattr(backends, "detect_hip_physical_device_arches", lambda: host)
    monkeypatch.setattr(
        backends, "_current_hip_device_ordinal", lambda: device_ordinal
    )
    for name in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising=False)
    for name, value in (env or {}).items():
        monkeypatch.setenv(name, value)
    return LLM("/nonexistent/incompatible-model.gguf", backend=backend)


def test_homogeneous_host_does_not_query_the_hip_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single-arch host must not initialize HIP just to run the check."""

    from hipengine.kernels import backends

    def _forbidden() -> int:
        raise AssertionError("hipGetDevice ran on a homogeneous host")

    monkeypatch.setattr(backends, "_current_hip_device_ordinal", _forbidden)
    assert selected_device_arches(detected_arches=TWO_CARD_GFX1100, env={}) == ("gfx1100",)
    assert selected_device_arches(detected_arches=("gfx1151",), env={}) == ("gfx1151",)


def test_public_llm_seam_refuses_before_generator_or_device_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refusal lands before any generator, weight load, or device work.

    ``_get_text_generator`` reads the GGUF index (a metadata read of the file
    header and tensor directory, not weights) and then resolves the backend.
    Reading that metadata is allowed and is stubbed here; everything after the
    backend resolution is sentineled so it cannot run unnoticed.
    """

    import hipengine.generation as generation
    from hipengine import LLM
    from hipengine.kernels import backends

    metadata_read = {"count": 0}

    def _stub_metadata(self):
        metadata_read["count"] += 1
        return None, SimpleNamespace(name="stub-model")

    def _forbidden(*args, **kwargs):
        raise AssertionError("device work ran before the capability refusal")

    monkeypatch.setattr(LLM, "_load_model_metadata", _stub_metadata)
    monkeypatch.setattr(generation, "resolve_text_generator", _forbidden)
    monkeypatch.setattr(backends, "load_backend_kernel_package", _forbidden)
    monkeypatch.setattr(backends, "_current_hip_device_ordinal", _forbidden)

    llm = _llm(monkeypatch, "hip_gfx1151", TWO_CARD_GFX1100)

    with pytest.raises(BackendDeviceCapabilityError):
        llm._get_text_generator()

    # The metadata read is the only thing that ran before the refusal.
    assert metadata_read["count"] == 1
    assert llm._text_generator is None


def test_public_llm_seam_validates_an_env_backend_force(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIPENGINE_BACKEND", "hip_gfx1151")
    llm = _llm(monkeypatch, "auto", ("gfx1100",))

    with pytest.raises(BackendDeviceCapabilityError):
        _ = llm.resolved_backend


def test_public_llm_seam_keeps_auto_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = _llm(monkeypatch, "auto", ("gfx1100",))

    assert llm.resolved_backend == "hip_gfx1100"
