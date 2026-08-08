from __future__ import annotations

from types import SimpleNamespace

import pytest

import hipengine.core.pm4.transport as transport_module
from hipengine.core.pm4.transport import (
    MissingSubmissionTransportError,
    create_graph_submission,
    create_graph_submission_context,
    register_builtin_submission_transports,
    resolve_submission_transport_factory,
    select_submission_transport,
)


def test_submission_transport_selection_is_default_hipgraph_and_environment_is_explicit() -> None:
    assert select_submission_transport(env={}) == "hipgraph"
    assert select_submission_transport(env={"HIPENGINE_SUBMISSION_TRANSPORT": " pm4 "}) == "pm4"
    assert (
        select_submission_transport("aql", env={"HIPENGINE_SUBMISSION_TRANSPORT": "pm4"}) == "aql"
    )
    with pytest.raises(ValueError, match="non-empty"):
        select_submission_transport("   ", env={})


def test_builtin_transport_registry_admits_native_only_on_exact_gfx1100_backend() -> None:
    register_builtin_submission_transports()

    assert callable(resolve_submission_transport_factory("hip_gfx1100", "hipgraph"))
    assert callable(resolve_submission_transport_factory("hip_gfx1151", "hipgraph"))
    assert callable(resolve_submission_transport_factory("hip_gfx1100", "aql"))
    assert callable(resolve_submission_transport_factory("hip_gfx1100", "pm4"))
    with pytest.raises(MissingSubmissionTransportError, match="hip_gfx1151.*pm4"):
        resolve_submission_transport_factory("hip_gfx1151", "pm4")


def test_hipgraph_submission_preserves_native_launch_and_checked_close() -> None:
    calls: list[tuple[object, ...]] = []
    runtime = SimpleNamespace(
        graph_instantiate=lambda graph: calls.append(("instantiate", graph)) or 91,
        graph_launch=lambda executable, stream: calls.append(("launch", executable, stream)),
        graph_exec_destroy=lambda executable: calls.append(("destroy", executable)),
    )

    submission = create_graph_submission(
        backend="hip_gfx1100",
        gfx_arch="gfx1100",
        runtime=runtime,
        graph=17,
        stream=23,
        transport="hipgraph",
    )
    submission.launch(23)
    provenance = submission.provenance()
    submission.close()
    submission.close()

    assert calls == [
        ("instantiate", 17),
        ("launch", 91, 23),
        ("destroy", 91),
    ]
    assert submission.graph_exec == 0
    assert provenance["transport"] == "hipgraph"
    assert provenance["backend"] == "hip_gfx1100"
    assert provenance["launches"] == 1
    assert provenance["native_fallbacks"] == 0


def test_pm4_submission_inspects_once_syncs_before_submit_and_never_launches_hip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []
    manifest = SimpleNamespace(
        gfx_arch="gfx1100",
        fingerprint="manifest-sha",
        nodes=(
            SimpleNamespace(hsaco_sha256="hsaco-b"),
            SimpleNamespace(hsaco_sha256="hsaco-a"),
            SimpleNamespace(hsaco_sha256="hsaco-a"),
        ),
    )

    class FakeExecutable:
        handle = 41

        def launch(self, name: str, *, timeout_seconds: float) -> None:
            calls.append(("native_launch", name, timeout_seconds))

        def provenance(self) -> dict[str, object]:
            return {"pm4_submissions": 1, "retired": True}

        def close(self) -> None:
            calls.append(("executable_close",))
            self.handle = 0

    class FakeContext:
        handle = 31

        def instantiate(
            self,
            observed_manifest,
            *,
            stateful_registers: bool = False,
            local_cache_dependencies: bool = False,
        ) -> FakeExecutable:
            calls.append(
                (
                    "native_instantiate",
                    observed_manifest.fingerprint,
                    stateful_registers,
                    local_cache_dependencies,
                )
            )
            return FakeExecutable()

        def provenance(self) -> dict[str, object]:
            return {"queue_id": 7, "usable": True}

        def close(self) -> None:
            calls.append(("context_close",))
            self.handle = 0

    fake_context = FakeContext()
    monkeypatch.setattr(
        transport_module,
        "inspect_hip_graph",
        lambda runtime, graph, *, gfx_arch, stream: calls.append(
            ("inspect", graph, gfx_arch, stream)
        )
        or manifest,
    )
    monkeypatch.setattr(
        transport_module.NativePm4Context,
        "create",
        lambda **kwargs: calls.append(("context_create", kwargs["pci_bdf"], kwargs["gfx_arch"]))
        or fake_context,
    )
    runtime = SimpleNamespace(
        device_pci_bus_id=lambda: "0000:03:00.0",
        stream_synchronize=lambda stream: calls.append(("stream_sync", stream)),
        graph_instantiate=lambda graph: pytest.fail("explicit PM4 must not instantiate HIP graph"),
        graph_launch=lambda executable, stream: pytest.fail(
            "explicit PM4 must not launch HIP graph"
        ),
    )

    submission = create_graph_submission(
        backend="hip_gfx1100",
        gfx_arch="gfx1100",
        runtime=runtime,
        graph=17,
        stream=23,
        transport="pm4",
        timeout_seconds=2.5,
    )
    submission.launch(29)
    provenance = submission.provenance()
    submission.close()

    assert calls == [
        ("inspect", 17, "gfx1100", 23),
        ("context_create", "0000:03:00.0", "gfx1100"),
        ("native_instantiate", "manifest-sha", False, False),
        ("stream_sync", 29),
        ("native_launch", "pm4", 2.5),
        ("executable_close",),
        ("context_close",),
    ]
    assert provenance["transport"] == "pm4"
    assert provenance["graph_fingerprint"] == "manifest-sha"
    assert provenance["hsaco_sha256"] == ["hsaco-a", "hsaco-b"]
    assert provenance["launches"] == 1
    assert provenance["native_fallbacks"] == 0
    assert provenance["context"]["queue_id"] == 7
    assert provenance["executable"]["pm4_submissions"] == 1


def test_native_submission_context_reuses_one_queue_across_graph_generations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []
    monkeypatch.setenv("HIPENGINE_PM4_STATEFUL_REGISTERS", "1")
    monkeypatch.setenv("HIPENGINE_PM4_LOCAL_CACHE_DEPENDENCIES", "1")
    manifest = SimpleNamespace(
        gfx_arch="gfx1100",
        fingerprint="shared",
        nodes=(SimpleNamespace(hsaco_sha256="hsaco"),),
    )

    class FakeExecutable:
        def __init__(self, handle: int) -> None:
            self.handle = handle

        def launch(self, name: str, *, timeout_seconds: float) -> None:
            calls.append(("launch", self.handle, name))

        def provenance(self) -> dict[str, object]:
            return {"pm4_submissions": 0, "retired": True}

        def close(self) -> None:
            calls.append(("executable_close", self.handle))
            self.handle = 0

    class FakeContext:
        handle = 71
        generation = 0

        def instantiate(
            self,
            observed_manifest,
            *,
            stateful_registers: bool = False,
            local_cache_dependencies: bool = False,
        ) -> FakeExecutable:
            self.generation += 1
            calls.append(
                (
                    "instantiate",
                    self.generation,
                    stateful_registers,
                    local_cache_dependencies,
                )
            )
            return FakeExecutable(80 + self.generation)

        def provenance(self) -> dict[str, object]:
            return {"queue_id": 9, "submissions": 0, "usable": True}

        def close(self) -> None:
            calls.append(("context_close",))
            self.handle = 0

    fake_context = FakeContext()
    monkeypatch.setattr(transport_module, "inspect_hip_graph", lambda *args, **kwargs: manifest)
    monkeypatch.setattr(
        transport_module.NativePm4Context,
        "create",
        lambda **kwargs: calls.append(("context_create",)) or fake_context,
    )
    runtime = SimpleNamespace(device_pci_bus_id=lambda: "0000:03:00.0")
    owner = create_graph_submission_context(
        backend="hip_gfx1100",
        gfx_arch="gfx1100",
        runtime=runtime,
        transport="pm4",
    )
    assert owner is not None

    first = create_graph_submission(
        backend="hip_gfx1100",
        gfx_arch="gfx1100",
        runtime=runtime,
        graph=17,
        stream=23,
        transport="pm4",
        submission_context=owner,
    )
    first.close()
    second = create_graph_submission(
        backend="hip_gfx1100",
        gfx_arch="gfx1100",
        runtime=runtime,
        graph=18,
        stream=24,
        transport="pm4",
        submission_context=owner,
    )
    second.close()

    assert owner.provenance()["children"] == 0
    assert owner.provenance()["generations"] == 2
    assert owner.provenance()["stateful_registers"] is True
    assert owner.provenance()["local_cache_dependencies"] is True
    assert owner.provenance()["context_create_ns"] >= 0
    assert owner.provenance()["last_graph_inspection_ns"] > 0
    assert owner.provenance()["last_native_instantiate_ns"] > 0
    assert fake_context.handle == 71
    owner.close()
    assert owner.provenance()["closed"] is True
    assert calls == [
        ("context_create",),
        ("instantiate", 1, True, True),
        ("executable_close", 81),
        ("instantiate", 2, True, True),
        ("executable_close", 82),
        ("context_close",),
    ]


def test_explicit_pm4_instantiation_failure_closes_context_without_hip_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    manifest = SimpleNamespace(gfx_arch="gfx1100", fingerprint="bad", nodes=())

    class RejectingContext:
        handle = 31

        def instantiate(
            self,
            observed_manifest,
            *,
            stateful_registers: bool = False,
            local_cache_dependencies: bool = False,
        ):
            calls.append("native_instantiate")
            raise RuntimeError("unsupported kernel descriptor")

        def close(self) -> None:
            calls.append("context_close")
            self.handle = 0

    monkeypatch.setattr(transport_module, "inspect_hip_graph", lambda *args, **kwargs: manifest)
    monkeypatch.setattr(
        transport_module.NativePm4Context, "create", lambda **kwargs: RejectingContext()
    )
    runtime = SimpleNamespace(
        device_pci_bus_id=lambda: "0000:03:00.0",
        graph_instantiate=lambda graph: calls.append("hip_fallback"),
    )

    with pytest.raises(RuntimeError, match="unsupported kernel descriptor"):
        create_graph_submission(
            backend="hip_gfx1100",
            gfx_arch="gfx1100",
            runtime=runtime,
            graph=17,
            stream=23,
            transport="pm4",
        )

    assert calls == ["native_instantiate", "context_close"]
