"""Route-scoped single-plane hidden alias for the non-DMS INT8 layer_outer route.

2026-09-08 engine-comparison follow-up (doc priority #1: long-prefill
hidden-buffer lifetime). The pure-INT8 direct-resident route plans the
"layer_outer_shared_oracle" prefill lifetime: two full-capacity BF16
hidden planes (2.69 GB at 128K on Qwen3.8-27B H5120) plus one shared
BF16 K/V oracle pair. The layer_outer DMS route already aliases its
planes (HIPENGINE_LAYER_OUTER_HIDDEN_ALIAS, adopted 2026-09-08 after a
GPU A/B with byte-identical decode logits); this gate extends the same
single-plane reuse to the non-DMS INT8 layer_outer route only.

The geometry-wide ``hidden_inplace_min_rows`` policy stays untouched:
ordinary dense prefill below the threshold keeps two planes. The INT8
route's own plane count is gated by
``HIPENGINE_INT8_LAYER_OUTER_HIDDEN_ALIAS`` (default ON since the
2026-09-09 adoption; env 0 is the rollback).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.runtime import qwen35_gguf_runner as gguf_runner
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
from hipengine.core.dtype import DType

from tests.test_unit_gguf_bulk_prefill_workspace_release import _fake_session


def _int8_layer_outer_session(monkeypatch, *, capacity: int = 8_192):
    session, freed = _fake_session(monkeypatch, capacity=capacity)
    # The real plan for this route carries mode="layer_outer_shared_oracle"
    # and a full-capacity hidden requirement.
    session.__dict__["_int8_prefill_lifetime_plan"] = SimpleNamespace(
        mode="layer_outer_shared_oracle",
        required_hidden_capacity=capacity,
    )
    session.__dict__["dms_prefill_mode"] = "dense_pool"
    return session, freed


def _chunk_outer_session(monkeypatch, *, capacity: int = 8_192):
    session, freed = _fake_session(monkeypatch, capacity=capacity)
    session.__dict__["_int8_prefill_lifetime_plan"] = SimpleNamespace(
        mode="chunk_outer_layer_local_oracles",
        required_hidden_capacity=768,
    )
    session.__dict__["dms_prefill_mode"] = "dense_pool"
    return session, freed


def test_int8_layer_outer_alias_env_gates_single_plane(monkeypatch) -> None:
    """Env-on aliases the two hidden planes on the INT8 layer_outer route."""

    monkeypatch.setenv(gguf_runner._INT8_LAYER_OUTER_HIDDEN_ALIAS_ENV, "1")
    session, _ = _int8_layer_outer_session(monkeypatch, capacity=8_192)
    session._allocate_bulk_prefill_workspace(SimpleNamespace())
    assert gguf_runner._int8_layer_outer_hidden_alias_enabled() is True
    assert (
        session._prefill_hidden_a.ptr == session._prefill_hidden_b.ptr
    ), "alias-enabled INT8 layer_outer route must reuse one physical plane"
    expected = 8_192 * 5120 * DType.BF16.itemsize
    assert session._prefill_hidden_a.nbytes == expected


def test_int8_layer_outer_alias_is_on_by_default(monkeypatch) -> None:
    """Adopted default aliases the planes; env 0 is the explicit rollback."""

    monkeypatch.delenv(gguf_runner._INT8_LAYER_OUTER_HIDDEN_ALIAS_ENV, raising=False)
    session, _ = _int8_layer_outer_session(monkeypatch, capacity=8_192)
    session._allocate_bulk_prefill_workspace(SimpleNamespace())
    assert gguf_runner._int8_layer_outer_hidden_alias_enabled() is True
    assert (
        session._prefill_hidden_a.ptr == session._prefill_hidden_b.ptr
    ), "default INT8 layer_outer route must reuse one physical hidden plane"


def test_int8_layer_outer_alias_env_off_keeps_two_planes(monkeypatch) -> None:
    """HIPENGINE_INT8_LAYER_OUTER_HIDDEN_ALIAS=0 restores the two-plane route."""

    monkeypatch.setenv(gguf_runner._INT8_LAYER_OUTER_HIDDEN_ALIAS_ENV, "0")
    session, _ = _int8_layer_outer_session(monkeypatch, capacity=8_192)
    session._allocate_bulk_prefill_workspace(SimpleNamespace())
    assert gguf_runner._int8_layer_outer_hidden_alias_enabled() is False
    assert session._prefill_hidden_a.ptr != session._prefill_hidden_b.ptr


def test_int8_layer_outer_alias_leaves_chunk_outer_route_alone(monkeypatch) -> None:
    """The chunk-outer INT8 route (chunk-sized hidden) is not aliased."""

    monkeypatch.setenv(gguf_runner._INT8_LAYER_OUTER_HIDDEN_ALIAS_ENV, "1")
    session, _ = _chunk_outer_session(monkeypatch, capacity=8_192)
    session._allocate_bulk_prefill_workspace(SimpleNamespace())
    assert session._prefill_hidden_a.nbytes == 768 * 5120 * DType.BF16.itemsize
    assert session._prefill_hidden_a.ptr != session._prefill_hidden_b.ptr


def test_int8_layer_outer_alias_leaves_bf16_dense_route_alone(monkeypatch) -> None:
    """The BF16 dense route (mode "bf16") ignores the INT8 alias gate."""

    monkeypatch.setenv(gguf_runner._INT8_LAYER_OUTER_HIDDEN_ALIAS_ENV, "1")
    session, _ = _fake_session(monkeypatch, capacity=8_192)
    session.__dict__["_int8_prefill_lifetime_plan"] = SimpleNamespace(
        mode="bf16",
        required_hidden_capacity=768,
    )
    session.__dict__["dms_prefill_mode"] = "dense_pool"
    session._allocate_bulk_prefill_workspace(SimpleNamespace())
    assert session._prefill_hidden_a.ptr != session._prefill_hidden_b.ptr


def test_int8_layer_outer_alias_alias_ledger_tracks_plane_once(monkeypatch) -> None:
    """The aliased plane is tracked exactly once in the buffer ledger."""

    monkeypatch.setenv(gguf_runner._INT8_LAYER_OUTER_HIDDEN_ALIAS_ENV, "1")
    session, _ = _int8_layer_outer_session(monkeypatch, capacity=8_192)
    session._allocate_bulk_prefill_workspace(SimpleNamespace())
    plane_ids = [
        id(buffer)
        for buffer in session._buffers
        if buffer.ptr == session._prefill_hidden_a.ptr
    ]
    assert len(plane_ids) == 1


def test_int8_layer_outer_alias_does_not_touch_dms_route_gate(monkeypatch) -> None:
    """The DMS layer_outer gate stays independently controllable."""

    monkeypatch.delenv(gguf_runner._INT8_LAYER_OUTER_HIDDEN_ALIAS_ENV, raising=False)
    monkeypatch.setenv(gguf_runner._LAYER_OUTER_HIDDEN_ALIAS_ENV, "0")
    # The INT8 gate follows its own default (ON since adoption) while the
    # DMS gate is independently rolled back.
    assert gguf_runner._int8_layer_outer_hidden_alias_enabled() is True
    assert gguf_runner._layer_outer_hidden_alias_enabled() is False


def test_resident_session_attributes_exist() -> None:
    assert hasattr(Qwen35GGUFResidentSession, "_allocate_bulk_prefill_workspace")


if __name__ == "__main__":
    pytest.main([__file__])
