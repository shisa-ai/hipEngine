"""Layer-outer DMS prefill must run without a dense BF16 pool.

Memory review target 4: the ``layer_outer`` DMS prefill mode replaces the
full dense BF16 KV pool with a single shared oracle pair over a
full-capacity hidden plane. Decisions are captured in a first pass, the
compact store is sized exactly from them, and a second pass packs each
completed layer before its oracle is reused.

CPU-testable contract:
- mode normalization/validation (requires external DMS metadata);
- the full-attention prefill scratch sources K/V from the shared oracle
  pair in layer_outer mode;
- the oracle pair is allocated once at full capacity and released;
- the pack sink fails closed without an active pack pass;
- the layer-outer hidden workspace covers the full prompt capacity;
- the probe threads ``--dms-prefill-mode`` into sessions and records it.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.core.dtype import DType
from hipengine.runtime import qwen35_gguf_runner as gguf_runner
from hipengine.runtime.qwen35_gguf_runner import (
    Qwen35GGUFResidentSession,
    _normalize_external_dms_prefill_mode,
)

from tests.test_qwen35_gguf_prefill_scratch_liveness import (
    _fake_dense_qwen36_runner,
    _install_fake_device,
)


def test_prefill_mode_normalization() -> None:
    assert _normalize_external_dms_prefill_mode("dense_pool") == "dense_pool"
    assert _normalize_external_dms_prefill_mode(" Layer_Outer ") == "layer_outer"
    with pytest.raises(ValueError, match="dms_prefill_mode"):
        _normalize_external_dms_prefill_mode("chunked")


def test_layer_outer_mode_requires_dms_metadata() -> None:
    with pytest.raises(ValueError, match="requires external DMS metadata"):
        Qwen35GGUFResidentSession.__post_init__  # noqa: B018 - documented attr
        session = object.__new__(Qwen35GGUFResidentSession)
        session.__dict__.update(dms_metadata_path=None, dms_prefill_mode="layer_outer")
        gguf_runner.Qwen35GGUFResidentSession.__post_init__(session)


def _layer_outer_session(monkeypatch, *, capacity: int = 73_728):
    _install_fake_device(monkeypatch)
    monkeypatch.setattr(gguf_runner, "free", lambda buffer, *, runtime=None: None)
    session = object.__new__(Qwen35GGUFResidentSession)
    session.__dict__.update(
        runner=_fake_dense_qwen36_runner(),
        runtime=SimpleNamespace(),
        scratch=SimpleNamespace(max_positions=capacity),
        backend="hip_gfx1100",
        dms_prefill_mode="layer_outer",
        _dms_prefill_oracle_pair=None,
        _dms_layer_outer_pack=None,
    )
    return session


def test_full_attention_scratch_uses_shared_oracle(monkeypatch) -> None:
    from dataclasses import make_dataclass

    session = _layer_outer_session(monkeypatch)
    Bulk = make_dataclass(
        "Bulk",
        [
            ("block_size", int),
            ("key_cache", object),
            ("value_cache", object),
            ("retained_key_cache", object),
            ("retained_value_cache", object),
            ("retained_append_spans", object),
            ("int8_kv_value_bf16", bool),
        ],
    )
    bulk_scratch = Bulk(
        block_size=256,
        key_cache=None,
        value_cache=None,
        retained_key_cache=None,
        retained_value_cache=None,
        retained_append_spans=None,
        int8_kv_value_bf16=True,
    )
    layer_scratch = session._full_attention_prefill_scratch_for_layer(
        bulk_scratch, layer_id=3
    )
    key_cache, value_cache = session._dms_prefill_oracle_pair
    assert layer_scratch is not bulk_scratch
    assert layer_scratch.block_size == 256
    assert layer_scratch.key_cache is key_cache
    assert layer_scratch.value_cache is value_cache
    assert layer_scratch.retained_key_cache is None
    assert layer_scratch.int8_kv_value_bf16 is False
    # The same pair is reused for the next layer (shared, not per-layer).
    again = session._full_attention_prefill_scratch_for_layer(
        bulk_scratch, layer_id=7
    )
    assert again.key_cache is key_cache
    # Oracle capacity: one full BF16 plane per K and V.
    expected = 73_728 * 4 * 256 * DType.BF16.itemsize
    assert key_cache.nbytes == expected
    assert value_cache.nbytes == expected


def test_oracle_release_and_pack_sink_fail_closed(monkeypatch) -> None:
    session = _layer_outer_session(monkeypatch)
    session._dms_layer_outer_oracle_pair()
    assert session._dms_prefill_oracle_pair is not None
    session._release_dms_prefill_oracle()
    assert session._dms_prefill_oracle_pair is None
    # Release is idempotent.
    session._release_dms_prefill_oracle()
    with pytest.raises(RuntimeError, match="pack sink"):
        session.pack_dms_prefill_layer(3, stream=0)


def test_layer_outer_hidden_workspace_covers_full_prompt(monkeypatch) -> None:
    from tests.test_gguf_bulk_prefill_workspace_release import _fake_session

    session, _ = _fake_session(monkeypatch, capacity=73_728)
    session.__dict__["dms_prefill_mode"] = "layer_outer"
    session._allocate_bulk_prefill_workspace(SimpleNamespace())
    # The hidden plane(s) cover the whole prompt (the real backend policy
    # additionally reuses one plane in place at these row counts).
    expected = 73_728 * 5120 * DType.BF16.itemsize
    assert session._prefill_hidden_a.nbytes == expected
    assert session._prefill_hidden_b.nbytes == expected


def test_layer_outer_hidden_alias_is_on_by_default(monkeypatch) -> None:
    """Adopted default aliases the planes; env 0 is the explicit rollback."""

    from tests.test_gguf_bulk_prefill_workspace_release import _fake_session

    monkeypatch.delenv(gguf_runner._LAYER_OUTER_HIDDEN_ALIAS_ENV, raising=False)
    session, _ = _fake_session(monkeypatch, capacity=8_192)
    session.__dict__["dms_prefill_mode"] = "layer_outer"
    session._allocate_bulk_prefill_workspace(SimpleNamespace())
    assert gguf_runner._layer_outer_hidden_alias_enabled() is True
    assert (
        session._prefill_hidden_a.ptr == session._prefill_hidden_b.ptr
    ), "default layer_outer route must reuse one physical hidden plane"
    # The aliased plane is tracked exactly once in the buffer ledger.
    plane_ids = [
        id(buffer)
        for buffer in session._buffers
        if buffer.ptr == session._prefill_hidden_a.ptr
    ]
    assert len(plane_ids) == 1


def test_layer_outer_hidden_alias_env_rolls_back_to_two_planes(monkeypatch) -> None:
    """HIPENGINE_LAYER_OUTER_HIDDEN_ALIAS=0 restores the two-plane route."""

    from tests.test_gguf_bulk_prefill_workspace_release import _fake_session

    monkeypatch.setenv(gguf_runner._LAYER_OUTER_HIDDEN_ALIAS_ENV, "0")
    session, _ = _fake_session(monkeypatch, capacity=8_192)
    session.__dict__["dms_prefill_mode"] = "layer_outer"
    session._allocate_bulk_prefill_workspace(SimpleNamespace())
    assert gguf_runner._layer_outer_hidden_alias_enabled() is False
    assert (
        session._prefill_hidden_a.ptr != session._prefill_hidden_b.ptr
    ), "env-off rollback must restore two distinct hidden planes"


def test_layer_outer_hidden_alias_env_gates_single_plane(monkeypatch) -> None:
    """Env-on aliases the two hidden planes for the layer_outer route only."""

    from tests.test_gguf_bulk_prefill_workspace_release import _fake_session

    monkeypatch.setenv(gguf_runner._LAYER_OUTER_HIDDEN_ALIAS_ENV, "1")
    session, _ = _fake_session(monkeypatch, capacity=8_192)
    session.__dict__["dms_prefill_mode"] = "layer_outer"
    session._allocate_bulk_prefill_workspace(SimpleNamespace())
    assert gguf_runner._layer_outer_hidden_alias_enabled() is True
    assert (
        session._prefill_hidden_a.ptr == session._prefill_hidden_b.ptr
    ), "alias-enabled layer_outer route must reuse one physical plane"
    expected = 8_192 * 5120 * DType.BF16.itemsize
    assert session._prefill_hidden_a.nbytes == expected


def test_layer_outer_hidden_alias_env_does_not_touch_dense_route(monkeypatch) -> None:
    """Ordinary (dense-pool) prefill ignores the route-scoped alias gate."""

    from tests.test_gguf_bulk_prefill_workspace_release import _fake_session

    monkeypatch.setenv(gguf_runner._LAYER_OUTER_HIDDEN_ALIAS_ENV, "1")
    session, _ = _fake_session(monkeypatch, capacity=8_192)
    session.__dict__["dms_prefill_mode"] = "dense_pool"
    session._allocate_bulk_prefill_workspace(SimpleNamespace())
    assert (
        session._prefill_hidden_a.ptr != session._prefill_hidden_b.ptr
    ), "the alias gate must not affect the ordinary prefill route"


def test_probe_threads_prefill_mode(monkeypatch, tmp_path):
    from scripts import qwen38_dms_concurrency_probe as probe
    from tests.test_dms_concurrency_runner_wiring import _install, _args

    constructions, session_kwargs = _install(monkeypatch)
    args = _args(tmp_path)
    args.dms_prefill_mode = "layer_outer"
    result = probe._run_cycle(args, 0, [[1, 2, 3]], [3], 2, "", [])
    assert all(
        kwargs.get("dms_prefill_mode") == "layer_outer"
        for kwargs in session_kwargs
    )
    assert result["runner_wiring"]["dms_prefill_mode"] == "layer_outer"
    args.dms_prefill_mode = "dense_pool"
    result = probe._run_cycle(args, 0, [[1, 2, 3]], [3], 2, "", [])
    assert result["runner_wiring"]["dms_prefill_mode"] == "dense_pool"
