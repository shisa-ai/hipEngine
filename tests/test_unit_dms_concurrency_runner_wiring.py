"""Single-session DMS probe cycles must use the private-C1 runner wiring.

A one-session (C1) cycle must not construct a shared
``Qwen35GGUFFullStackRunner``: sharing forces device token-embedding
placement (``shared_runner_device_fallback``) and disables the selective
small-weight arena (``shared_runner_fallback``), bypassing the existing
private-C1 memory optimizations. Multi-session cycles keep exactly one
shared runner, which is the required multi-row placement policy.
"""
from types import SimpleNamespace

import numpy as np

from scripts import qwen38_dms_concurrency_probe as probe


def _install(monkeypatch):
    constructions = []
    session_kwargs = []

    class Session:
        def __init__(self, *args, **kwargs):
            session_kwargs.append(kwargs)
            self._dms_backend = SimpleNamespace(
                observability_snapshot=lambda: {
                    "capacity": {},
                    "extent_pool": {
                        "capacity_slots": 1,
                        "free_slots": 0,
                        "allocation_failures": 0,
                    },
                    "ledger": {"active_reservations": 1},
                }
            )

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def prefill(self, prompt, **kwargs):
            pass

        def step(self, token, **kwargs):
            return SimpleNamespace(
                token_id=token + 1, logits=np.array([float(token), 1.0], dtype=np.float32)
            )

    def runner(*args, **kwargs):
        constructions.append(kwargs)
        return SimpleNamespace(close=lambda: None)

    monkeypatch.setattr(probe, "Qwen35GGUFResidentSession", Session)
    monkeypatch.setattr(probe, "Qwen35GGUFFullStackRunner", runner)
    monkeypatch.setattr(probe, "memory_stats", lambda: {"current_allocated_bytes": 0})
    return constructions, session_kwargs


def _args(tmp_path):
    return SimpleNamespace(
        model="fixture",
        metadata="fixture",
        backend="hip_gfx1100",
        codec="int8_evaluation",
        verify_c1=False,
        cancel_after_steps=None,
        output=tmp_path / "result.json",
    )


def test_single_session_cycle_uses_private_runner(monkeypatch, tmp_path):
    constructions, session_kwargs = _install(monkeypatch)
    result = probe._run_cycle(_args(tmp_path), 0, [[1, 2, 3]], [3], 2, "", [])
    assert constructions == [], "C1 cycles must not construct a shared runner"
    assert session_kwargs and all(
        kwargs.get("shared_runner") is None for kwargs in session_kwargs
    ), "the single session must own its runner (private-C1 placement policies)"
    wiring = result["runner_wiring"]
    assert wiring["shared_runner"] is False
    assert wiring["sessions"] == 1


def test_multi_session_cycle_shares_one_runner(monkeypatch, tmp_path):
    constructions, session_kwargs = _install(monkeypatch)
    result = probe._run_cycle(_args(tmp_path), 0, [[1, 2], [3, 4]], [2, 2], 2, "", [])
    assert len(constructions) == 1, "multi-session cycles keep exactly one shared runner"
    assert session_kwargs and all(
        kwargs.get("shared_runner") is not None for kwargs in session_kwargs
    )
    wiring = result["runner_wiring"]
    assert wiring["shared_runner"] is True
    assert wiring["sessions"] == 2


def test_runner_wiring_records_placement_labels(monkeypatch, tmp_path):
    _install(monkeypatch)
    result = probe._run_cycle(_args(tmp_path), 0, [[1, 2]], [2], 1, "", [])
    wiring = result["runner_wiring"]
    for key in (
        "shared_runner",
        "session_owns_runner",
        "token_embedding_placement",
        "host_token_embedding_reason",
        "small_weight_arena_enabled",
        "small_weight_arena_reason",
    ):
        assert key in wiring, f"runner wiring must label {key}"
