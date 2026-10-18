"""Unit tests for the Gemma 4 control-smoke session adapter (piece B).

These run against a stub runner so the contract itself -- call shape,
lifecycle, and the honest rejections -- is what is under test, not weight
loading. The piece-B wiring test that drives a real runner lives separately
because it needs the fixture artifact.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from hipengine.runtime.gemma4_session import Gemma4ForwardResult, Gemma4ResidentSession


class StubRunner:
    """Minimal stand-in recording exactly what the session asked of it."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.position = 0
        self.closed = False

    def forward(self, token_ids, *, apply_softcap: bool = True):
        self.calls.append(("forward", tuple(int(t) for t in token_ids)))
        self.position += len(token_ids)
        return np.arange(7, dtype=np.float32)

    def next_token(self, logits) -> int:
        self.calls.append(("next_token",))
        return 3

    def reset(self) -> None:
        self.calls.append(("reset",))
        self.position = 0

    def close(self) -> None:
        self.calls.append(("close",))
        self.closed = True


def _session(runner=None) -> tuple[Gemma4ResidentSession, StubRunner]:
    stub = runner or StubRunner()
    return Gemma4ResidentSession(stub, backend="hip_gfx1100", target_arch="gemma4"), stub


def test_unit_gemma4_session_prefill_shape_and_delegate():
    session, stub = _session()
    result = session.prefill([11, 12, 13])

    assert isinstance(result, Gemma4ForwardResult)
    assert stub.calls == [("forward", (11, 12, 13)), ("next_token",)]
    assert stub.position == 3
    assert result.token_id == 3
    assert result.logits.shape == (7,)
    assert session.position == 3


def test_unit_gemma4_session_step_is_a_one_token_forward():
    session, stub = _session()
    result = session.step(42)

    assert stub.calls == [("forward", (42,)), ("next_token",)]
    assert result.token_id == 3
    # The runner treats a decode step as a prefill of one token, which is why
    # both entry points may delegate to the same call.
    assert stub.position == 1


def test_unit_gemma4_session_exposes_the_smokes_runner_reads():
    session, _ = _session()
    assert session.backend == "hip_gfx1100"
    assert session.target_arch == "gemma4"
    assert isinstance(session.runner, StubRunner)


def test_unit_gemma4_session_reset_and_context_manager_close():
    with _session()[0] as session:
        session.prefill([1])
        session.reset()
        assert session.position == 0
    # closed by __exit__
    with pytest.raises(RuntimeError, match="closed"):
        session.step(5)


def test_unit_gemma4_session_close_is_idempotent_and_reaches_the_runner():
    session, stub = _session()
    session.close()
    session.close()
    assert stub.closed
    assert stub.calls.count(("close",)) == 1
    with pytest.raises(RuntimeError, match="closed"):
        session.prefill([1])


def test_unit_gemma4_session_rejects_empty_prompt_and_unsupported_flags():
    session, _ = _session()
    with pytest.raises(ValueError, match="at least one token"):
        session.prefill([])
    with pytest.raises(NotImplementedError, match="return_logits=False"):
        session.step(1, return_logits=False)
    # The Qwen-shaped bulk/GDN keywords must be accepted, since the call site
    # is shared -- accepting them is the contract, interpreting them is not.
    session.prefill([1, 2], use_bulk=False, bulk_attention_mode="expl",
                    capture_hidden_seed_fp32=True)


def test_unit_gemma4_session_requires_a_runner():
    with pytest.raises(ValueError, match="runner must not be None"):
        Gemma4ResidentSession(None, backend="b", target_arch="g")


def test_unit_gemma4_session_module_stays_torch_free():
    source = open("hipengine/runtime/gemma4_session.py", encoding="utf-8").read()
    assert "import torch" not in source
    assert importlib.util.find_spec("hipengine.runtime.gemma4_session") is not None