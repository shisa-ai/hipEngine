"""The NextN hidden-batch step must supply the split-K workspace it needs.

A multi-choice group past a certain live context failed before commit with
``long-context packed AR decode requires a row-sized split-K workspace``: the
draft path ran the same row-bulk attention as the target but passed no
workspace, so the batch decode refused once its K/V walk had to split. The
refusal was recovered as an autoregressive fallback, which is why the group
looked like it had simply stopped speculating.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from hipengine.runtime import qwen35_gguf_runner as runner_mod
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

SPLIT_THRESHOLD = 1024


class _Workspace:
    def __init__(self, rows: int, max_context_len: int) -> None:
        self.rows = rows
        self.num_splits = (max_context_len + 255) // 256


@dataclass
class _LayerScratch:
    """The step rebuilds this with ``dataclasses.replace``."""

    cos_table: int = 0
    sin_table: int = 0


def _bare_session(runner, scratch, position: int) -> Qwen35GGUFResidentSession:
    session = object.__new__(Qwen35GGUFResidentSession)
    session.runner = runner
    session.scratch = scratch
    # The step falls back to a real HIP runtime when this is unset, which would
    # make a bookkeeping test depend on a device.
    session.runtime = SimpleNamespace()
    session._position = position
    return session


def _batch(*, live_count: int, ensure_calls: list, decode_calls: list):
    """Two draft rows whose only real behaviour is argument bookkeeping."""

    runner = SimpleNamespace(
        weights=SimpleNamespace(
            config=SimpleNamespace(layer_types=("full_attention",)),
        ),
        _run_full_attention_decode_batch_layer_rows=lambda *args, **kwargs: (
            decode_calls.append(kwargs)
        ),
        _packed_decode_metadata_kernel=lambda: None,
    )
    scratch = SimpleNamespace(cos_table=1, sin_table=2)
    position = live_count - 1
    first = _bare_session(runner, scratch, position)
    second = _bare_session(runner, scratch, position)
    first._bulk_prefill_scratch = SimpleNamespace()
    first._prefill_hidden_b = SimpleNamespace(ptr=7)
    first._ensure_packed_verify_workspace = lambda **kwargs: (
        SimpleNamespace(max_live_count=live_count),
        SimpleNamespace(
            for_packed_verify_layout=lambda *args, **kwargs: SimpleNamespace(
                rows=2,
                start=0,
                prefill_spans=SimpleNamespace(max_live_count=live_count),
            )
        ),
    )
    first._sync_packed_decode_initial_state = lambda *args, **kwargs: None
    first._scatter_packed_decode_state = lambda *args, **kwargs: None
    first._packed_full_attention_scratch_for_layer = lambda *args, **kwargs: (
        _LayerScratch()
    )
    first._ensure_packed_ar_attention_workspace = lambda **kwargs: (
        ensure_calls.append(kwargs)
        or _Workspace(kwargs["rows"], kwargs["max_context_len"])
    )
    return first, [first, second]


def _step(session, sessions, *, position: int) -> None:
    session.step_hidden_batch_native(
        3,
        sessions=sessions,
        positions=[position, position],
        output_hidden_ptr=0,
        logits_ptr=0,
        score_output=False,
        synchronize=False,
    )


@pytest.fixture(autouse=True)
def _layout_rebind(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runner_mod, "_rebind_packed_verify_layout_pages", lambda layout, state: layout
    )


def test_hidden_batch_step_allocates_the_split_workspace_past_the_threshold() -> None:
    ensure_calls: list[dict] = []
    decode_calls: list[dict] = []
    session, sessions = _batch(
        live_count=2048, ensure_calls=ensure_calls, decode_calls=decode_calls
    )

    _step(session, sessions, position=2047)

    assert [call["max_context_len"] for call in ensure_calls] == [2048], (
        "the draft step must size the workspace from its live span"
    )
    assert decode_calls[0]["split_workspace"] is not None


def test_hidden_batch_step_keeps_the_unsplit_route_below_the_threshold() -> None:
    ensure_calls: list[dict] = []
    decode_calls: list[dict] = []
    session, sessions = _batch(
        live_count=256, ensure_calls=ensure_calls, decode_calls=decode_calls
    )

    _step(session, sessions, position=255)

    assert ensure_calls == [], "a short live span must not allocate split buffers"
    assert decode_calls[0]["split_workspace"] is None


def test_hidden_batch_step_allocates_exactly_at_the_threshold() -> None:
    ensure_calls: list[dict] = []
    decode_calls: list[dict] = []
    session, sessions = _batch(
        live_count=SPLIT_THRESHOLD, ensure_calls=ensure_calls, decode_calls=decode_calls
    )

    _step(session, sessions, position=SPLIT_THRESHOLD - 1)

    assert [call["rows"] for call in ensure_calls] == [2]
    assert decode_calls[0]["split_workspace"] is not None


def test_packed_target_verifier_admits_a_live_context_past_the_split_threshold() -> None:
    """The verifier's own guard contradicted the workspace it allocates below it."""

    with open(runner_mod.__file__) as handle:
        text = handle.read()

    assert "packed target verifier currently requires context < 1024" not in text
