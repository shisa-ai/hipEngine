"""CPU tests for which blocks the Inkling 2 session sends through the resident traversal (no GPU)."""

from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.runtime.inkling2_mini_reference import Inkling2TextSession


def _session(**overrides):
    """A session stand-in whose traversal, resident block and head only record what they were asked."""

    calls = {"resident": [], "traverse": 0}

    def resident(session, rows):
        calls["resident"].append(np.array(rows, dtype=np.float32).copy())
        # One hidden row per input row plus the residual-stack block of each visit.
        return rows + 100.0, [rows + 200.0 + index for index in range(3)]

    def traverse(h, capture):
        calls["traverse"] += 1
        return h - 1.0, [h + index for index in range(3)], None
        yield  # pragma: no cover - a generator that stops at once

    model = SimpleNamespace(
        resident_block=resident,
        resident_decode=False,
        project_logits=lambda hidden: hidden,
        cap_logits=lambda logits: logits,
        plan=SimpleNamespace(visits=(0, 1, 2)),
    )
    session = SimpleNamespace(model=model, _undo=None, position=7, _traverse=traverse,
                              _output_hidden=lambda h, blocks, **options: h + sum(blocks))
    for name, value in overrides.items():
        if name.startswith("model_"):
            setattr(model, name[len("model_"):], value)
        else:
            setattr(session, name, value)
    return session, calls


H = np.arange(2 * 3, dtype=np.float32).reshape(2, 3)
LAST = H[-1] + 100.0 + sum(H[-1] + 200.0 + index for index in range(3))


def test_committed_prefill_blocks_take_the_resident_traversal_and_project_the_last_row():
    session, calls = _session()
    logits = Inkling2TextSession._forward(session, H, 2, logits=True, capture=None, last_only=True)
    assert len(calls["resident"]) == 1 and np.array_equal(calls["resident"][0], H)
    assert calls["traverse"] == 0 and session.position == 9
    assert logits.shape == (3,) and np.array_equal(logits, LAST)


def test_earlier_prompt_blocks_run_the_traversal_without_projecting_a_row():
    session, calls = _session()
    assert Inkling2TextSession._forward(session, H, 2, logits=False, capture=None, last_only=True) is None
    assert len(calls["resident"]) == 1 and calls["traverse"] == 0 and session.position == 9


@pytest.mark.parametrize(
    "overrides, width, logits, last_only",
    [
        ({}, 1, True, True),  # a single token is the decode path, gated by resident_decode
        ({}, 2, True, False),  # step_block projects every row for verification
        ({"capture": lambda boundary, h: None}, 2, True, True),  # tap capture needs each visit on the host
        ({"_undo": object()}, 2, True, True),  # a provisional block must be able to roll back
        ({"model_resident_block": None}, 2, True, True),  # strict has no resident traversal
    ],
)
def test_nothing_else_takes_the_block_traversal(overrides, width, logits, last_only):
    session, calls = _session(**overrides)
    Inkling2TextSession._forward(session, H, width, logits=logits, capture=overrides.get("capture"),
                                 last_only=last_only)
    assert calls["resident"] == [] and calls["traverse"] == 1 and session.position == 7 + width


def test_scalar_decode_still_takes_its_own_resident_route():
    session, calls = _session(model_resident_decode=True)
    logits = Inkling2TextSession._forward(session, H[0], 1, logits=True, capture=None, last_only=False)
    assert len(calls["resident"]) == 1 and calls["traverse"] == 0 and session.position == 8
    assert logits.shape == (3,) and np.array_equal(logits, H[0] + 100.0 + sum(H[0] + 200.0 + index for index in range(3)))
