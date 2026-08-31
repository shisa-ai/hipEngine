from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "gguf_t16_prefill_row_sweep.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("gguf_t16_prefill_row_sweep", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_counterbalanced_schedule_reverses_every_arm_and_pair_bias() -> None:
    sweep = _load_module()

    schedule = sweep._counterbalanced_schedule(
        ("shared_b", "single_wave", "smallm"), order_pairs=2
    )

    assert schedule == [
        {
            "pass_index": 0,
            "pair_index": 0,
            "direction": "forward",
            "arms": ("shared_b", "single_wave", "smallm"),
        },
        {
            "pass_index": 1,
            "pair_index": 0,
            "direction": "reverse",
            "arms": ("smallm", "single_wave", "shared_b"),
        },
        {
            "pass_index": 2,
            "pair_index": 1,
            "direction": "reverse",
            "arms": ("smallm", "single_wave", "shared_b"),
        },
        {
            "pass_index": 3,
            "pair_index": 1,
            "direction": "forward",
            "arms": ("shared_b", "single_wave", "smallm"),
        },
    ]


def test_timed_returns_an_immutable_output_snapshot(monkeypatch) -> None:
    sweep = _load_module()

    class FakeBuffers:
        tiles_ptr = 22

        def __init__(self) -> None:
            self.output = np.zeros(4, dtype=np.uint16)

        def sync(self) -> None:
            return None

    buffers = FakeBuffers()

    def fake_read_out(_buffers, _out_buf, _rows, _out_features):
        return _buffers.output

    monkeypatch.setattr(sweep, "_read_out", fake_read_out)

    def first_arm(*_args):
        buffers.output[:] = 7

    def second_arm(*_args):
        buffers.output[:] = 19

    _, first_capture = sweep._timed(
        first_arm,
        buffers,
        x_ptr=11,
        out_ptr=33,
        rows=1,
        in_features=4,
        out_features=4,
        reps=2,
        warmup=1,
        capture=True,
        out_buf=object(),
    )
    _, second_capture = sweep._timed(
        second_arm,
        buffers,
        x_ptr=11,
        out_ptr=33,
        rows=1,
        in_features=4,
        out_features=4,
        reps=2,
        warmup=1,
        capture=True,
        out_buf=object(),
    )

    assert np.array_equal(first_capture, np.full(4, 7, dtype=np.uint16))
    assert np.array_equal(second_capture, np.full(4, 19, dtype=np.uint16))
