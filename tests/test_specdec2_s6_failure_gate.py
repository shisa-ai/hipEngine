from __future__ import annotations

from types import SimpleNamespace

from hipengine.generation.qwen35_gguf_mtp2 import Qwen35GGUFMTP2Adapter
from scripts.specdec2_s6_failure_gate import (
    _failure_phase_specs,
    _outcomes,
    _recent_rows,
)


def test_failure_gate_selects_only_requested_completed_rows() -> None:
    snapshot = {
        "runner": {
            "routes": {
                "recent_completed": [
                    {"request_id": 1},
                    {"request_id": 2},
                    {"request_id": 3},
                ]
            }
        }
    }

    assert _recent_rows(snapshot, {1, 3}) == [
        {"request_id": 1},
        {"request_id": 3},
    ]


def test_failure_gate_captures_fatal_per_request_outcomes() -> None:
    class Handle:
        def __init__(self, result=None, error=None) -> None:
            self._result = result
            self._error = error

        def result(self, timeout: int):
            assert timeout == 180
            if self._error is not None:
                raise self._error
            return self._result

    outputs, errors = _outcomes(
        (
            Handle(result=SimpleNamespace(generated_token_ids=(1, 2))),
            Handle(error=RuntimeError("postcommit")),
        )
    )

    assert outputs == ((1, 2), None)
    assert errors == (None, "RuntimeError:postcommit")


def test_failure_gate_injects_the_current_bounded_accept_readback() -> None:
    phases = _failure_phase_specs()
    readback = next(row for row in phases if row[0] == "readback")

    assert readback[1] is Qwen35GGUFMTP2Adapter
    assert readback[2] == "_read_target_batch_accept"
    assert readback[4] is True
