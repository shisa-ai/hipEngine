"""Token-serial GGUF verification stays an independent state/scoring control."""
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core.memory import DeviceBuffer
from hipengine.runtime import qwen35_gguf_runner as runner


def _session(monkeypatch):
    calls = []
    session = runner.Qwen35GGUFResidentSession.__new__(runner.Qwen35GGUFResidentSession)
    session.runner = SimpleNamespace(weights=object(), hidden_size=3, vocab_size=100)
    session.scratch = SimpleNamespace(max_positions=64, hidden_seed_fp32=DeviceBuffer(0x2000, 12))
    session.runtime = SimpleNamespace(
        memcpy_async=lambda *args: calls.append(("copy", args)),
        device_synchronize=lambda: calls.append(("sync",)),
    )
    session._position = 10
    session.use_gemv_decode = True
    session._verify_hidden_seed_buf = DeviceBuffer(0x3000, 36)
    session._ensure_verify_block_buffers = lambda rows, **kwargs: calls.append(("buffers", rows))
    session._ensure_verify_linear_state_row_buffers = lambda rows, **kwargs: calls.append(("state_buffers", rows))
    session._normalize_layer_output_capture = lambda layers: ()
    session._record_current_linear_state_row = lambda row, **kwargs: calls.append(("state_row", row))

    def step(token, **kwargs):
        calls.append(("step", token, session._position, kwargs))
        session._position += 1
        return SimpleNamespace(token_id=token + 1)

    def hidden(token, **kwargs):
        calls.append(("hidden", token, kwargs))

    session.step = step
    session._run_token_to_final_hidden = hidden
    monkeypatch.setattr(runner, "host_array_ptr", lambda array: array)

    def readback(array, *args, **kwargs):
        array[:] = np.arange(array.size, dtype=np.float32).reshape(array.shape)
        calls.append(("readback", array.shape))

    monkeypatch.setattr(runner, "copy_device_to_host", readback)
    return session, calls


@pytest.mark.parametrize("state_only", [False, True])
@pytest.mark.parametrize("capture", [False, True])
def test_serial_verifier_preserves_score_or_echo_and_absolute_row_order(monkeypatch, state_only, capture):
    session, calls = _session(monkeypatch)
    result = session.verify_target_block_serial_exact(
        [4, 7, 9], advance_state_only=state_only, capture_linear_state_rows=capture,
    )
    assert result.input_token_ids == [4, 7, 9]
    assert result.token_ids == ([4, 7, 9] if state_only else [5, 8, 10])
    assert result.start_position == 10 and session.position == 13
    assert result.linear_state_rows_captured is capture
    assert result.final_linear_state_committed is True
    assert session._verify_hidden_seed_rows_populated == 3
    assert session._hidden_seed_fp32_populated is True
    np.testing.assert_array_equal(result.hidden_seeds, np.arange(9, dtype=np.float32).reshape(3, 3))
    assert [call for call in calls if call[0] == "state_row"] == (
        [("state_row", row) for row in range(3)] if capture else []
    )
    assert [(call[1][0], call[1][1], call[1][2]) for call in calls if call[0] == "copy"] == [
        (0x3000 + row * 12, 0x2000, 12) for row in range(3)
    ]
    if state_only:
        assert not any(call[0] == "step" for call in calls)
        assert [(call[1], call[2]["position"]) for call in calls if call[0] == "hidden"] == [
            (4, 10), (7, 11), (9, 12),
        ]
    else:
        assert not any(call[0] == "hidden" for call in calls)
        assert [(call[1], call[2]) for call in calls if call[0] == "step"] == [
            (4, 10), (7, 11), (9, 12),
        ]
    assert calls[-2:] == [("sync",), ("readback", (3, 3))]


@pytest.mark.parametrize("tokens,stream,position,error", [
    ([4], 1, 10, "default stream"),
    ([], 0, 10, "non-empty"),
    ([4, -1], 0, 10, "token_id -1"),
    ([4, 100], 0, 10, "token_id 100"),
    ([4, 7], 0, 63, "exceeds cache capacity"),
])
def test_serial_verifier_refuses_invalid_input_before_work_or_mutation(monkeypatch, tokens, stream, position, error):
    session, calls = _session(monkeypatch)
    session._position = position
    with pytest.raises(ValueError, match=error):
        session.verify_target_block_serial_exact(tokens, stream=stream, capture_linear_state_rows=True)
    assert calls == []
    assert session.position == position
