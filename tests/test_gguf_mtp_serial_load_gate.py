from __future__ import annotations

from scripts.gguf_mtp_serial_load_gate import _ids


def test_load_gate_extracts_authoritative_choice_ids() -> None:
    body = {
        "choices": [
            {"hipengine": {"generated_token_ids": [1, 2]}},
            {"hipengine": {"generated_token_ids": [3]}},
        ]
    }
    assert _ids(body) == [[1, 2], [3]]
