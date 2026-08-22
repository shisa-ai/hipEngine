from __future__ import annotations

from scripts.gguf_mtp_rollout_gate import _ids


def test_rollout_gate_reads_authoritative_ids() -> None:
    body = {"choices": [{"hipengine": {"generated_token_ids": [7, 8]}}]}
    assert _ids(body) == [7, 8]
