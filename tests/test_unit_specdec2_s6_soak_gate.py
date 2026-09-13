from __future__ import annotations

from types import SimpleNamespace

from scripts.specdec2_s6_soak_gate import _outputs


class _Handle:
    def __init__(self, tokens):
        self.tokens = tokens

    def result(self, *, timeout):
        assert timeout == 180
        return SimpleNamespace(generated_token_ids=self.tokens)


def test_soak_gate_normalizes_all_child_output_ids() -> None:
    assert _outputs((_Handle((1, 2)), _Handle([3, 4]))) == (
        (1, 2),
        (3, 4),
    )
