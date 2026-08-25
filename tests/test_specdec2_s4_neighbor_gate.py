from __future__ import annotations

from dataclasses import dataclass

from scripts.specdec2_s4_neighbor_gate import _controlled_token_rows


@dataclass(frozen=True)
class _Result:
    token_ids: list[int]


def test_controlled_neighbor_rows_force_full_accept_beside_root_reject() -> None:
    rows = _controlled_token_rows(
        (_Result([9, 8, 7]), _Result([6, 5, 4])),
        ((101, 102), (201, 202)),
        vocab_size=1000,
    )

    assert rows[0].token_ids == [101, 102, 7]
    assert rows[1].token_ids[0] == 202
    assert rows[1].token_ids[0] != 201
