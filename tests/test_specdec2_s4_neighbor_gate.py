from __future__ import annotations

from dataclasses import dataclass
import ctypes

import numpy as np

from scripts.specdec2_s4_neighbor_gate import (
    _controlled_device_token_rows,
    _controlled_token_rows,
)


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


@dataclass(frozen=True)
class _Tensor:
    values: np.ndarray

    @property
    def ptr(self) -> int:
        return int(self.values.ctypes.data)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape


@dataclass(frozen=True)
class _DeviceResult:
    target_top1: _Tensor


def test_controlled_neighbor_rows_rewrite_device_target_ids_in_place() -> None:
    target_rows = (
        np.array([9, 8, 7], dtype=np.int32),
        np.array([6, 5, 4], dtype=np.int32),
    )
    results = tuple(_DeviceResult(_Tensor(row)) for row in target_rows)

    def copy_to_host(destination: int, source: object, nbytes: int, **_: object) -> None:
        ctypes.memmove(int(destination), int(source.ptr), int(nbytes))

    def copy_to_device(destination: object, source: int, nbytes: int, **_: object) -> None:
        ctypes.memmove(int(destination.ptr), int(source), int(nbytes))

    output = _controlled_device_token_rows(
        results,
        ((101, 102), (201, 202)),
        runtime=object(),
        vocab_size=1000,
        copy_to_host=copy_to_host,
        copy_to_device=copy_to_device,
    )

    assert output == list(results)
    assert target_rows[0].tolist() == [101, 102, 7]
    assert target_rows[1][0] == 202
    assert target_rows[1][0] != 201
