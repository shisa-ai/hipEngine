from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import numpy as np
import pytest

from hipengine.quant.iu4_ffn_pfs import (
    IU4FFNSpec,
    PFSFormatError,
    open_iu4_ffn_pfs,
    pfs_s4_to_n16_k32_tiles,
)
from hipengine.quant.iu4_s4 import unpack_s4
from hipengine.quant.registry import resolve_quant


def _pfs_weight_layout(values: np.ndarray) -> np.ndarray:
    q = np.asarray(values, dtype=np.int8)
    unsigned = (q.astype(np.int16) & 0xF).astype(np.uint8)
    return np.ascontiguousarray(
        unsigned[:, 0::2] | (unsigned[:, 1::2] << np.uint8(4))
    )


def _write_fixture(path: Path, *, corrupt_first_offset: bool = False) -> tuple[IU4FFNSpec, dict[str, np.ndarray]]:
    spec = IU4FFNSpec(layers=2, hidden=64, intermediate=128)
    rng = np.random.default_rng(0x1A4F)
    logical: dict[str, np.ndarray] = {}
    entries: list[tuple[int, int, int, int, int, int, bytes]] = []
    for layer in range(spec.layers):
        gate = rng.integers(-8, 8, size=(2 * spec.intermediate, spec.hidden), dtype=np.int8)
        down = rng.integers(-8, 8, size=(spec.hidden, spec.intermediate), dtype=np.int8)
        logical[f"gate.{layer}"] = gate
        logical[f"down.{layer}"] = down
        gate_scales = np.linspace(0.01, 0.02, 2 * spec.intermediate, dtype=np.float32)
        down_scales = np.linspace(0.02, 0.03, spec.hidden, dtype=np.float32)
        gate_sums = gate.sum(axis=1, dtype=np.int32)
        down_sums = down.sum(axis=1, dtype=np.int32)
        entries.extend(
            [
                (layer, 10, 4, 2, 2 * spec.intermediate, spec.hidden, _pfs_weight_layout(gate).tobytes()),
                (layer, 11, 2, 1, 2 * spec.intermediate, 1, gate_scales.tobytes()),
                (layer, 12, 3, 1, 2 * spec.intermediate, 1, gate_sums.tobytes()),
                (layer, 13, 4, 2, spec.hidden, spec.intermediate, _pfs_weight_layout(down).tobytes()),
                (layer, 14, 2, 1, spec.hidden, 1, down_scales.tobytes()),
                (layer, 15, 3, 1, spec.hidden, 1, down_sums.tobytes()),
            ]
        )
    entry_count = len(entries)
    data_offset = ((64 + entry_count * 64 + 4095) // 4096) * 4096
    next_offset = data_offset
    table = bytearray()
    payload = bytearray()
    for index, (layer, kind, dtype, rank, rows, cols, data) in enumerate(entries):
        offset = next_offset + (16 if corrupt_first_offset and index == 0 else 0)
        table += struct.pack(
            "<HHBBHIIQQ32s",
            layer,
            kind,
            dtype,
            rank,
            0,
            rows,
            cols,
            offset,
            len(data),
            b"\0" * 32,
        )
        payload += data
        next_offset += len(data)
    file_bytes = data_offset + len(payload)
    header = struct.pack(
        "<8sIIIIQQQQQ",
        b"PFSIU4F\0",
        1,
        64,
        64,
        entry_count,
        64,
        entry_count * 64,
        data_offset,
        file_bytes,
        0,
    )
    path.write_bytes(header + table + b"\0" * (data_offset - 64 - len(table)) + payload)
    return spec, logical


def _tiles_to_logical(tiles: np.ndarray, *, rows: int, cols: int) -> np.ndarray:
    words = np.ascontiguousarray(
        tiles.view(np.uint32).reshape(rows // 16, cols // 32, 16, 4)
        .transpose(0, 2, 1, 3)
        .reshape(rows, cols // 8)
    )
    return unpack_s4(words.view(np.uint8).reshape(rows, cols // 2))


def test_pfs_product_identity_is_explicit_t3_quant() -> None:
    quant = resolve_quant("iu4_s4_kairic_ffn_v1")

    assert quant.weight_storage == "pfsiu4f_s4_rowmajor_per_output_scale_sum"
    assert quant.activation_preprocess == "dynamic_u4_block_hadamard1024_per_row"


def test_pfs_parser_validates_and_converts_release_layout(tmp_path: Path) -> None:
    path = tmp_path / "tiny.pfs"
    spec, logical = _write_fixture(path)
    expected_sha = hashlib.sha256(path.read_bytes()).hexdigest()

    with open_iu4_ffn_pfs(path, spec=spec, expected_sha256=expected_sha) as sidecar:
        layer = sidecar.layer(1)
        gate_tiles = pfs_s4_to_n16_k32_tiles(layer.gate_weight)
        down_tiles = pfs_s4_to_n16_k32_tiles(layer.down_weight)

        assert np.array_equal(
            _tiles_to_logical(gate_tiles, rows=2 * spec.intermediate, cols=spec.hidden),
            logical["gate.1"],
        )
        assert np.array_equal(
            _tiles_to_logical(down_tiles, rows=spec.hidden, cols=spec.intermediate),
            logical["down.1"],
        )
        assert np.array_equal(layer.gate_sums, logical["gate.1"].sum(axis=1, dtype=np.int32))
        assert np.array_equal(layer.down_sums, logical["down.1"].sum(axis=1, dtype=np.int32))
        assert layer.gate_scales.shape == (2 * spec.intermediate,)
        assert layer.down_scales.shape == (spec.hidden,)


def test_pfs_parser_rejects_hash_and_noncontiguous_entries(tmp_path: Path) -> None:
    valid = tmp_path / "valid.pfs"
    spec, _ = _write_fixture(valid)
    with pytest.raises(PFSFormatError, match="SHA-256"):
        open_iu4_ffn_pfs(valid, spec=spec, expected_sha256="0" * 64)

    invalid = tmp_path / "invalid.pfs"
    _write_fixture(invalid, corrupt_first_offset=True)
    with pytest.raises(PFSFormatError, match="contiguous"):
        open_iu4_ffn_pfs(invalid, spec=spec)
