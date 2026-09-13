from __future__ import annotations

import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hipengine.loading import GGUFReader, MissingGGUFTensorError, scan_gguf
from hipengine.quant.gguf import GGMLQuantizationType, GGUFValueType


def _gguf_string(value: str) -> bytes:
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def _gguf_scalar(value_type: GGUFValueType, value: Any) -> bytes:
    fmt = {
        GGUFValueType.UINT8: "B",
        GGUFValueType.INT8: "b",
        GGUFValueType.UINT16: "H",
        GGUFValueType.INT16: "h",
        GGUFValueType.UINT32: "I",
        GGUFValueType.INT32: "i",
        GGUFValueType.FLOAT32: "f",
        GGUFValueType.BOOL: "?",
        GGUFValueType.UINT64: "Q",
        GGUFValueType.INT64: "q",
        GGUFValueType.FLOAT64: "d",
    }[value_type]
    return struct.pack(f"<{fmt}", value)


def _gguf_value(value_type: GGUFValueType, value: Any) -> bytes:
    if value_type == GGUFValueType.STRING:
        return _gguf_string(str(value))
    if value_type == GGUFValueType.ARRAY:
        item_type, items = value
        out = bytearray(struct.pack("<IQ", int(item_type), len(items)))
        for item in items:
            out += _gguf_value(item_type, item)
        return bytes(out)
    return _gguf_scalar(value_type, value)


def _align_up(value: int, alignment: int) -> int:
    remainder = value % alignment
    return value if remainder == 0 else value + alignment - remainder


def _write_tiny_gguf(path: Path) -> None:
    alignment = 32
    dense = np.arange(6, dtype=np.float32).reshape(2, 3)
    bf16 = np.asarray([0x3F80, 0xC020], dtype=np.uint16)
    q8 = np.concatenate(
        [
            np.asarray([2.0], dtype=np.float16).view(np.uint8),
            np.arange(-16, 16, dtype=np.int8).view(np.uint8),
        ]
    )
    tensors = [
        ("dense.weight", (2, 3), GGMLQuantizationType.F32, dense.tobytes()),
        ("bf16.weight", (2,), GGMLQuantizationType.BF16, bf16.tobytes()),
        ("q8.weight", (1, 32), GGMLQuantizationType.Q8_0, q8.tobytes()),
    ]

    tensor_blob = bytearray()
    tensor_records = []
    for name, shape, qtype, payload in tensors:
        offset = _align_up(len(tensor_blob), alignment)
        tensor_blob += b"\x00" * (offset - len(tensor_blob))
        tensor_records.append((name, shape, qtype, offset))
        tensor_blob += payload

    metadata = [
        ("general.architecture", GGUFValueType.STRING, "toy"),
        ("general.alignment", GGUFValueType.UINT32, alignment),
        ("general.file_type", GGUFValueType.UINT32, 7),
        ("toy.context_length", GGUFValueType.UINT32, 16),
        ("tokenizer.ggml.tokens", GGUFValueType.ARRAY, (GGUFValueType.STRING, ["a", "b"])),
        ("tokenizer.ggml.scores", GGUFValueType.ARRAY, (GGUFValueType.FLOAT32, [0.25, -0.5])),
    ]

    header = bytearray()
    header += b"GGUF"
    header += struct.pack("<IQQ", 3, len(tensors), len(metadata))
    for key, value_type, value in metadata:
        header += _gguf_string(key)
        header += struct.pack("<I", int(value_type))
        header += _gguf_value(value_type, value)
    for name, shape, qtype, offset in tensor_records:
        ggml_shape = tuple(reversed(shape))
        header += _gguf_string(name)
        header += struct.pack("<I", len(ggml_shape))
        header += struct.pack(f"<{len(ggml_shape)}Q", *ggml_shape)
        header += struct.pack("<IQ", int(qtype), offset)

    data_start = _align_up(len(header), alignment)
    header += b"\x00" * (data_start - len(header))
    path.write_bytes(bytes(header) + bytes(tensor_blob))


def test_scan_gguf_parses_metadata_tensor_table_and_lazy_data(tmp_path: Path) -> None:
    path = tmp_path / "tiny.gguf"
    _write_tiny_gguf(path)

    info = scan_gguf(path)

    assert info.path == path.resolve()
    assert info.version == 3
    assert info.alignment == 32
    assert info.architecture == "toy"
    assert info.file_type == 7
    assert info.file_type_name == "MOSTLY_Q8_0"
    assert info.metadata["tokenizer.ggml.tokens"] == ["a", "b"]
    assert info.metadata["tokenizer.ggml.scores"] == pytest.approx([0.25, -0.5])
    assert info.tensor_count == 3
    assert info.names_with_prefix("dense") == ("dense.weight",)

    dense_info, bf16_info, q8_info = info.require(["dense.weight", "bf16.weight", "q8.weight"])
    assert dense_info.shape == (2, 3)
    assert dense_info.ggml_shape == (3, 2)
    assert dense_info.ggml_type_name == "F32"
    assert dense_info.nbytes == 24
    assert bf16_info.byte_shape == (2,)
    assert q8_info.byte_shape == (1, 34)
    assert q8_info.nbytes == 34

    reader = GGUFReader(path)
    np.testing.assert_allclose(
        reader.tensor_data("dense.weight"), np.arange(6, dtype=np.float32).reshape(2, 3)
    )
    np.testing.assert_allclose(
        reader.dequantize_tensor("bf16.weight"), np.asarray([1.0, -2.5], dtype=np.float32)
    )
    np.testing.assert_allclose(
        reader.dequantize_tensor("q8.weight")[0], np.arange(-16, 16, dtype=np.float32) * 2.0
    )


def test_scan_gguf_reports_missing_tensor_cleanly(tmp_path: Path) -> None:
    path = tmp_path / "tiny.gguf"
    _write_tiny_gguf(path)
    info = scan_gguf(path)

    with pytest.raises(MissingGGUFTensorError, match="missing required GGUF tensors: absent"):
        info.require(["dense.weight", "absent"])


def test_gguf_loading_helpers_do_not_import_torch(tmp_path: Path) -> None:
    had_torch = "torch" in sys.modules
    path = tmp_path / "tiny.gguf"
    _write_tiny_gguf(path)

    scan_gguf(path)

    if not had_torch:
        assert "torch" not in sys.modules
