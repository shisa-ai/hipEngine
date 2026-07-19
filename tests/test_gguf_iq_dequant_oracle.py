"""Validate hipEngine IQ3_XXS/IQ4_XS/Q3_K NumPy dequantizers vs llama.cpp.

The committed fixture ``tests/fixtures/gguf/q3km_iq_dequant_oracle.json``
contains real expert-tensor rows from ``Qwen3.6-35B-A3B-UD-Q3_K_M.gguf``
plus expected float32 values produced by llama.cpp ``dequantize_row_iq3_xxs``
/ ``dequantize_row_iq4_xs`` / ``dequantize_row_q3_K`` (via
``scripts/gguf_q3km_dequant_oracle_fixture.py``). Both implementations must
agree bit-exactly: the dequant math is integer table lookups scaled by exact
fp16->fp32 conversions, so any mismatch is a layout/porting bug.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from hipengine.quant.gguf import GGMLQuantizationType, dequantize_gguf_data, quant_layout

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "gguf" / "q3km_iq_dequant_oracle.json"


def _load() -> dict:
    with FIXTURE.open() as fh:
        return json.load(fh)


def test_oracle_fixture_covers_ud_q3_k_m_expert_types() -> None:
    fixture = _load()
    seen = {entry["ggml_type"] for entry in fixture["tensors"]}
    assert seen == {"IQ3_XXS", "IQ4_XS", "Q3_K"}


def test_real_rows_match_llamacpp_oracle_bit_exact() -> None:
    fixture = _load()
    for entry in fixture["tensors"]:
        qtype = GGMLQuantizationType[entry["ggml_type"]]
        layout = quant_layout(qtype)
        rows = len(entry["rows"])
        row_nbytes = len(base64.b64decode(entry["row_bytes_b64"])) // rows
        raw = np.frombuffer(base64.b64decode(entry["row_bytes_b64"]), dtype=np.uint8)
        raw = raw.reshape(rows, row_nbytes)
        expected = np.frombuffer(
            base64.b64decode(entry["expected_f32_b64"]), dtype=np.float32
        ).reshape(entry["expected_shape"])
        assert expected.shape == (rows, row_nbytes // layout.type_size * 256)

        out = dequantize_gguf_data(raw, qtype)

        assert out.shape == expected.shape, entry["name"]
        np.testing.assert_array_equal(out, expected, err_msg=f"{entry['name']} rows vs llama.cpp oracle")
