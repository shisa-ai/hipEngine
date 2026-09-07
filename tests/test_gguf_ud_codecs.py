"""Exact CPU codec comparisons against independently compiled llama.cpp."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from hipengine.quant.gguf import GGMLQuantizationType, dequantize_gguf_data, quant_layout

FIXTURES = Path(__file__).parent / 'fixtures/gguf_ud'
FORMATS = ('IQ3_S', 'IQ2_S', 'IQ4_NL', 'IQ4_XS', 'IQ3_XXS', 'IQ2_XS', 'Q3_K')


def test_ud_fixture_provenance():
    manifest = json.loads((FIXTURES / 'manifest.json').read_text())
    assert manifest['commit'] == '17252c769a63c1cb650ce98ae309cf4de0da7778'
    assert manifest['license'] == 'MIT'
    assert hashlib.sha256((FIXTURES / 'synthetic.npz').read_bytes()).hexdigest() == manifest['fixture_sha256']
    assert 'ggml/src/ggml-quants.c' in manifest['source_sha256']
    assert set(manifest['formats']) == set(FORMATS)


@pytest.mark.parametrize('name', FORMATS)
def test_ud_codec_exact_f32_and_row_boundaries(name):
    qtype = GGMLQuantizationType[name]
    with np.load(FIXTURES / 'synthetic.npz') as fixture:
        raw, expected = fixture[name + '_raw'], fixture[name + '_f32']
    layout = quant_layout(qtype)
    assert raw.shape == (64, layout.type_size)
    for rows in (1, 2, 8, 64):
        actual = dequantize_gguf_data(raw.reshape(rows, -1), qtype)
        assert actual.shape == (rows, expected.size // rows)
        np.testing.assert_array_equal(actual.view(np.uint32), expected.reshape(rows, -1).view(np.uint32))


def test_ud_fixture_exercises_all_new_codebooks_signs_and_high_bits():
    with np.load(FIXTURES / 'synthetic.npz') as fixture:
        iq2, iq3 = fixture['IQ2_S_raw'], fixture['IQ3_S_raw']
    for raw, count, low_start, high_start, group, bits, signs in (
        (iq2, 1024, 2, 66, 4, 2, slice(34, 66)),
        (iq3, 512, 2, 66, 8, 1, slice(74, 106)),
    ):
        low = raw[:, low_start:low_start + 8 * group].astype(np.uint16).reshape(64, 8, group)
        high = (raw[:, high_start:high_start+8, None] >> (bits * np.arange(group))) & ((1 << bits) - 1)
        indices = low | (high << 8)
        assert set(indices.ravel()) == set(range(count))
        nonzero = np.ascontiguousarray(raw[:, :2]).view('<f2').ravel() != 0
        assert set(indices[nonzero].ravel()) == set(range(count))
        assert set(raw[:, signs].ravel()) == set(range(256))
        scales = raw[:, 74:82] if count == 1024 else raw[:, 106:110]
        assert set((scales & 15).ravel()) == set(range(16))
        assert set((scales >> 4).ravel()) == set(range(16))
