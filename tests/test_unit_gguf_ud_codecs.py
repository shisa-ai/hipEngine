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
    actual = dequantize_gguf_data(raw.reshape(2, 4, -1), qtype)
    assert actual.shape == (2, 4, expected.size // 8)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.reshape(actual.shape).view(np.uint32))


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


@pytest.mark.parametrize('name', FORMATS)
def test_ud_storage_sizes_and_super_scale_corners(name):
    sizes = {'IQ3_S': 110, 'IQ2_S': 82, 'IQ4_NL': 18, 'IQ4_XS': 136,
             'IQ3_XXS': 98, 'IQ2_XS': 74, 'Q3_K': 110}
    layout = quant_layout(GGMLQuantizationType[name])
    assert layout.type_size == sizes[name]
    assert layout.block_size == (32 if name == 'IQ4_NL' else 256)
    with np.load(FIXTURES / 'synthetic.npz') as fixture:
        raw = fixture[name + '_raw']
    offset = 108 if name == 'Q3_K' else 0
    scales = np.ascontiguousarray(raw[:, offset:offset + 2]).view('<u2')
    assert set(scales.ravel()) == {0, 0x8000, 1, 0x0400, 0x3c00, 0xbc00, 0x7bff, 0x3555}


@pytest.mark.parametrize('name', ('IQ2_XS', 'IQ3_XXS', 'IQ4_XS', 'IQ4_NL', 'Q3_K'))
def test_ud_existing_packed_selector_coverage_at_nonzero_scale(name):
    with np.load(FIXTURES / 'synthetic.npz') as fixture:
        raw = fixture[name + '_raw']
    offset = 108 if name == 'Q3_K' else 0
    nonzero = np.ascontiguousarray(raw[:, offset:offset + 2]).view('<f2').ravel() != 0
    raw = raw[nonzero]
    if name == 'IQ2_XS':
        words = np.ascontiguousarray(raw[:, 2:66]).view('<u2')
        assert set((words & 511).ravel()) == set(range(512))
        assert set((words >> 9).ravel()) == set(range(128))
        scales = raw[:, 66:74]
        assert set((scales & 15).ravel()) | set((scales >> 4).ravel()) == set(range(16))
    elif name == 'IQ3_XXS':
        assert set(raw[:, 2:66].ravel()) == set(range(256))
        aux = np.ascontiguousarray(raw[:, 66:98]).view('<u4')
        for shift in (0, 7, 14, 21):
            assert set(((aux >> shift) & 127).ravel()) == set(range(128))
        assert set((aux >> 28).ravel()) == set(range(16))
    elif name in ('IQ4_XS', 'IQ4_NL'):
        start = 8 if name == 'IQ4_XS' else 2
        assert set((raw[:, start:] & 15).ravel()) == set(range(16))
        assert set((raw[:, start:] >> 4).ravel()) == set(range(16))
        if name == 'IQ4_XS':
            high = raw[:, 2].astype(np.uint16) | (raw[:, 3].astype(np.uint16) << 8)
            scales = []
            for j in range(8):
                low = (raw[:, 4 + j // 2] >> (4 * (j % 2))) & 15
                scales.extend((low | (((high >> (2 * j)) & 3) << 4)).tolist())
            assert set(scales) == set(range(64))
    else:
        # Each Q3 subblock uses two low quant bits and one independent hmask bit.
        for bit in range(8):
            assert set(((raw[:, :32] >> bit) & 1).ravel()) == {0, 1}
        for shift in (0, 2, 4, 6):
            assert set(((raw[:, 32:96] >> shift) & 3).ravel()) == set(range(4))
        scales = []
        for j in range(16):
            low = (raw[:, 96 + j % 8] >> (4 * (j // 8))) & 15
            high = (raw[:, 104 + j % 4] >> (2 * (j // 4))) & 3
            scales.extend((low | (high << 4)).tolist())
        assert set(scales) == set(range(64))


REAL = json.loads((FIXTURES / 'real_rows.json').read_text())


def test_real_fixture_provenance_and_coverage():
    assert REAL['commit'] == '17252c769a63c1cb650ce98ae309cf4de0da7778'
    assert hashlib.sha256((FIXTURES / 'real_rows.npz').read_bytes()).hexdigest() == REAL['fixture_sha256']
    assert {e['type'] for e in REAL['entries']} == set(FORMATS)
    assert {e['shape'][-1] for e in REAL['entries']} == {5120, 17408}
    pins = json.loads((FIXTURES.parents[2] / 'docs/UD-QUANTS-U0-IDENTITY.json').read_text())['files']
    for entry in REAL['entries']:
        pin = next(p for p in pins if p['file'] == entry['model'])
        assert entry['published_sha256'] == pin['published_sha256']
        assert entry['rows'] == [0, entry['shape'][0] - 1]


@pytest.mark.parametrize('entry', REAL['entries'], ids=lambda e: e['model'] + ':' + e['tensor'])
def test_ud_real_rows_exact_f32(entry):
    with np.load(FIXTURES / 'real_rows.npz') as fixture:
        raw = fixture[entry['key'] + '_raw']
        expected = fixture[entry['key'] + '_f32']
    actual = dequantize_gguf_data(raw, GGMLQuantizationType[entry['type']])
    assert actual.shape == (2, entry['shape'][-1])
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
