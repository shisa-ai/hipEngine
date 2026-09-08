"""Row-slab parity for the strict dense IQ kernel.

`R` (rows per block) changes only which prompt rows share a workgroup. The
per-(row, column) k ownership, FMA order, wave32 shuffle tree and serial
wave-0..3 sum are untouched, so every slab must be bit-identical to R=1 —
including on ragged tails, where the trailing rows of the last block are
padding that must never be stored.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc,
)
from tests._ud_hip import ud_hip_backend

FIXTURE = Path(__file__).parent / 'fixtures/gguf_ud'
ENTRIES = json.loads((FIXTURE / 'real_rows.json').read_text())['entries']
QUANTS = sorted({entry['type'] for entry in ENTRIES})
# 3/5/7/9/15/17/33 are ragged against at least one supported slab.
ROWS = (1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 32, 33)


def bf16(x):
    bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


@pytest.fixture(scope='module')
def library(ud_hip_backend):
    from hipengine.kernels.hip_gfx1100.quant.gguf_iq_dense import build_gguf_iq_dense
    version_file = os.environ.get('HIPENGINE_COMPILER_VERSION_FILE')
    return build_gguf_iq_dense(
        compiler_version=Path(version_file).read_text() if version_file else None,
        require_cached=os.environ.get('HIPENGINE_REQUIRE_CACHED_BUILD') == '1')


def _project(library, quant, raw, x, rows, k, n, output, row_batch):
    """Run one projection into a canaried buffer and return the live rows."""
    from hipengine.kernels.hip_gfx1100.quant.gguf_iq_dense import launch
    dtype = np.float32 if output == 'f32' else np.uint16
    host = np.full((rows + 2, n), 123, dtype=dtype)
    buffers = []
    try:
        for array in (x, raw, host):
            buf = malloc(array.nbytes)
            buffers.append(buf)
            copy_host_to_device(buf, host_array_ptr(array), array.nbytes)
        launch(buffers[0].ptr, buffers[1].ptr, buffers[2].ptr + host.strides[0],
               rows, k, n, quant='gguf_' + quant.lower(), output=output,
               library=library, row_batch=row_batch)
        copy_device_to_host(host_array_ptr(host), buffers[2], host.nbytes)
    finally:
        for buf in reversed(buffers):
            free(buf)
    # Padding rows in the final slab must not write past the live rows.
    assert np.all(host[[0, -1]] == 123), 'row-slab padding wrote outside the output'
    return host[1:-1]


@pytest.mark.parametrize('quant', QUANTS)
@pytest.mark.parametrize('rows', ROWS)
@pytest.mark.parametrize('output', ('f32', 'bf16'))
def test_row_slabs_are_bit_identical_to_single_row_blocks(
        library, ud_hip_backend, quant, rows, output):
    from hipengine.kernels.hip_gfx1100.quant.gguf_iq_dense import ROW_BATCHES
    entry = next(e for e in ENTRIES if e['type'] == quant)
    with np.load(FIXTURE / 'real_rows.npz') as fixture:
        raw = np.ascontiguousarray(fixture[entry['key'] + '_raw'][[0, 1, 0]])
        k = fixture[entry['key'] + '_f32'].shape[1]
    x = bf16(np.random.default_rng(53).normal(0, 0.125, (rows, k)))
    baseline = _project(library, quant, raw, x, rows, k, 3, output, 1)
    bits = np.uint32 if output == 'f32' else np.uint16
    for row_batch in ROW_BATCHES[1:]:
        actual = _project(library, quant, raw, x, rows, k, 3, output, row_batch)
        np.testing.assert_array_equal(
            actual.view(bits), baseline.view(bits),
            err_msg=f'{quant} rows={rows} R={row_batch} diverged from R=1')


@pytest.mark.parametrize('row_batch', (0, 3, 5, 16, -1))
def test_launch_rejects_unsupported_row_batches(row_batch, monkeypatch):
    from hipengine.kernels.hip_gfx1100.quant import gguf_iq_dense as dense
    monkeypatch.setattr(dense, '_default_library',
                        lambda: pytest.fail('built before validation'))
    with pytest.raises(ValueError):
        dense.launch(1, 2, 3, rows=8, in_features=256, out_features=3,
                     quant='gguf_iq3_s', output='f32', row_batch=row_batch)


@pytest.mark.parametrize('rows,expected', [(1, 1), (2, 2), (3, 2), (4, 4), (7, 4),
                                           (8, 8), (9, 8), (512, 8)])
def test_default_slab_is_the_largest_one_the_prompt_fills(rows, expected):
    from hipengine.kernels.hip_gfx1100.quant.gguf_iq_dense import _row_batch
    assert _row_batch(rows) == expected
