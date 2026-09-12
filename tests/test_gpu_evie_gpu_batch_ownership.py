"""Batched FP32/FP16 ownership gates, including distinct document pixels.

These enforce exact isolation at fixed shape, not cross-width byte equality
or a general retrieval-quality envelope. One model is resident at a time.
"""

import ctypes
from pathlib import Path

import numpy as np
import pytest

FIXTURE = Path(__file__).parent / "fixtures/evie/evie_4p5b_doc_query.npz"


@pytest.fixture(scope="module", params=["fp32", "fp16"])
def runner(request):
    try:
        hip = ctypes.CDLL("libamdhip64.so")
    except OSError:
        pytest.skip("ROCm/HIP unavailable")
    count = ctypes.c_int()
    hip.hipGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    hip.hipGetDeviceCount.restype = ctypes.c_int
    if hip.hipGetDeviceCount(ctypes.byref(count)) != 0 or count.value == 0:
        pytest.skip("No HIP device")
    from hipengine.loading.hf_cache import resolve_model_path
    from hipengine.loading.evie import load_evie_model
    from hipengine.runtime.evie import EvieRunner
    try:
        snapshot = resolve_model_path("tencent/EVIE-4.5B")
    except (FileNotFoundError, ValueError):
        pytest.skip("EVIE snapshot unavailable")
    if not snapshot.is_dir():
        pytest.skip("EVIE snapshot unavailable")
    loaded = load_evie_model(snapshot, runtime=None, precision=request.param)
    try:
        result = EvieRunner(loaded, precision=request.param)
    except BaseException:
        loaded.free()
        raise
    try:
        yield result
    finally:
        result.close()


@pytest.fixture(scope="module")
def oracle():
    if not FIXTURE.is_file():
        pytest.skip("EVIE fixture unavailable")
    with np.load(FIXTURE) as data:
        return {name: data[name] for name in data.files}


def _padded(ids):
    return (
        np.pad(ids, (2, 3)),
        np.pad(np.ones(len(ids), dtype=np.int64), (2, 3)),
    )


def test_ragged_padded_queries_isolate_both_slots_and_repeat(runner, oracle):
    a = oracle["query_input_ids"][0]
    b = a[:5].copy()
    c = (b + 17) % 100_000
    one = lambda ids: (ids, np.ones(len(ids), dtype=np.int64))
    ab = runner.encode_queries([one(a), one(b)])
    ac = runner.encode_queries([one(a), one(c)])
    np.testing.assert_array_equal(ab[0], ac[0])
    assert not np.array_equal(ab[1], ac[1])
    ba = runner.encode_queries([one(b), one(a)])
    ca = runner.encode_queries([one(c), one(a)])
    np.testing.assert_array_equal(ba[1], ca[1])
    padded = runner.encode_queries([_padded(a), _padded(b)])
    repeat = runner.encode_queries([one(a), one(b)])
    for expected, masked, again in zip(ab, padded, repeat):
        assert np.isfinite(expected).all()
        np.testing.assert_array_equal(expected, masked)
        np.testing.assert_array_equal(expected, again)


def test_distinct_document_pixels_isolate_both_slots_and_repeat(runner, oracle):
    ids = oracle["input_ids"][0]
    mask = oracle["attention_mask"][0]
    pixels = oracle["pixel_values"][0]
    grid = oracle["image_grid_thw"]
    a = (ids, mask, pixels, grid)
    b = (ids, mask, pixels * np.float32(0.5), grid)
    c = (ids, mask, -pixels * np.float32(0.5), grid)
    ab = runner.encode_documents([a, b])
    ac = runner.encode_documents([a, c])
    np.testing.assert_array_equal(ab[0], ac[0])
    assert not np.array_equal(ab[1], ac[1])
    ba = runner.encode_documents([b, a])
    ca = runner.encode_documents([c, a])
    np.testing.assert_array_equal(ba[1], ca[1])
    padded_ids, padded_mask = _padded(ids)
    padded = runner.encode_documents([
        (padded_ids, padded_mask, pixels, grid),
        (padded_ids, padded_mask, b[2], grid),
    ])
    repeat = runner.encode_documents([a, b])
    for expected, masked, again in zip(ab, padded, repeat):
        assert np.isfinite(expected).all()
        np.testing.assert_array_equal(expected, masked)
        np.testing.assert_array_equal(expected, again)


def test_ragged_documents_preserve_short_pages_visual_ownership(runner, oracle):
    ids = oracle["input_ids"][0]
    mask = oracle["attention_mask"][0]
    pixels = oracle["pixel_values"][0]
    grid = oracle["image_grid_thw"]
    image_id = runner.spec.image_token_id
    image_rows = np.flatnonzero(ids == image_id)
    # A valid 4x4 patch grid produces four merged image tokens. Keep the
    # document template and use distinct pixels, so row-count/offset bugs
    # cannot hide behind equal-size pages or identical images.
    short_ids = np.concatenate([
        ids[:image_rows[0]], np.full(4, image_id, dtype=np.int64),
        ids[image_rows[-1] + 1:],
    ])
    short_pixels = pixels[:16].copy() * np.float32(0.25)
    small_grid = np.array([[1, 4, 4]], dtype=np.int64)
    small = (short_ids, np.ones(len(short_ids), dtype=np.int64), short_pixels, small_grid)
    a = (ids, mask, pixels, grid)
    c = (ids, mask, -pixels, grid)
    ab = runner.encode_documents([a, small])
    cb = runner.encode_documents([c, small])
    assert ab[1].shape == (len(short_ids), 128)
    assert np.isfinite(ab[1]).all()
    np.testing.assert_array_equal(ab[1], cb[1])
    assert not np.array_equal(ab[0], cb[0])
    ba = runner.encode_documents([small, a])
    bc = runner.encode_documents([small, c])
    np.testing.assert_array_equal(ba[0], bc[0])
