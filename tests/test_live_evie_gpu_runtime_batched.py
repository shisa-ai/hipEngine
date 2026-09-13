"""RED/gate tests for the EVIE batched query encode (segment isolation).

The isolation contract (review entry 648bca): replacing or resizing a
neighbor query must not change a query's own embeddings. Identical-query
repetition alone does not prove this (duplicated keys/values can mask a
missing segment mask), so the primary tests use *distinct* neighbors.

Numerical contract: isolation is exact — with identical batch shape the
unchanged query's rows are bit-equal; with a different batch shape
(different GEMM m, hence different rocBLAS tiling) a small arithmetic
delta is allowed and pinned by measurement. Repeatability of a fixed
batch is bit-exact. Cross-batch arithmetic parity is explicitly NOT
required.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "evie" / "evie_4p5b_doc_query.npz"
PINNED_MODEL_ID = "tencent/EVIE-4.5B"

if not FIXTURE.is_file():
    pytest.skip("EVIE oracle fixture not present", allow_module_level=True)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


if not _hip_available():
    pytest.skip("ROCm/HIP runtime not available", allow_module_level=True)


@pytest.fixture(scope="module")
def runner():
    from hipengine.loading.evie import load_evie_model
    from hipengine.loading.hf_cache import resolve_model_path

    try:
        snapshot = resolve_model_path(PINNED_MODEL_ID)
    except Exception:
        pytest.skip(f"{PINNED_MODEL_ID} not in local HF cache")
    if not snapshot.is_dir():
        pytest.skip(f"{PINNED_MODEL_ID} not in local HF cache")

    from hipengine.runtime.evie import EvieRunner

    loaded = load_evie_model(snapshot, runtime=None, precision="fp32")
    r = EvieRunner(loaded)
    yield r
    r.close()


@pytest.fixture(scope="module")
def fixture() -> dict[str, np.ndarray]:
    with np.load(FIXTURE) as data:
        return {k: data[k] for k in data.files}


def _queries(fixture: dict[str, np.ndarray]):
    """A (the oracle query) plus distinct same-length and ragged neighbors."""

    a_ids = fixture["query_input_ids"][0]
    a_mask = np.ones(len(a_ids), dtype=np.int64)
    # distinct, same length: shift the interior token ids
    b_ids = a_ids.copy()
    b_ids[3:-3] = (b_ids[3:-3] + 7) % 100_000
    c_ids = a_ids.copy()
    c_ids[3:-3] = (c_ids[3:-3] + 13) % 100_000
    return a_ids, a_mask, b_ids, c_ids


def _emb_len(runner, ids):
    return len(ids)


def test_batched_single_query_matches_sequential_bitexactly(runner, fixture) -> None:
    a_ids, a_mask, _, _ = _queries(fixture)
    out = runner.encode_queries([(a_ids, a_mask)])
    ref = runner.encode_query(a_ids, a_mask)
    assert len(out) == 1
    np.testing.assert_array_equal(out[0], ref)


def test_batched_identical_repeats_match_each_other(runner, fixture) -> None:
    a_ids, a_mask, _, _ = _queries(fixture)
    out = runner.encode_queries([(a_ids, a_mask)] * 3)
    for o in out[1:]:
        np.testing.assert_array_equal(out[0], o)
    ref = runner.encode_query(a_ids, a_mask)
    np.testing.assert_array_equal(out[0], ref)


def test_distinct_neighbor_same_shape_is_bit_exact(runner, fixture) -> None:
    """Primary isolation oracle: A|B vs A|C with len(B) == len(C), B != C."""

    a_ids, a_mask, b_ids, c_ids = _queries(fixture)
    m = np.ones(len(b_ids), dtype=np.int64)
    ab = runner.encode_queries([(a_ids, a_mask), (b_ids, m)])
    ac = runner.encode_queries([(a_ids, a_mask), (c_ids, m)])
    np.testing.assert_array_equal(ab[0], ac[0])
    # sanity: the distinct neighbors really differ
    assert not np.array_equal(ab[1], ac[1])


def test_distinct_neighbor_ragged_shape_is_isolated(runner, fixture) -> None:
    """A followed by a very different, shorter C: A stays within tolerance."""

    a_ids, a_mask, _, c_ids = _queries(fixture)
    short = c_ids[:-6]
    m_short = np.ones(len(short), dtype=np.int64)
    ab = runner.encode_queries([(a_ids, a_mask), (c_ids, np.ones(len(c_ids), dtype=np.int64))])
    a_short = runner.encode_queries([(a_ids, a_mask), (short, m_short)])
    ref = runner.encode_query(a_ids, a_mask)
    # same-shape batch is bit-exact vs sequential
    np.testing.assert_array_equal(ab[0], ref)
    # ragged batch differs only by GEMM m (tiling); pin a tight bound
    delta = np.abs(a_short[0] - ref)
    assert delta.max() <= 2e-5, delta.max()


def test_short_segments_match_sequential(runner, fixture) -> None:
    """Boundary-impulse isolation: segment lengths 1, 2, 3, 5 plus normal."""

    a_ids, a_mask, _, _ = _queries(fixture)
    base = a_ids[5:25]
    seqs = [(base[:1], None), (base[:2], None), (base[:3], None), (base[:5], None), (base, None)]
    out = runner.encode_queries(seqs)
    for (ids, _), emb in zip(seqs, out):
        assert emb.shape == (len(ids), 128)
        ref = runner.encode_query(ids, np.ones(len(ids), dtype=np.int64))
        np.testing.assert_allclose(emb, ref, atol=2e-5, rtol=0)


def test_batched_repeatability(runner, fixture) -> None:
    a_ids, a_mask, b_ids, _ = _queries(fixture)
    m = np.ones(len(b_ids), dtype=np.int64)
    o1 = runner.encode_queries([(a_ids, a_mask), (b_ids, m)])
    o2 = runner.encode_queries([(a_ids, a_mask), (b_ids, m)])
    np.testing.assert_array_equal(o1[0], o2[0])
    np.testing.assert_array_equal(o1[1], o2[1])
