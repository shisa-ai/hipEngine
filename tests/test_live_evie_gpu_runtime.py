"""GPU parity test for the EVIE-4.5B torch-free HIP runtime.

Requires the cached ``tencent/EVIE-4.5B`` snapshot, a working ROCm stack,
and the oracle fixture from ``scripts/evie_oracle_torch.py``. Skipped
otherwise (no network, no GPU, or missing fixture).
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "evie" / "evie_4p5b_doc_query.npz"
PINNED_MODEL_ID = "tencent/EVIE-4.5B"

if not FIXTURE.is_file():
    pytest.skip("EVIE doc/query fixture not present", allow_module_level=True)


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
    from hipengine.loading.hf_cache import resolve_model_path

    try:
        snapshot = resolve_model_path(PINNED_MODEL_ID)
    except Exception:
        pytest.skip(f"{PINNED_MODEL_ID} not in local HF cache")
    if not snapshot.is_dir():
        pytest.skip(f"{PINNED_MODEL_ID} not in local HF cache")

    from hipengine.loading.evie import load_evie_model
    from hipengine.runtime.evie import EvieRunner

    loaded = load_evie_model(snapshot, runtime=None)
    r = EvieRunner(loaded)
    yield r
    r.close()


@pytest.fixture(scope="module")
def fixture() -> dict[str, np.ndarray]:
    with np.load(FIXTURE) as data:
        return {k: data[k] for k in data.files}


def test_query_embeddings_match_oracle(runner, fixture) -> None:
    got = runner.encode_query(
        fixture["query_input_ids"][0], fixture["query_attention_mask"][0]
    )
    ref = fixture["query_embeddings_128"][0]
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, atol=2e-4, rtol=0)


def test_visual_features_match_oracle(runner, fixture) -> None:
    spec = runner.spec
    n_img = int((fixture["input_ids"][0] == spec.image_token_id).sum())
    n = len(fixture["pixel_values"][0])
    scratch = runner._scratch_for(max(len(fixture["input_ids"][0]), n))
    visual = runner.vision_forward(
        fixture["pixel_values"][0], fixture["image_grid_thw"], scratch
    )
    got = runner._to_host(visual.ptr, n_img * 2560).reshape(n_img, 2560)
    ref = fixture["visual_features"]
    np.testing.assert_allclose(got, ref, atol=2e-4, rtol=0)


def test_doc_embeddings_and_maxsim_match_oracle(runner, fixture) -> None:
    from hipengine.runtime.evie import maxsim

    doc = runner.encode_document(
        fixture["input_ids"][0],
        fixture["attention_mask"][0],
        fixture["pixel_values"][0],
        fixture["image_grid_thw"],
    )
    ref = fixture["doc_embeddings_128"][0]
    np.testing.assert_allclose(doc, ref, atol=2e-4, rtol=0)

    query = runner.encode_query(
        fixture["query_input_ids"][0], fixture["query_attention_mask"][0]
    )
    score = maxsim(query, doc)
    assert abs(score - float(fixture["maxsim_score"][0, 0])) < 1e-4

    doc2 = runner.encode_document(
        fixture["input_ids"][0],
        fixture["attention_mask"][0],
        fixture["pixel_values"][0],
        fixture["image_grid_thw"],
    )
    np.testing.assert_array_equal(doc, doc2)


def test_state_rezero_clears_recycled_nan(runner, fixture) -> None:
    """The conv/GDN re-zero must clear a NaN, not multiply it by zero.

    Both state buffers are persistent and re-zeroed before every layer with a
    GPU-side kernel. A scale-by-zero kernel (``x * 0.0``) is a no-op for a NaN
    (``NaN * 0 == NaN``), so a NaN left in recycled device memory survives the
    re-zero and turns the whole encoder's output into NaN -- which is what the
    Surya runner, which shares this idiom, did after a long test suite had run.
    A fresh process only masked it because hipMalloc returns zeroed pages for
    allocations this size.
    """

    args = (
        fixture["input_ids"][0],
        fixture["attention_mask"][0],
        fixture["pixel_values"][0],
        fixture["image_grid_thw"],
    )
    ref = runner.encode_document(*args)
    assert np.isfinite(ref).all(), "unpoisoned encode is not finite"

    states = [
        buf
        for buf in (runner._zero_conv_state, runner._gdn_state_zero)
        if buf is not None
    ]
    assert len(states) == 2, "expected the conv and GDN zero-state buffers"
    for buf in states:
        # 0xFFFFFFFF is a quiet NaN in fp32
        runner.runtime.memset(buf.ptr, 0xFF, buf.nbytes)
    runner.runtime.device_synchronize()

    after = runner.encode_document(*args)
    assert np.isfinite(after).all(), (
        "state re-zeroing let NaN survive into the encoder: "
        f"{int((~np.isfinite(after)).sum())} of {after.size} values are not finite"
    )
    np.testing.assert_array_equal(after, ref)

