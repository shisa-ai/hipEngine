"""EVIE-8B parity tests (strict fp32 vs the torch oracle + fp16 tier).

8B is the single-head variant of the family: ``custom_text_proj`` is
Linear(4096, 4096) with full-dim (4096-d) multi-vector embeddings — no
Prefix-MRL d128 head. GDN (32/16 heads x 128, conv 4), attention (16 q
heads x 256, GQA 4, interval 4), and weight naming match the 4.5B
contracts.

Measured parity (fp32, vs the torch fp32 oracle on the fixture):
doc max|d| ~1.3e-4, query max|d| ~6e-7, visual features ~6e-3 — the
same gate class as 4.5B.

The fp16 tier is weaker than 4.5B's: the 8B vision tower amplifies fp16
activation-storage noise (doc cos mean ~0.99, min ~0.5 on a few tokens;
rocBLAS f16 GEMM results can also vary slightly across process loads
due to address-dependent split-k accumulation). fp32 is the
recommended precision for EVIE-8B; fp16 is gated at a coarser,
self-consistent floor.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "evie" / "evie_8b_doc_query.npz"
PINNED_MODEL_ID = "tencent/EVIE-8B"

if not FIXTURE.is_file():
    pytest.skip("EVIE-8B oracle fixture not present", allow_module_level=True)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


if not _hip_available():
    pytest.skip("ROCm/HIP runtime not available", allow_module_level=True)


def _runner(precision: str):
    from hipengine.loading.evie import load_evie_model
    from hipengine.loading.hf_cache import resolve_model_path
    from hipengine.runtime.evie import EvieRunner

    try:
        snapshot = resolve_model_path(PINNED_MODEL_ID)
    except Exception:
        pytest.skip(f"{PINNED_MODEL_ID} not in local HF cache")
    if not Path(snapshot).is_dir():
        pytest.skip(f"{PINNED_MODEL_ID} not in local HF cache")
    return EvieRunner(
        load_evie_model(snapshot, runtime=None, precision=precision),
        precision=precision,
    )


@pytest.fixture(scope="module")
def runner():
    r = _runner("fp32")
    yield r
    r.close()


@pytest.fixture(scope="module")
def fixture() -> dict[str, np.ndarray]:
    with np.load(FIXTURE) as data:
        return {k: data[k] for k in data.files}


def _cos_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.sum(a * b, axis=-1)


def test_8b_spec_geometry(runner) -> None:
    spec = runner.spec
    assert spec.hidden_size == 4096
    assert spec.num_layers == 32
    assert spec.vision_hidden_size == 1152
    assert spec.vision_depth == 27
    assert spec.vision_intermediate_size == 4304
    assert spec.vision_out_hidden_size == 4096
    assert spec.proj_dim == 4096
    assert spec.default_head == 4096
    assert spec.head_dims == (4096,)
    assert runner.VISION_HEAD_DIM == 72


def test_8b_query_embeddings_match_oracle(runner, fixture) -> None:
    got = runner.encode_query(
        fixture["query_input_ids"][0], fixture["query_attention_mask"][0]
    )
    ref = fixture["query_embeddings_128"][0]
    assert got.shape == (len(ref), 4096)
    np.testing.assert_allclose(got, ref, atol=2e-4, rtol=0)


def test_8b_doc_embeddings_match_oracle(runner, fixture) -> None:
    from hipengine.runtime.evie import maxsim

    doc = runner.encode_document(
        fixture["input_ids"][0],
        fixture["attention_mask"][0],
        fixture["pixel_values"][0],
        fixture["image_grid_thw"],
    )
    ref = fixture["doc_embeddings_128"][0]
    assert doc.shape == (len(ref), 4096)
    np.testing.assert_allclose(doc, ref, atol=2e-4, rtol=0)

    query = runner.encode_query(
        fixture["query_input_ids"][0], fixture["query_attention_mask"][0]
    )
    score = maxsim(query, doc)
    assert abs(score - float(fixture["maxsim_score"][0, 0])) < 1e-3

    doc2 = runner.encode_document(
        fixture["input_ids"][0],
        fixture["attention_mask"][0],
        fixture["pixel_values"][0],
        fixture["image_grid_thw"],
    )
    np.testing.assert_array_equal(doc, doc2)


def test_8b_fp16_doc_quality_tier() -> None:
    """fp16 tier: deterministic within a process, coarse agreement with fp32.

    The 8B vision tower amplifies fp16 storage noise (doc cos min can
    dip to ~0.5 on a few tokens — see the module docstring); the gate is
    the attainable floor, not 4.5B-class parity.
    """

    with np.load(FIXTURE) as fx:
        embs = {}
        for prec in ("fp32", "fp16"):
            r = _runner(prec)
            d1 = r.encode_document(
                fx["input_ids"][0],
                fx["attention_mask"][0],
                fx["pixel_values"][0],
                fx["image_grid_thw"],
            )
            d2 = r.encode_document(
                fx["input_ids"][0],
                fx["attention_mask"][0],
                fx["pixel_values"][0],
                fx["image_grid_thw"],
            )
            r.close()
            assert np.isfinite(d1).all()
            assert np.array_equal(d1, d2), "doc encode must be deterministic"
            embs[prec] = d1
        c = _cos_rows(embs["fp32"], embs["fp16"])
        assert c.mean() >= 0.98, f"fp16-vs-fp32 doc cos mean {c.mean()}"
