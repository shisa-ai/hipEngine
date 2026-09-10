"""Production fp16-path parity tests for the EVIE GPU runtime.

Kept in a separate module from ``test_evie_gpu_runtime`` (strict fp32) so the
two full-weight-set runners are never resident at the same time: pytest tears
down the fp32 module's module-scoped fixture before this module's fixture
creates the fp16 runner. Co-resident multi-model runs on this GTT/iGPU host
have caused GPU stalls and a host panic (see worklog 2026-09-09 entries).

The gate follows docs/MODEL-EVIE.md's production envelope: per-token embedding
cosine vs the fp32 oracle and MaxSim delta, plus bit-repeatability of repeated
encodes.
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

    loaded = load_evie_model(snapshot, runtime=None, precision="fp16")
    from hipengine.runtime.evie import EvieRunner

    r = EvieRunner(loaded, precision="fp16")
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


def test_fp16_query_and_doc_embeddings_match_oracle(runner, fixture) -> None:
    qry = runner.encode_query(
        fixture["query_input_ids"][0], fixture["query_attention_mask"][0]
    )
    doc = runner.encode_document(
        fixture["input_ids"][0],
        fixture["attention_mask"][0],
        fixture["pixel_values"][0],
        fixture["image_grid_thw"],
    )

    cq = _cos_rows(qry, fixture["query_embeddings_128"][0])
    cd = _cos_rows(doc, fixture["doc_embeddings_128"][0])
    # Manifest-pinned production envelope (docs/EXECUTION-PROFILES.md
    # sec 5 / 6.4): the fp16 GEMM path is class T1 (local implementation
    # drift); the recorded manifest measured over 3 bit-identical runs is
    # q mean/min 0.9999986/0.9999884, d mean/min 0.9999500/0.9985462.
    # These bounds are the recorded values less a tiny cross-build slack,
    # not independent quality thresholds; a future path that shifts
    # parity must re-run the full profile adjudication.
    assert cq.mean() >= 0.9999985, cq.mean()
    assert cd.mean() >= 0.999949, cd.mean()
    assert cq.min() >= 0.999988, cq.min()
    assert cd.min() >= 0.99854, cd.min()


def test_fp16_maxsim_within_half_percent(runner, fixture) -> None:
    from hipengine.runtime.evie import maxsim

    qry = runner.encode_query(
        fixture["query_input_ids"][0], fixture["query_attention_mask"][0]
    )
    doc = runner.encode_document(
        fixture["input_ids"][0],
        fixture["attention_mask"][0],
        fixture["pixel_values"][0],
        fixture["image_grid_thw"],
    )
    # manifest-pinned: measured 5.91077 vs oracle 5.91139 (delta 6.2e-4)
    score = maxsim(qry, doc)
    ref = float(fixture["maxsim_score"][0, 0])
    assert abs(score - ref) <= 8e-4, (score, ref)


def test_fp16_doc_encode_bit_repeatability(runner, fixture) -> None:
    ids, mask = fixture["input_ids"][0], fixture["attention_mask"][0]
    pv, grid = fixture["pixel_values"][0], fixture["image_grid_thw"]
    # docs/EXECUTION-PROFILES.md sec 6.4: repeat identically for at
    # least three fixed-seed runs
    runs = [runner.encode_document(ids, mask, pv, grid) for _ in range(3)]
    for d in runs[1:]:
        np.testing.assert_array_equal(runs[0], d)


def test_fp16_query_encode_bit_repeatability(runner, fixture) -> None:
    runs = [
        runner.encode_query(
            fixture["query_input_ids"][0], fixture["query_attention_mask"][0]
        )
        for _ in range(3)
    ]
    for q in runs[1:]:
        np.testing.assert_array_equal(runs[0], q)
