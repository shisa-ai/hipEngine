from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hipengine.loading.hf_cache import resolve_model_path

FIXTURE = Path(__file__).parent / "fixtures" / "evie" / "evie_4p5b_doc_query.npz"
PINNED_MODEL_ID = "tencent/EVIE-4.5B"

if not FIXTURE.is_file():
    pytest.skip("EVIE doc/query fixture not present", allow_module_level=True)


def _cached_snapshot() -> Path | None:
    try:
        path = resolve_model_path(PINNED_MODEL_ID)
    except Exception:
        return None
    return path if path.is_dir() else None


@pytest.fixture(scope="module")
def weights():
    snapshot = _cached_snapshot()
    if snapshot is None:
        pytest.skip(f"{PINNED_MODEL_ID} not in local HF cache")
    from hipengine.kernels.cpu_reference.evie import EvieWeights

    return EvieWeights.load(str(snapshot))


@pytest.fixture(scope="module")
def fixture() -> dict[str, np.ndarray]:
    with np.load(FIXTURE) as data:
        return {k: data[k] for k in data.files}


def test_vision_tower_matches_oracle(weights, fixture) -> None:
    from hipengine.kernels.cpu_reference.evie import EvieSpec, vision_forward

    spec = EvieSpec()
    got = vision_forward(weights, spec, fixture["pixel_values"][0], fixture["image_grid_thw"])
    ref = fixture["visual_features"]
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, atol=2e-4, rtol=0)


def test_query_hidden_matches_oracle(weights, fixture) -> None:
    from hipengine.kernels.cpu_reference.evie import EvieSpec, lm_rope_positions, text_forward

    spec = EvieSpec()
    ids = fixture["query_input_ids"]
    mask = fixture["query_attention_mask"]
    pos = lm_rope_positions(ids[0], mask[0], np.zeros((0, 3), dtype=int), spec)
    assert pos.shape == (3, ids.shape[1])
    got = text_forward(weights, spec, ids, pos)
    ref = fixture["query_hidden"]
    np.testing.assert_allclose(got, ref, atol=2e-4, rtol=0)


def test_doc_embeddings_and_maxsim_match_oracle(weights, fixture) -> None:
    from hipengine.kernels.cpu_reference.evie import (
        EvieSpec,
        lm_rope_positions,
        maxsim_scores,
        project_embeddings,
        text_forward,
        vision_forward,
    )

    spec = EvieSpec()
    vis = vision_forward(weights, spec, fixture["pixel_values"][0], fixture["image_grid_thw"])
    ids = fixture["input_ids"]
    mask = fixture["attention_mask"]
    pos = lm_rope_positions(ids[0], mask[0], fixture["image_grid_thw"], spec)
    hidden = text_forward(weights, spec, ids, pos, visual_features=vis)
    doc = project_embeddings(weights, hidden, mask, 128)
    np.testing.assert_allclose(doc, fixture["doc_embeddings_128"], atol=1e-4, rtol=0)

    qids = fixture["query_input_ids"]
    qmask = fixture["query_attention_mask"]
    qpos = lm_rope_positions(qids[0], qmask[0], np.zeros((0, 3), dtype=int), spec)
    q_hidden = text_forward(weights, spec, qids, qpos)
    query = project_embeddings(weights, q_hidden, qmask, 128)
    np.testing.assert_allclose(query, fixture["query_embeddings_128"], atol=1e-4, rtol=0)

    score = float(maxsim_scores(query[0], doc[0]))
    ref_score = float(fixture["maxsim_score"][0, 0])
    assert abs(score - ref_score) < 1e-4


def test_spec_layer_kinds() -> None:
    from hipengine.kernels.cpu_reference.evie import EvieSpec

    spec = EvieSpec()
    full = [i for i in range(spec.num_layers) if spec.is_full_attention(i)]
    assert full == [3, 7, 11, 15, 19, 23, 27, 31]
