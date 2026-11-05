"""Live gate: Muse Glimmer CPU reference vs the llama.cpp per-node oracle.

The oracle fixture was captured from llama-debug build 12005 (e0f142d5d) on
the same Q4_K_M artifact; see fixtures/muse_glimmer/llamacpp_debug_e0f142d5d.json
for the exact commands. Sampled hidden-state values are printed to 4 decimals
by llama-debug; fp32 accumulation-order drift grows with depth, so tolerances
are per-node and declared here before measurement.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hipengine.kernels.cpu_reference.muse_glimmer import (
    MuseGlimmerConfig,
    forward,
    gguf_weight_source,
    rmsnorm,
)
from hipengine.loading.gguf import GGUFReader

MODEL = Path(
    "/models/gguf/Muse-Glimmer-30B-GGUF/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf"
)
FIXTURE = (
    Path(__file__).parent.parent
    / "fixtures"
    / "muse_glimmer"
    / "llamacpp_debug_e0f142d5d.json"
)


@pytest.fixture(scope="module")
def oracle() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def reference_run(oracle: dict):
    if not MODEL.exists():
        pytest.skip(f"local Muse Glimmer GGUF not found: {MODEL}")
    reader = GGUFReader(MODEL)
    config = MuseGlimmerConfig.from_gguf_info(reader.info, vocab_size=202048)
    get = gguf_weight_source(reader)
    logits, captures, _ = forward(
        oracle["token_ids"], get, config, capture_layers=(1, 2, 4, 5, 52)
    )
    embd = get("token_embd.weight")[np.asarray(oracle["token_ids"])]
    return config, get, logits, captures, rmsnorm(embd, None, config.rms_eps)


def _edge6(row: np.ndarray) -> list[float]:
    return [float(v) for v in (row[0], row[1], row[2], row[-3], row[-2], row[-1])]


def test_embedding_norm_matches_oracle_exactly(reference_run, oracle) -> None:
    _, _, _, _, embd_norm = reference_run
    ref = oracle["nodes"]["embd_norm"]["sample_rows"]
    for token in range(len(oracle["token_ids"])):
        np.testing.assert_allclose(_edge6(embd_norm[token]), ref[token], atol=5e-5)


@pytest.mark.parametrize(
    ("layer", "capture", "atol"),
    [(0, 1, 0.02), (1, 2, 0.02), (3, 4, 0.03), (4, 5, 0.03)],
)
def test_layer_outputs_match_oracle(reference_run, oracle, layer, capture, atol) -> None:
    _, _, _, captures, _ = reference_run
    ref = oracle["nodes"][f"l_out-{layer}"]["sample_rows"]
    for token in range(len(oracle["token_ids"])):
        np.testing.assert_allclose(
            _edge6(captures[capture][token]), ref[token], atol=atol
        )


def test_final_layer_output_matches_oracle(reference_run, oracle) -> None:
    _, _, _, captures, _ = reference_run
    ref = oracle["nodes"]["l_out-51"]["sample_rows"][-1]
    np.testing.assert_allclose(_edge6(captures[52][-1]), ref, atol=0.35)


def test_final_logits_argmax_and_top20(reference_run, oracle) -> None:
    _, _, logits, _, _ = reference_run
    final = logits[-1]
    expected = oracle["final_logits"]
    assert int(final.argmax()) == expected["argmax"]
    top20 = set(int(i) for i in np.argsort(final)[::-1][:20])
    assert top20 == {row["id"] for row in expected["top20"]}
    for row in expected["top20"]:
        assert abs(float(final[row["id"]]) - row["logit"]) < 0.15
