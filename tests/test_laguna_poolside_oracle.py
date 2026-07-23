from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest

FIXTURES = Path(__file__).parent / "fixtures"
TEMPLATE_FIXTURE = FIXTURES / "laguna_poolside_v1_template.json"
ORACLE_FIXTURE = FIXTURES / "laguna_poolside_v1_oracle.json"
Q2_XL_ORACLE_FIXTURE = FIXTURES / "laguna_poolside_q2_xl_v1_oracle.json"
_ORACLES = (
    (ORACLE_FIXTURE, "7da520c5f44bc3c79d4eeebfd1151ba7114c5d7568e72a995638417093c5753f"),
    (Q2_XL_ORACLE_FIXTURE, "8fe1170f012723f6f7d6c9b08d8f928b0b3d8bffc32926f33a930148a1d62679"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(("oracle_fixture", "expected_model_sha256"), _ORACLES)
def test_laguna_poolside_oracle_provenance_matches_template(
    oracle_fixture: Path,
    expected_model_sha256: str,
) -> None:
    template = json.loads(TEMPLATE_FIXTURE.read_text())
    oracle = json.loads(oracle_fixture.read_text())
    prompt = next(
        case for case in template["cases"] if case["name"] == oracle["prompt"]["case"]
    )

    assert oracle["source"]["commit"] == template["poolside_llama_commit"]
    assert oracle["model"]["sha256"] == expected_model_sha256
    assert hashlib.sha256(prompt["rendered"].encode()).hexdigest() == oracle["prompt"][
        "rendered_sha256"
    ]
    token_bytes = b"".join(struct.pack("<i", token_id) for token_id in prompt["token_ids"])
    assert hashlib.sha256(token_bytes).hexdigest() == oracle["prompt"][
        "token_ids_i32le_sha256"
    ]
    assert len(prompt["token_ids"]) == oracle["prompt"]["token_count"]
    assert oracle["prompt"]["pass_as_token_ids"] is True
    assert oracle["server"]["target_only"] is True
    assert oracle["server"]["context_length"] == 4_096
    assert oracle["server"]["flash_attention"] is False
    assert oracle["server"]["mmap"] is False


@pytest.mark.parametrize("oracle_fixture", (ORACLE_FIXTURE, Q2_XL_ORACLE_FIXTURE))
def test_laguna_poolside_first_token_distribution_is_complete(
    oracle_fixture: Path,
) -> None:
    oracle = json.loads(oracle_fixture.read_text())
    first = oracle["first_token"]
    distribution = first["full_distribution"]
    path = FIXTURES / distribution["path"]
    logprobs = np.load(path, allow_pickle=False)

    assert _sha256(path) == distribution["npy_sha256"]
    assert hashlib.sha256(logprobs.tobytes()).hexdigest() == distribution[
        "dense_bytes_sha256"
    ]
    assert logprobs.dtype == np.dtype("<f4")
    assert list(logprobs.shape) == distribution["shape"]
    assert logprobs.size == distribution["count"] == oracle["model"]["vocab_size"]
    assert int(np.isfinite(logprobs).sum()) == distribution["finite_count"]
    assert int(np.argmax(logprobs)) == first["id"]
    assert float(logprobs[first["id"]]) == first["logprob"]
    np.testing.assert_allclose(
        np.exp(logprobs.astype(np.float64)).sum(),
        distribution["probability_sum_float64"],
        rtol=0.0,
        atol=1e-12,
    )
    assert distribution["fresh_process_repeat_exact"] is True
    assert distribution["fresh_process_repeat_changed_values"] == 0

    for expected in first["top10"]:
        assert float(logprobs[expected["id"]]) == expected["logprob"]


@pytest.mark.parametrize("oracle_fixture", (ORACLE_FIXTURE, Q2_XL_ORACLE_FIXTURE))
def test_laguna_poolside_greedy_oracle_has_two_fresh_start_repeats(
    oracle_fixture: Path,
) -> None:
    oracle = json.loads(oracle_fixture.read_text())
    greedy = oracle["greedy32"]

    assert len(greedy["token_ids"]) == 32
    assert greedy["token_ids"][0] == oracle["first_token"]["id"]
    assert greedy["text"].startswith(oracle["first_token"]["text"])
    assert greedy["fresh_process_repeat_exact"] is True
    assert greedy["fresh_process_repeat_count"] == 2
    assert len(greedy["measured_runs"]) == 2
    assert all(run["decode_tokens"] == 32 for run in greedy["measured_runs"])
