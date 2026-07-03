from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.gguf_mtp_parity_precheck import (
    build_parity_precheck,
    compare_sampling_settings,
    load_sampling_settings,
    stable_json_sha256,
)


HIPENGINE_D32_TOKEN_FIXTURE = Path(
    "benchmarks/fixtures/hipengine_gguf_prompt_tokens_qwen36_35b_a3b_ud_q4_k_m_d32.json"
)
LLAMACPP_HIP_D32_TOKEN_FIXTURE = Path(
    "benchmarks/fixtures/llamacpp_hip_prompt_tokens_qwen36_35b_a3b_ud_q4_k_m_d32.json"
)
B1_SAMPLING_FIXTURE = Path("benchmarks/fixtures/gguf_mtp_b1_sampling_greedy_seed12345.json")


def _inventory(*, token_ids: list[int] | None = None, name: str = "p0") -> dict[str, object]:
    tokens = [1, 2, 3] if token_ids is None else token_ids
    return {
        "schema": 1,
        "kind": "hipengine_gguf_prompt_token_inventory",
        "prompts": [
            {
                "name": name,
                "token_ids": tokens,
                "token_ids_sha256": "synthetic",
                "rendered_sha256": "prompt-hash",
            }
        ],
    }


def test_sampling_settings_compare_requires_both_when_requested() -> None:
    comparison = compare_sampling_settings(None, None, require_sampling=True)

    assert comparison["checked"] is False
    assert comparison["passed"] is False
    assert comparison["reason"] == "sampling settings were not provided"

    one_sided = compare_sampling_settings({"target": {"temperature": 0}}, None)

    assert one_sided["checked"] is True
    assert one_sided["passed"] is False
    assert one_sided["mismatches"] == [
        {"path": "<root>", "hipengine": "present", "llamacpp": "missing"}
    ]


def test_sampling_settings_compare_reports_nested_mismatches_and_hashes() -> None:
    hipengine = {
        "target": {"temperature": 0.0, "seed": 123},
        "draft": {"top_k": 10, "selection": "greedy_top1_from_topk"},
    }
    llamacpp = {
        "target": {"temperature": 0.0, "seed": 123},
        "draft": {"top_k": 8, "selection": "greedy_top1_from_topk"},
        "server": "llama.cpp",
    }

    comparison = compare_sampling_settings(hipengine, llamacpp)

    assert comparison["checked"] is True
    assert comparison["passed"] is False
    assert comparison["hipengine_sampling_sha256"] == stable_json_sha256(hipengine)
    assert comparison["llamacpp_sampling_sha256"] == stable_json_sha256(llamacpp)
    assert comparison["mismatches"] == [
        {"path": "draft.top_k", "hipengine": 10, "llamacpp": 8},
        {"path": "server", "hipengine": None, "llamacpp": "llama.cpp"},
    ]


def test_parity_precheck_passes_matching_token_ids_and_sampling() -> None:
    sampling = {
        "target": {"temperature": 0.0, "seed": 1234},
        "draft": {"top_k": 10, "selection": "greedy_top1_from_topk"},
    }

    precheck = build_parity_precheck(
        hipengine_token_inventory=_inventory(),
        llamacpp_token_inventory=_inventory(),
        hipengine_sampling=sampling,
        llamacpp_sampling=dict(sampling),
        require_sampling=True,
    )

    assert precheck["kind"] == "gguf_mtp_parity_precheck"
    assert precheck["all_pass"] is True
    assert precheck["token_ids"]["all_match"] is True
    assert precheck["sampling"]["passed"] is True


def test_parity_precheck_fails_token_or_sampling_mismatch() -> None:
    precheck = build_parity_precheck(
        hipengine_token_inventory=_inventory(token_ids=[1, 2, 3]),
        llamacpp_token_inventory=_inventory(token_ids=[1, 7, 3]),
        hipengine_sampling={"target": {"temperature": 0.0}},
        llamacpp_sampling={"target": {"temperature": 0.25}},
        require_sampling=True,
        context_tokens=1,
    )

    assert precheck["all_pass"] is False
    assert precheck["token_ids"]["mismatches"][0]["first_mismatch_index"] == 1
    assert precheck["sampling"]["mismatches"] == [
        {"path": "target.temperature", "hipengine": 0.0, "llamacpp": 0.25}
    ]


def test_load_sampling_settings_accepts_wrapped_or_plain_sampling(tmp_path: Path) -> None:
    wrapped = tmp_path / "wrapped.json"
    plain = tmp_path / "plain.json"
    wrapped.write_text(json.dumps({"schema": 1, "sampling": {"target": {"temperature": 0.0}}}))
    plain.write_text(json.dumps({"target": {"temperature": 0.0}}))

    assert load_sampling_settings(wrapped) == {"target": {"temperature": 0.0}}
    assert load_sampling_settings(plain) == {"target": {"temperature": 0.0}}


def test_committed_b1_sampling_fixture_is_self_consistent() -> None:
    payload = json.loads(B1_SAMPLING_FIXTURE.read_text())
    sampling = load_sampling_settings(B1_SAMPLING_FIXTURE)

    assert payload["schema"] == 1
    assert payload["kind"] == "gguf_mtp_sampling_settings"
    assert payload["name"] == "gguf_mtp_b1_greedy_seed12345"
    assert sampling["target"] == {
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "min_p": 0.0,
        "seed": 12345,
    }
    assert sampling["draft"] == {
        "budget": "B1",
        "draft_max": 1,
        "selection": "greedy_top1",
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "min_p": 0.0,
        "seed": 12345,
    }
    assert sampling["request"]["max_tokens"] == 128
    assert sampling["request"]["prompt_render"] == "raw"
    assert sampling["request"]["stream"] is False
    assert sampling["request"]["cache_prompt"] is False
    assert sampling["comparability"] == {
        "accepted_per_output_denominator": "accepted_target_tokens / emitted_output_tokens",
        "accept_per_draft_denominator": "accepted_draft_tokens / proposed_draft_tokens",
        "requires_token_id_parity": True,
        "requires_numeric_gate": True,
    }
    assert payload["llamacpp_request_mapping"]["server"] == {
        "spec_type": "draft-mtp",
        "spec_draft_n_max": 1,
    }
    assert payload["hipengine_request_mapping"]["draft"] == {
        "budget": 1,
        "selection": "greedy_top1",
    }


def test_parity_precheck_cli_fails_on_mismatch_when_requested(tmp_path: Path) -> None:
    hip = tmp_path / "hip.json"
    llama = tmp_path / "llama.json"
    sampling = tmp_path / "sampling.json"
    out = tmp_path / "precheck.json"
    hip.write_text(json.dumps(_inventory(token_ids=[1, 2, 3])))
    llama.write_text(json.dumps(_inventory(token_ids=[1, 7, 3])))
    sampling.write_text(json.dumps({"target": {"temperature": 0.0}}))

    result = subprocess.run(
        [
            sys.executable,
            "scripts/gguf_mtp_parity_precheck.py",
            "--hipengine-token-inventory",
            str(hip),
            "--llamacpp-token-inventory",
            str(llama),
            "--hipengine-sampling",
            str(sampling),
            "--llamacpp-sampling",
            str(sampling),
            "--require-sampling",
            "--fail-on-mismatch",
            "--out",
            str(out),
        ],
        cwd=Path.cwd(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 1
    payload = json.loads(out.read_text())
    assert payload["all_pass"] is False
    assert payload["sampling"]["passed"] is True
    assert payload["token_ids"]["mismatches"]


def test_parity_precheck_passes_committed_hipengine_vs_llamacpp_token_fixtures() -> None:
    precheck = build_parity_precheck(
        hipengine_token_inventory=json.loads(HIPENGINE_D32_TOKEN_FIXTURE.read_text()),
        llamacpp_token_inventory=json.loads(LLAMACPP_HIP_D32_TOKEN_FIXTURE.read_text()),
    )

    assert precheck["all_pass"] is True
    assert precheck["token_ids"]["all_match"] is True
    assert precheck["token_ids"]["mismatches"] == []
    assert precheck["sampling"]["checked"] is False
    assert precheck["sampling"]["passed"] is True


def test_parity_precheck_with_committed_b1_sampling_passes_real_token_fixtures() -> None:
    sampling = load_sampling_settings(B1_SAMPLING_FIXTURE)

    precheck = build_parity_precheck(
        hipengine_token_inventory=json.loads(HIPENGINE_D32_TOKEN_FIXTURE.read_text()),
        llamacpp_token_inventory=json.loads(LLAMACPP_HIP_D32_TOKEN_FIXTURE.read_text()),
        hipengine_sampling=sampling,
        llamacpp_sampling=dict(sampling),
        require_sampling=True,
    )

    assert precheck["all_pass"] is True
    assert precheck["token_ids"]["all_match"] is True
    assert precheck["token_ids"]["mismatches"] == []
    assert precheck["sampling"]["checked"] is True
    assert precheck["sampling"]["passed"] is True
    assert precheck["sampling"]["mismatches"] == []


def test_parity_precheck_cli_passes_matching_committed_hipengine_fixture(tmp_path: Path) -> None:
    out = tmp_path / "precheck.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/gguf_mtp_parity_precheck.py",
            "--hipengine-token-inventory",
            str(HIPENGINE_D32_TOKEN_FIXTURE),
            "--llamacpp-token-inventory",
            str(HIPENGINE_D32_TOKEN_FIXTURE),
            "--hipengine-sampling",
            str(B1_SAMPLING_FIXTURE),
            "--llamacpp-sampling",
            str(B1_SAMPLING_FIXTURE),
            "--require-sampling",
            "--fail-on-mismatch",
            "--out",
            str(out),
        ],
        cwd=Path.cwd(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(out.read_text())
    assert payload["all_pass"] is True
    assert payload["token_ids"]["compared_prompts"] == 9
    assert payload["sampling"]["passed"] is True
