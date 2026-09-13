"""Retention packets must fail on incomplete or nonexact model matrices."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "benchmarks/results/2026-09-14-journey-ple"


def tool():
    spec = importlib.util.spec_from_file_location("ple_evidence", ARTIFACT / "assemble.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture():
    samples = []
    for i in range(12):
        for mode, slot in (("before", 0), ("after", 1)):
            samples.append(dict(
                case_id=f"case{i}", mode=mode, repetition=0, sequence_slot=slot,
                category="code", prompt_tokens=512, prefill_ms=10., decode_ms=20.,
                decode_transitions=128, client_wall_s=.03, output_token_count=129,
                output_token_ids_sha256="ids", final_logits_sha256="logits",
                final_state_sha256="state", memory_delta={},
            ))
    return dict(
        status="completed", kind="qwen4exp_ple_complete_model_ab",
        host={"machine_id": tool().HOST}, source={"head": "0" * 40},
        model_identity={"fingerprint": {"value": tool().FINGERPRINT}},
        command=[], method="mmap_random", script_sha256="unavailable",
        protocol={"repetitions_per_arm": 1, "cache": "warm"},
        samples=samples, after_close={"active_allocations": 0, "current_allocated_bytes": 0},
        manifest_sha256="manifest",
    )


@pytest.mark.parametrize("fault", ["missing_sample", "bad_state", "leak"])
def test_compactor_rejects_invalid_model_evidence(tmp_path, fault):
    data = fixture()
    if fault == "missing_sample":
        data["samples"].pop()
    elif fault == "bad_state":
        data["samples"][0]["final_state_sha256"] = "different"
    else:
        data["after_close"]["active_allocations"] = 1
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        tool().compact(path)
