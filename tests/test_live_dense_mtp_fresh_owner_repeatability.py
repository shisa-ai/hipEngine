"""Fresh-owner dense MTP control repeatability through the public API."""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path

import pytest

from hipengine import LLM, SamplingParams
from hipengine.core.memory import memory_stats
from hipengine.kernels.backends import resolve_backend
from hipengine.runtime.qwen35_gguf_mtp import _StateJournal


MODEL = Path(os.environ.get(
    "HIPENGINE_TEST_DENSE_GGUF_MODEL", "/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf",
))


def test_dense_mtp_fresh_owners_repeat_acceptance_controls(monkeypatch):
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        pytest.skip("HIP runtime is not available")
    if not MODEL.exists():
        pytest.skip(f"local GGUF fixture not found: {MODEL}")
    backend = resolve_backend("auto")
    if backend not in {"hip_gfx1100", "hip_gfx1151"}:
        pytest.skip("dense GGUF MTP requires a gfx11 backend")
    prompts = Path(__file__).resolve().parents[1] / "benchmarks/prompts/laguna-target-ar-code-general-ja-heldout.jsonl"
    prompt = next(json.loads(line) for line in prompts.read_text().splitlines()
                  if json.loads(line)["id"] == "heldout_general_ja_speculative")
    text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
                   for m in prompt["messages"]) + "<|im_start|>assistant\n"
    params = SamplingParams(max_tokens=64, temperature=0, ignore_eos=True, speculative_mtp=True)
    original = _StateJournal._copy_initial_state
    expected = None
    initial_memory = memory_stats()["current_allocated_bytes"]
    keys = ("cycles", "accepted_draft_tokens", "generated_draft_tokens",
            "selected_depth_histogram", "failure_reason_counts", "ar_step_reason_counts")
    for batched in (False, True, True, False, False, True):
        monkeypatch.setattr(_StateJournal, "_copy_initial_state",
                            original if batched else lambda self, **kwargs: False)
        llm = LLM(str(MODEL), backend=backend, kv_storage="bf16",
                  max_sequence_length=2048, prefix_cache="off",
                  speculative_candidate_budget=3)
        try:
            llm.generate_detailed(["List three uses of a hash table."], params)
            for repeat in range(3):
                row = llm.generate_detailed([text], params)[0]
                controls = row.telemetry.to_json_dict()["diagnostics"]["specdec2_mtp2"]
                assert controls["cycles"] > 0
                assert controls["recoverable_failures"] == 0
                observed = (tuple(row.generated_token_ids), {key: controls[key] for key in keys})
                if expected is None:
                    expected = observed
                assert observed == expected, {"batched": batched, "repeat": repeat, "controls": controls}
        finally:
            llm.close()
        assert memory_stats()["current_allocated_bytes"] == initial_memory
