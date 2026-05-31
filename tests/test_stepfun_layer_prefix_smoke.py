from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path

import pytest

from scripts.stepfun_layer_prefix_smoke import main

DEFAULT_STEPFUN_GGUF_DIR = Path("/data/models/gguf")


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


HIP_AVAILABLE = _hip_available()


def _stepfun_gguf_dir() -> Path:
    root = Path(os.environ.get("HIPENGINE_STEPFUN_GGUF_DIR", DEFAULT_STEPFUN_GGUF_DIR))
    paths = tuple(sorted(root.glob("Step-3.7-flash-Q3_K_L-*.gguf")))
    if len(paths) != 3:
        pytest.skip(
            "StepFun GGUF Q3_K_L shards not found; set HIPENGINE_STEPFUN_GGUF_DIR "
            "to a directory containing Step-3.7-flash-Q3_K_L-00001..00003.gguf"
        )
    return root


def test_stepfun_layer_prefix_smoke_dry_run_plans_all_layers_without_hip(
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _stepfun_gguf_dir()

    rc = main(
        [
            "--dry-run-plan",
            "--model-dir",
            str(root),
            "--layer-count",
            "45",
            "--message",
            "hello",
            "--pretty",
        ]
    )

    assert rc == 0
    output = capsys.readouterr()
    assert output.err == ""
    payload = json.loads(output.out)
    assert payload["status"] == "planned"
    assert payload["scope"] == "layers_0_44_prefix_no_skipped_layers"
    assert payload["layer_count"] == 45
    assert payload["skipped_layers"] == []
    assert payload["selected_slot_count"] == 753
    assert payload["selected_slots"][:3] == [
        "root.token_embedding",
        "root.output_norm",
        "root.lm_head",
    ]
    assert "root.rope_freqs" not in payload["selected_slots"]
    assert "layers.44.ffn_down_shexp" in payload["selected_slots"]
    assert payload["no_vision_projector_mtp_slots"] is True
    assert payload["resident_weight_nbytes"] > 102_000_000_000
    assert "no HIP runtime was initialized" in payload["note"]


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_stepfun_layer_prefix_smoke_outputs_partial_prompt_json(capsys: pytest.CaptureFixture[str]) -> None:
    root = _stepfun_gguf_dir()

    rc = main(
        [
            "--model-dir",
            str(root),
            "--layer-count",
            "1",
            "--message",
            "hello",
            "--pretty",
        ]
    )

    assert rc == 0
    output = capsys.readouterr()
    assert output.err == ""
    payload = json.loads(output.out)
    assert payload["status"] == "partial_prompt_smoke"
    assert payload["layer_count"] == 1
    assert payload["scope"] == "layers_0_0_prefix_only_layers_1_44_skipped"
    assert payload["selected_slot_count"] == 15
    assert payload["selected_slots"][:3] == [
        "root.token_embedding",
        "root.output_norm",
        "root.lm_head",
    ]
    assert payload["no_vision_projector_mtp_slots"] is True
    assert payload["prompt_length"] > 0
    assert payload["layer_hidden_shape"] == [payload["prompt_length"], 4096]
    assert payload["logits_shape"] == [1, 128896]
    assert set(payload["sampled_logits"]) == {"0", "1", "128007", "128895"}
    assert payload["resident_weight_nbytes"] > 0
    assert payload["memory_stats_after_free"]["active_allocations"] == 0
    assert payload["memory_stats_after_free"]["current_allocated_bytes"] == 0
