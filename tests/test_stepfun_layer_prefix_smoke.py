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
            "--stream-chunk-layers",
            "1",
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
    streaming_plan = payload["streaming_plan"]
    assert streaming_plan["chunk_layers"] == 1
    assert streaming_plan["chunk_count"] == 45
    assert streaming_plan["root_slots"] == [
        "root.token_embedding",
        "root.output_norm",
        "root.lm_head",
    ]
    assert streaming_plan["peak_resident_weight_nbytes"] < payload["resident_weight_nbytes"]
    assert streaming_plan["max_chunk"]["layer_count"] == 1


def test_stepfun_layer_prefix_smoke_dry_run_writes_output_file(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    root = _stepfun_gguf_dir()
    output_path = tmp_path / "prefix-plan.json"

    rc = main(
        [
            "--dry-run-plan",
            "--model-dir",
            str(root),
            "--layer-count",
            "45",
            "--message",
            "hello",
            "--output",
            str(output_path),
            "--pretty",
        ]
    )

    assert rc == 0
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == ""
    payload = json.loads(output_path.read_text())
    assert payload["status"] == "planned"
    assert payload["scope"] == "layers_0_44_prefix_no_skipped_layers"
    assert payload["command"].endswith(f"--output {output_path} --pretty")
    assert payload["selected_slot_count"] == 753
    assert payload["resident_weight_nbytes"] > 102_000_000_000


def test_stepfun_layer_prefix_smoke_budget_guard_blocks_before_hip(
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _stepfun_gguf_dir()

    with pytest.raises(MemoryError, match="max-resident-weight-gib"):
        main(
            [
                "--model-dir",
                str(root),
                "--layer-count",
                "1",
                "--message",
                "hello",
                "--max-resident-weight-gib",
                "0.001",
            ]
        )

    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == ""


@pytest.mark.skipif(not HIP_AVAILABLE, reason="HIP runtime is not available")
def test_stepfun_layer_prefix_smoke_outputs_chunked_partial_prompt_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _stepfun_gguf_dir()

    rc = main(
        [
            "--model-dir",
            str(root),
            "--layer-count",
            "1",
            "--message",
            "hello",
            "--stream-chunk-layers",
            "1",
            "--max-resident-weight-gib",
            "4",
            "--pretty",
        ]
    )

    assert rc == 0
    output = capsys.readouterr()
    assert output.err == ""
    payload = json.loads(output.out)
    assert payload["status"] == "partial_prompt_smoke"
    assert payload["execution_mode"] == "chunked"
    assert payload["stream_chunk_layers"] == 1
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
    assert payload["peak_resident_weight_nbytes"] == payload["resident_weight_nbytes"]
    assert payload["all_resident_weight_nbytes"] == payload["resident_weight_nbytes"]
    assert payload["chunk_records"] == [
        {
            "start_layer": 0,
            "end_layer_exclusive": 1,
            "slot_count": 12,
            "layer_weight_nbytes": 113_687_552,
            "peak_with_roots_nbytes": 1_235_614_720,
        }
    ]
    assert "hip_free_after_generation_before_free_gib" in payload
    assert payload["memory_stats_before_free"]["active_allocations"] == 3
    assert payload["memory_stats_after_free"]["active_allocations"] == 0
    assert payload["memory_stats_after_free"]["current_allocated_bytes"] == 0
