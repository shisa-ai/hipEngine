"""Tests for scripts/llamacpp_vulkan_perf_summary.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "llamacpp_vulkan_perf_summary.py"
    module_name = "_llamacpp_vulkan_perf_summary_test_module"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


SCRIPT = _load_script_module()


@pytest.mark.parametrize(
    "name,expected",
    [
        ("MUL_MAT_ID q4_K m=512 n=8 k=2048", "llama_selected_q4"),
        ("MUL_MAT_ID_VEC q4_K m=512 n=8 k=2048", "llama_selected_q4"),
        ("MUL_MAT_ID q5_K m=2048 n=8 k=512", "llama_selected_q5"),
        ("MUL_MAT_ID_MUL MUL_MAT_ID_VEC q5_K m=2048 n=8 k=512", "llama_selected_q5"),
        ("MUL_MAT_ID q6_K m=2048 n=8 k=512", "llama_selected_q6"),
        ("MUL_MAT_ID q5_1 m=2560 n=10 k=640", "llama_selected_q51"),
        ("MUL_MAT_ID_MUL MUL_MAT_ID_VEC q5_1 m=2560 n=10 k=640", "llama_selected_q51"),
        ("MUL_MAT_ID q8_0 m=2560 n=10 k=640", "llama_selected_q8"),
        ("MUL_MAT_ID_VEC f32 m=2560 n=10 k=640", "llama_selected_other"),
        ("MUL_MAT q8_0 m=2048 n=1 k=4096", "llama_dense_q8"),
        ("MUL_MAT_ADD MUL_MAT_VEC q8_0 m=2048 n=1 k=4096", "llama_dense_q8"),
        ("MUL_MAT_VEC q6_K m=248320 n=1 k=2048", "llama_lm_head"),
        ("MUL_MAT f32 m=256 n=1 k=2048", "llama_f32_matmul"),
        ("MUL_MAT_VEC f32 m=256 n=1 k=2048", "llama_f32_matmul"),
        ("GATED_DELTA_NET", "llama_gdn"),
        ("SSM_CONV_SILU SSM_CONV", "llama_linear_attn_conv"),
        ("FLASH_ATTN_EXT dst(256,16,512,1)", "llama_flash_attn"),
        ("RMS_NORM_MUL RMS_NORM(2048,1,1,1)", "llama_norm"),
        ("TOPK_MOE_EARLY_SOFTMAX_NORM SOFT_MAX", "llama_router_topk"),
        ("CONCAT", "llama_copy_layout"),
        ("MULTI_ADD ADD", "llama_elementwise"),
        ("unknown", "other"),
    ],
)
def test_classify_operation(name: str, expected: str) -> None:
    assert SCRIPT.classify_operation(name) == expected


def test_parse_and_aggregate_sections(tmp_path: Path) -> None:
    log = tmp_path / "vulkan.stderr"
    log.write_text(
        "ggml_vulkan: Found 1 Vulkan devices:\n"
        "----------------\n"
        "Vulkan Timings:\n"
        "ADD: 1 x 10.0 us = 10.0 us\n"
        "MUL_MAT q8_0 m=2048 n=1 k=4096: 2 x 20.0 us = 40.0 us (100 GFLOPS/s)\n"
        "Total time: 50.0 us.\n"
        "----------------\n"
        "Vulkan Timings:\n"
        "ADD: 1 x 5.0 us = 5.0 us\n"
        "MUL_MAT q8_0 m=2048 n=1 k=4096: 2 x 15.0 us = 30.0 us (120 GFLOPS/s)\n"
        "GATED_DELTA_NET: 1 x 7.0 us = 7.0 us\n"
        "Total time: 42.0 us.\n"
    )

    sections = SCRIPT.parse_perf_log(log)
    assert len(sections) == 2
    assert sections[0].total_us == pytest.approx(50.0)
    assert sections[1].operations[1].dispatches == 2

    payload = SCRIPT.build_summary(log, label="unit", command="llama-bench", discard_first_sections=1, top=10)
    assert payload["schema"] == SCRIPT.SCHEMA
    assert payload["classifier_version"] == SCRIPT.CLASSIFIER_VERSION
    assert payload["sections_found"] == 2
    assert payload["selected_section_count"] == 1
    assert payload["total_gpu_ms"] == pytest.approx(0.042)
    assert payload["avg_section_gpu_ms"] == pytest.approx(0.042)
    assert payload["total_dispatches"] == 4
    families = {row["family"]: row for row in payload["families"]}
    assert families["llama_dense_q8"]["total_ms"] == pytest.approx(0.030)
    assert families["llama_gdn"]["total_ms"] == pytest.approx(0.007)
    assert families["llama_elementwise"]["total_ms"] == pytest.approx(0.005)
    assert families["llama_dense_q8"]["share_of_total"] == pytest.approx(30.0 / 42.0)

    output = tmp_path / "summary.json"
    rc = SCRIPT.main(
        [
            "--log",
            str(log),
            "--json",
            str(output),
            "--label",
            "unit",
            "--discard-first-sections",
            "1",
        ]
    )
    assert rc == 0
    assert json.loads(output.read_text())["total_gpu_ms"] == pytest.approx(0.042)


def test_build_summary_can_select_only_last_sections(tmp_path: Path) -> None:
    log = tmp_path / "decode.stderr"
    log.write_text(
        "Vulkan Timings:\nADD: 1 x 9.0 us = 9.0 us\nTotal time: 9.0 us.\n"
        "Vulkan Timings:\nADD: 1 x 7.0 us = 7.0 us\nTotal time: 7.0 us.\n"
        "Vulkan Timings:\nADD: 1 x 5.0 us = 5.0 us\nTotal time: 5.0 us.\n"
    )
    payload = SCRIPT.build_summary(
        log,
        label="decode",
        command=None,
        discard_first_sections=1,
        select_last_sections=1,
        top=10,
    )
    assert payload["sections_found"] == 3
    assert payload["discard_first_sections"] == 1
    assert payload["select_last_sections"] == 1
    assert payload["selected_section_count"] == 1
    assert payload["total_gpu_ms"] == pytest.approx(0.005)


def test_parse_rejects_incomplete_section(tmp_path: Path) -> None:
    log = tmp_path / "incomplete.stderr"
    log.write_text("Vulkan Timings:\nADD: 1 x 5.0 us = 5.0 us\n")
    with pytest.raises(ValueError, match="incomplete Vulkan timing section"):
        SCRIPT.parse_perf_log(log)


def test_build_summary_rejects_discarding_all_sections(tmp_path: Path) -> None:
    log = tmp_path / "one.stderr"
    log.write_text("Vulkan Timings:\nADD: 1 x 5.0 us = 5.0 us\nTotal time: 5.0 us.\n")
    with pytest.raises(ValueError, match="no selected Vulkan timing sections"):
        SCRIPT.build_summary(log, label="unit", command=None, discard_first_sections=1, top=10)
