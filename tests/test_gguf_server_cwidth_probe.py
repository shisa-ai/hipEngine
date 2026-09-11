"""CPU tests for the C-width baseline harness (roadmap P5 / reviewer item 4).

The 2026-09-10 review found the harness broken in ways a GPU run would only
surface late: a ``NameError`` on a removed variable that fired *after* a
successful first-width run (so the artifact was never written), one lane's
completion count applied to every lane, decode rates derived from requested
tokens, and no measured serial control. These tests pin the replacement's
accounting rules without a device or a server.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_harness():
    path = REPO_ROOT / "scripts" / "gguf_server_cwidth_probe.py"
    spec = importlib.util.spec_from_file_location("gguf_server_cwidth_probe", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def harness():
    return _load_harness()


def _metrics(
    *,
    decode: int = 0,
    prefill: int = 0,
    fallbacks: dict[str, int] | None = None,
    routes: dict[str, int] | None = None,
    manifest: dict[str, str] | None = None,
) -> dict[str, float]:
    out: dict[str, float] = {
        "hipengine_resident_work_decode_total": float(decode),
        "hipengine_resident_work_prefill_total": float(prefill),
    }
    for reason, value in (fallbacks or {}).items():
        out[f'hipengine_resident_fallback_total{{reason="{reason}"}}'] = float(value)
    for route, value in (routes or {}).items():
        out[f'hipengine_resident_route_total{{route="{route}"}}'] = float(value)
    if manifest is not None:
        labels = ",".join(f'{k}="{v}"' for k, v in sorted(manifest.items()))
        out[f"hipengine_resident_route_manifest_info{{{labels}}}"] = 1.0
    return out


def _payload(prompt: int, completion: int) -> dict:
    return {"usage": {"prompt_tokens": prompt, "completion_tokens": completion}}


def test_parse_prometheus_keeps_finite_numbers_and_labels(harness) -> None:
    text = "\n".join(
        [
            "# HELP thing a thing",
            "# TYPE thing counter",
            "thing 3",
            'labeled{route="packed_decode"} 7',
            'labeled{route="serial"} NaN',
            'labeled{route="other"} +Inf',
            "broken line without value",
            "  ",
        ]
    )
    parsed = harness.parse_prometheus(text)
    assert parsed["thing"] == 3.0
    assert parsed['labeled{route="packed_decode"}'] == 7.0
    # Non-finite samples are dropped so the artifact stays JSON-valid.
    assert 'labeled{route="serial"}' not in parsed
    assert 'labeled{route="other"}' not in parsed


def test_labeled_counter_extracts_single_label_series(harness) -> None:
    metrics = _metrics(fallbacks={"serial_c1_per_row": 4, "other": 2})
    assert harness.labeled_counter(
        metrics, "hipengine_resident_fallback_total"
    ) == {"serial_c1_per_row": 4.0, "other": 2.0}


def test_usage_counts_reports_missing_as_negative(harness) -> None:
    assert harness._usage_counts(_payload(2048, 63)) == (2048, 63)
    assert harness._usage_counts({}) == (-1, -1)
    assert harness._usage_counts({"usage": {"prompt_tokens": "x"}}) == (-1, -1)


def test_phase_record_uses_each_lanes_own_counts(harness) -> None:
    """Heterogeneous completion counts must not be flattened to lane 0."""

    record = harness.build_phase_record(
        width=3,
        phase="concurrent",
        wall_s=10.0,
        lane_results=[
            (3.0, _payload(2048, 60)),
            (9.0, _payload(2048, 63)),
            (9.5, _payload(2048, 62)),
        ],
        lane_errors=[],
        metrics_before=_metrics(decode=100, prefill=10),
        metrics_after=_metrics(decode=130, prefill=13),
    )
    assert record["status"] == "pass"
    assert record["completion_tokens_reported"] == [60, 63, 62]
    assert record["completion_tokens_total"] == 185
    assert record["prompt_tokens_total"] == 6144
    # Complete-request throughput counts every token of every lane.
    assert record["complete_request_throughput_tok_s"] == round(6329 / 10.0, 3)
    assert record["per_request_wall_s"] == [3.0, 9.0, 9.5]
    # Measured model steps, not a requested-token estimate.
    assert record["measured_decode_steps"] == 30
    assert record["measured_prefill_steps"] == 3
    assert record["measured_decode_steps_per_s"] == 3.0
    # No decode-only tok/s field of any spelling.
    assert not any("aggregate" in key for key in record)


def test_phase_record_records_width_and_fallback_counters(harness) -> None:
    record = harness.build_phase_record(
        width=2,
        phase="concurrent",
        wall_s=8.0,
        lane_results=[(4.0, _payload(2048, 40)), (4.0, _payload(2048, 40))],
        lane_errors=[],
        metrics_before=_metrics(
            fallbacks={"serial_c1_per_row": 1},
            routes={"kv_live_spans_int8_batch": 1},
        ),
        metrics_after=_metrics(
            fallbacks={"serial_c1_per_row": 5, "packed_decode_width_unqualified": 2},
            routes={"kv_live_spans_int8_batch": 3, "kv_live_spans_int8_serial": 4},
            manifest={
                "claim_level": "diagnostic",
                "kv_attention_source": "int8_direct",
                "logical_c": "2",
                "mode": "serial_c1_per_row",
            },
        ),
    )
    assert record["fallback_reasons_delta"]["serial_c1_per_row"] == 4.0
    assert record["fallback_reasons_delta"]["packed_decode_width_unqualified"] == 2.0
    assert record["route_counts_delta"]["kv_live_spans_int8_batch"] == 2.0
    assert record["route_counts_delta"]["kv_live_spans_int8_serial"] == 4.0
    assert record["execution_manifest"]["mode"] == "serial_c1_per_row"
    assert record["execution_manifest"]["logical_c"] == "2"


def test_phase_record_marks_failed_requests(harness) -> None:
    record = harness.build_phase_record(
        width=2,
        phase="concurrent",
        wall_s=1.0,
        lane_results=[(1.0, _payload(2048, 5))],
        lane_errors=["lane 1: connection reset"],
        metrics_before=_metrics(),
        metrics_after=_metrics(),
    )
    assert record["status"] == "request_failed"
    assert record["completed_lanes"] == 1
    assert record["expected_lanes"] == 2
    assert record["errors"] == ["lane 1: connection reset"]
    # A failed phase carries no throughput claim.
    assert "complete_request_throughput_tok_s" not in record


def test_phase_record_marks_missing_usage(harness) -> None:
    record = harness.build_phase_record(
        width=1,
        phase="serial",
        wall_s=2.0,
        lane_results=[(2.0, {"choices": []})],
        lane_errors=[],
        metrics_before=_metrics(),
        metrics_after=_metrics(),
    )
    assert record["status"] == "usage_missing"
    assert "complete_request_throughput_tok_s" not in record


def test_phase_record_tolerates_missing_metrics(harness) -> None:
    """A server that does not expose the counters must not break the artifact."""

    record = harness.build_phase_record(
        width=1,
        phase="concurrent",
        wall_s=3.0,
        lane_results=[(3.0, _payload(2048, 50))],
        lane_errors=[],
        metrics_before={},
        metrics_after={},
    )
    assert record["status"] == "pass"
    assert record["measured_decode_steps"] is None
    assert record["measured_prefill_steps"] is None
    assert record["measured_decode_steps_per_s"] is None
    assert record["fallback_reasons_delta"] == {}
    assert "execution_manifest" not in record


def test_serial_reference_is_a_ratio_of_two_measured_rates(harness) -> None:
    serial = {"status": "pass", "complete_request_throughput_tok_s": 40.0}
    concurrent = {"status": "pass", "complete_request_throughput_tok_s": 41.0}
    out = harness.serial_reference(serial, concurrent)
    assert out["serial_complete_request_rate_tok_s"] == 40.0
    assert out["concurrent_over_serial_ratio"] == 1.025


def test_serial_reference_absent_when_control_failed(harness) -> None:
    out = harness.serial_reference(
        {"status": "request_failed"},
        {"status": "pass", "complete_request_throughput_tok_s": 41.0},
    )
    assert out == {}


def test_artifact_is_json_serializable(harness) -> None:
    """The whole assembled artifact must dump cleanly - the old NameError
    path produced no artifact at all."""

    concurrent = harness.build_phase_record(
        width=2,
        phase="concurrent",
        wall_s=9.0,
        lane_results=[(9.0, _payload(2048, 63)), (9.0, _payload(2048, 63))],
        lane_errors=[],
        metrics_before=_metrics(decode=10, prefill=1),
        metrics_after=_metrics(decode=70, prefill=3),
    )
    serial = harness.build_phase_record(
        width=2,
        phase="serial",
        wall_s=18.0,
        lane_results=[(9.0, _payload(2048, 63)), (9.0, _payload(2048, 63))],
        lane_errors=[],
        metrics_before=_metrics(decode=70, prefill=3),
        metrics_after=_metrics(decode=130, prefill=5),
    )
    record = {
        "status": concurrent["status"],
        "concurrent": concurrent,
        "serial_control": serial,
    }
    record.update(harness.serial_reference(serial, concurrent))
    artifact = {"kind": "test", "widths": {"2": record}}
    text = json.dumps(artifact, indent=2, default=str)
    assert json.loads(text)["widths"]["2"]["concurrent"]["status"] == "pass"


def test_harness_has_no_removed_variable_reference(harness) -> None:
    """Regression guard for the review's NameError (removed ``aggregate``)."""

    source = (
        REPO_ROOT / "scripts" / "gguf_server_cwidth_probe.py"
    ).read_text(encoding="utf-8")
    # ``aggregate`` may appear only inside the withdrawn-claim prose, never as
    # an expression. Check the f-string print block specifically.
    assert "f\"{aggregate:" not in source
    assert "aggregate_est" not in source


def test_vram_free_gib_reads_the_selected_card(harness, monkeypatch) -> None:
    payload = json.dumps(
        {
            "card0": {
                "VRAM Total Memory (B)": str(48 * 1024**3),
                "VRAM Total Used Memory (B)": str(8 * 1024**3),
            },
            "card1": {
                "VRAM Total Memory (B)": str(24 * 1024**3),
                "VRAM Total Used Memory (B)": str(24 * 1024**3),
            },
        }
    )

    def fake_run(cmd, **kwargs):
        return type("R", (), {"returncode": 0, "stdout": payload})()

    monkeypatch.setattr(harness.subprocess, "run", fake_run)
    assert harness.vram_free_gib("0") == pytest.approx(40.0, abs=0.01)
    assert harness.vram_free_gib("1") == pytest.approx(0.0, abs=0.01)
    assert harness.vram_free_gib(None) == pytest.approx(40.0, abs=0.01)


def test_vram_free_gib_returns_none_when_unavailable(harness, monkeypatch) -> None:
    def fake_run(cmd, **kwargs):
        raise OSError("rocm-smi missing")

    monkeypatch.setattr(harness.subprocess, "run", fake_run)
    assert harness.vram_free_gib("0") is None

    def fake_bad_json(cmd, **kwargs):
        return type("R", (), {"returncode": 0, "stdout": "not json"})()

    monkeypatch.setattr(harness.subprocess, "run", fake_bad_json)
    assert harness.vram_free_gib("0") is None


def test_harness_guards_against_a_device_still_held_by_another_process(harness) -> None:
    """A phase measured with no VRAM headroom must be skipped, not published."""

    source = (REPO_ROOT / "scripts" / "gguf_server_cwidth_probe.py").read_text(
        encoding="utf-8"
    )
    assert "insufficient_vram" in source
    assert "min_vram_free_gib" in source
    # The guard must run before the server is launched for that width.
    guard = source.index("insufficient_vram")
    launch = source.index("srv = launch_server(width)")
    assert guard < launch
