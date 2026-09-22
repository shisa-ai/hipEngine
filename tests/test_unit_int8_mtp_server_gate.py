import json
from copy import deepcopy

import pytest

from scripts.int8_mtp_server_gate import (
    assert_result,
    assert_token_exact,
    stream_result,
)
from scripts.int8_mtp_prefix_gate import assert_restored_pair


def _events(*, cycles=1, mirror=0):
    metadata = {
        "generated_token_ids": [1, 2],
        "timing": {"mtp_cycles_count": cycles},
        "diagnostics": {"kv_layout": {
            "storage_dtype": "int8_per_token_head", "scale_dtype": "fp32",
            "kv_attention_source": "bf16_mirror" if mirror else "int8_direct",
            "persistent_bf16_mirror_bytes": mirror,
        }},
    }
    return [
        "data: " + json.dumps({"choices": [{"finish_reason": "length", "hipengine": metadata}]}),
        "data: " + json.dumps({"choices": [], "usage": {
            "prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5,
        }}),
        "data: [DONE]",
    ]


def test_server_gate_accepts_complete_compact_int8_stream():
    result = stream_result(_events())
    assert_result(result, speculative=True, compact=True)
    assert result["ids"] == [1, 2]


@pytest.mark.parametrize("mutation", ["no_done", "no_usage", "duplicate_finish", "after_done", "error"])
def test_server_gate_rejects_broken_stream(mutation):
    events = _events()
    if mutation == "no_done":
        events.pop()
    elif mutation == "no_usage":
        events.pop(1)
    elif mutation == "duplicate_finish":
        events.insert(1, events[0])
    elif mutation == "after_done":
        events.append(events[0])
    else:
        events.insert(0, 'data: {"error":{"code":"generation_failed"}}')
    with pytest.raises(AssertionError):
        stream_result(events)


def test_server_gate_cannot_mistake_mirror_or_ar_for_compact_mtp():
    with pytest.raises(AssertionError):
        assert_result(stream_result(_events(mirror=256)), speculative=True, compact=True)
    with pytest.raises(AssertionError):
        assert_result(stream_result(_events(cycles=0)), speculative=True, compact=True)
    with pytest.raises(AssertionError):
        assert_result(stream_result(_events(cycles=1)), speculative=False, compact=True)


def test_server_gate_reports_first_token_exactness_mismatch():
    with pytest.raises(AssertionError, match="token_exactness_mismatch") as excinfo:
        assert_token_exact([10, 11, 12], [10, 13, 12], prompt="code_lru_cache", endpoint="/v1/completions")
    assert excinfo.value.args[0]["index"] == 1
    assert excinfo.value.args[0]["reference_token"] == 11
    assert excinfo.value.args[0]["candidate_token"] == 13


    result = stream_result(_events())
    result["usage"]["completion_tokens"] = 3
    with pytest.raises(AssertionError):
        assert_result(result, speculative=True, compact=True)


@pytest.mark.parametrize("failure", [None, "miss", "partial", "ar", "ids"])
def test_prefix_gate_requires_hit_restoration_and_matching_output(failure):
    baseline = {
        "generated_token_ids": [1, 2],
        "timing": {"mtp_cycles_count": 0},
        "diagnostics": {"prefix_cache": {"hit": True, "reused_tokens": 512}},
    }
    candidate = deepcopy(baseline)
    candidate["timing"]["mtp_cycles_count"] = 1
    if failure == "miss":
        candidate["diagnostics"]["prefix_cache"]["hit"] = False
    elif failure == "partial":
        candidate["diagnostics"]["prefix_cache"]["reused_tokens"] = 256
    elif failure == "ar":
        candidate["timing"]["mtp_cycles_count"] = 0
    elif failure == "ids":
        candidate["generated_token_ids"] = [3]
    if failure is None:
        assert assert_restored_pair(baseline, candidate, 512) == 1
    else:
        with pytest.raises(AssertionError):
            assert_restored_pair(baseline, candidate, 512)
