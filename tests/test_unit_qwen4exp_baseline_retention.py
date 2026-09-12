import copy
import json

import pytest

from scripts.qwen4exp_baseline_retention import compact_capture, digest, validate_samples, variance_report
from scripts.qwen4exp_canonical_ar_bench import DEFAULT_FIXTURE, load_fixture, token_ids_sha256
from scripts.qwen4exp_framework_family_refresh import HOST_ID, PIN


def samples():
    fixture, _ = load_fixture(DEFAULT_FIXTURE)
    return [
        dict(case_id=c["id"], category=c["category"], prompt_tokens=c["prompt_tokens"],
             prompt_token_ids_sha256=token_ids_sha256(c["prompt_token_ids"]),
             repetition=r, prefill_ms=100., decode_ms=200.,
             prefill_tok_s=c["prompt_tokens"]*10, decode_tok_s=640.,
             decode_transitions=128, client_wall_s=.3, output_token_count=129,
             output_token_ids=[1]*129, output_token_ids_sha256=token_ids_sha256([1]*129))
        for c in fixture["cases"] for r in range(3)
    ]


def test_complete_samples():
    assert validate_samples(samples())["all_cases_deterministic"]


def test_variance_keeps_workloads_separate():
    rows = samples()
    report = variance_report(rows, validate_samples(rows))
    assert report["prefill_tok_s"]["max_within_case_cv"] == 0
    rows[0]["prefill_ms"] = 200.
    rows[0]["prefill_tok_s"] /= 2
    report = variance_report(rows, validate_samples(rows))
    case = rows[0]["case_id"]
    assert report["prefill_tok_s"]["cases"][case]["last_vs_first_percent"] == 100.
    assert report["prefill_tok_s"]["cases_above_two_percent_cv"] == [case]
    assert report["decode_tok_s"]["max_within_case_cv"] == 0


@pytest.mark.parametrize("failure", ["missing", "duplicate", "prompt", "output_hash",
                                    "transitions", "time", "nondeterministic"])
def test_reject_invalid_samples(failure):
    rows = copy.deepcopy(samples())
    if failure == "missing":
        rows.pop()
    elif failure == "duplicate":
        rows[1] = rows[0].copy()
    elif failure == "prompt":
        rows[0]["prompt_token_ids_sha256"] = "wrong"
    elif failure == "output_hash":
        rows[0]["output_token_ids_sha256"] = "wrong"
    elif failure == "transitions":
        rows[0]["decode_transitions"] = 127
    elif failure == "time":
        rows[0]["prefill_ms"] = float("nan")
    else:
        rows[0]["output_token_ids"] = [2]*129
        rows[0]["output_token_ids_sha256"] = token_ids_sha256([2]*129)
    with pytest.raises(ValueError):
        validate_samples(rows)


@pytest.mark.parametrize("failure", [None, "hash", "host", "source", "leak", "kv", "server"])
def test_capture_identity_and_lifecycle(tmp_path, failure):
    _, fixture_hash = load_fixture(DEFAULT_FIXTURE)
    source = dict(head="hipengine-test", tracked_clean=True)
    host = dict(machine_id=HOST_ID)
    controller = dict(status="captured", source=source, host=host, stages=[],
                      binary_hashes={"server": "binary"})
    for label in ("hipengine", "halo-box-vulkan", "halo-box-hip"):
        raw = dict(
            engine=label, status="completed", source=source, host=host,
            fixture_sha256=fixture_hash, samples=samples(), protocol={},
        )
        if label == "hipengine":
            raw.update(profile=dict(requested="production", fell_back_to_strict=False),
                       memory_after_close=dict(active_allocations=0, current_allocated_bytes=0))
            if failure == "leak":
                raw["memory_after_close"]["active_allocations"] = 1
        else:
            raw.update(source=dict(head=PIN, tracked_clean=True), server_returncode=0,
                       server_binary_sha256="binary", command=["server","-ctk","bf16","-ctv","bf16"])
            if failure == "host":
                raw["host"] = dict(machine_id="another-host")
            if failure == "source":
                raw["source"]["head"] = "wrong"
            if failure == "kv":
                raw["command"][-1] = "f16"
            if failure == "server":
                raw["server_returncode"] = 1
        path = tmp_path / (label + ".json")
        path.write_text(json.dumps(raw))
        controller["stages"].append(dict(
            engine=label, path=str(path), sha256=digest(path), elapsed_seconds=1.))
    if failure == "hash":
        controller["stages"][0]["sha256"] = "wrong"
    path = tmp_path / "controller.json"
    path.write_text(json.dumps(controller))
    if failure:
        with pytest.raises(ValueError):
            compact_capture(path)
    else:
        packet = compact_capture(path)
        assert len(packet["engines"]) == 3
        assert packet["vulkan_over_hipengine"]["4096"]["prefill_tok_s_weighted"] == 1.
