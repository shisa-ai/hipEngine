"""Contracts for prepared HTTP phase boundaries and native denominators."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts import qwen4exp_http_mtp_economics as http


def _body():
    phases = {
        "schema": 1, "complete": True, "prompt_tokens": 101,
        "cached_prompt_tokens": 100, "new_prompt_tokens": 1,
        "generated_tokens": 5, "decode_transitions": 4,
        "prefill_wall_ms": 10.0, "decode_wall_ms": 200.0,
        "request_wall_ms": 215.0, "cache_restore_ms": 2.0,
    }
    return {
        "choices": [{"hipengine": {"diagnostics": {"request_phases": phases}}}],
        "usage": {"prompt_tokens": 101, "completion_tokens": 5,
                  "prompt_tokens_details": {"cached_tokens": 100}},
        "hipengine": {"token_accounting": {
            "choice_generated_token_ids": [[1, 2, 3, 4, 5]],
            "choice_generated_tokens": [5], "total_generated_tokens": 5,
        }},
    }


def test_native_phase_rates_keep_outputs_and_transitions_distinct():
    rates = http.native_phase_metrics(_body())
    assert rates["decode_output_tok_s"] == 25.0
    assert rates["decode_transition_tok_s"] == 20.0
    assert rates["prefill_new_tok_s"] == 100.0
    assert rates["counts"]["cached_prompt_tokens"] == 100


@pytest.mark.parametrize("key,value", [
    ("cached_prompt_tokens", 99), ("new_prompt_tokens", 2),
    ("prompt_tokens", 102), ("generated_tokens", 6), ("decode_transitions", 5),
    ("complete", False), ("decode_wall_ms", 0.0),
    ("decode_wall_ms", float("nan")), ("prefill_wall_ms", float("inf")),
    ("cache_restore_ms", -1), ("prompt_tokens", True), ("request_wall_ms", 199.0),
])
def test_phase_metadata_must_agree_with_native_accounting(key, value):
    body = _body()
    body["choices"][0]["hipengine"]["diagnostics"]["request_phases"][key] = value
    with pytest.raises(ValueError):
        http.native_phase_metrics(body)


def test_absent_phases_are_unavailable_not_zero_cost():
    body = _body()
    body["choices"][0].pop("hipengine")
    assert http.native_phase_metrics(body) is None


def test_cohort_wall_and_sum_of_request_rates_are_different():
    phase = http.native_phase_metrics(_body())
    result = http.summarize_phases([{"phase_metrics": phase}, {"phase_metrics": phase}])
    assert result["sum_request_decode_output_tok_s"] == 50.0
    assert result["sum_request_decode_transition_tok_s"] == 40.0
    assert http.summarize_phases([{"phase_metrics": phase}, {"phase_metrics": None}]) is None


def test_preparation_uses_same_route_and_prefix_without_final_token(monkeypatch):
    original = [{"id": "a", "category": "code", "prompt_token_ids": [1, 2, 3]},
                {"id": "b", "category": "prose", "prompt_token_ids": [4, 5, 6, 7]}]
    frozen = deepcopy(original)
    seen = []

    def one(client, url, model, row, mtp, tokens):
        seen.append((row["prompt_token_ids"], mtp, tokens))
        return {"status": 200, "speculative_requested": mtp, "speculative_used": mtp}

    monkeypatch.setattr(http, "_one", one)
    rows, preparation = http.prepare_wave_rows(None, "http://localhost", "m", original, [False, True])
    assert rows == original == frozen
    assert seen == [([1, 2], False, 1), ([4, 5, 6], True, 1)]
    assert len(preparation) == 2


def test_preparation_requires_real_route_execution(monkeypatch):
    monkeypatch.setattr(http, "_one", lambda *args: {"status": 400})
    with pytest.raises(ValueError, match="prepar"):
        http.prepare_wave_rows(None, "http://localhost", "m",
                               [{"id": "a", "prompt_token_ids": [1, 2]}], [False])


def test_chat_preparation_uses_server_rendering_and_exact_ids(monkeypatch):
    calls = []
    def post(url, *, json):
        calls.append((url, json))
        body = ({"text": "<user>hello</user>", "token_count": 3}
                if url.endswith("count_tokens") else {"token_ids": [1, 2, 3], "token_count": 3})
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: body)
    monkeypatch.setattr(http, "_one", lambda *args: {
        "status": 200, "speculative_requested": False, "speculative_used": False,
    })
    rows, _ = http.prepare_wave_rows(SimpleNamespace(post=post), "http://localhost", "m",
                                    [{"id": "a", "messages": [{"role": "user", "content": "hello"}]}], [False])
    assert rows[0]["prompt_token_ids"] == [1, 2, 3]
    assert calls[0][1]["enable_thinking"] is False
    assert calls[1][1] == {"text": "<user>hello</user>"}


@pytest.mark.parametrize("ids", [[], [1], [1, True], [1, -2], [1, 2.0]])
def test_preparation_rejects_unusable_token_rows(ids):
    with pytest.raises(ValueError, match="token"):
        http.prepare_wave_rows(None, "http://localhost", "m", [{"id": "a", "prompt_token_ids": ids}], [False])


@pytest.mark.parametrize("cached,new", [(0, 101), (99, 2), (101, 0)])
def test_prepared_measurement_refuses_cache_misses_or_wrong_checkpoint(cached, new):
    result = {"phase_metrics": {"counts": {"prompt_tokens": 101,
              "cached_prompt_tokens": cached, "new_prompt_tokens": new}}}
    with pytest.raises(ValueError, match="prepared"):
        http.require_prepared_hit(result, expected_prompt_tokens=101)


def test_prepared_measurement_accepts_the_declared_single_token_suffix():
    http.require_prepared_hit({"phase_metrics": {"counts": {"prompt_tokens": 101,
        "cached_prompt_tokens": 100, "new_prompt_tokens": 1}}}, expected_prompt_tokens=101)


def test_missing_phase_telemetry_cannot_claim_preparation():
    with pytest.raises(ValueError, match="prepared"):
        http.require_prepared_hit({"phase_metrics": None}, expected_prompt_tokens=101)


def test_http_refuses_short_completion_even_with_consistent_usage():
    body = _body()
    client = SimpleNamespace(post=lambda *args, **kwargs: SimpleNamespace(status_code=200, json=lambda: body))
    with pytest.raises(ValueError, match="requested 6 output tokens, received 5"):
        http._one(client, "http://localhost", "m", {"id": "a", "messages": []}, False, 6)


def test_prefix_reply_can_use_natural_eos_without_weakening_measurement_counts():
    body = _body()
    payloads = []
    def post(*args, **kwargs):
        payloads.append(kwargs["json"])
        return SimpleNamespace(status_code=200, json=lambda: body)
    client = SimpleNamespace(post=post)
    result = http._one(client, "http://localhost", "m", {"id": "prefix", "messages": []}, False, 8,
                       ignore_eos=False, require_max_tokens=False)
    assert payloads[0]["ignore_eos"] is False
    assert result["completion_tokens"] == 5
    assert result["diagnostics"] == body["choices"][0]["hipengine"]["diagnostics"]
    with pytest.raises(ValueError, match="requested 8"):
        http._one(client, "http://localhost", "m", {"id": "measured", "messages": []}, False, 8,
                  ignore_eos=False)
    with pytest.raises(ValueError, match="requested 4"):
        http._one(client, "http://localhost", "m", {"id": "overflow", "messages": []}, False, 4,
                  require_max_tokens=False)


def test_prepared_wave_finishes_all_preparation_before_any_measurement(monkeypatch):
    import httpx

    calls = []
    class Client:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def close(self):
            pass
    monkeypatch.setattr(httpx, "Client", Client)
    def one(client, url, model, row, flag, tokens):
        calls.append((row["id"], tokens))
        return {"id": row["id"], "status": 200, "completion_tokens": tokens, "wall_s": 1.0,
                "speculative_requested": flag, "speculative_used": flag,
                "phase_metrics": http.native_phase_metrics(_body())}
    monkeypatch.setattr(http, "_one", one)
    rows = [{"id": name, "prompt_token_ids": list(range(101))} for name in ("a", "b")]
    result = http._wave(rows, "http://localhost", "m", [False, True], 5, 1.0, prepared=True)
    assert calls[:2] == [("a", 1), ("b", 1)]
    assert sorted(calls[2:]) == [("a", 5), ("b", 5)]
    assert result["completion_tokens"] == 10  # preparation's two outputs do not enter this count
    assert len(result["preparation"]) == 2
    assert result["phase_rates"]["sum_request_decode_output_tok_s"] == 50.0


def test_cli_counterbalances_repeats_and_retains_failed_preparation(monkeypatch, tmp_path):
    import json
    from scripts import qwen4exp_mtp_inprocess_bench as bench

    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"id": "a", "prompt_token_ids": [1, 2, 3]}) + "\n")
    output = tmp_path / "output.json"
    calls = []
    monkeypatch.setattr(bench, "source_snapshot", lambda: {"head": "fixed"})
    def wave(rows, url, model, flag, tokens, timeout, **options):
        calls.append((flag, options))
        if flag:
            raise ValueError("preparation cache miss")
        return {"wall_s": 1.0, "completion_tokens": 5, "cases": [{
            "status": 200, "speculative_requested": False, "speculative_used": False,
        }]}
    monkeypatch.setattr(http, "_wave", wave)
    assert http.main(["--prompts", str(prompts), "--output", str(output), "--prepared", "--no-warmup"]) == 1
    assert [flag for flag, _ in calls] == [False, True, True, False, False, True]
    assert all(options == {"prepared": True} for _, options in calls)
    artifact = json.loads(output.read_text())
    assert artifact["preparation"] == "prefix_minus_one"
    assert artifact["repetitions"] == 3
    assert artifact["status"] == "invalid" and artifact["ratios"] == {}
    assert artifact["waves"]["mtp"][0]["cases"][0]["error"] == "preparation cache miss"
