from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts import qwen4exp_http_mtp_economics as http
from scripts import qwen4exp_mtp_inprocess_bench as bench


class FakeRunner:
    def __init__(self):
        self.calls = []

    def reset(self):
        self.calls.append(("reset", {}))

    def prefill(self, prompt, **kwargs):
        self.calls.append(("prefill", kwargs))
        return SimpleNamespace(token_id=4)

    def step(self, token, **kwargs):
        self.calls.append(("step", kwargs))
        return SimpleNamespace(token_id=token + 1)


def test_ar_trajectory_uses_public_compact_capture_by_default():
    runner = FakeRunner()
    assert bench.run_ar(runner, [1, 2], 3, capture_mode="compact") == [4, 5, 6]
    assert runner.calls == [
        ("reset", {}),
        ("prefill", {"capture_logits": False, "capture_target_hidden": False}),
        ("step", {"capture_logits": False, "capture_target_hidden": False, "token_id_resident": True}),
        ("step", {"capture_logits": False, "capture_target_hidden": False, "token_id_resident": True}),
    ]


def test_ar_full_capture_is_an_explicit_diagnostic():
    runner = FakeRunner()
    assert bench.run_ar(runner, [1], 1, capture_mode="full") == [4]
    assert runner.calls == [("reset", {}), ("prefill", {})]


@pytest.mark.parametrize("tokens", [0, -1])
def test_ar_rejects_empty_output_budget_before_mutation(tokens):
    runner = FakeRunner()
    with pytest.raises(ValueError, match="positive"):
        bench.run_ar(runner, [1], tokens, capture_mode="compact")
    assert runner.calls == []


def test_pair_order_is_counterbalanced_across_prompts_and_repeats():
    orders = [bench.pair_order(case, rep) for case in range(10) for rep in range(3)]
    assert orders.count(("ar", "mtp")) == 15
    assert orders.count(("mtp", "ar")) == 15
    assert bench.pair_order(0, 0) != bench.pair_order(0, 1)


def test_pair_records_actual_tokens_and_full_phase_census():
    runner = FakeRunner()
    output = SimpleNamespace(
        generated_token_ids=(4, 5, 6),
        telemetry=SimpleNamespace(diagnostics={
            "proposed_draft_tokens": 2, "accepted_draft_tokens": 1,
            "draft_acceptance": 0.5, "target_verify": "state_rows",
            "phase_census": {"cycles": 1, "proposal": {"ms": 0.5}},
        }),
    )
    calls = []

    def mtp(request):
        calls.append("mtp")
        return [output]

    generator = SimpleNamespace(runner=runner, generate_speculative_mtp_detailed=mtp)
    request = SimpleNamespace(prompts=((1, 2),), max_tokens=3)
    ticks = iter([0.0, 2.0, 3.0, 7.0])
    row = bench.measure_pair(
        generator, request, case={"id": "p", "category": "code"},
        repetition=2, order=("mtp", "ar"), capture_mode="compact",
        clock=lambda: next(ticks),
    )
    assert calls == ["mtp"]
    assert row["order"] == ["mtp", "ar"]
    assert row["repetition"] == 2
    assert row["ar_seconds"] == 4.0 and row["mtp_seconds"] == 2.0
    assert row["ar_output_tokens"] == row["mtp_output_tokens"] == 3
    assert row["ar_token_ids"] == row["mtp_token_ids"] == [4, 5, 6]
    assert row["phase_census"] == output.telemetry.diagnostics["phase_census"]


def test_pair_refuses_a_short_mtp_completion():
    output = SimpleNamespace(generated_token_ids=(4,), telemetry=SimpleNamespace(diagnostics={}))
    generator = SimpleNamespace(
        runner=FakeRunner(), generate_speculative_mtp_detailed=lambda _: [output],
    )
    request = SimpleNamespace(prompts=((1,),), max_tokens=3)
    with pytest.raises(ValueError, match="MTP.*3.*1"):
        bench.measure_pair(generator, request, case={"id": "p", "category": "code"},
                           repetition=0, order=("mtp", "ar"), capture_mode="compact")


def test_summary_uses_actual_counts_and_reports_repeat_determinism():
    rows = [dict(id="p", category="code", ar_seconds=2.0, mtp_seconds=1.0,
                 ar_output_tokens=3, mtp_output_tokens=3, proposed=2, accepted=1,
                 ar_token_ids=[4, 5, 6], mtp_token_ids=[4, 5, 6], identical_to_ar=True)]
    rows.append({**rows[0], "mtp_token_ids": [4, 5, 7], "identical_to_ar": False})
    full = bench.summarize_rows(rows)["full"]
    assert full["ar_tok_s"] == 1.5 and full["mtp_tok_s"] == 3.0
    assert full["prompts"] == 1 and full["samples"] == 2
    assert full["ar_deterministic"] is True
    assert full["mtp_deterministic"] is False


def test_parser_defaults_to_compact_repeated_measurements():
    args = bench.build_parser().parse_args(["--output", "out.json"])
    assert args.capture_mode == "compact"
    assert args.warmups == 1 and args.repetitions == 3


@pytest.mark.parametrize("mtp", [False, True])
def test_http_pins_both_routes_and_retains_request_intent(mtp):
    payloads = []

    def post(url, *, json):
        from hipengine.server.api import ServerConfig, _validate_generation_request
        from hipengine.server.schemas import ChatCompletionRequest

        _validate_generation_request(ServerConfig(model="m"), ChatCompletionRequest(**json))
        payloads.append(json)
        return SimpleNamespace(status_code=200, json=lambda: {
            "choices": [{"finish_reason": "length"}],
            "usage": {"completion_tokens": 4, "prompt_tokens": 2},
            "hipengine": {
                "speculative_mtp": {"used": mtp},
                "token_accounting": {
                    "choice_generated_token_ids": [[1, 2, 3, 4]],
                    "choice_generated_tokens": [4], "total_generated_tokens": 4,
                },
            },
        })

    result = http._one(SimpleNamespace(post=post), "http://localhost", "m",
                       {"id": "p", "category": "code", "messages": [{"role": "user", "content": "Hi"}]}, mtp, 4)
    assert payloads[0]["speculative_mtp"] is mtp
    assert payloads[0]["ignore_eos"] is True
    assert result["speculative_requested"] is mtp


@pytest.mark.parametrize("requested,used", [(True, False), (False, True), (True, None), (False, None)])
def test_http_rejects_each_route_mismatch_or_missing_witness(requested, used):
    assert not http.route_matches_request({"status": 200, "speculative_requested": requested,
                                          "speculative_used": used})


def test_http_accepts_a_valid_mixed_cohort_per_request():
    assert all(http.route_matches_request({"status": 200, "speculative_requested": flag,
                                          "speculative_used": flag}) for flag in (True, False))
    assert not http.route_matches_request({"status": 500})


def test_source_snapshot_detects_edits_with_unchanged_dirty_status(monkeypatch):
    monkeypatch.setattr(bench, "_git_metadata", lambda _: {"head": "abc", "tracked_clean": False})
    diffs = iter([b"first version", b"second version"])
    monkeypatch.setattr(bench.subprocess, "check_output", lambda *a, **k: next(diffs))
    assert bench.source_snapshot() != bench.source_snapshot()


def test_http_requires_exact_ids_not_usage_alone():
    with pytest.raises(ValueError, match="exact"):
        http.exact_completion_count({"usage": {"completion_tokens": 3}})


@pytest.mark.parametrize("field,value", [
    ("total_generated_tokens", 4), ("choice_generated_tokens", [4]),
    ("choice_generated_token_ids", [[1, -2, 3]]),
])
def test_http_rejects_forged_token_denominators(field, value):
    accounting = {"choice_generated_token_ids": [[1, 2, 3]],
                  "choice_generated_tokens": [3], "total_generated_tokens": 3}
    accounting[field] = value
    with pytest.raises(ValueError):
        http.exact_completion_count({"usage": {"completion_tokens": 3},
                                     "hipengine": {"token_accounting": accounting}})


def test_http_invalid_route_suppresses_speedup_ratio(monkeypatch, tmp_path):
    path = tmp_path / "output.json"
    def wave(rows, url, model, mtp, max_tokens, timeout):
        return {"wall_s": 1.0, "completion_tokens": 2, "cases": [
            {"status": 200, "speculative_requested": mtp, "speculative_used": False},
        ]}
    monkeypatch.setattr(http, "_load", lambda _: [{"id": "p", "category": "code"}])
    monkeypatch.setattr(http, "_wave", wave)
    assert http.main(["--no-warmup", "--output", str(path)]) == 1
    import json
    artifact = json.loads(path.read_text())
    assert artifact["status"] == "invalid"
    assert artifact["ratios"] == {}
