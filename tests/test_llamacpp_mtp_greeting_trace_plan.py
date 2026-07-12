from __future__ import annotations

from scripts.llamacpp_mtp_greeting_trace_plan import build_trace_plan


def test_greeting_trace_plan_uses_verbose_draft_mtp_flags() -> None:
    plan = build_trace_plan()
    cmd = plan["server_command"]

    assert "--spec-type" in cmd
    assert "draft-mtp" in cmd
    assert "--spec-draft-n-max" in cmd
    assert cmd[cmd.index("--spec-draft-n-max") + 1] == "2"
    assert "--no-spec-draft-backend-sampling" in cmd
    assert "--reasoning" in cmd
    assert cmd[cmd.index("--reasoning") + 1] == "off"
    assert "--log-verbosity" in cmd
    assert cmd[cmd.index("--log-verbosity") + 1] == "5"
    assert "--no-log-prefix" in cmd
    assert "--no-log-timestamps" in cmd


def test_greeting_trace_plan_request_and_parser_match_needed_oracle() -> None:
    plan = build_trace_plan()

    assert plan["request_endpoint"].endswith("/v1/chat/completions")
    assert plan["request_payload"]["messages"] == [
        {"role": "user", "content": "Write a short greeting."}
    ]
    assert plan["request_payload"]["temperature"] == 0.0
    assert plan["request_payload"]["top_k"] == 1
    assert plan["metadata"]["request"]["top_k_candidates"] == 10
    assert plan["metadata"]["request"]["reasoning"] == "off"
    assert plan["metadata"]["native_reference_artifact"].endswith(
        "mtp-bench-1781845600-b2-greeting-topk-diagnostic.json"
    )
    parser_cmd = plan["parser_command"]
    assert parser_cmd[:2] == ["python3", "scripts/llamacpp_mtp_draft_trace.py"]
    assert "--top-k" in parser_cmd
    assert parser_cmd[parser_cmd.index("--top-k") + 1] == "10"
