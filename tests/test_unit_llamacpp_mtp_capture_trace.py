from __future__ import annotations

import json
from pathlib import Path

from scripts.llamacpp_mtp_capture_trace import (
    health_endpoint_for_request,
    load_plan,
    response_path_for_plan,
    write_plan_metadata,
)


def test_health_endpoint_for_chat_completions_request() -> None:
    assert (
        health_endpoint_for_request("http://127.0.0.1:8093/v1/chat/completions")
        == "http://127.0.0.1:8093/health"
    )


def test_trace_capture_plan_paths_and_metadata(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    metadata_path = tmp_path / "meta.json"
    trace_path = tmp_path / "trace.json"
    plan = {
        "metadata_path": str(metadata_path),
        "metadata": {"request": {"prompt_name": "greeting"}},
        "trace_json": str(trace_path),
    }
    plan_path.write_text(json.dumps(plan))

    loaded = load_plan(plan_path)
    written = write_plan_metadata(loaded)

    assert written == metadata_path
    assert json.loads(metadata_path.read_text()) == plan["metadata"]
    assert response_path_for_plan(loaded) == tmp_path / "trace.response.json"
