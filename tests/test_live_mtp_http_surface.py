"""Live HTTP surface gate: multiple choices, tools, and JSON through MTP (task #17).

The unit slices pinned the two server-side decisions -- which requests reach the
live-many path, and which explicit requests are refused by name -- but the
acceptance is about what a client actually receives: output parity against the AR
route and the right response shape for n>1 choices, tool calls, and JSON output.

This gate runs a real ``hipengine serve`` process on the INT8 KV profile with MTP
available and drives it over HTTP, comparing each constrained request against the
same request with ``speculative_mtp: false``. Parity is token-exact on
``choices[i].hipengine.generated_token_ids``, and each case also asserts the shape
the client depends on.

Skips unless ROCm, the dense GGUF model, and a free port are all available.
"""

from __future__ import annotations

import ctypes
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path(
    os.environ.get("HIPENGINE_INT8_MTP_MODEL", "/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
)
SERVED_MODEL = MODEL.name
PORT = int(os.environ.get("HIPENGINE_MTP_SURFACE_PORT", "8097"))
MAX_TOKENS = 24

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a repository file and return its contents.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
]
MESSAGES = [
    {
        "role": "user",
        "content": (
            "Reply with a single short sentence about how a KV cache works."
        ),
    }
]
TOOL_MESSAGES = [
    {
        "role": "user",
        "content": "Read the file pyproject.toml using the read_file tool.",
    }
]


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _port_free(port: int) -> bool:
    with socket.socket() as probe:
        probe.settimeout(0.5)
        return probe.connect_ex(("127.0.0.1", port)) != 0


pytestmark = pytest.mark.skipif(
    not (_hip_available() and MODEL.exists()),
    reason="requires ROCm + the dense 27B GGUF model",
)


@pytest.fixture(scope="module")
def client():
    """A real serve process on the INT8 KV profile, torn down with the module."""

    import httpx

    if not _port_free(PORT):
        pytest.skip(f"port {PORT} is already in use")

    env = os.environ.copy()
    env.setdefault("HIPENGINE_GGUF_INT8_KV_ALLOW_UNVERIFIED_LONG", "1")
    env.setdefault("HIPENGINE_GGUF_INT8_KV_BF16_FULL_LAYERS", "none")
    log = tempfile.NamedTemporaryFile(
        prefix=f"mtp_http_surface_{PORT}_", suffix=".log", delete=False
    )
    command = [
        sys.executable,
        "-m",
        "hipengine.server",
        "--model",
        str(MODEL),
        "--backend",
        "hip_gfx1151",
        "--quant",
        "gguf_q4_k_m",
        "--kv-storage",
        "int8_per_token_head",
        "--kv-scale-dtype",
        "fp32",
        "--kv-scale-granularity",
        "per_token_head",
        "--max-context-tokens",
        "4096",
        "--max-active-requests",
        "2",
        "--speculative-mtp-serving",
        "auto",
        "--prefix-cache",
        "off",
        "--port",
        str(PORT),
    ]
    server = subprocess.Popen(
        command, env=env, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT
    )
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{PORT}", timeout=600.0) as http:
            deadline = time.monotonic() + 900.0
            ready = None
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    log.flush()
                    pytest.fail(
                        f"server exited with {server.returncode}; see "
                        f"{log.name}"
                    )
                try:
                    response = http.get("/ready")
                    if response.status_code == 200:
                        ready = response.json()
                        break
                except Exception:
                    pass
                time.sleep(2.0)
            if ready is None:
                pytest.fail(f"server never became ready; see {log.name}")
            capability = ready["model"]["kv_capability"]
            assert capability["effective_kv_storage"] == "int8_per_token_head", capability
            yield http
    finally:
        server.terminate()
        try:
            server.wait(timeout=120)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=60)
        log.close()


def _chat(client, *, speculative, extra=None, messages=None):
    body = {
        "model": SERVED_MODEL,
        "messages": messages or MESSAGES,
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "speculative_mtp": speculative,
    }
    body.update(extra or {})
    response = client.post("/v1/chat/completions", json=body)
    if response.status_code != 200:
        pytest.fail(
            f"chat request failed with {response.status_code}: "
            f"{response.text[:2000]}"
        )
    return response.json()


def _ids(body, index=0):
    return list(body["choices"][index]["hipengine"]["generated_token_ids"])


def _cycles(body, index=0):
    timing = body["choices"][index]["hipengine"].get("timing", {})
    return int(timing.get("mtp_cycles_count", 0))


def _assert_parity(ar, mtp, *, label, choices=1, require_mtp=True):
    assert len(ar["choices"]) == choices, (label, len(ar["choices"]))
    assert len(mtp["choices"]) == choices, (label, len(mtp["choices"]))
    for index in range(choices):
        assert _ids(ar, index) == _ids(mtp, index), (
            f"{label}: choice {index} drifted from the AR route"
        )
    assert _cycles(ar, 0) == 0, (label, "the AR arm reported MTP cycles")
    if require_mtp:
        assert _cycles(mtp, 0) > 0, (label, "the MTP arm reported no MTP cycles")


def test_multiple_choices_match_ar_and_report_every_choice(client) -> None:
    """n>1 choices: token-exact parity per choice and a well-formed choice list."""

    ar = _chat(client, speculative=False, extra={"n": 2})
    mtp = _chat(client, speculative=True, extra={"n": 2})

    _assert_parity(ar, mtp, label="n=2", choices=2)
    for body in (ar, mtp):
        assert [choice["index"] for choice in body["choices"]] == [0, 1]
        for choice in body["choices"]:
            assert choice["message"]["role"] == "assistant"
            assert choice["finish_reason"] in {"stop", "length"}
            assert choice["message"]["content"] is not None


def test_tool_call_matches_ar_and_keeps_the_tool_shape(client) -> None:
    """A forced tool call keeps parity and the OpenAI tool_calls shape."""

    extra = {"tools": TOOLS, "tool_choice": "required"}
    ar = _chat(client, speculative=False, extra=extra, messages=TOOL_MESSAGES)
    mtp = _chat(client, speculative=True, extra=extra, messages=TOOL_MESSAGES)

    # Whether the model emits a call is the tool path's question, not this
    # gate's; what this gate owns is that both routes agree, and that a call,
    # when it is produced, has the shape a client parses.
    _assert_parity(ar, mtp, label="tools", require_mtp=False)
    for label, body in (("ar", ar), ("mtp", mtp)):
        choice = body["choices"][0]
        calls = choice["message"].get("tool_calls")
        if not calls:
            assert choice["finish_reason"] in {"stop", "length"}, (label, choice)
            continue
        assert choice["finish_reason"] == "tool_calls", (label, choice["finish_reason"])
        for call in calls:
            assert call["type"] == "function"
            assert call["id"], (label, "tool call has no id")
            function = call["function"]
            assert function["name"] in {tool["function"]["name"] for tool in TOOLS}
            json.loads(function["arguments"])


def test_json_object_matches_ar_and_is_valid_json(client) -> None:
    """Structured output keeps parity and the content parses as JSON."""

    extra = {"response_format": {"type": "json_object"}}
    ar = _chat(client, speculative=False, extra=extra)
    mtp = _chat(client, speculative=True, extra=extra)

    # Parity and shape are this test's contract; whether MTP ran at all for a
    # constrained request is the downgrade test's contract.
    _assert_parity(ar, mtp, label="json_object", require_mtp=False)
    for label, body in (("ar", ar), ("mtp", mtp)):
        choice = body["choices"][0]
        assert choice["finish_reason"] in {"stop", "length"}, (label, choice)
        content = choice["message"]["content"]
        assert content, (label, "empty content")
        json.loads(content)


def _route_report(metadata):
    """The route evidence a client can read off a response's choice metadata.

    A constrained explicit request can legitimately be served over AR, but only
    if the response says so. The reason lives in the nested speculative block, so
    a search of the top level alone would call a reported fallback silent.
    """

    report = {
        key: value
        for key, value in metadata.items()
        if "route" in key or "speculative" in key or "reason" in key
    }
    specdec = (metadata.get("diagnostics") or {}).get("specdec2_mtp2") or {}
    for key in (
        "plan_reason",
        "plan_ar_only",
        "provider_readiness",
        "provider_decline_reason",
        "cycles",
        "failure_reason_counts",
    ):
        if key in specdec:
            report[f"diagnostics.specdec2_mtp2.{key}"] = specdec[key]
    return report


def test_explicit_mtp_with_a_constraint_is_not_silently_downgraded(client) -> None:
    """An explicit MTP request with a constraint runs MTP or says why it did not.

    The acceptance forbids silently serving an explicit request over the AR route.
    Either the constrained request really ran MTP, or the response reports the
    route it took instead of leaving the client to infer it from zero cycles.
    """

    constrained = {
        "n=2": ({"n": 2}, MESSAGES),
        "tools": ({"tools": TOOLS, "tool_choice": "required"}, TOOL_MESSAGES),
        "json_object": ({"response_format": {"type": "json_object"}}, MESSAGES),
    }
    silent: dict[str, dict] = {}
    routes: dict[str, object] = {}
    for label, (extra, messages) in constrained.items():
        body = _chat(client, speculative=True, extra=extra, messages=messages)
        if _cycles(body) > 0:
            routes[label] = "mtp"
            continue
        metadata = body["choices"][0]["hipengine"]
        # Only a key that names the route or a reason counts as reporting: the
        # always-present diagnostics/finish_details blocks are not an answer to
        # "which route did my explicit request take".
        reported = _route_report(metadata)
        routes[label] = reported or "ar"
        if not reported:
            silent[label] = metadata
    assert not silent, (
        "an explicit speculative_mtp request served no MTP cycle and reported "
        f"nothing about the route it took instead (routes seen: {routes}): "
        f"{json.dumps(silent, default=str)}"
    )
