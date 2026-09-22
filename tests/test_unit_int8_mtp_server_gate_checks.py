"""The server gate's repeatability and isolation checks must be able to fail.

The INT8 MTP server gate (`scripts/int8_mtp_server_gate.py`) runs the public
HTTP matrix against a live server, so it cannot run here. Its *checks* can,
though: this module drives the real script against a fake server that speaks
the same wire shape, and asserts that each new check fires on a violation.

That distinction matters. A determinism check that only ever passes is
indistinguishable from no check at all, and the whole point of the acceptance's
"identical ids across two schedules" and "isolation check with a neighbouring
request" is to catch state that leaks between requests. So each check is
exercised in both directions:

- a well-behaved server passes the gate;
- a server whose ids depend on how many requests it has served fails the
  repeatability check;
- a server whose ids are contaminated by a concurrent request fails the
  isolation check.

The fake server is deliberately minimal: it implements only what the gate
reads, so a change to the gate's wire expectations shows up here as a failure
rather than as silent agreement.
"""

from __future__ import annotations

import http.server
import json
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "scripts" / "int8_mtp_server_gate.py"

_KV_LAYOUT = {
    "storage_dtype": "int8_per_token_head",
    "scale_dtype": "fp32",
    "kv_attention_source": "int8_direct",
    "persistent_bf16_mirror_bytes": 0,
}


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class _FakeServer(http.server.ThreadingHTTPServer):
    """A minimal OpenAI-shaped server whose id behaviour is configurable."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, mode: str) -> None:
        super().__init__(("127.0.0.1", 0), _FakeHandler)
        self.mode = mode
        self.visits: dict[str, int] = {}
        self.in_flight = 0
        # Latched once two requests are ever in flight together, and read after
        # the handler's overlap window closes, so both parties of an overlap see
        # the contamination. Keying off the instantaneous count instead made
        # this a race: whichever request woke first saw it and the other did not.
        self.leaked = False
        self.lock = threading.Lock()

    @property
    def base_url(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{host}:{port}"

    def next_ids(self, prompt: str) -> list[int]:
        with self.lock:
            visits = self.visits.get(prompt, 0)
            self.visits[prompt] = visits + 1
            leaked = self.leaked
        if self.mode == "nondeterministic":
            # Stable inside one pass over a prompt (the gate runs an AR/MTP/
            # stream triple, which must agree) but different on the next pass,
            # so only the second-schedule comparison can catch it.
            return [100 + visits // 3, 200 + visits // 3]
        if self.mode == "leak" and leaked:
            # Contaminated by whatever else was in flight.
            return [999, 999]
        # Deterministic in the prompt, so any schedule reproduces it.
        digest = sum(ord(char) for char in prompt)
        return [digest % 1000, (digest * 7) % 1000]


class _FakeHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:  # keep the test output quiet
        pass

    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - http.server naming
        if self.path != "/ready":
            self._json({"error": {"message": "not found"}}, status=404)
            return
        self._json(
            {
                "ready": True,
                "model": {
                    "kv_capability": {
                        "effective_kv_storage": "int8_per_token_head",
                        "diagnostic_override": None,
                    }
                },
                "startup": {"eager_load": False},
                "queue": {"active_requests": 0, "depth": 0},
            }
        )

    def do_POST(self) -> None:  # noqa: N802 - http.server naming
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        prompt = str(payload.get("prompt") or json.dumps(payload.get("messages", [])))
        speculative = bool(payload.get("speculative_mtp"))
        streamed = bool(payload.get("stream"))

        server: _FakeServer = self.server  # type: ignore[assignment]
        with server.lock:
            server.in_flight += 1
            if server.in_flight > 1:
                server.leaked = True
        try:
            # Hold the request briefly so a concurrent neighbour genuinely
            # overlaps; without this the isolation check could race past.
            time.sleep(0.05)
            ids = server.next_ids(prompt)
        finally:
            with server.lock:
                server.in_flight -= 1

        choice = {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "ok"},
            "hipengine": {
                "generated_token_ids": ids,
                "diagnostics": {"kv_layout": dict(_KV_LAYOUT)},
                "timing": {"mtp_cycles_count": 3 if speculative else 0},
            },
        }
        usage = {
            "prompt_tokens": 5,
            "completion_tokens": len(ids),
            "total_tokens": 5 + len(ids),
        }

        if not streamed:
            self._json({"choices": [choice], "usage": usage})
            return

        events = [
            {"choices": [{"index": 0, "delta": {"content": "ok"}}]},
            {"choices": [choice]},
            {"choices": [], "usage": usage},
        ]
        body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
        body += "data: [DONE]\n\n"
        raw = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture()
def fake_server():
    servers: list[_FakeServer] = []
    threads: list[threading.Thread] = []

    def start(mode: str) -> _FakeServer:
        server = _FakeServer(mode)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append(server)
        threads.append(thread)
        return server

    try:
        yield start
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=5.0)


def _run_gate(server: _FakeServer, tmp_path: Path, *, extra: list[str] | None = None):
    report_path = tmp_path / "report.json"
    completed = subprocess.run(
        [
            sys.executable, str(GATE),
            "--base-url", server.base_url,
            "--limit", "2",
            "--timeout", "30",
            "--json", str(report_path),
            *(extra or []),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    return completed, report


def test_gate_passes_against_a_well_behaved_server(fake_server, tmp_path: Path) -> None:
    """The positive control, and the determinism checks' happy path."""

    server = fake_server("ok")
    completed, report = _run_gate(server, tmp_path)

    assert completed.returncode == 0, completed.stderr[-2000:]
    assert report["passed"] is True
    assert report["determinism"]["skipped"] is False
    assert report["determinism"]["second_schedule"] == "reversed"
    assert report["determinism"]["repeatability_rows"] == 2
    assert report["determinism"]["isolation"]["ids"]


def test_gate_fails_when_ids_depend_on_request_history(fake_server, tmp_path: Path) -> None:
    """The repeatability check must fire on a second-schedule mismatch.

    This is the failure the acceptance's "identical ids across two schedules"
    is aimed at: state that leaks forward between requests.
    """

    server = fake_server("nondeterministic")
    completed, report = _run_gate(server, tmp_path)

    assert completed.returncode == 1
    assert report["passed"] is False
    assert "repeatability" in report["error"]


def test_gate_fails_when_a_neighbour_contaminates_the_target(
    fake_server, tmp_path: Path
) -> None:
    """The isolation check must fire when concurrent requests interfere."""

    server = fake_server("leak")
    completed, report = _run_gate(server, tmp_path)

    assert completed.returncode == 1
    assert report["passed"] is False
    assert "isolation" in report["error"]


def test_skip_determinism_opts_out_and_says_so(fake_server, tmp_path: Path) -> None:
    """A diagnostic run may skip the checks, but the report must record that.

    A skipped check that left the report looking complete would let a
    determinism-free run be read as full evidence.
    """

    server = fake_server("nondeterministic")
    completed, report = _run_gate(server, tmp_path, extra=["--skip-determinism"])

    assert completed.returncode == 0, completed.stderr[-2000:]
    assert report["passed"] is True
    assert report["determinism"] == {"skipped": True}
    assert report["determinism_required"] is False
