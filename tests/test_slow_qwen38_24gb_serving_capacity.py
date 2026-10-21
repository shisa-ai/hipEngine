"""`hipengine serve` keeps its 24 GB context: no silent capacity regressions.

Every v0.5-v0.8 release regressed 24 GB serving without a failing test: the
capacity numbers came from harnesses and from 48 GB cards with a raised
reserve, and neither sees an allocation the sizing model does not price. This
test runs the real server on a real 24 GB-class GPU, the way a user does
(automatic context, default prefix cache and MTP), and fails when an automatic
context drops below its recorded floor or a near-full-depth prompt no longer
serves.

Run it on an RX 7900 XTX before a release (about one hour; the DMS case alone
prefills over 200K tokens):

    HIP_VISIBLE_DEVICES=1 uv run pytest tests/test_slow_qwen38_24gb_serving_capacity.py -v

It skips, naming why, when no 24 GB-class GPU is visible or the model is
absent. Floors are the automatic contexts recorded in
`benchmarks/results/2026-10-22-xtx-24gb-dms-clamp-hot-vocab.json`, less about
3-5% for driver and firmware variation. Raise a floor when capacity improves;
lowering one is a recorded product decision, never a test fix.
"""

from __future__ import annotations

import ctypes
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pytest

MODEL = Path(os.environ.get("HIPENGINE_CAPACITY_MODEL", "/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"))
DMS_METADATA = Path(
    os.environ.get(
        "HIPENGINE_CAPACITY_DMS_METADATA",
        "/models/dms/qwen38-27b-q4km-dms-w8192-local/dms_metadata.json",
    )
)
RUN_COMMAND = "HIP_VISIBLE_DEVICES=<24 GB GPU> uv run pytest tests/test_slow_qwen38_24gb_serving_capacity.py"


@dataclass(frozen=True)
class Case:
    name: str
    args: tuple[str, ...]
    floor_tokens: int
    concurrent: int = 0
    dms: bool = False


CASES = (
    # Automatic capacity: the largest of (4, 2, 1) slots whose context still
    # reaches 32K, so on 24 GB this serves two requests at once.
    Case("bf16_mtp_default", (), 34_000, concurrent=4),
    Case("bf16_mtp_c1", ("--max-active-requests", "1"), 59_000),
    Case("int8_mtp", ("--kv-storage", "int8_per_token_head"), 69_000, concurrent=4),
    Case(
        "bf16_mtp_off_c1",
        ("--speculative-mtp-serving", "off", "--max-active-requests", "1"),
        80_000,
    ),
    Case(
        "int8_mtp_off_c1",
        ("--kv-storage", "int8_per_token_head", "--speculative-mtp-serving", "off",
         "--max-active-requests", "1"),
        95_000,
    ),
    Case(
        "dms_int8_mtp",
        ("--kv-storage", "int8_per_token_head", "--dms-metadata-path", str(DMS_METADATA)),
        250_000,
        dms=True,
    ),
)


def _visible_device_total_gib() -> float | None:
    try:
        lib = ctypes.CDLL("libamdhip64.so")
    except OSError:
        return None
    free, total = ctypes.c_size_t(), ctypes.c_size_t()
    if lib.hipMemGetInfo(ctypes.byref(free), ctypes.byref(total)) != 0:
        return None
    return total.value / 1024**3


def _require_24gb_card() -> None:
    total = _visible_device_total_gib()
    if total is None:
        pytest.skip(f"no ROCm device visible; run on a 24 GB GPU: {RUN_COMMAND}")
    if not 22.0 <= total <= 26.0:
        pytest.skip(
            f"visible GPU has {total:.1f} GiB, not a 24 GB-class card; a larger card "
            f"cannot show 24 GB regressions: {RUN_COMMAND}"
        )
    if not MODEL.is_file():
        pytest.skip(f"model not found at {MODEL} (set HIPENGINE_CAPACITY_MODEL)")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _get(port: int, path: str, timeout: float = 30.0) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=timeout) as resp:
        return json.load(resp)


def _chat(port: int, model: str, content: str, max_tokens: int, timeout: float) -> dict:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        return {"http_status": exc.code, "error": exc.read().decode(errors="replace")[:600]}


class _Server:
    def __init__(self, args: tuple[str, ...], log_path: Path) -> None:
        self.port = _free_port()
        self.log_path = log_path
        self._log = open(log_path, "w")
        self.process = subprocess.Popen(
            [sys.executable, "-m", "hipengine.cli", "serve", "--model", str(MODEL),
             "--port", str(self.port), *args],
            stdout=self._log,
            stderr=subprocess.STDOUT,
        )

    def wait_ready(self, timeout: float = 1800.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise AssertionError(f"server exited during startup; log: {self.log_path}")
            try:
                if _get(self.port, "/ready", timeout=5).get("ready"):
                    return
            except (OSError, ValueError):
                pass
            time.sleep(5)
        raise AssertionError(f"server not ready after {timeout:.0f}s; log: {self.log_path}")

    def close(self) -> None:
        self.process.terminate()
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=30)
        self._log.close()


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_24gb_serve_keeps_its_automatic_context(case: Case, tmp_path: Path) -> None:
    _require_24gb_card()
    if case.dms and not DMS_METADATA.is_file():
        pytest.skip(f"DMS metadata not found at {DMS_METADATA} (set HIPENGINE_CAPACITY_DMS_METADATA)")
    server = _Server(case.args, tmp_path / f"{case.name}.log")
    try:
        server.wait_ready()
        model = _get(server.port, "/v1/models")["data"][0]
        context = int(model["hipengine"]["context"]["effective_max_context_tokens"])
        assert context >= case.floor_tokens, (
            f"{case.name}: automatic context {context} fell below the {case.floor_tokens}-token "
            f"24 GB floor; log: {server.log_path}"
        )
        # About ten tokens per repetition; 97% of the context leaves room for
        # the chat template and the output.
        target = int(context * 0.97)
        prompt = "Count the words. " + "The quick brown fox jumps over the lazy dog. " * (target // 10)
        deep = _chat(server.port, model["id"], prompt, 16, timeout=5400)
        assert "choices" in deep, f"{case.name}: full-depth request failed: {deep}; log: {server.log_path}"
        assert deep["usage"]["prompt_tokens"] >= int(context * 0.95)
        assert deep["usage"]["completion_tokens"] >= 1
        if case.concurrent:
            short = "Summarize: " + "The quick brown fox jumps over the lazy dog. " * 200
            with ThreadPoolExecutor(case.concurrent) as pool:
                results = list(pool.map(
                    lambda _: _chat(server.port, model["id"], short, 32, timeout=1200),
                    range(case.concurrent),
                ))
            failed = [result for result in results if "choices" not in result]
            assert not failed, f"{case.name}: concurrent requests failed: {failed[:1]}"
        assert _get(server.port, "/ready").get("ready"), f"{case.name}: server not ready after requests"
        print(
            f"{case.name}: automatic context {context}, served {deep['usage']['prompt_tokens']} "
            f"prompt tokens (floor {case.floor_tokens})"
        )
    finally:
        server.close()
