"""CPU-only contract tests for scripts/gguf_server_allocation_probe.py.

The probe exists to record what the *server* decided and what it cost, so these
tests pin three things without a GPU, a server or a model: the argv/environment
the probe builds for each axis combination, the raw payloads it keeps, and the
artifact it still writes when a step fails. Every network, process and clock
boundary is monkeypatched; nothing here imports hipengine.
"""
from __future__ import annotations

import io
import json
import threading
import urllib.error
from pathlib import Path

import pytest

from scripts import gguf_server_allocation_probe as probe

POLICY_KWARGS = {"model": Path("/models/x.gguf"), "port": 18000}


def make_args(**overrides):
    argv: list[str] = []
    for key, value in overrides.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            argv.append(flag)
        elif value is not None and value is not False:
            argv += [flag, str(value)]
    return probe.parse_args(argv)


def policy_for(**overrides):
    args = make_args(**overrides)
    return args, probe.resolve_policy(args, {})


def command_for(**overrides) -> list[str]:
    args, policy = policy_for(**overrides)
    return probe.build_server_command(args, policy)


class FakeServer:
    """Minimal ``subprocess.Popen`` stand-in for teardown paths."""

    def __init__(self, *, wait_exc=None, kill_exc=None, exit_code=None):
        self.returncode = exit_code
        self._wait_exc = wait_exc
        self._kill_exc = kill_exc
        self.signalled = False
        self.killed = False

    def poll(self):
        return self.returncode

    def send_signal(self, _signal):
        self.signalled = True

    def wait(self, timeout=None):
        if self._wait_exc is not None:
            exc, self._wait_exc = self._wait_exc, None
            raise exc
        self.returncode = 0
        return 0

    def kill(self):
        self.killed = True
        if self._kill_exc is not None:
            raise self._kill_exc


# --------------------------------------------------------------------------
# Arguments and policy
# --------------------------------------------------------------------------


def test_parse_args_defaults_keep_historical_configuration():
    args = probe.parse_args([])
    assert args.kv_storage == "int8_per_token_head"
    assert args.speculative_mtp_serving == "off"
    assert args.prefix_cache == "off"
    assert args.max_context_tokens is None  # "auto" resolves to an omission
    assert args.max_active_requests == 1
    assert args.prompt_rows == 2048
    assert args.legacy_int8_diagnostic_overrides is False


@pytest.mark.parametrize("raw", ["0", "-4", "many"])
def test_positive_int_or_auto_rejects_non_positive_or_text(raw):
    with pytest.raises(SystemExit):
        probe.parse_args(["--max-context-tokens", raw])


def test_positive_int_or_auto_accepts_auto_and_int():
    assert probe._positive_int_or_auto("auto") is None
    assert probe._positive_int_or_auto(" 4096 ") == 4096


def test_automatic_axes_reach_the_server_as_omissions():
    command = command_for(max_context_tokens="auto", max_active_requests="auto",
                          prefix_cache="default")
    assert "--max-context-tokens" not in command
    assert "--max-active-requests" not in command
    assert "--prefix-cache" not in command
    # --kv-storage is always passed so an inherited variable cannot decide it.
    assert command[command.index("--kv-storage") + 1] == "int8_per_token_head"


def test_pinned_axes_reach_the_server_as_literals():
    command = command_for(max_context_tokens="16384", max_active_requests="2",
                          prefix_cache="radix")
    assert command[command.index("--max-context-tokens") + 1] == "16384"
    assert command[command.index("--max-active-requests") + 1] == "2"
    assert command[command.index("--prefix-cache") + 1] == "radix"


def test_scale_flags_are_int8_only():
    assert "--kv-scale-granularity" in command_for()
    bf16 = command_for(kv_storage="bf16")
    assert "--kv-scale-granularity" not in bf16
    assert "--kv-scale-dtype" not in bf16


def test_speculative_axis_is_always_explicit():
    command = command_for(speculative_mtp_serving="auto")
    assert command[command.index("--speculative-mtp-serving") + 1] == "auto"


def test_dms_arguments_are_mutually_exclusive():
    args = make_args(dms_metadata="m.json", dms_metadata_path="dms")
    with pytest.raises(probe.PolicyError, match="not both"):
        probe.resolve_policy(args, {})


def test_compact_dms_is_left_to_the_server_to_validate():
    """The probe must not re-derive the server's admission policy."""

    args, policy = policy_for(dms_metadata="meta.json", kv_storage="int8_per_token_head",
                              max_context_tokens="32768")
    assert policy["dms_metadata"] == "meta.json"
    command = probe.build_server_command(args, policy)
    assert command[command.index("--dms-metadata") + 1] == "meta.json"
    assert command[command.index("--dms-prefill-mode") + 1] == "dense_pool"


def test_online_dms_path_is_passed_through():
    command = command_for(dms_metadata_path="dms-dir")
    assert command[command.index("--dms-metadata-path") + 1] == "dms-dir"
    assert "--dms-prefill-mode" not in command


@pytest.mark.parametrize(
    "env, kwargs, axis",
    [
        ({"HIPENGINE_MAX_CONTEXT_TOKENS": "8192"}, {"max_context_tokens": "auto"},
         "--max-context-tokens"),
        ({"HIPENGINE_MAX_ACTIVE_REQUESTS": "4"}, {"max_active_requests": "auto"},
         "--max-active-requests"),
        ({"HIPENGINE_PREFIX_CACHE": "radix"}, {"prefix_cache": "default"},
         "--prefix-cache"),
    ],
)
def test_environment_pinning_an_automatic_axis_is_refused(env, kwargs, axis):
    args = make_args(**kwargs)
    with pytest.raises(probe.PolicyError, match=axis):
        probe.resolve_policy(args, env)


def test_automatic_context_disabled_in_environment_is_refused():
    args = make_args(max_context_tokens="auto")
    with pytest.raises(probe.PolicyError, match="HIPENGINE_GGUF_AUTO_CONTEXT"):
        probe.resolve_policy(args, {"HIPENGINE_GGUF_AUTO_CONTEXT": "0"})


def test_pinned_axis_is_not_refused_by_a_pinned_environment():
    args = make_args(max_context_tokens="4096")
    policy = probe.resolve_policy(args, {"HIPENGINE_MAX_CONTEXT_TOKENS": "8192"})
    assert policy["max_context_tokens"] == 4096


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------


def test_default_run_injects_no_environment():
    args, policy = policy_for()
    env, record = probe.build_server_env(args, policy, {"PATH": "/bin"})
    assert env == {"PATH": "/bin"}
    assert record["overrides"] == {}


def test_every_declared_capture_key_is_recorded():
    """Every variable the probe says it captures is captured, and nothing else.

    The key list is taken from the module so the test cannot drift from the
    probe's own declaration; the probe, not the test, decides which variables
    describe a measurement.
    """

    base = {name: f"v{i}" for i, name in enumerate(probe._CAPTURED_ENV_KEYS)}
    base["PATH"] = "/bin"
    args, policy = policy_for()
    _, record = probe.build_server_env(args, policy, base)
    assert set(record["inherited"]) == set(probe._CAPTURED_ENV_KEYS)
    assert "PATH" not in record["inherited"]


def test_unrelated_measurement_variables_are_recorded_by_name():
    base = {"GPU_MAX_HW_QUEUES": "1", "HSA_SCRATCH_SINGLE_LIMIT": "4294967296"}
    args, policy = policy_for()
    _, record = probe.build_server_env(args, policy, base)
    assert record["inherited"]["GPU_MAX_HW_QUEUES"] == "1"
    assert record["inherited"]["HSA_SCRATCH_SINGLE_LIMIT"] == "4294967296"


def test_legacy_overrides_are_opt_in_and_recorded():
    args, policy = policy_for(legacy_int8_diagnostic_overrides=True)
    env, record = probe.build_server_env(args, policy, {})
    assert env[probe._INT8_ALLOW_UNVERIFIED_LONG_ENV] == "1"
    assert env[probe._INT8_BF16_FULL_LAYERS_ENV] == "none"
    assert set(record["overrides"]) == {
        probe._INT8_ALLOW_UNVERIFIED_LONG_ENV, probe._INT8_BF16_FULL_LAYERS_ENV
    }
    assert all(entry["effect"] for entry in record["overrides"].values())


def test_legacy_overrides_are_ignored_for_bf16_and_said_so():
    args, policy = policy_for(legacy_int8_diagnostic_overrides=True, kv_storage="bf16")
    env, record = probe.build_server_env(args, policy, {})
    assert probe._INT8_ALLOW_UNVERIFIED_LONG_ENV not in env
    assert probe._INT8_BF16_FULL_LAYERS_ENV not in env
    assert "ignored" in next(iter(record["overrides"].values()))["value"]


# --------------------------------------------------------------------------
# Raw documents and response classification
# --------------------------------------------------------------------------


def test_document_keeps_the_whole_payload():
    payload = {"ready": True, "kv_pool": {"slots": [1, 2]}, "extra": {"deep": [None]}}
    doc = probe.document(200, payload, None)
    assert doc == {"http_status": 200, "error": None, "payload": payload}


def test_http_json_keeps_the_body_of_an_error_status(monkeypatch):
    body = b'{"ready": false, "reason": "kv_pool_exhausted"}'
    error = urllib.error.HTTPError("http://x/ready", 503, "busy", {}, io.BytesIO(body))

    def raise_error(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(probe.urllib.request, "urlopen", raise_error)
    status, payload, message = probe._http_json("http://x/ready")
    assert status == 503 and message is None
    assert payload == {"ready": False, "reason": "kv_pool_exhausted"}


def test_http_json_reports_transport_failure(monkeypatch):
    def raise_error(*_args, **_kwargs):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(probe.urllib.request, "urlopen", raise_error)
    status, payload, message = probe._http_json("http://x/ready")
    assert status is None and payload is None and "URLError" in message


def test_http_json_flags_a_non_json_body(monkeypatch):
    class FakeResponse:
        status = 200

        def read(self):
            return b"<html>not json</html>"

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    monkeypatch.setattr(probe.urllib.request, "urlopen", lambda *a, **k: FakeResponse())
    status, payload, message = probe._http_json("http://x/ready")
    assert status == 200 and payload is None and "JSONDecodeError" in message


def test_readiness_requires_the_ready_flag():
    assert probe.readiness_accepts_requests(200, {"ready": True}) is True
    assert probe.readiness_accepts_requests(200, {"ready": False}) is False
    assert probe.readiness_accepts_requests(503, {"ready": True}) is False
    assert probe.readiness_accepts_requests(200, ["ready"]) is False


def completion(text="ok", finish="stop", completion_tokens=8, **extra):
    payload = {
        "choices": [{"text": text, "finish_reason": finish}],
        "usage": {"completion_tokens": completion_tokens},
    }
    payload.update(extra)
    return payload


def test_outcome_accepts_a_completed_generation():
    assert probe.completion_outcome(200, completion())["ok"] is True


@pytest.mark.parametrize(
    "status, payload, reason",
    [
        (None, None, "no_http_status"),
        (500, {"error": {"message": "boom"}}, "http_error_status"),
        (200, ["not", "an", "object"], "response_not_an_object"),
        (200, completion(error={"message": "rejected"}), "error_payload"),
        (200, {"usage": {"completion_tokens": 1}}, "no_choices"),
        (200, {"choices": [], "usage": {"completion_tokens": 1}}, "no_choices"),
        (200, {"choices": ["text"], "usage": {"completion_tokens": 1}},
         "choice_not_an_object"),
        (200, completion(text=None), "choice_missing_text"),
        (200, completion(finish=None), "choice_incomplete_finish_reason"),
        (200, {"choices": [{"text": "x", "finish_reason": "stop"}]}, "usage_missing"),
        (200, completion(completion_tokens=0), "usage_missing_completion_tokens"),
    ],
)
def test_outcome_rejects_unusable_responses(status, payload, reason):
    outcome = probe.completion_outcome(status, payload)
    assert outcome["ok"] is False and outcome["reason"] == reason


# --------------------------------------------------------------------------
# The probe body
# --------------------------------------------------------------------------


def test_probe_records_raw_documents_and_server_metadata(monkeypatch):
    ready = {"ready": True, "kv_pool": {"high_water": 1}, "phases": {"load": 1.5}}
    caps = {"object": "hipengine.capabilities", "kv": {"storage": ["bf16"]},
            "device": {"arch": "gfx1100"}, "admission": {"max_active_requests": 1}}
    body = {
        "choices": [{"text": "hello world", "finish_reason": "stop"}],
        "usage": {"completion_tokens": 8, "prompt_tokens": 2048},
        "hipengine": {
            "kv_storage": "int8_per_token_head",
            "route": {"prefill": "packed_slot_local", "decode": "paged_attention"},
            "mtp": {"accepted": 0, "proposed": 0, "draft_tokens": []},
            "prompt_accounting": {"slabs": 2, "rows": 2048},
        },
    }
    gauges = {
        "hipengine_kv_pool_current_bytes": 2.0 * 2**30,
        "hipengine_kv_pool_high_water_observed_bytes": 3.0 * 2**30,
        "hipengine_kv_pool_pinned_pages": 7.0,
        "hipengine_resident_packed_workspace_current_bytes": 1.5 * 2**30,
        "hipengine_resident_packed_workspace_leased_pool_bytes": 0.5 * 2**30,
        "hipengine_resident_prefill_oracle_owner_bytes": 0.25 * 2**30,
        "hipengine_resident_prefill_oracle_observed_peak_bytes": 0.75 * 2**30,
        "hipengine_resident_prefill_oracle_observed_peak_owners": 2.0,
        "hipengine_resident_prefill_oracle_owners": 1.0,
        "hipengine_resident_prefill_hidden_owner_bytes": 0.125 * 2**30,
        "hipengine_resident_kv_total_bytes": 2.5 * 2**30,
    }
    monkeypatch.setattr(probe, "_scrape", lambda *_a, **_k: dict(gauges))
    monkeypatch.setattr(probe, "_http_json",
                        lambda url, **_k: (200, ready if url.endswith("/ready") else caps,
                                           None))
    monkeypatch.setattr(probe, "_request_once",
                        lambda *_a, **_k: (200, json.dumps(body), None))
    monkeypatch.setattr(probe.time, "sleep", lambda _s: None)

    args = make_args(prompt_rows="2048", decode_tokens="8")
    result = {"command": [], "identity": {}}
    # The real readiness path runs: /ready is answered 200 with the document.
    probe._probe_running_server(args, [1, 2, 3], result, FakeServer(), 0.0)

    assert result["status"] == "pass"
    # Raw documents, not a hand-picked summary of them.
    assert result["readiness_before"]["payload"] == ready
    assert result["capabilities_before"]["payload"] == caps
    assert result["capabilities_after"]["payload"] == caps
    assert result["readiness_after"]["payload"] == ready
    # Raw server metadata, not a guessed subset.
    assert result["response"]["hipengine"] == body["hipengine"]
    assert result["response"]["usage"] == body["usage"]
    assert result["response"]["finish_reason"] == "stop"
    # Per-domain peaks stay separate; they are never summed.
    assert result["domains_gib"]["pool_high_water"] == pytest.approx(3.0)
    assert result["domains_gib"]["workspace_leased_pool"] == pytest.approx(0.5)
    assert result["domains_gib"]["pool_pinned_pages"] == 7.0
    assert result["during_request_peaks"]["hipengine_kv_pool_high_water_observed_bytes"] == (
        3.0 * 2**30
    )
    assert "sum" not in json.dumps(result["domains_gib"])


def test_probe_failure_keeps_the_response_body(monkeypatch):
    monkeypatch.setattr(probe, "_scrape", lambda *_a, **_k: {})
    monkeypatch.setattr(probe, "_http_json", lambda *_a, **_k: (200, {}, None))
    monkeypatch.setattr(probe, "_request_once",
                        lambda *_a, **_k: (400, '{"error": {"message": "too long"}}', "HTTPError"))
    monkeypatch.setattr(probe, "_await_ready", lambda *_a, **_k: True)
    monkeypatch.setattr(probe.time, "sleep", lambda _s: None)

    result = {"command": [], "identity": {}}
    probe._probe_running_server(make_args(), [1], result, FakeServer(), 0.0)
    assert result["status"] == "request_failed"
    assert result["response"]["outcome"]["reason"] == "http_error_status"
    assert "too long" in result["response"]["body_head"]


def test_probe_returns_without_measuring_when_metrics_are_unavailable(monkeypatch):
    monkeypatch.setattr(probe, "_scrape", lambda *_a, **_k: None)
    monkeypatch.setattr(probe, "_http_json", lambda *_a, **_k: (200, {}, None))
    monkeypatch.setattr(probe, "_await_ready", lambda *_a, **_k: True)
    result = {"command": [], "identity": {}}
    probe._probe_running_server(make_args(), [1], result, FakeServer(), 0.0)
    assert result["status"] == "metrics_unavailable"


def test_poller_thread_stops_when_the_request_raises(monkeypatch):
    monkeypatch.setattr(probe, "_scrape", lambda *_a, **_k: {})
    monkeypatch.setattr(probe, "_http_json", lambda *_a, **_k: (200, {}, None))
    monkeypatch.setattr(probe, "_await_ready", lambda *_a, **_k: True)

    def explode(*_args, **_kwargs):
        raise RuntimeError("request exploded")

    monkeypatch.setattr(probe, "_request_once", explode)
    before = threading.active_count()
    result = {"command": [], "identity": {}}
    with pytest.raises(RuntimeError):
        probe._probe_running_server(make_args(), [1], result, FakeServer(), 0.0)
    assert threading.active_count() <= before


def test_poller_is_not_started_before_readiness(monkeypatch):
    monkeypatch.setattr(probe, "_await_ready", lambda *_a, **_k: False)
    monkeypatch.setattr(probe, "_scrape", lambda *_a, **_k: pytest.fail("scraped early"))
    result = {"status": "startup_timeout", "command": [], "identity": {}}
    probe._probe_running_server(make_args(), [1], result, FakeServer(), 0.0)
    assert result["status"] == "startup_timeout"


# --------------------------------------------------------------------------
# Readiness, teardown, artifact
# --------------------------------------------------------------------------


def test_await_ready_records_the_document_and_polls(monkeypatch):
    monkeypatch.setattr(probe.time, "sleep", lambda _s: None)
    reads = iter([(503, {"ready": False}, None), (200, {"ready": True}, None)])
    monkeypatch.setattr(probe, "_http_json", lambda *_a, **_k: next(reads))
    args = make_args(startup_timeout="30")
    result: dict = {}
    assert probe._await_ready(FakeServer(), args, result) is True
    assert result["ready_polls"] == 2
    assert result["readiness_before"]["payload"] == {"ready": True}


def test_await_ready_times_out(monkeypatch):
    monkeypatch.setattr(probe.time, "sleep", lambda _s: None)
    monkeypatch.setattr(probe, "_http_json", lambda *_a, **_k: (503, {"ready": False}, None))
    args = make_args(startup_timeout="0.01")
    result: dict = {}
    assert probe._await_ready(FakeServer(), args, result) is False
    assert result["status"] == "startup_timeout"
    assert result["readiness_before"]["http_status"] == 503


def test_await_ready_reports_a_server_that_exited(monkeypatch):
    monkeypatch.setattr(probe.time, "sleep", lambda _s: None)
    args = make_args(startup_timeout="30")
    result: dict = {}
    assert probe._await_ready(FakeServer(exit_code=3), args, result) is False
    assert result["status"] == "server_exited_during_startup"
    assert result["server_exit_code"] == 3


def test_teardown_reports_a_clean_stop():
    result: dict = {}
    server = FakeServer()
    probe._teardown(server, result)
    assert result["server_teardown"] == "clean" and server.signalled


def test_teardown_kills_after_a_hang():
    result: dict = {}
    server = FakeServer(wait_exc=probe.subprocess.TimeoutExpired("cmd", 60))
    probe._teardown(server, result)
    assert result["server_teardown"] == "killed_after_timeout" and server.killed


def test_teardown_never_raises():
    result: dict = {}
    server = FakeServer(wait_exc=probe.subprocess.TimeoutExpired("cmd", 60),
                        kill_exc=OSError("gone"))
    probe._teardown(server, result)
    assert result["server_teardown"].startswith("kill_failed")


def test_log_tail_is_bounded_and_keeps_the_end(tmp_path):
    log = tmp_path / "server.log"
    log.write_text("\n".join(f"line {i}" for i in range(500)), encoding="utf-8")
    tail = probe._log_tail(log, max_lines=10)
    assert tail[-1] == "line 499" and len(tail) <= 10
    assert probe._log_tail(tmp_path / "missing.log") is None


def test_finish_probe_writes_the_artifact_and_fails_a_bad_run(tmp_path):
    path = tmp_path / "out.json"
    result = {"status": "pass", "server_teardown": "clean", "probe_stamp": "s"}
    assert probe.finish_probe(result, path, stdout=io.StringIO()) == 0
    assert json.loads(path.read_text())["artifact_path"] == str(path)

    failed = {"status": "request_failed", "server_teardown": "clean", "probe_stamp": "s"}
    assert probe.finish_probe(failed, path, stdout=io.StringIO()) == 1


def test_finish_probe_defaults_to_a_temporary_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(probe.tempfile, "gettempdir", lambda: str(tmp_path))
    result = {"status": "policy_rejected", "probe_stamp": "20260101T000000"}
    assert probe.finish_probe(result, None, stdout=io.StringIO()) == 1
    written = tmp_path / "gguf-server-allocation-probe-20260101T000000.json"
    assert json.loads(written.read_text())["status"] == "policy_rejected"


def test_finish_probe_reports_an_unwritable_artifact(tmp_path):
    result = {"status": "pass", "server_teardown": "clean", "probe_stamp": "s"}
    blocker = tmp_path / "file"
    blocker.write_text("x")
    stderr = io.StringIO()
    assert probe.finish_probe(result, blocker / "nested" / "out.json",
                              stdout=io.StringIO(), stderr=stderr) == 1
    assert "failed to write artifact" in stderr.getvalue()


# --------------------------------------------------------------------------
# Failure containment in run_probe
# --------------------------------------------------------------------------


def test_run_probe_records_an_unopenable_log(tmp_path):
    result = {"command": ["server"], "identity": {}}
    probe.run_probe(make_args(), {}, [1], result, tmp_path / "missing" / "server.log")
    assert result["status"] == "server_log_unavailable"


def test_run_probe_records_a_launch_failure(tmp_path, monkeypatch):
    def explode(*_args, **_kwargs):
        raise OSError("no such binary")

    monkeypatch.setattr(probe.subprocess, "Popen", explode)
    result = {"command": ["server"], "identity": {}}
    probe.run_probe(make_args(), {}, [1], result, tmp_path / "server.log")
    assert result["status"] == "server_launch_failed"
    assert "no such binary" in result["error"]
    assert result["server_log_tail"] is not None


def test_run_probe_records_an_internal_error_and_still_tears_down(tmp_path, monkeypatch):
    monkeypatch.setattr(probe.subprocess, "Popen", lambda *_a, **_k: FakeServer())
    monkeypatch.setattr(probe, "_probe_running_server",
                        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")))
    result = {"command": ["server"], "identity": {}}
    probe.run_probe(make_args(), {}, [1], result, tmp_path / "server.log")
    assert result["status"] == "probe_error" and "boom" in result["error"]
    assert result["server_teardown"] == "clean"
    assert result["server_log_tail"] == []


# --------------------------------------------------------------------------
# Prompt and identity
# --------------------------------------------------------------------------


def test_prompt_is_deterministic_for_a_given_shape():
    first, digest = probe.build_prompt(make_args(prompt_rows="64"))
    second, digest_again = probe.build_prompt(make_args(prompt_rows="64"))
    assert first == second and digest == digest_again and len(first) == 64


def test_prompt_uses_the_parity_fixture_for_2048_rows():
    ids, digest = probe.build_prompt(make_args(prompt_rows="2048"))
    assert len(ids) == 2048 and len(digest) == 64


def test_identity_records_host_and_tree():
    record = probe.identity()
    assert set(record) == {"hostname", "platform", "head", "dirty", "probe_script_sha256"}
    assert len(record["probe_script_sha256"]) == 64


# --------------------------------------------------------------------------
# main()
# --------------------------------------------------------------------------


def test_main_writes_an_artifact_when_policy_is_rejected(tmp_path):
    path = tmp_path / "rejected.json"
    code = probe.main(["--dms-metadata", "a.json", "--dms-metadata-path", "b",
                       "--json", str(path)])
    assert code == 1
    artifact = json.loads(path.read_text())
    assert artifact["status"] == "policy_rejected"
    assert "not both" in artifact["error"]
    assert artifact["identity"]["probe_script_sha256"]
