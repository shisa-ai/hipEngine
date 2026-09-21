#!/usr/bin/env python3
"""Check INT8 MTP HTTP cancellation, deadlines, stops and concurrent reuse."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import httpx

from int8_mtp_server_gate import assert_result, blocking_result


def drain(client, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get("/ready")
        response.raise_for_status()
        state = response.json()
        queue = state["queue"]
        if not queue["active_requests"] and not queue["depth"] and not queue["worker_active"]:
            assert state["engine_service"]["serving"]
            return state
        time.sleep(0.05)
    raise AssertionError("server did not drain after request retirement")


def run(args):
    report = {"scope": "http_lifecycle", "passed": False, "checks": {}}
    try:
        with httpx.Client(base_url=args.base_url, timeout=300) as client:
            state = drain(client)
            capability = state["model"]["kv_capability"]
            report["kv_capability"] = capability
            assert capability["effective_kv_storage"] == "int8_per_token_head"
            if capability.get("diagnostic_override"):
                assert args.allow_kv_diagnostic_override
            base = {
                "model": args.model, "max_tokens": 24, "temperature": 0,
                "prompt": "Write a Python function that returns the sum of two integers.",
            }

            def generate(**changes):
                response = client.post("/v1/completions", json={**base, **changes})
                response.raise_for_status()
                return response.json()

            baseline_body = generate(speculative_mtp=False)
            baseline = blocking_result(baseline_body)
            assert_result(baseline, speculative=False, compact=True)
            automatic = blocking_result(generate())
            assert_result(automatic, speculative=True, compact=True)
            assert automatic["ids"] == baseline["ids"]
            report["checks"]["automatic_and_explicit_off"] = True

            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(generate, speculative_mtp=value) for value in (True, False)]
                concurrent = [blocking_result(future.result()) for future in futures]
            assert all(row["ids"] == baseline["ids"] for row in concurrent)
            assert concurrent[1]["cycles"] == 0
            drain(client)
            report["checks"]["concurrent_mixed_ar_mtp"] = True

            # Multi-choice in one request is a realized group of `n` rows. It is
            # admitted at the artifact's qualified direct-INT8 width, so both
            # choices must speculate and still match the autoregressive ids.
            multi = generate(speculative_mtp=True, n=2)
            assert len(multi["choices"]) == 2, multi
            multi_cycles = [
                int(choice["hipengine"]["timing"]["mtp_cycles_count"])
                for choice in multi["choices"]
            ]
            assert all(cycles > 0 for cycles in multi_cycles), multi_cycles
            multi_ids = [
                choice["hipengine"]["generated_token_ids"]
                for choice in multi["choices"]
            ]
            assert all(ids == baseline["ids"] for ids in multi_ids), {
                "baseline_ids": baseline["ids"],
                "multi_ids": multi_ids,
                "multi_cycles": multi_cycles,
            }
            many = generate(speculative_mtp=False, n=2)
            assert len(many["choices"]) == 2
            assert all(choice["hipengine"]["generated_token_ids"] == baseline["ids"] for choice in many["choices"])
            report["checks"]["multichoice_speculates_and_matches_ar"] = True

            text = baseline_body["choices"][0]["text"]
            assert len(text) >= 12
            stop_at = len(text) // 2
            stop_text = text[stop_at:stop_at + 8]
            stopped = blocking_result(generate(speculative_mtp=False, stop=[stop_text]))
            assert stopped["finish_reason"] == "stop"
            response = client.post("/v1/completions", json={
                **base, "speculative_mtp": True, "stop": [stop_text],
            })
            if response.status_code == 501:
                assert "mtp_sampling_unsupported" in response.text, response.text
                report["checks"]["explicit_stop_mtp_capability_error"] = True
            else:
                response.raise_for_status()
                speculative_stop = blocking_result(response.json())
                assert stopped["ids"] == speculative_stop["ids"]
                assert speculative_stop["finish_reason"] == "stop"
                assert speculative_stop["cycles"] > 0
                report["checks"]["stop_inside_speculation"] = True

            before = drain(client)
            token_seen = False
            with client.stream("POST", "/v1/completions", json={
                **base, "speculative_mtp": True, "stream": True, "max_tokens": 256,
                "stream_options": {"include_usage": True, "include_hipengine": True},
            }) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    event = json.loads(line[6:])
                    assert "error" not in event, event
                    if any(choice.get("text") for choice in event.get("choices", [])):
                        token_seen = True
                        break
            assert token_seen
            after = drain(client)
            assert client.get("/health").status_code == 200
            assert after["kv_capacity"]["pool"]["refcounted_pages"] == before["kv_capacity"]["pool"]["refcounted_pages"]
            report["checks"]["disconnect_reclaim"] = True

            response = client.post("/v1/completions", json={
                **base, "speculative_mtp": True, "max_tokens": 256, "timeout_ms": 1,
            })
            assert response.status_code == 408, response.text
            assert response.json()["error"]["code"] == "deadline_exceeded"
            drain(client)
            report["checks"]["deadline_reclaim"] = True

            reused = blocking_result(generate(speculative_mtp=True))
            assert reused["ids"] == baseline["ids"]
            assert_result(reused, speculative=True, compact=True)
            report["checks"]["post_cancel_deadline_reuse"] = True
            report["final_queue"] = drain(client)["queue"]
        report["passed"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8098")
    parser.add_argument("--model", default="int8-mtp")
    parser.add_argument("--allow-kv-diagnostic-override", action="store_true")
    parser.add_argument("--json", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
