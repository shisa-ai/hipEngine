#!/usr/bin/env python3
"""Same-artifact full-category Flash-Next AR/MTP request-wall baseline."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.qwen4exp_canonical_ar_bench import (
    _git_metadata, _host_metadata, _post_json, _wait_for_health, sha256_path,
    token_ids_sha256,
)
from scripts.qwen4exp_mtp_phase_census import DEFAULT_MODEL, DEFAULT_SIDECAR, DEFAULT_PROMPTS

HELDOUT = {
    "code_markdown_table", "general_en_explain", "general_ja_explain",
    "mixed_ja_en_review",
}


def validate_output(tokens, count):
    if not isinstance(tokens, (list, tuple)) or len(tokens) != count:
        raise ValueError("unexpected generated token count")
    if any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError("generated tokens must be nonnegative integers")
    return list(tokens)


def validate_local_mtp(diagnostics):
    proposed = diagnostics.get("proposed_draft_tokens")
    if type(proposed) is not int or proposed <= 0:
        raise ValueError("MTP arm did not report real draft proposals")


def wait_server(process, host, port, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        code = process.poll()
        if code is not None:
            raise RuntimeError(f"server exited during startup: {code}")
        try:
            _wait_for_health(host, port, min(2, deadline - time.monotonic()))
            return
        except TimeoutError:
            continue
    raise TimeoutError("server startup deadline exceeded")


def summarize(rows, *, expected_ids, repetitions):
    expected = {(name, mode, rep) for name in expected_ids
                for mode in ("ar", "mtp") for rep in range(repetitions)}
    cells = {}
    for row in rows:
        key = row["id"], row["mode"], row["repetition"]
        if key in cells:
            raise ValueError("duplicate measurement cell")
        if not math.isfinite(row["seconds"]) or row["seconds"] <= 0:
            raise ValueError("invalid request wall time")
        cells[key] = row
    if set(cells) != expected:
        raise ValueError("incomplete measurement matrix")
    groups = defaultdict(list)
    deterministic = True
    exact = True
    for name in expected_ids:
        for mode in ("ar", "mtp"):
            reference = cells[name, mode, 0]["tokens"]
            deterministic &= all(cells[name, mode, rep]["tokens"] == reference
                                 for rep in range(repetitions))
        exact &= all(cells[name, "ar", rep]["tokens"] == cells[name, "mtp", rep]["tokens"]
                     for rep in range(repetitions))
    for row in rows:
        for group in ("full", row["split"], "category:" + row["category"]):
            groups[group].append(row)
    result = {"deterministic": bool(deterministic), "exact_vs_own_ar": bool(exact)}
    for name, group in groups.items():
        seconds = {mode: sum(r["seconds"] for r in group if r["mode"] == mode)
                   for mode in ("ar", "mtp")}
        counts = {mode: sum(len(r["tokens"]) for r in group if r["mode"] == mode)
                  for mode in ("ar", "mtp")}
        if counts["ar"] != counts["mtp"]:
            raise ValueError("AR/MTP output counts differ")
        result[name] = {
            "ar_seconds": seconds["ar"], "mtp_seconds": seconds["mtp"],
            "ar_request_tok_s": counts["ar"] / seconds["ar"],
            "mtp_request_tok_s": counts["mtp"] / seconds["mtp"],
            "mtp_over_ar": seconds["ar"] / seconds["mtp"],
        }
    return result


def prompt_rows(model, prompts, max_tokens):
    from hipengine.loading.gguf import discover_gguf_files, load_gguf_index
    from hipengine.tokenization.gguf import Qwen4ExpGGUFTokenizer

    tokenizer = Qwen4ExpGGUFTokenizer.from_gguf_info(
        load_gguf_index(discover_gguf_files(model)[0]))
    source = [json.loads(line) for line in prompts.read_text().splitlines() if line.strip()]
    canonical = [json.loads(line) for line in DEFAULT_PROMPTS.read_text().splitlines()
                 if line.strip()]
    if source != canonical:
        raise ValueError("use the complete committed ten-prompt category suite")
    result = []
    for row in source:
        text = "\n".join(str(message["content"]) for message in row["messages"])
        tokens = list(map(int, tokenizer.encode(text)))
        if not tokens or len(tokens) + max_tokens > 1024:
            raise ValueError("prompt plus output exceeds the 1024-token MTP scope")
        result.append({
            "id": row["id"], "category": row["category"],
            "split": "heldout" if row["id"] in HELDOUT else "train",
            "prompt_token_ids": tokens, "prompt_sha256": token_ids_sha256(tokens),
        })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("hipengine", "llamacpp"), required=True)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--server-bin", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--server-arg", action="append", default=[])
    parser.add_argument("--port", type=int, default=18158)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_tokens < 2 or args.repetitions < 1 or not 1 <= args.budget <= 4:
        parser.error("require max-tokens>=2, repetitions>=1 and budget in 1..4")
    if args.engine == "llamacpp" and (not args.server_bin or not args.source_root):
        parser.error("llamacpp needs --server-bin and --source-root")
    if args.output.exists():
        raise FileExistsError(args.output)
    from scripts.qwen4exp_framework_family_refresh import model_identity

    identity = model_identity(args.model_root)
    sidecar_hash = sha256_path(args.sidecar)
    if sidecar_hash != "9db03a687670608286e99b563fcc86d0ee76c8dd863f64b2afc0b54eb0eb975d":
        raise ValueError("MTP sidecar does not match the frozen Q8_0 artifact")
    cases = prompt_rows(args.model_root, args.prompts, args.max_tokens)
    report = {
        "schema": 1, "kind": "qwen4exp_journey_mtp_request_baseline",
        "status": "running", "performance_claim": False,
        "command": [sys.executable, *sys.argv], "source": _git_metadata(ROOT),
        "host": _host_metadata(), "engine": args.engine,
        "model_root": str(args.model_root), "quant": "UD-Q4_K_XL", "kv": "bf16",
        "model_identity": identity,
        "sidecar": str(args.sidecar), "sidecar_sha256": sidecar_hash,
        "prompt_file_sha256": sha256_path(args.prompts), "cases": cases,
        "protocol": {
            "timing": "synchronized complete request wall; includes prefill; excludes load",
            "max_tokens": args.max_tokens, "budget": args.budget,
            "context_capacity": 1024, "warmups_per_case_per_arm": 1,
            "repetitions": args.repetitions, "temperature": 0, "top_k": 1,
            "true_ar": True, "cache_prompt": False,
            "prompt_rendering": "raw joined message contents; exact shared token arrays",
            "order": "HE rotates AR/MTP by case/repetition; external sequential resident arms",
        },
        "environment": {k: v for k, v in os.environ.items()
                        if k.startswith(("HIPENGINE_", "LLAMA_", "GGML_", "HSA_", "GPU_", "DEBUG_HIP_"))
                        or k in ("LD_LIBRARY_PATH", "HIP_PATH", "ROCM_PATH")},
        "samples": [], "warmups": [], "server_commands": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def measure(call, case, mode, rep):
        start = time.perf_counter()
        tokens, diagnostics = call(case, mode)
        elapsed = time.perf_counter() - start
        row = {k: case[k] for k in ("id", "category", "split", "prompt_sha256")}
        row.update(mode=mode, repetition=rep, seconds=elapsed,
                   tokens=validate_output(tokens, args.max_tokens), diagnostics=diagnostics)
        report["warmups" if rep < 0 else "samples"].append(row)
        save()
        print(f"{mode} rep={rep} {case['id']} {elapsed:.3f}s", flush=True)

    save()
    try:
        if args.engine == "hipengine":
            from hipengine import LLM, SamplingParams
            from hipengine.core.hip import get_hip_runtime
            from hipengine.core.memory import memory_stats
            from hipengine.execution_profiles import ExecutionProfile, resolve_runtime_profile
            from hipengine.generation.qwen4_exp_profiles import (
                QWEN4_EXP_MODEL, QWEN4_EXP_BACKEND, QWEN4_EXP_QUANTS,
            )

            llm = LLM(str(args.model_root), backend="hip_gfx1151",
                      quant="gguf_ud_q4_k_xl", execution_profile="production",
                      max_sequence_length=1024, speculative_provider="qwen4_exp_mtp",
                      draft_model=str(args.sidecar), speculative_candidate_budget=args.budget)
            params = SamplingParams(max_tokens=args.max_tokens, temperature=0,
                                    top_k=1, ignore_eos=True)
            runtime = get_hip_runtime()

            def call(case, mode):
                method = (llm.generate_detailed if mode == "ar"
                          else llm.generate_speculative_mtp_detailed)
                output = method([case["prompt_token_ids"]], params)[0]
                runtime.device_synchronize()
                diagnostics = dict(output.telemetry.diagnostics) if output.telemetry else {}
                if mode == "mtp":
                    validate_local_mtp(diagnostics)
                return list(output.generated_token_ids), diagnostics

            try:
                llm.prepare(max_sequence_length=1024)
                resolved = resolve_runtime_profile(
                    model=QWEN4_EXP_MODEL, backend=QWEN4_EXP_BACKEND,
                    quant=QWEN4_EXP_QUANTS[1], profile=ExecutionProfile.PRODUCTION)
                report["profile"] = {
                    "requested": "production", "manifest_sha256": resolved.manifest_sha256,
                    "strict_manifest_sha256": resolved.strict_manifest_sha256,
                    "fell_back_to_strict": resolved.fell_back_to_strict,
                }
                for rep in range(-1, args.repetitions):
                    for index, case in enumerate(cases):
                        modes = ("ar", "mtp") if (index + rep) % 2 else ("mtp", "ar")
                        for mode in modes:
                            measure(call, case, mode, rep)
            finally:
                llm.close()
                report["memory_after_close"] = memory_stats()
        else:
            from hipengine.loading.gguf import discover_gguf_files

            report["comparator_source"] = _git_metadata(args.source_root)
            report["binary_sha256"] = sha256_path(args.server_bin)
            for mode in ("ar", "mtp"):
                command = [str(args.server_bin), "-m", str(discover_gguf_files(args.model_root)[0]),
                           "--host", "127.0.0.1", "--port", str(args.port),
                           "--parallel", "1", "--no-webui", *args.server_arg]
                if mode == "mtp":
                    command += ["--spec-type", "draft-mtp", "--spec-draft-model", str(args.sidecar),
                                "--spec-draft-ngl", "999", "--spec-draft-n-max", str(args.budget)]
                report["server_commands"].append(command)
                log_path = args.output.with_suffix(f".{mode}.server.log")
                process = None
                with log_path.open("wb") as log:
                    try:
                        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                        wait_server(process, "127.0.0.1", args.port, 600)

                        def call(case, _mode):
                            response = _post_json("127.0.0.1", args.port, "/completion", {
                                "prompt": case["prompt_token_ids"], "n_predict": args.max_tokens,
                                "temperature": 0, "top_k": 1, "seed": 0, "ignore_eos": True,
                                "cache_prompt": False, "return_tokens": True,
                            }, 600)
                            if response.get("timings", {}).get("prompt_n") != len(case["prompt_token_ids"]):
                                raise ValueError("external prompt count/cache mismatch")
                            return response.get("tokens"), {
                                k: response[k] for k in ("timings", "n_drafted", "n_drafted_accepted")
                                if k in response}

                        for rep in range(-1, args.repetitions):
                            for case in cases:
                                measure(call, case, mode, rep)
                    finally:
                        if process:
                            process.terminate()
                            try:
                                process.wait(timeout=30)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait()
                        report.setdefault("logs", {})[str(log_path)] = sha256_path(log_path)
        report["summary"] = summarize(report["samples"],
                                     expected_ids=[c["id"] for c in cases],
                                     repetitions=args.repetitions)
        report["status"] = "completed_diagnostic"
    except Exception as exc:
        report["status"] = "blocked"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
