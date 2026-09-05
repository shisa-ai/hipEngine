"""Capture and reconcile same-host Qwen4Exp HIP/Vulkan semantic owner evidence."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hipengine.benchmark.provenance import collect_model_identity
from scripts.llamacpp_vulkan_perf_summary import parse_perf_text
from scripts.qwen4exp_canonical_ar_bench import (
    DEFAULT_FIXTURE,
    _git_metadata,
    _host_metadata,
    _wait_for_health,
    load_fixture,
)
from scripts.qwen4exp_llamacpp_exact_profile import (
    _completion_payload,
    _decode_prompt,
    _request,
    _select_case,
    _terminate_profiled_process,
)
from scripts.qwen4exp_vulkan_owner_build import PIN, digest

HOST_ID = "55ea6c509d0b49eea8de7094a1023668"
TAXONOMY = "qwen4exp-semantic-owners-v1"
OWNERS = {"moe", "linear", "gr_read", "qsa", "gdn", "ple", "boundary"}
EMPTY_OPS = {"NONE", "VIEW", "RESHAPE", "PERMUTE", "TRANSPOSE"}
OWNER_RE = re.compile(r"^HE_OWNER (0x[0-9a-f]+) (\w+) (\w+) (\S+) (\S+)$")
NODES_RE = re.compile(r" HE_NODES=(0x[0-9a-f]+(?:,0x[0-9a-f]+)*)$")
MODEL_REVISION = "8bdc666649440e9bdc97e16f3f75782c98478ff5"
MODEL_FINGERPRINT = "fb1f2fbf73d588c9ac27f24bade5663bd3da8ac1862f62ee5bf457578a88ec53"
SHARED_SLOTS = {"shared_gate", "shared_up", "shared_down", "shared_expert_gate"}
SHARED_WEIGHTS = re.compile(r"(?:^|\.)ffn_(?:gate|up|down|gate_inp)_shexp\.weight$")


def model_identity(root):
    identity = collect_model_identity(root, revision=MODEL_REVISION)
    if identity["fingerprint"]["value"] != MODEL_FINGERPRINT:
        raise ValueError("model directory does not match the frozen four-shard payload")
    return identity


def check_host():
    if socket.gethostname() != "gfx1151" or Path("/etc/machine-id").read_text().strip() != HOST_ID:
        raise RuntimeError("this campaign requires the pinned Framework physical host")


def decode_name(value):
    return "" if value == "-" else bytes.fromhex(value).decode("utf-8")


def annotated_sections(text):
    """Resolve each timing against metadata known at that point, not the final graph."""
    owners = {}
    sections = []
    current = None
    snapshot = None
    for line in text.splitlines():
        match = OWNER_RE.match(line.strip())
        if match:
            ptr, owner, op, name, weight = match.groups()
            if owner not in OWNERS:
                raise ValueError(f"unknown semantic owner {owner}")
            source_owner = owner
            weight_name = decode_name(weight)
            if owner == "linear" and SHARED_WEIGHTS.search(weight_name):
                owner = "moe"
            owners[ptr] = {
                "owner": owner,
                "source_owner": source_owner,
                "op": op,
                "name": decode_name(name),
                "weight": weight_name,
            }
        if line.strip() == "Vulkan Timings:":
            if current is not None:
                raise ValueError("nested timing section")
            current, snapshot = [line], owners.copy()
        elif current is not None:
            current.append(line)
            if line.strip().startswith("Total time:"):
                parsed = parse_perf_text("\n".join(current))[0]
                if not math.isfinite(parsed.total_us) or parsed.total_us <= 0:
                    raise ValueError("invalid Vulkan total time")
                rows = []
                seen = set()
                for op in parsed.operations:
                    if (
                        not math.isfinite(op.total_us)
                        or op.total_us < 0
                        or not math.isfinite(op.average_us)
                        or op.average_us < 0
                        or op.dispatches <= 0
                    ):
                        raise ValueError("invalid Vulkan operation timing")
                    ids = NODES_RE.search(op.name)
                    node_ids = ids.group(1).split(",") if ids else []
                    nodes = []
                    missing = []
                    for ptr in node_ids:
                        if ptr in seen:
                            raise ValueError(f"node counted twice in one graph: {ptr}")
                        seen.add(ptr)
                        if ptr in snapshot:
                            nodes.append({"id": ptr, **snapshot[ptr]})
                        else:
                            missing.append(ptr)
                    families = {n["owner"] for n in nodes if n["op"] not in EMPTY_OPS}
                    owner = (
                        next(iter(families))
                        if len(families) == 1 and not missing
                        else "mixed_fused"
                        if len(families) > 1 and not missing
                        else "unclassified"
                    )
                    rows.append(
                        {
                            "operation": op.name,
                            "ms": op.total_us / 1000,
                            "dispatches": op.dispatches,
                            "owner": owner,
                            "families": sorted(families),
                            "nodes": nodes,
                            "missing_node_ids": missing,
                        }
                    )
                total = sum(row["ms"] for row in rows)
                if abs(total - parsed.total_us / 1000) > max(0.01, len(rows) * 0.00001):
                    raise ValueError("Vulkan timing lines do not reconcile to total")
                sections.append({"total_ms": parsed.total_us / 1000, "rows": rows})
                current = None
    if current is not None:
        raise ValueError("unfinished timing section")
    return sections


def summarize_sections(sections):
    if not sections:
        raise ValueError("no request-bounded Vulkan sections")
    totals = defaultdict(float)
    unresolved = []
    for section in sections:
        for row in section["rows"]:
            totals[row["owner"]] += row["ms"]
            if row["owner"] in {"unclassified", "mixed_fused"}:
                unresolved.append(row)
    total = sum(s["total_ms"] for s in sections)
    unresolved_ms = totals["unclassified"] + totals["mixed_fused"]
    return {
        "taxonomy": TAXONOMY,
        "total_device_ms": total,
        "owner_ms": dict(totals),
        "unresolved_ms": unresolved_ms,
        "coverage_fraction": 1 - unresolved_ms / total if total else 0,
        "matched_gap_eligible": unresolved_ms == 0,
        "unresolved": unresolved,
        "sections": sections,
    }


def capture_vulkan(args):
    check_host()
    source = args.source_root.resolve()
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip() != PIN:
        raise ValueError("halo-box source pin mismatch")
    fixture, fixture_hash = load_fixture(DEFAULT_FIXTURE)
    cases = [_select_case(fixture, name) for name in args.case_id]
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path = output.with_suffix(".server.log")
    if log_path.exists():
        raise FileExistsError(log_path)
    env = os.environ.copy()
    env.pop("GGML_VK_PERF_LOGGER_CONCURRENT", None)
    if args.reference_only:
        if args.owner_build:
            raise ValueError("reference-only requires the uninstrumented binary")
        for key in (
            "GGML_VK_PERF_LOGGER",
            "GGML_VK_PERF_LOGGER_FREQUENCY",
            "HIPENGINE_VK_OWNER_TRACE",
        ):
            env.pop(key, None)
    else:
        env["GGML_VK_PERF_LOGGER"] = "1"
        env["GGML_VK_PERF_LOGGER_FREQUENCY"] = "1"
        env["HIPENGINE_VK_OWNER_TRACE"] = "1"
    if args.owner_build:
        owned = args.owner_build.resolve()
        env["LD_LIBRARY_PATH"] = str(owned) + ":" + env.get("LD_LIBRARY_PATH", "")
        env["GGML_BACKEND_PATH"] = str(owned)
    command = [
        str(args.server_bin.resolve()),
        "-m",
        str(args.model.resolve()),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--parallel",
        "1",
        "--no-webui",
        "-ngl",
        "999",
        "-fa",
        "on",
        "-ctk",
        "bf16",
        "-ctv",
        "bf16",
        "-c",
        "4352",
        "-b",
        "8192",
        "-ub",
        "2048",
        "-t",
        "4",
    ]
    report = {
        "schema": 1,
        "kind": "qwen4exp_vulkan_semantic_capture",
        "status": "running",
        "performance_claim": False,
        "taxonomy": TAXONOMY,
        "host": _host_metadata(),
        "source": _git_metadata(ROOT),
        "comparator_source": _git_metadata(source),
        "command": command,
        "server_sha256": digest(args.server_bin),
        "fixture_sha256": fixture_hash,
        "model": str(args.model.resolve()),
        "log": str(log_path),
        "cases": [],
        "logger_concurrent": False,
        "logger_frequency": 1,
    }
    report["controller_sha256"] = digest(Path(__file__))
    report["model_identity"] = model_identity(args.model.resolve().parent)
    report["quant"] = "UD-Q4_K_XL"
    report["kv_dtype"] = "BF16"
    report["reference_only"] = args.reference_only
    report["environment"] = {
        key: env.get(key)
        for key in (
            "GGML_VK_PERF_LOGGER",
            "GGML_VK_PERF_LOGGER_FREQUENCY",
            "HIPENGINE_VK_OWNER_TRACE",
            "GGML_BACKEND_PATH",
            "LD_LIBRARY_PATH",
        )
    }
    if args.owner_build:
        report["instrumentation"] = json.loads((args.owner_build / "build.json").read_text())
        for path, expected in report["instrumentation"]["files"].items():
            if digest(path) != expected:
                raise ValueError(f"instrumentation build input/output changed: {path}")
    process = None
    with log_path.open("wb") as log:
        try:
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True
            )
            _wait_for_health(args.host, args.port, args.startup_timeout)
            maps = Path(f"/proc/{process.pid}/maps").read_text()
            if args.owner_build:
                for lib in ("libllama.so.0.3.0", "libggml-vulkan.so.0.22.0"):
                    path = str(args.owner_build.resolve() / lib)
                    if path not in maps:
                        raise RuntimeError(f"instrumented library not loaded: {path}")
            report["loaded_libraries"] = sorted(
                {
                    line.split()[-1]
                    for line in maps.splitlines()
                    if "libllama" in line or "libggml-vulkan" in line
                }
            )
            for case in cases:
                prompt = case["prompt_token_ids"]
                warm = _request(args, _completion_payload(prompt, n_predict=1, cache_prompt=False))
                start = (
                    0
                    if args.reference_only
                    else len(annotated_sections(log_path.read_text(errors="replace")))
                )
                prefill = _request(
                    args, _completion_payload(prompt, n_predict=1, cache_prompt=False)
                )
                after_prefill = (
                    0
                    if args.reference_only
                    else len(annotated_sections(log_path.read_text(errors="replace")))
                )
                decode = _request(
                    args,
                    _completion_payload(
                        _decode_prompt(case, prefill["raw_response"]),
                        n_predict=1,
                        cache_prompt=True,
                    ),
                )
                sections = (
                    []
                    if args.reference_only
                    else annotated_sections(log_path.read_text(errors="replace"))
                )
                if prefill["response"]["prompt_n"] != case["prompt_tokens"]:
                    raise ValueError("prefill prompt count mismatch")
                if decode["response"]["prompt_n"] != 1:
                    raise ValueError("decode must evaluate exactly one appended root")
                if warm["response"]["output_token_ids"] != prefill["response"]["output_token_ids"]:
                    raise ValueError("Vulkan prefill repeat mismatch")
                prefill.pop("raw_response")
                decode.pop("raw_response")
                report["cases"].append(
                    {
                        "id": case["id"],
                        "prompt_tokens": case["prompt_tokens"],
                        "prompt_token_ids_sha256": case["prompt_token_ids_sha256"],
                        "prefill": prefill,
                        "decode": decode,
                        "prefill_profile": None
                        if args.reference_only
                        else summarize_sections(sections[start:after_prefill]),
                        "decode_profile": None
                        if args.reference_only
                        else summarize_sections(sections[after_prefill:]),
                        "section_bounds": [start, after_prefill, len(sections)],
                    }
                )
                output.write_text(json.dumps(report, indent=2) + "\n")
                print(case["id"], "captured", flush=True)
            report["status"] = "captured"
        except Exception as exc:
            report["status"] = "failed"
            report["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if process:
                _terminate_profiled_process(process)
                report["returncode"] = process.returncode
            report["log_sha256"] = digest(log_path)
            output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def normalize_hip_roles(raw):
    owners = defaultdict(float)
    for row in raw["roles"]:
        if not math.isfinite(float(row["ms"])) or float(row["ms"]) < 0:
            raise ValueError("invalid HIP owner timing")
        family = row["name"].split(":", 1)[0]
        if family == "linear" and row["name"].rsplit(".", 1)[-1] in SHARED_SLOTS:
            family = "moe"
        family = {
            "qsa_prefill": "qsa",
            "qsa_decode": "qsa",
            "prefill_boundary": "boundary",
            "decode_boundary": "boundary",
        }.get(family, family)
        if family not in OWNERS:
            raise ValueError(f"unrecognized HIP owner {family}")
        owners[family] += float(row["ms"])
    if raw["unattributed_ms"] != 0:
        raise ValueError("HIP trace has unattributed work")
    total = sum(owners.values())
    if abs(total - raw["attributed_ms"]) > 0.01:
        raise ValueError("HIP owner sums do not reconcile")
    return {
        "taxonomy": TAXONOMY,
        "owner_ms": dict(owners),
        "total_device_ms": total,
        "profiled_window_ms": raw["window_ms"],
        "coverage_fraction": 1.0,
        "matched_gap_eligible": True,
    }


def capture_hipengine(args):
    from scripts.qwen4exp_role_analyze import analyze

    check_host()
    if not _git_metadata(ROOT)["tracked_clean"]:
        raise ValueError("commit the validated collector before frozen HIP captures")
    fixture, fixture_hash = load_fixture(DEFAULT_FIXTURE)
    cases = [_select_case(fixture, name) for name in args.case_id]
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    run_root = output.with_suffix(".raw")
    run_root.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": 1,
        "kind": "qwen4exp_hip_semantic_capture",
        "status": "running",
        "performance_claim": False,
        "taxonomy": TAXONOMY,
        "host": _host_metadata(),
        "source": _git_metadata(ROOT),
        "fixture_sha256": fixture_hash,
        "model_identity": model_identity(args.model_root),
        "quant": "UD-Q4_K_XL",
        "kv_dtype": "BF16",
        "cases": [],
    }
    try:
        for case in cases:
            tokens = case["prompt_tokens"]
            for phase in ("prefill", "decode"):
                directory = run_root / f"{case['id']}-{phase}"
                directory.mkdir()
                raw_output = directory / "child.json"
                helper = (
                    "qwen4exp_profile_gap.py"
                    if phase == "prefill"
                    else "qwen4exp_context_decode_profile.py"
                )
                command = [
                    sys.executable,
                    str(ROOT / "scripts" / helper),
                    "--model-root",
                    str(args.model_root),
                    "--case-id",
                    case["id"],
                    "--profile",
                    "--role-markers",
                    "--compiler-version-file",
                    str(args.compiler_version_file),
                    "--require-cached-build",
                    "--output",
                    str(raw_output),
                ]
                if phase == "prefill":
                    command += ["--mode", "prefill", "--repetitions", "1"]
                    markers = [f"qwen4exp_prefill_p{tokens}_0"]
                else:
                    command += ["--live-count", str(tokens + 1), "--repetitions", "3"]
                    markers = [f"qwen4exp_decode_live{tokens + 1}_rep{i}" for i in range(3)]
                profiled = [
                    str(args.rocprof_bin),
                    "--kernel-trace",
                    "--hip-trace",
                    "--marker-trace",
                    "--output-format",
                    "csv",
                    "-d",
                    str(directory / "trace"),
                    "--",
                    *command,
                ]
                env = os.environ.copy()
                env["HIPENGINE_REQUIRE_CACHED_BUILD"] = "1"
                env["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)
                env["HIPENGINE_HIP_ARCH"] = "gfx1151"
                with (directory / "process.log").open("wb") as log:
                    subprocess.run(
                        profiled,
                        cwd=ROOT,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                raw = json.loads(raw_output.read_text())
                closed = raw.get(
                    "memory_after_close", raw.get("lifecycle", {}).get("after_close", {})
                )
                if (
                    closed.get("active_allocations") != 0
                    or closed.get("current_allocated_bytes") != 0
                ):
                    raise ValueError("HIP capture did not close tracked ownership")
                if phase == "decode" and raw.get("status") != "passed":
                    raise ValueError("HIP decode repeat/lifecycle/source gate failed")
                profiles = []
                for marker in markers:
                    roles = analyze(directory / "trace" / socket.gethostname(), marker)
                    (directory / f"{marker}.roles.json").write_text(
                        json.dumps(roles, indent=2) + "\n"
                    )
                    profiles.append(normalize_hip_roles(roles))
                count = len(profiles)
                averaged = {
                    owner: sum(p["owner_ms"].get(owner, 0) for p in profiles) / count
                    for owner in sorted(OWNERS)
                }
                report["cases"].append(
                    {
                        "id": case["id"],
                        "phase": phase,
                        "prompt_tokens": tokens,
                        "prompt_token_ids_sha256": case["prompt_token_ids_sha256"],
                        "live_count": tokens + 1 if phase == "decode" else None,
                        "command": profiled,
                        "raw_path": str(raw_output),
                        "raw_sha256": digest(raw_output),
                        "raw": raw,
                        "profile": {
                            "owner_ms": averaged,
                            "total_device_ms": sum(averaged.values()),
                            "profiled_window_ms": sum(p["profiled_window_ms"] for p in profiles)
                            / count,
                            "matched_gap_eligible": True,
                            "coverage_fraction": 1.0,
                            "repetitions": count,
                        },
                    }
                )
                output.write_text(json.dumps(report, indent=2) + "\n")
                print(case["id"], phase, "captured", flush=True)
        report["status"] = "captured"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def check_capture_identity(left, right):
    for name in ("taxonomy", "fixture_sha256", "quant", "kv_dtype"):
        if left.get(name) != right.get(name) or left.get(name) is None:
            raise ValueError(f"capture mismatch or missing field: {name}")
    if (
        left["host"]["machine_id"] != right["host"]["machine_id"]
        or left["host"]["machine_id"] != HOST_ID
    ):
        raise ValueError("physical host mismatch")
    if left["model_identity"]["fingerprint"] != right["model_identity"]["fingerprint"]:
        raise ValueError("model fingerprint mismatch")
    if left["status"] != "captured" or right["status"] != "captured":
        raise ValueError("capture incomplete")


def join_captures(hip, vk):
    check_capture_identity(hip, vk)
    comparisons = []
    for row in hip["cases"]:
        other = next(c for c in vk["cases"] if c["id"] == row["id"])
        phase = row["phase"]
        if row["prompt_tokens"] != other["prompt_tokens"]:
            raise ValueError("workload shape mismatch")
        if row["prompt_token_ids_sha256"] != other["prompt_token_ids_sha256"]:
            raise ValueError("prompt identity mismatch")
        right = other[f"{phase}_profile"]
        if (
            not right
            or not right["matched_gap_eligible"]
            or not row["profile"]["matched_gap_eligible"]
        ):
            raise ValueError("unresolved or missing semantic ownership")
        if phase == "decode" and (
            row["live_count"] != other["prompt_tokens"] + 1
            or other["decode"]["response"]["prompt_n"] != 1
        ):
            raise ValueError("decode context mismatch")
        left = row["profile"]
        owners = []
        for owner in sorted(OWNERS):
            he_ms, vk_ms = left["owner_ms"].get(owner, 0), right["owner_ms"].get(owner, 0)
            owners.append(
                {
                    "owner": owner,
                    "hipengine_ms": he_ms,
                    "vulkan_ms": vk_ms,
                    "hip_over_vulkan": he_ms / vk_ms if vk_ms else None,
                }
            )
        comparisons.append(
            {
                "id": row["id"],
                "phase": phase,
                "owners": owners,
                "hipengine_device_ms": left["total_device_ms"],
                "vulkan_device_ms": right["total_device_ms"],
                "hipengine_profiled_window_ms": left["profiled_window_ms"],
            }
        )
    return {
        "schema": 1,
        "kind": "qwen4exp_semantic_owner_comparison",
        "taxonomy": TAXONOMY,
        "performance_claim": False,
        "comparisons": comparisons,
        "limits": [
            "HIP kernel durations versus Vulkan query intervals: semantic alignment, not identical instruments.",
            "MoE is the complete FFN including shared projections and router; fine source roles remain available.",
            "Unprofiled canonical AR remains the throughput source of truth.",
            "Fixed-live decode profiles are not tg128-average family costs.",
        ],
    }


def refresh_vulkan_sections(capture):
    log = Path(capture["log"])
    if digest(log) != capture["log_sha256"]:
        raise ValueError("Vulkan source log changed")
    sections = annotated_sections(log.read_text(errors="replace"))
    for case in capture["cases"]:
        start, split, end = case["section_bounds"]
        if not 0 <= start < split < end <= len(sections):
            raise ValueError("invalid request section bounds")
        case["prefill_profile"] = summarize_sections(sections[start:split])
        case["decode_profile"] = summarize_sections(sections[split:end])
    return capture


def compare_reference(reference, measured):
    # Reference captures need no owner labels, but must pin the same executable,
    # payload path, fixture and physical host as the instrumented library run.
    for key in ("fixture_sha256", "model", "server_sha256"):
        if not reference.get(key) or reference[key] != measured.get(key):
            raise ValueError(f"instrumentation reference mismatch: {key}")
    if reference["host"]["machine_id"] != HOST_ID or measured["host"]["machine_id"] != HOST_ID:
        raise ValueError("instrumentation reference host mismatch")
    if any(
        c["comparator_source"]["head"] != PIN or c["status"] != "captured"
        for c in (reference, measured)
    ):
        raise ValueError("instrumentation reference source/status mismatch")
    if not reference.get("reference_only") or measured.get("reference_only"):
        raise ValueError("reference must have profiling disabled")
    for capture in (reference, measured):
        argv = capture["command"]
        for flag in ("-ctk", "-ctv"):
            if argv[argv.index(flag) + 1] != "bf16":
                raise ValueError("instrumentation reference KV mismatch")
    for case in measured["cases"]:
        old = next(c for c in reference["cases"] if c["id"] == case["id"])
        for phase in ("prefill", "decode"):
            if (
                old[phase]["response"]["output_token_ids"]
                != case[phase]["response"]["output_token_ids"]
            ):
                raise ValueError(f"instrumentation output mismatch: {case['id']} {phase}")


def baseline_commands(queue, run_root):
    script = str(ROOT / "scripts/qwen4exp_canonical_ar_bench.py")
    common = [
        "--fixture",
        str(ROOT / queue["fixture"]["path"]),
        "--warmups",
        "1",
        "--repetitions",
        "3",
    ]
    model_root = queue["model"]["root"]
    commands = [
        (
            "hipengine",
            [
                sys.executable,
                script,
                "hipengine",
                "--model-root",
                model_root,
                "--execution-profile",
                "production",
                "--prefill-chunk-size",
                "512",
                "--compiler-version-file",
                "/tmp/hipengine-hipcc-version.txt",
                "--require-cached-build",
                "--output",
                str(run_root / "hipengine.json"),
                *common,
            ],
        )
    ]
    for index, backend in enumerate(("vulkan", "hip")):
        label = f"halo-box-{backend}"
        binary = queue["comparator"][f"{backend}_binary"]
        argv = [
            sys.executable,
            script,
            "llamacpp",
            "--server-bin",
            binary,
            "--source-root",
            queue["comparator"]["source"],
            "--engine-label",
            label,
            "--model",
            str(Path(model_root) / "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"),
            "--port",
            str(18138 + index),
            "--output",
            str(run_root / f"{label}.json"),
            "--server-log",
            str(run_root / f"{label}.server.log"),
            *common,
        ]
        argv += [f"--server-arg={arg}" for arg in queue["comparator"]["server_args"]]
        commands.append((label, argv))
    return commands


def run_baselines(args):
    check_host()
    queue = json.loads(args.queue.read_text())
    if queue["host"]["machine_id"] != HOST_ID or queue["comparator"]["commit"] != PIN:
        raise ValueError("queue identity mismatch")
    model_identity(Path(queue["model"]["root"]))
    if digest(ROOT / queue["fixture"]["path"]) != queue["fixture"]["sha256"]:
        raise ValueError("queue fixture mismatch")
    source = _git_metadata(ROOT)
    if not source["tracked_clean"]:
        raise ValueError("freeze a clean tracked revision before baselines")
    run_root = args.output.resolve().with_suffix(".raw")
    comparator = _git_metadata(Path(queue["comparator"]["source"]))
    if comparator["head"] != PIN or not comparator["tracked_clean"]:
        raise ValueError("comparator source pin/cleanliness mismatch")
    binary_paths = set()
    for backend in ("vulkan", "hip"):
        binary = Path(queue["comparator"][f"{backend}_binary"])
        binary_paths.add(binary.resolve())
        binary_paths.update(p.resolve() for p in binary.parent.glob("lib*.so*"))
    binary_hashes = {str(p): digest(p) for p in sorted(binary_paths)}
    if args.resume:
        report = json.loads(args.output.read_text())
        if (
            report["source"] != source
            or report["queue_sha256"] != digest(args.queue)
            or report["binary_hashes"] != binary_hashes
            or report["host"]["machine_id"] != HOST_ID
        ):
            raise ValueError("resume does not match the frozen run")
        for stage in report["stages"]:
            if digest(stage["path"]) != stage["sha256"]:
                raise ValueError("completed baseline artifact changed")
        report["status"] = "running"
        report.pop("error", None)
    else:
        if args.output.exists():
            raise FileExistsError(args.output)
        run_root.mkdir(parents=True, exist_ok=False)
        report = {
            "schema": 1,
            "kind": "qwen4exp_framework_baseline_refresh",
            "status": "running",
            "source": source,
            "host": _host_metadata(),
            "queue_sha256": digest(args.queue),
            "binary_hashes": binary_hashes,
            "stages": [],
        }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    try:
        for label, argv in baseline_commands(queue, run_root):
            if any(stage["engine"] == label for stage in report["stages"]):
                continue
            if _git_metadata(ROOT) != source:
                raise ValueError("source changed after baseline freeze")
            if any(digest(path) != expected for path, expected in binary_hashes.items()):
                raise ValueError("baseline binary/library changed after freeze")
            env = os.environ.copy()
            for name in (
                "GGML_VK_PERF_LOGGER",
                "GGML_VK_PERF_LOGGER_CONCURRENT",
                "GGML_VK_PERF_LOGGER_FREQUENCY",
                "HIPENGINE_VK_OWNER_TRACE",
            ):
                env.pop(name, None)
            env["HIPENGINE_COMPILER_VERSION_FILE"] = "/tmp/hipengine-hipcc-version.txt"
            if label != "hipengine":
                backend = label.removeprefix("halo-box-")
                binary_dir = str(Path(queue["comparator"][f"{backend}_binary"]).parent)
                env["GGML_BACKEND_PATH"] = binary_dir
                env["LD_LIBRARY_PATH"] = binary_dir + ":" + env.get("LD_LIBRARY_PATH", "")
            else:
                env.pop("GGML_BACKEND_PATH", None)
                env["HIPENGINE_HIP_ARCH"] = "gfx1151"
            start = time.monotonic()
            with (run_root / f"{label}.process.log").open("wb") as log:
                subprocess.run(
                    argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
                )
            result_path = run_root / f"{label}.json"
            result = json.loads(result_path.read_text())
            if len(result["samples"]) != 36:
                raise ValueError("baseline must contain all 36 measured samples")
            report["stages"].append(
                {
                    "engine": label,
                    "command": argv,
                    "elapsed_seconds": time.monotonic() - start,
                    "path": str(result_path),
                    "sha256": digest(result_path),
                }
            )
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(label, "baseline complete", flush=True)
        if _git_metadata(ROOT) != source:
            raise ValueError("source changed during baselines")
        report["status"] = "captured"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


def render_comparison(result):
    lines = []
    for comparison in result["comparisons"]:
        lines += [
            f"### {comparison['id']} {comparison['phase']}",
            "",
            "| Owner | hipEngine (ms) | halo-box Vulkan (ms) | HE / Vulkan |",
            "| --- | ---: | ---: | ---: |",
        ]
        for row in comparison["owners"]:
            ratio = (
                f"{row['hip_over_vulkan']:.3f}x" if row["hip_over_vulkan"] is not None else "n/a"
            )
            label = "MoE/FFN (routed + shared)" if row["owner"] == "moe" else row["owner"]
            lines.append(
                f"| {label} | {row['hipengine_ms']:.3f} | {row['vulkan_ms']:.3f} | {ratio} |"
            )
        lines += [
            "",
            "HIP kernel sums and Vulkan query intervals are diagnostic device costs, not unprofiled wall rates.",
            "",
        ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    capture = sub.add_parser("capture-vulkan")
    capture.add_argument("--source-root", type=Path, required=True)
    capture.add_argument("--server-bin", type=Path, required=True)
    capture.add_argument("--owner-build", type=Path)
    capture.add_argument("--reference-only", action="store_true")
    capture.add_argument("--model", type=Path, required=True)
    capture.add_argument("--case-id", action="append", required=True)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--host", default="127.0.0.1")
    capture.add_argument("--port", type=int, default=18137)
    capture.add_argument("--startup-timeout", type=float, default=1800)
    capture.add_argument("--request-timeout", type=float, default=600)
    hip = sub.add_parser("capture-hipengine")
    hip.add_argument("--model-root", type=Path, required=True)
    hip.add_argument("--case-id", action="append", required=True)
    hip.add_argument("--compiler-version-file", type=Path, required=True)
    hip.add_argument("--rocprof-bin", type=Path, default=Path("rocprofv3"))
    hip.add_argument("--output", type=Path, required=True)
    join = sub.add_parser("join")
    join.add_argument("--hipengine", type=Path, required=True)
    join.add_argument("--vulkan", type=Path, required=True)
    join.add_argument("--reference", type=Path, required=True)
    join.add_argument("--output", type=Path, required=True)
    join.add_argument("--markdown", type=Path)
    baseline = sub.add_parser("baselines")
    baseline.add_argument(
        "--queue",
        type=Path,
        default=ROOT / "benchmarks/results/2026-09-05-framework-qwen4exp-family-refresh-queue.json",
    )
    baseline.add_argument("--output", type=Path, required=True)
    baseline.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.mode == "capture-vulkan":
        capture_vulkan(args)
    elif args.mode == "capture-hipengine":
        capture_hipengine(args)
    elif args.mode == "baselines":
        run_baselines(args)
    else:
        he, vk, reference = [
            json.loads(p.read_text()) for p in (args.hipengine, args.vulkan, args.reference)
        ]
        compare_reference(reference, vk)
        refresh_vulkan_sections(vk)
        result = join_captures(he, vk)
        result["sources"] = {
            str(p): digest(p) for p in (args.hipengine, args.vulkan, args.reference)
        }
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        if args.markdown:
            args.markdown.write_text(render_comparison(result))


if __name__ == "__main__":
    main()
