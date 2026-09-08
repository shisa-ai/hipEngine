from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import platform
import os
import sys

ROOT = Path("/home/lhl/hipEngine-main")
RAW = Path("/tmp/engine-compare-final")
HELDOUT = {"code_markdown_table", "general_en_explain", "general_ja_explain", "mixed_ja_en_review"}
artifact = {
    "schema": 1, "kind": "rx7900xtx_three_engine_comparison", "date": "2026-09-08",
    "status": "complete", "performance_claim": False,
    "model": {"path": "/models/gguf/Qwen3.8-27B-Q4_K_M.gguf", "quant": "Q4_K_M",
              "bytes": 17106773984,
              "sha256": "7b2aec3b9ababdfd75aa17552ee95607d866e44decf547f6f12fcef85cc89f1b"},
    "hardware": {"host": "epyc", "gpu": "AMD Radeon RX 7900 XTX", "arch": "gfx1100",
                 "physical_gpu_index": 1, "pci": "0000:10:00.0",
                 "unique_id": "0xcc4d02090dc9c3ff", "vram_bytes": 25753026560,
                 "hipcc": Path("/tmp/hipengine-hipcc-version.txt").read_text()},
    "hipengine_fix_commit": "6c01f1f1c",
    "hipengine_fix_worklog": "worklog/entries/20260908T120624.010092Z-lhl-repair-c1-mtp-native-scratch-and-fallback-owners-716c2b.md",
    "protocols": {
        "pp_tg": "BF16 KV, repeated token 9707, 512/8192 prompt and 128 timed decode transitions, three measured runs. hipEngine direct resident graph decode, one full warmup plus one decode warmup token. External server internal phase timers, short request warmup, 129 visible outputs, b4096/ub1024, capacity8704.",
        "mtp_short": "All ten canonical prompts, all four categories and six-train/four-heldout split, raw token IDs, greedy, 25 visible outputs/24 timed transitions, three repetitions, capacity1024, explicit fixed B3 and a true no-MTP AR baseline.",
        "mtp_long": "Same full suite and capacity, 129 visible outputs/128 timed transitions, one repetition, explicit B3 versus true AR.",
        "capacity": "No KV eviction or weight offload; entire repeated-token prompt plus eight timed decode transitions must complete. BF16 hipEngine point has one additional decode warmup. Observed bounds, not operational reserves or exhaustive maxima. No new long-context task-quality gate.",
        "profile": "Current resident selectors / explicit native MTP with AR-ID equality checks; no new named-profile arithmetic or donor-kernel promotion.",
        "isolation": "Final timing and repaired 128K point use a detached hipEngine snapshot, <=128MiB idle admission, and 200ms KFD process-group checks. Foreign GPU1 use aborts the run. Other work may use GPU0.",
    },
    "prompt_fixture": {
        "path": "benchmarks/prompts/mtpbench-code-general-ja.jsonl",
        "sha256": hashlib.sha256((ROOT / "benchmarks/prompts/mtpbench-code-general-ja.jsonl").read_bytes()).hexdigest(),
        "heldout_ids": sorted(HELDOUT),
    },
    "runs": {}, "output_id_rows": {}, "prompt_id_rows": {},
    "pp_tg_summary": {}, "mtp_summary": {}, "capacity": [],
}
artifact["hardware"]["platform"] = platform.platform()
artifact["hardware"]["python_executable"] = sys.executable
artifact["hardware"]["python_version"] = sys.version
artifact["hardware"]["library_path"] = os.environ.get("LD_LIBRARY_PATH")


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ids_ref(ids):
    assert ids and all(type(t) is int and 0 <= t < 248320 for t in ids)
    digest = hashlib.sha256(b"".join(t.to_bytes(8, "little", signed=True) for t in ids)).hexdigest()
    assert artifact["output_id_rows"].setdefault(digest, ids) == ids
    return digest


def case(name, success=True):
    matches = []
    for path in RAW.glob(f"{name}*-monitor.json"):
        suffix = path.name[len(name):-len("-monitor.json")]
        if suffix and not suffix.startswith("-retry"):
            continue
        monitor = json.loads(path.read_text())
        if success and monitor["returncode"] != 0:
            continue
        assert not monitor["foreign_gpu1_owners"]
        assert monitor["source_clean"]
        assert monitor["memory"]["baseline_bytes"] < 128 * (1 << 20)
        matches.append((monitor["started_unix"], path, monitor))
    assert matches, f"missing completed case: {name}"
    _, monitor_path, monitor = max(matches)
    data_path = monitor_path.with_name(monitor_path.name.replace("-monitor.json", ".json"))
    data = json.loads(data_path.read_text()) if data_path.exists() else {}
    evidence = dict(monitor=monitor, raw_path=str(data_path),
                    raw_sha256=fingerprint(data_path) if data_path.exists() else None)
    artifact["runs"][name] = evidence
    return data, evidence


def external_rows(data):
    result = []
    for row in data["rows"]:
        ids = row["output_ids"]
        timing = row["timings"]
        assert len(ids) == timing["predicted_n"] == row["decode_transitions"] + 1
        assert timing["predicted_ms"] > 0
        if "prompt_ids" in row:
            prior = artifact["prompt_id_rows"].setdefault(row["id"], row["prompt_ids"])
            assert prior == row["prompt_ids"]
        result.append(dict(
            id=row["id"], category=row["category"], run=row["repetition"],
            prompt_tokens=row["prompt_tokens"], output_count=len(ids), output_ids_ref=ids_ref(ids),
            transitions=row["decode_transitions"], decode_seconds=timing["predicted_ms"] / 1000,
            prefill_seconds=timing["prompt_ms"] / 1000,
            accepted=timing.get("draft_n_accepted", 0), drafted=timing.get("draft_n", 0),
        ))
    return result


def hip_rows(rows):
    result = []
    for row in rows:
        result.append(dict(
            id=row["id"], category=row["category"], run=row["run"],
            prompt_tokens=row["prompt_tokens"], output_count=row["visible_outputs"],
            output_ids_ref=ids_ref(row["token_ids"]), transitions=row["timed_transitions"],
            decode_seconds=row["decode_seconds"], prefill_seconds=row["prefill_seconds"],
            accepted=row.get("accepted_draft_tokens", 0), drafted=row.get("proposed_draft_tokens", 0),
            verify_modes=dict(Counter(c["target_verify_mode"] for c in row.get("cycle_records", []))),
            exact_greedy_match=row.get("exact_greedy_match"),
        ))
    return result


def metrics(rows):
    return dict(requests=len(rows), transitions=sum(r["transitions"] for r in rows),
                decode_seconds=sum(r["decode_seconds"] for r in rows),
                tok_s=sum(r["transitions"] for r in rows) / sum(r["decode_seconds"] for r in rows),
                accepted=sum(r["accepted"] for r in rows), drafted=sum(r["drafted"] for r in rows))


def paired_summary(ar, mtp):
    assert len(ar) == len(mtp)
    oracle = {(r["id"], r["run"]): r for r in ar}
    checks = []
    for row in mtp:
        reference = oracle[(row["id"], row["run"])]
        assert reference["prompt_tokens"] == row["prompt_tokens"]
        checks.append(dict(id=row["id"], run=row["run"],
                           exact=reference["output_ids_ref"] == row["output_ids_ref"]))
    a, m = metrics(ar), metrics(mtp)
    result = dict(ar=a, mtp=m, ratio=m["tok_s"] / a["tok_s"], checks=checks,
                  exact_matches=sum(c["exact"] for c in checks))
    result["categories"] = {}
    for category in sorted({r["category"] for r in ar}):
        aa, mm = metrics([r for r in ar if r["category"] == category]), metrics([r for r in mtp if r["category"] == category])
        result["categories"][category] = dict(ar=aa, mtp=mm, ratio=mm["tok_s"] / aa["tok_s"])
    result["splits"] = {}
    for name, heldout in (("train", False), ("heldout", True)):
        aa = metrics([r for r in ar if (r["id"] in HELDOUT) == heldout])
        mm = metrics([r for r in mtp if (r["id"] in HELDOUT) == heldout])
        result["splits"][name] = dict(ar=aa, mtp=mm, ratio=mm["tok_s"] / aa["tok_s"])
    for label, rows in (("ar", ar), ("mtp", mtp)):
        repeated = len({r["run"] for r in rows}) > 1
        result[label]["repeat_stable"] = (
            all(r["output_ids_ref"] == next(v for v in rows if v["id"] == r["id"])["output_ids_ref"] for r in rows)
            if repeated else None
        )
    return result


for horizon in ("short", "long"):
    data, evidence = case(f"hipengine-c1-{horizon}")
    assert not data["provenance"]["dirty"]
    assert data["correctness"]["all_exact_greedy"] and data["correctness"]["all_gpu_accept_match_cpu"]
    assert data["memory_after_close"]["active_allocations"] == 0
    ar = hip_rows(data["rows"]["true_ar"])
    mtp = hip_rows(data["rows"]["mtp"]["3"])
    evidence.update(provenance=data["provenance"], correctness=data["correctness"],
                    memory_after_close=data["memory_after_close"], ar=ar, mtp=mtp)
    artifact["mtp_summary"][f"hipEngine-{horizon}"] = paired_summary(ar, mtp)

for name, ar_case, mtp_case in (
    ("nasone32-short", "nasone32-ar", "nasone32-mtp"),
    ("nasone32-sequential-short", "nasone32-ar-sequential", "nasone32-mtp-sequential"),
    ("nasone32-long", "nasone32-ar-long", "nasone32-mtp-long"),
    ("strix-llama.cpp-short", "strix-llama-ar", "strix-llama-mtp"),
    ("strix-llama.cpp-long", "strix-llama-ar-long", "strix-llama-mtp-long"),
):
    rows = []
    for label in (ar_case, mtp_case):
        data, evidence = case(label)
        assert data["status"] == "complete"
        selected = external_rows(data)
        evidence.update(server_command=data["server_command"], source_commit=data["source_commit"],
                        kv=data["kv"], mode=data["mode"], rows=selected)
        rows.append(selected)
    artifact["mtp_summary"][name] = paired_summary(*rows)

for n in (512, 8192):
    data, evidence = case(f"hipengine-p{n}")
    assert not data["shipping_ar_route_mismatch"]
    assert data["summary"]["finite_final_logits_all"]
    assert all(r["effective_graph_replay_decode"] for r in data["runs"] if r["measured"])
    evidence.update(provenance=data["provenance"], summary=data["summary"])
    artifact["pp_tg_summary"][f"hipEngine-{n}"] = dict(
        pp=data["summary"]["prefill_tok_s"]["median"], tg=data["summary"]["decode_tok_s"]["median"])

for name, label in (("nasone32", "nasone32-shapes"), ("strix-llama.cpp", "strix-llama-shapes")):
    data, evidence = case(label)
    evidence.update(server_command=data["server_command"], source_commit=data["source_commit"],
                    rows=external_rows(data), kv=data["kv"])
    for n in (512, 8192):
        selected = [r for r in data["rows"] if r["prompt_tokens"] == n]
        artifact["pp_tg_summary"][f"{name}-{n}"] = dict(
            pp=statistics.median(r["prefill_tok_s"] for r in selected),
            tg=statistics.median(r["decode_tok_s"] for r in selected))

data, evidence = case("hipengine-pure-int8-128k")
assert data["summary"]["finite_final_logits_all"]
families = data["persistent_session_memory"]["snapshots"]["before_close"]["owned_session_breakdown"]["families"]
assert families["decode_scratch"]["by_component_bytes"]["full_attention_bf16_mirrors"] == 0
evidence.update(provenance=data["provenance"], memory_families={
    k: {f:v for f,v in value.items() if f != "bulk_prefill_scratch_census"}
    for k,value in families.items()
})
evidence["correctness"] = {
    "finite_final_logits_all": data["summary"]["finite_final_logits_all"],
    "final_token_ids": data["summary"]["final_token_ids"],
}
evidence["lifecycle_memory"] = data["persistent_session_memory"]["summary"]
artifact["capacity"].append(dict(engine="hipEngine", kv="pure INT8 / FP32 scales",
                                prompt_tokens=131072, passed=True, source_case="hipengine-pure-int8-128k",
                                peak_gib=evidence["monitor"]["memory"]["peak_gib"]))

# Earlier uncontaminated capacity points have unchanged AR code or pinned external source.
for label, engine, kv, prompt, passed in (
    ("hip-bf16-112k", "hipEngine", "BF16", 114688, True),
    ("hip-bf16-128k", "hipEngine", "BF16", 131072, False),
    ("hip-pure-int8-139264", "hipEngine", "pure INT8 / FP32 scales", 139264, False),
):
    monitor_path = Path(f"/tmp/nasone-{label}-monitor.json")
    monitor = json.loads(monitor_path.read_text())
    assert (monitor["returncode"] == 0) == passed
    artifact["runs"][label] = dict(monitor=monitor, raw_monitor_sha256=fingerprint(monitor_path))
    if passed:
        result_path = Path(f"/tmp/nasone-{label}.json")
        result = json.loads(result_path.read_text())
        assert result["summary"]["finite_final_logits_all"]
        assert result["summary"]["final_token_ids"] == [9707]
        artifact["runs"][label].update(
            raw_sha256=fingerprint(result_path), provenance=result["provenance"],
            summary=result["summary"], lifecycle_memory=result["persistent_session_memory"]["summary"],
        )
    if not passed:
        log = Path(f"/tmp/nasone-{label}.log").read_text()
        assert "out of memory" in log
        artifact["runs"][label]["error_tail"] = log[-1600:]
    artifact["capacity"].append(dict(engine=engine, kv=kv, prompt_tokens=prompt,
                                    passed=passed, source_case=label, peak_gib=monitor["memory"]["peak_gib"]))

for label, kv, prompt, passed in (
    ("bf16-112k", "BF16", 114688, True), ("bf16-128k", "BF16", 131072, False),
    ("q8-192k", "Q8_0", 196608, True), ("q8-224k", "Q8_0", 229376, False),
):
    path = Path(f"/tmp/nasone-{label}.json")
    data = json.loads(path.read_text())
    assert (data["status"] == "complete") == passed
    assert data["memory"]["baseline_bytes"] < 128 * (1 << 20)
    if passed:
        assert len(data["rows"]) == 1
        assert data["rows"][0]["prompt_tokens"] == prompt
        assert data["rows"][0]["decode_transitions"] == 8
        assert data["rows"][0]["output_ids"] == [9707] * 9
    artifact["runs"][f"nasone32-{label}"] = dict(raw_sha256=fingerprint(path), data=data)
    if not passed:
        log = path.with_suffix(".log").read_text()
        assert "out of memory" in log
        artifact["runs"][f"nasone32-{label}"]["error_tail"] = log[-1600:]
    artifact["capacity"].append(dict(engine="nasone32", kv=kv, prompt_tokens=prompt, passed=passed,
                                    source_case=f"nasone32-{label}", peak_gib=data["memory"]["peak_gib"]))

for high, low in (
    ("strix-llama-bf16-128k", "strix-llama-bf16-112k"),
    ("strix-llama-q8-224k", "strix-llama-q8-192k"),
):
    _, high_evidence = case(high, success=False)
    if high_evidence["monitor"]["returncode"] != 0:
        case(low)

for label, kv, prompt in (
    ("strix-llama-bf16-128k", "BF16", 131072), ("strix-llama-bf16-112k", "BF16", 114688),
    ("strix-llama-q8-224k", "Q8_0", 229376), ("strix-llama-q8-192k", "Q8_0", 196608),
):
    if not list(RAW.glob(f"{label}*-monitor.json")):
        continue
    data, evidence = case(label, success=False)
    passed = evidence["monitor"]["returncode"] == 0
    evidence["data"] = data
    if passed:
        assert data["status"] == "complete" and len(data["rows"]) == 1
        assert data["rows"][0]["prompt_tokens"] == prompt
        assert data["rows"][0]["decode_transitions"] == 8
        assert data["rows"][0]["output_ids"] == [9707] * 9
    if not passed:
        logs = [RAW / f"{label}.log", RAW / f"{label}-driver.log"]
        log = "\n".join(p.read_text(errors="replace") for p in logs if p.exists())
        assert "out of memory" in log.lower() or "failed to allocate" in log.lower()
        evidence["error_tail"] = log[-1800:]
    artifact["capacity"].append(dict(engine="strix-llama.cpp", kv=kv, prompt_tokens=prompt,
                                    passed=passed, source_case=label,
                                    peak_gib=evidence["monitor"]["memory"]["peak_gib"]))

artifact["strix_source"] = json.loads((RAW / "strix-source.json").read_text())
screens = {}
for label in ("ar-clean", "mtp-clean", "adaptive-clean", "adaptive-floor1", "ar-long", "mtp-long", "adaptive-floor1-long"):
    path = Path(f"/tmp/nasone-{label}.json")
    data = json.loads(path.read_text())
    assert data["status"] == "complete"
    assert data["memory"]["baseline_bytes"] < 128 * (1 << 20)
    screens[label] = external_rows(data)
    artifact["runs"][f"nasone32-screen-{label}"] = dict(
        raw_sha256=fingerprint(path), source_commit=data["source_commit"],
        command=data["command"], server_command=data["server_command"],
        rows=screens[label], memory=data["memory"])
artifact["nasone32_adaptive_screen"] = {
    "default_minimum": 3, "maximum": 3,
    "fixed_short": paired_summary(screens["ar-clean"], screens["mtp-clean"]),
    "readme_adaptive_short": paired_summary(screens["ar-clean"], screens["adaptive-clean"]),
    "floor1_short": paired_summary(screens["ar-clean"], screens["adaptive-floor1"]),
    "fixed_long": paired_summary(screens["ar-long"], screens["mtp-long"]),
    "floor1_long": paired_summary(screens["ar-long"], screens["adaptive-floor1-long"]),
}
artifact["capacity_note"] = "Only completed, uncontaminated points are published. The earlier overlapped hipEngine 128K attempt is replaced by the detached-snapshot rerun. Historical capacity AR code is unaffected by the MTP-only repair."
artifact["build_commands"] = {
    "nasone32": "cmake -S /tmp/llama-rdna3-nasone32 -B /tmp/llama-rdna3-nasone32/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang '-DCMAKE_HIP_FLAGS=-mllvm --amdgpu-unroll-threshold-local=600' -DGGML_HIP=ON -DGGML_HIP_GRAPHS=ON -DAMDGPU_TARGETS=gfx1100 -DLLAMA_BUILD_TESTS=OFF",
    "strix-llama.cpp": "cmake -S /tmp/strix-llama-head -B /tmp/strix-llama-head/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang '-DCMAKE_HIP_FLAGS=-mllvm --amdgpu-unroll-threshold-local=600' -DGGML_HIP=ON -DGGML_HIP_GRAPHS=ON -DAMDGPU_TARGETS=gfx1100 -DGPU_TARGETS=gfx1100 -DLLAMA_BUILD_TESTS=OFF",
    "nasone32-build": "cmake --build /tmp/llama-rdna3-nasone32/build --target llama-bench llama-server -j 16",
    "strix-llama.cpp-build": "cmake --build /tmp/strix-llama-head/build --target llama-bench llama-server -j 16",
}
destination = ROOT / "benchmarks/results/2026-09-08-rx7900xtx-engine-comparison.json"


def compact_json(value, depth=0):
    pad = "  " * depth
    child_pad = pad + "  "
    if isinstance(value, dict):
        if not value:
            return "{}"
        return "{\n" + ",\n".join(
            child_pad + json.dumps(k) + ": " + compact_json(v, depth + 1)
            for k, v in value.items()
        ) + "\n" + pad + "}"
    if isinstance(value, list) and any(isinstance(v, (dict, list)) for v in value):
        return "[\n" + ",\n".join(child_pad + compact_json(v, depth + 1) for v in value) + "\n" + pad + "]"
    return json.dumps(value, ensure_ascii=True)


destination.write_text(compact_json(artifact) + "\n")
print(json.dumps({k:artifact[k] for k in ("pp_tg_summary", "mtp_summary", "capacity")}, indent=2))
