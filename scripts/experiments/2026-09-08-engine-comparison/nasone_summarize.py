import hashlib
import json
from pathlib import Path
import statistics

root = Path("/home/lhl/hipEngine-main")
out = {
    "schema": 1, "date": "2026-09-08", "kind": "rx7900xtx_nasone32_comparison",
    "status": "complete_diagnostic_comparison", "performance_claim": False,
    "runtime_provenance": {
        "core_reference_commit": "307e7633f300453f440e26ef0d6ce99c6b26490e",
        "verification": "git diff 307e7633f HEAD --stat -- hipengine: empty after measurements",
        "execution_profile": "legacy resident default selectors; not a newly certified named production profile",
        "variant_manifest_hash": None,
        "manifest_note": "These comparison harnesses do not emit a unified selected-variant manifest; no promotion is claimed.",
    },
    "host": "epyc", "hardware": {
        "device": "AMD Radeon RX 7900 XTX", "hip_visible_devices": "1",
        "pci": "0000:10:00.0", "unique_id": "0xcc4d02090dc9c3ff",
        "kfd_gpu_id": 33912, "vram_bytes": 25753026560, "arch": "gfx1100",
        "hipcc_version": Path("/tmp/hipengine-hipcc-version.txt").read_text(),
        "LD_LIBRARY_PATH": "/opt/rocm/lib:/opt/rocm/lib:/opt/rocm/lib",
    },
    "model": {
        "path": "/models/gguf/Qwen3.8-27B-Q4_K_M.gguf", "quant": "Q4_K_M",
        "bytes": 17106773984,
        "sha256": "7b2aec3b9ababdfd75aa17552ee95607d866e44decf547f6f12fcef85cc89f1b",
    },
    "scope": [
        "Single GPU; local Qwen3.8 dense Q4_K_M, not the quoted Qwen3.6, Q8, Next or tensor-parallel workloads.",
        "External default arithmetic vs legacy resident hipEngine selectors; no new production-profile or kernel promotion.",
        "Synthetic repeated 9707 for fixed-length pp/tg and capacity. Capacity proves execution/accounting, not task quality.",
        "Natural suite uses identical raw token arrays, all ten category/heldout prompts, greedy 25 outputs/24 timed transitions.",
        "External b4096/u1024; no auto fit or weight offload. Context bounds are configuration-specific and not proven maxima.",
        "No cross-engine teacher-forced KL gate was run. Output equality is reported, not substituted for a quality gate.",
    ],
    "external": {}, "hipengine": {}, "monitors": {}, "attempts": {},
    "nasone32_build": {
        "repository": "https://github.com/nasone32/llama.cpp-RDNA3-7900xtx-opt",
        "commit": "7dc2f0cb28326816f67f6b979008383344e2038b",
        "source_clean": True,
        "configure": "cmake -S /tmp/llama-rdna3-nasone32 -B /tmp/llama-rdna3-nasone32/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang '-DCMAKE_HIP_FLAGS=-mllvm --amdgpu-unroll-threshold-local=600' -DGGML_HIP=ON -DGGML_HIP_GRAPHS=ON -DAMDGPU_TARGETS=gfx1100 -DLLAMA_BUILD_TESTS=OFF",
        "build": "cmake --build /tmp/llama-rdna3-nasone32/build --target llama-bench llama-server -j 16",
    },
    "host_controls": {
        "gpu1_ownership_scope": "200 ms polling of KFD VRAM owners above 1 MiB; no claim about undetected shorter overlaps.",
        "gpu0": "Other user-owned work ran on GPU0 during later rows; left untouched.",
        "clocks": "Default, not locked; ablations are screening evidence, not balanced optimization promotion.",
        "ownership_poll_seconds": 0.2,
        "vram_poll_seconds": 0.02,
    },
}


def read(name):
    path = Path("/tmp") / f"nasone-{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def metadata(name, payload):
    path = Path("/tmp") / f"nasone-{name}.json"
    return dict(raw_path=str(path), raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                **payload)


for name in (
    "ar-clean", "mtp-clean", "adaptive-clean", "ar-sequential", "mtp-sequential",
    "ar-long", "mtp-long", "adaptive-long",
    "adaptive-floor1", "adaptive-floor1-long",
    "shapes-clean", "q8-shapes", "gdn-fp32", "gdn-sequential-shape", "bf16-128k", "bf16-112k", "q8-224k", "q8-192k",
):
    payload = read(name)
    if payload is not None:
        out["external"][name] = metadata(name, payload)
        if payload["status"] == "failed":
            log = Path("/tmp") / f"nasone-{name}.log"
            out["external"][name]["error_log_tail"] = log.read_text()[-2400:]

for name in ("adaptive", "shapes", "ar", "mtp"):
    payload = read(name)
    if payload:
        out["attempts"][name] = dict(
            status="INVALID_contaminated" if name in ("adaptive", "shapes") else "initial_diagnostic",
            memory=payload["memory"], raw_path=f"/tmp/nasone-{name}.json",
            reason="pre-existing GPU1 VRAM" if name in ("adaptive", "shapes") else "superseded by monitored repeats",
        )
out["attempts"]["hip-native-context1024"] = {
    "status": "failed_warmup_no_timing",
    "command": "HIP_VISIBLE_DEVICES=1 HIPENGINE_HIP_ARCH=gfx1100 HIPENGINE_GGUF_DECODE_REPACK=1 python3 scripts/qwen36_dense_gguf_suite.py --model /models/gguf/Qwen3.8-27B-Q4_K_M.gguf --max-new-tokens 25 --candidate-budgets 3 --target-verify-mode native --runs 1 --limit 10 --warmup --max-sequence-length 1024 --compiler-version-file /tmp/hipengine-hipcc-version.txt --output /tmp/nasone-hipengine-natural.json",
    "error_log_tail": Path("/tmp/nasone-hipengine-natural.log").read_text()[-3500:],
}

for name in (
    "hip-p512-clean", "hip-p8192-clean", "hip-bf16-128k", "hip-bf16-112k", "hip-bf16-96k", "hip-bf16-80k",
    "hip-int8-128k", "hip-int8-112k", "hip-ar-only", "hip-native-retry",
    "hip-pure-int8-129024", "hip-pure-int8-131072", "hip-pure-int8-114688",
    "hip-pure-int8-139264",
):
    payload = read(name)
    if payload:
        if "summary" in payload and "runs" in payload:
            keys = ("provenance", "command", "prompt_length", "decode_tokens", "warmup_decode_tokens",
                    "warmup_runs", "measured_runs", "max_sequence_length", "kv_storage_dtype",
                    "kv_policy", "graph_replay_decode", "public_ar_profile",
                    "shipping_ar_route_mismatch", "summary", "gguf_tensor_inventory_hash")
            compact = {k: payload[k] for k in keys if k in payload}
            compact["runs"] = [
                {k: row[k] for k in ("run_index", "measured", "throughput", "correctness_sanity",
                                    "effective_graph_replay_decode", "decode_graph_disabled_reason",
                                    "prefill_chunk_sizes") if k in row}
                for row in payload["runs"]
            ]
            persistent = payload.get("persistent_session_memory", {})
            compact["lifecycle_memory"] = persistent.get("summary", {})
            families = persistent.get("snapshots", {}).get("before_close", {}).get(
                "owned_session_breakdown", {}).get("families", {})
            compact["memory_families"] = {
                k: {field: value for field, value in v.items()
                    if field != "bulk_prefill_scratch_census"}
                for k, v in families.items()
            }
            out["hipengine"][name] = metadata(name, compact)
        else:
            out["hipengine"][name] = metadata(name, payload)
    monitor = read(name + "-monitor")
    if monitor:
        out["monitors"][name] = monitor
        if name in {"hip-int8-128k", "hip-int8-112k"}:
            monitor["layout_classification"] = "default hybrid: 8 BF16-prefix layers plus 8 INT8 layers; source-resolved"
            monitor["excluded_from_pure_int8_comparison"] = True
        if monitor["returncode"]:
            log = Path("/tmp") / f"nasone-{name}.log"
            out["monitors"][name]["error_log_tail"] = log.read_text()[-3500:]

heldout = {"code_markdown_table", "general_en_explain", "general_ja_explain", "mixed_ja_en_review"}


def external_metrics(rows):
    return dict(
        rows=len(rows), tokens_per_second=sum(r["decode_transitions"] for r in rows) * 1000 /
        sum(r["timings"]["predicted_ms"] for r in rows),
        accepted=sum(r["timings"].get("draft_n_accepted", 0) for r in rows),
        drafted=sum(r["timings"].get("draft_n", 0) for r in rows),
    )


out["natural_summary"] = {}
for name in ("ar-clean", "mtp-clean", "adaptive-clean", "ar-sequential", "mtp-sequential",
             "ar-long", "mtp-long", "adaptive-long", "adaptive-floor1", "adaptive-floor1-long"):
    if name not in out["external"]:
        continue
    rows = out["external"][name]["rows"]
    repeated = all(sum(x["id"] == r["id"] for x in rows) >= 2 for r in rows)
    out["natural_summary"][name] = {
        "all": external_metrics(rows),
        "heldout": external_metrics([r for r in rows if r["id"] in heldout]),
        "train": external_metrics([r for r in rows if r["id"] not in heldout]),
        "categories": {c: external_metrics([r for r in rows if r["category"] == c])
                       for c in sorted({r["category"] for r in rows})},
        "repeat_check_available": repeated,
        "repeat_deterministic": (all(r["output_ids"] == next(x for x in rows if x["id"] == r["id"])["output_ids"]
                                    for r in rows) if repeated else None),
    }
    reference_name = "ar-sequential" if name.endswith("sequential") else "ar-long" if name.endswith("long") else "ar-clean"
    if reference_name in out["external"]:
        reference = {(r["id"], r["repetition"]): r for r in out["external"][reference_name]["rows"]}
        checks = []
        for row in rows:
            ar = reference[(row["id"], row["repetition"])]
            assert row["prompt_ids"] == ar["prompt_ids"]
            checks.append(dict(id=row["id"], repetition=row["repetition"],
                               exact=ar["output_ids"] == row["output_ids"]))
        out["natural_summary"][name]["own_ar_checks"] = checks

rows = out["hipengine"].get("hip-ar-only", {}).get("rows", [])
if rows:
    out["natural_summary"]["hipengine_ar"] = {
        "tokens_per_second": sum(r["timed_transitions"] for r in rows) / sum(r["decode_seconds"] for r in rows),
        "categories": {c: sum(r["timed_transitions"] for r in rows if r["category"] == c) /
                       sum(r["decode_seconds"] for r in rows if r["category"] == c)
                       for c in sorted({r["category"] for r in rows})},
        "repeat_deterministic": all(r["token_ids"] == next(x for x in rows if x["id"] == r["id"])["token_ids"]
                                    for r in rows),
    }
    if "ar-clean" in out["external"]:
        ref = {(r["id"], r["repetition"]): r for r in out["external"]["ar-clean"]["rows"]}
        checks = []
        for row in rows:
            peer = ref[(row["id"], row["repetition"])]
            assert row["prompt_ids"] == peer["prompt_ids"]
            checks.append(dict(id=row["id"], repetition=row["repetition"],
                               exact=row["token_ids"] == peer["output_ids"]))
        out["natural_summary"]["hipengine_ar"]["external_ar_checks"] = checks

out["shape_summary"] = {}
for name in ("shapes-clean", "q8-shapes", "gdn-fp32", "gdn-sequential-shape"):
    if name in out["external"]:
        rows = out["external"][name]["rows"]
        out["shape_summary"][name] = {
            str(n): dict(pp_median=statistics.median(r["prefill_tok_s"] for r in rows if r["prompt_tokens"] == n),
                         tg_median=statistics.median(r["decode_tok_s"] for r in rows if r["prompt_tokens"] == n))
            for n in sorted({r["prompt_tokens"] for r in rows})
        }
out["ownership_transitions"] = [
    json.loads(line) for line in Path("/tmp/nasone-ownership.jsonl").read_text().splitlines()
]
out["host_controls"]["maximum_observed_gpu1_owners"] = max(
    len(row["owners"]) for row in out["ownership_transitions"]
)
overlaps = [row for row in out["ownership_transitions"] if len(row["owners"]) > 1]
assert all(
    any("nasone-hip-pure-int8-131072.json" in owner["command"] for owner in row["owners"])
    for row in overlaps
)
out["host_controls"]["overlap_rows"] = overlaps
if overlaps:
    affected = out["hipengine"]["hip-pure-int8-131072"]
    affected["evidence_classification"] = "capacity_pass_with_brief_foreign_gpu1_allocation"
    affected["timing_valid"] = False
    affected["memory_scope_note"] = (
        "Whole-card peak includes a foreign approximately 103 MB allocation observed "
        "2026-09-08 19:32:17.486-19:32:19.496 JST. The completed capacity/finiteness result "
        "survives; throughput and isolated-process peak attribution do not. The 126K pass "
        "has no observed overlap."
    )
out["natural_prompt_ids"] = {}
out["output_id_rows"] = {}
out["id_encoding"] = "Each row's output_ids_sha256_i64 resolves to the complete output_id_rows entry; hashes use signed little-endian int64 bytes."
for group in (out["external"], out["hipengine"]):
    for payload in group.values():
        for row in payload.get("rows", []):
            tokens = row.pop("prompt_ids", None)
            if tokens is not None:
                prior = out["natural_prompt_ids"].setdefault(row["id"], tokens)
                assert prior == tokens
                row["prompt_ids_sha256_i64"] = hashlib.sha256(
                    b"".join(int(t).to_bytes(8, "little", signed=True) for t in tokens)
                ).hexdigest()
            outputs = row.pop("output_ids", row.pop("token_ids", None))
            if outputs is not None:
                digest = hashlib.sha256(
                    b"".join(int(t).to_bytes(8, "little", signed=True) for t in outputs)
                ).hexdigest()
                assert out["output_id_rows"].setdefault(digest, outputs) == outputs
                row["output_ids_sha256_i64"] = digest
                row["output_token_count"] = len(outputs)
            if "timings" in row:
                row["timings"] = {k: v for k, v in row["timings"].items()
                                  if k in {"prompt_n", "prompt_ms", "predicted_n", "predicted_ms",
                                           "draft_n", "draft_n_accepted"}}
            row.pop("decode_tok_s", None)
            row.pop("prefill_tok_s", None)
Path("/tmp/nasone-survey-summary.json").write_text(json.dumps(out, indent=2) + "\n")
destination = root / "benchmarks/results/2026-09-08-rx7900xtx-nasone32-survey.json"
destination.write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(dict(natural=out["natural_summary"], shapes=out["shape_summary"]), indent=2))
