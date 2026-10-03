#!/usr/bin/env python3
"""Assemble the durable compact artifact for the gfx1151 35B INT8-KV screen."""
import hashlib
import json
import subprocess
from pathlib import Path

RAW = Path("/tmp/reprof/int8kv")
OUT = Path("/home/lhl/hipEngine/benchmarks/results/2026-10-03-gfx1151-qwen36-35b-a3b-int8-kv-quality-blocked")
OUT.mkdir(parents=True, exist_ok=True)

summary = json.load(open(RAW / "SUMMARY.json"))

SHAPES = {"512/8": "A-int8-fp16-512x8.json", "1024/8": "A-int8-fp16-1024x8.json", "4096/16": "A-int8-fp16-4096x16.json"}
RUNS = {
    "int8_per_token_head_fp16": {
        "512/8": "A-int8-fp16-512x8.json",
        "1024/8": "A-int8-fp16-1024x8.json",
        "1024/8_repeat": "A-int8-fp16-1024x8-repeat.json",
        "4096/16": "A-int8-fp16-4096x16.json",
        "4096/16_repeat": "A-int8-fp16-4096x16-repeat.json",
    },
    "int8_per_token_head_fp32": {"512/8": "B-int8-fp32-512x8.json", "1024/8": "B-int8-fp32-1024x8.json"},
    "tail4_hadamard_group32_fp16": {"512/8": "C-tail4-512x8.json"},
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prompt_rows(raw):
    out = {}
    for row in raw["engines"]["gguf"]["rows"]:
        out[row["id"]] = {
            "category": row["category"],
            "profile": row["profile"],
            "split": row["split"],
            "positions": row["positions"],
            "mean_kl": row["mean_kl"],
            "max_kl": row["max_kl"],
            "top1_agreement": row["top1_agreement"],
            "top1_mismatch_count": row["top1_mismatch_count"],
            "passed": row["passed"],
            "reference_logits_sha256": row["reference_logits_sha256"],
            "candidate_logits_sha256": row["candidate_logits_sha256"],
            "reference_top1": row["reference_top1"],
            "candidate_top1": row["candidate_top1"],
        }
    return out


# --- screening.json ---------------------------------------------------------
screening = {
    "schema": 1,
    "kind": "gfx1151_qwen36_35b_a3b_int8_decode_kv_quality_screen",
    "date": "2026-10-03",
    "status": "correctness_blocked",
    "publishable": False,
    "performance_claim": False,
    "question": (
        "Is an INT8 decode KV copy (int8_per_token_head, the gufo-matching representation) retainable for "
        "Qwen3.6-35B-A3B on gfx1151 under the production numerical gate?"
    ),
    "decision": summary["decision"],
    "model": {
        "path": "/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        "sha256": "0b21525e972670ed59e1812e170b27c26355381f0656ecc4e25617ece7dac58b",
        "size_bytes": 22663387424,
        "quant": "gguf_ud_q4_k_m",
    },
    "host": {
        "name": "gfx1151",
        "gpu": "AMD Radeon 8060S Graphics",
        "arch": "gfx1151",
        "cu": 40,
        "memory": "125 GiB unified LPDDR5X",
        "machine_id": "55ea6c509d0b49eea8de7094a1023668",
        "rocm": "HIP version: 7.15.26333-0000000",
        "note": "Physical Framework Desktop host; independent lane from the W7900 gfx1100 KV results.",
    },
    "software": {
        "hipengine_commit": "4fb6a04d29eca59018b6b0d5651bc2d79829867a",
        "backend": "hip_gfx1151",
        "target_arch": "gfx1151",
        "harness": "scripts/qwen35_native_mixed_kv_suite.py",
        "source_tree_state": "no edits under hipengine/ or kernels/ during the run",
        "compiler_version_first_line": summary["model"] and "HIP version: 7.15.26333-0000000",
    },
    "prompt_suite": {
        "path": "benchmarks/prompts/mtpbench-code-general-ja.jsonl",
        "sha256": "fac920be5e691fec2cb70fd8b7eedddab8926b89d6a1627f62ec4f441d86084a",
        "natural_rows": 10,
        "category_counts": {"code": 4, "general_en": 2, "general_ja": 2, "mixed_ja_en": 2},
        "harness_control": "mixed_v1 (mixed_synthetic, repeated 36-token unit)",
        "cases_per_invocation": 11,
    },
    "protocol": {
        "teacher_forcing": True,
        "quality_gate": {"mean_kl_max_per_prompt": 0.05, "top1_agreement_min_per_prompt": 0.9, "all_prompts_must_pass": True, "finite_logits": True},
        "shape_order": ["512/8", "1024/8", "4096/16"],
        "stop_rule": "stop a representation after its first failed shape",
        "flags": [
            "--diagnostic-kv-capability",
            "--require-no-bf16-mirror",
            "--max-sequence-length 32786",
            "--compiler-version-file /tmp/reprof/compiler.txt",
            "--require-cached-build",
        ],
        "note": "Timings in the screening runs are diagnostic only; the retained runtime comparison is runtime.json.",
    },
    "stop_rule_trace": [
        {"candidate": "int8_per_token_head + fp16 scales", "512/8": "pass", "1024/8": "pass", "4096/16": "FAIL", "first_failure": "4096/16"},
        {"candidate": "int8_per_token_head + fp32 scales", "512/8": "pass", "1024/8": "FAIL", "4096/16": "not run", "first_failure": "1024/8"},
        {"candidate": "tail4_hadamard_group32 + fp16 scales", "512/8": "FAIL", "1024/8": "not run", "4096/16": "not run", "first_failure": "512/8"},
    ],
    "candidates": {},
    "binding_failure": {
        "candidate": "int8_per_token_head_fp16",
        "shape": "4096/16",
        "prompt": "mixed_v1",
        "category": "mixed_synthetic",
        "mean_kl": 0.1596586351679316,
        "max_kl": 2.69385,
        "top1_agreement": 0.9411764705882353,
        "positions": 17,
        "top1_mismatches": 1,
        "mechanism": (
            "Single-position event, not diffuse drift: 17 positions, summed KL 2.7142 against a max of 2.6939, so "
            "99.25% of the per-prompt KL mass sits at position 4 where the reference top-1 is 271 and the candidate "
            "top-1 is 30523; the other 16 positions average 1.27e-3. The control is a repeated 36-token unit, so its "
            "reference distributions are near-one-hot and one rank flip costs KL ~2.7."
        ),
    },
    "natural_prompt_outcome": {
        "512/8": "10/10 natural prompts pass; worst prompt mean KL 1.223e-3 (mixed_ja_en_translate), min top-1 1.0",
        "1024/8": "10/10 natural prompts pass; worst prompt mean KL ~1.3e-3, min top-1 1.0",
        "4096/16": "10/10 natural prompts pass at 100% top-1 with mean KL <= 1.09e-3 (code 3.1e-4, general_en 2.4e-4, general_ja 7.1e-4, mixed_ja_en 1.0e-3)",
        "reading": (
            "The 35B route is materially better behaved on natural prompts than the gfx1151 27B route, which rejected "
            "at 1024/8 with a failing natural prompt (mixed_ja_en_review). The 35B blocker is confined to the synthetic "
            "repeated-token control."
        ),
    },
    "why_not_promoted": (
        "The declared gate requires every prompt in the suite to pass, and mixed_v1 is in the suite. Waiving it for this "
        "artifact would apply a weaker gate to a second model than the one that closed the gfx1151 27B lane, and the "
        "measured upside does not justify re-cutting the gate: the runtime gain is +1.44%/+1.50%/+1.59% at depths "
        "0/4096/16384 (flat, not widening) and the net owned-session memory reduction is 146,761,728 B (0.64%) at "
        "18,688 retained rows. See runtime.json and memory.json."
    ),
    "evidence": {
        "raw_runs": "the child run directory /tmp/reprof/int8kv/ (ephemeral); per-prompt rows are reproduced below",
        "summary_raw_sha256": sha256_file(RAW / "SUMMARY.json"),
    },
}

for name, runs in RUNS.items():
    entry = {"shapes": {}}
    for shape, fname in runs.items():
        raw = json.load(open(RAW / fname))
        rows = prompt_rows(raw)
        entry["shapes"][shape] = {
            "artifact": fname,
            "status": raw["status"],
            "passed": raw["passed"],
            "prompt_length": raw["prompt_length"],
            "decode_steps": raw["decode_steps"],
            "elapsed_seconds": raw["elapsed_seconds"],
            "per_prompt": rows,
        }
    screening["candidates"][name] = entry

# attach the aggregate view straight from the child summary so no number is retyped
for name, cand in summary["candidates"].items():
    if name in screening["candidates"]:
        screening["candidates"][name]["aggregate_from_summary"] = {
            shape: {"passed": s["passed"], "aggregate": s["aggregate"], "failing_prompts": s["failing_prompts"], "by_category": s["by_category"]}
            for shape, s in cand["shapes"].items()
        }

(OUT / "screening.json").write_text(json.dumps(screening, indent=1) + "\n")

# --- runtime.json -----------------------------------------------------------
runtime = {
    "schema": 1,
    "kind": "gfx1151_qwen36_35b_a3b_int8_decode_kv_runtime",
    "date": "2026-10-03",
    "status": "diagnostic_complete",
    "publishable": False,
    "performance_claim": False,
    "question": "Does the INT8 decode KV copy move decode throughput, and does its advantage widen with depth?",
    "protocol": {
        "name": "gufo cached-prefix protocol",
        "turn_tokens": 2048,
        "decode_tokens": 128,
        "depths": [0, 4096, 16384],
        "depth_note": "depth is an untimed cached prefix; the timed request is a 2048-token new turn plus 128 decode tokens",
        "one_resident_session": True,
        "warm_iteration_discarded": True,
        "measured_repetitions": 2,
        "int8_graph_env": "HIPENGINE_GGUF_INT8_KV_DECODE_GRAPH=1",
        "decode_graph_min_replay_steps": 128,
    },
    "tg_tok_s": summary["runtime_gufo_cached_prefix_protocol"]["tg_tok_s"],
    "deltas_pct": summary["runtime_gufo_cached_prefix_protocol"]["deltas_pct"],
    "depth_drop_0_to_16384_pct": summary["runtime_gufo_cached_prefix_protocol"]["depth_drop_0_to_16384_pct"],
    "reading": (
        "INT8 KV is worth +1.44%/+1.50%/+1.59% against the BF16 graph route at depths 0/4096/16384: real but flat, and "
        "an order of magnitude too small to explain the 1.23x/1.26x/1.40x decode ratio attributed to gufo. The "
        "KV-width hypothesis for the depth-widening gap is therefore not supported by this measurement. The BF16 eager "
        "route is the one that loses depth (-5.78%), and graph admission removes that loss (-0.14%), which is a "
        "separate and already-known effect."
    ),
    "raw": summary["runtime_gufo_cached_prefix_protocol"],
}
(OUT / "runtime.json").write_text(json.dumps(runtime, indent=1) + "\n")

# --- memory.json ------------------------------------------------------------
memory = {
    "schema": 1,
    "kind": "gfx1151_qwen36_35b_a3b_int8_decode_kv_memory",
    "date": "2026-10-03",
    "status": "diagnostic_complete",
    "publishable": False,
    "performance_claim": False,
    "context_rows": 18688,
    "accounting": summary["memory_at_capacity_18688"],
    "reading": (
        "The retained KV payload halves (-49.61%), but KV is a small share of the owned session at this depth, so the "
        "net owned-session reduction is 146,761,728 B (0.64%) and the graph run peak falls 108,488,704 B (0.101 GiB). "
        "The no-mirror prefill hidden buffer grows 42,991,616 B on the INT8 route. This is a bandwidth play, not a "
        "capacity play, on a host with 125 GiB of unified memory."
    ),
}
(OUT / "memory.json").write_text(json.dumps(memory, indent=1) + "\n")

# --- determinism.json -------------------------------------------------------
det = {
    "schema": 1,
    "kind": "gfx1151_qwen36_35b_a3b_int8_decode_kv_determinism",
    "date": "2026-10-03",
    "status": "diagnostic_complete",
    "publishable": False,
    "performance_claim": False,
    "result": summary["determinism"],
    "raw_pairing": {},
}
for shape in ["1024/8", "4096/16"]:
    a = json.load(open(RAW / RUNS["int8_per_token_head_fp16"][shape]))
    b = json.load(open(RAW / RUNS["int8_per_token_head_fp16"][shape + "_repeat"]))
    ra = prompt_rows(a)
    rb = prompt_rows(b)
    det["raw_pairing"][shape] = {
        "same_candidate_logits": all(ra[k]["candidate_logits_sha256"] == rb[k]["candidate_logits_sha256"] for k in ra),
        "same_reference_logits": all(ra[k]["reference_logits_sha256"] == rb[k]["reference_logits_sha256"] for k in ra),
        "same_per_prompt_metrics": all(
            ra[k]["mean_kl"] == rb[k]["mean_kl"] and ra[k]["max_kl"] == rb[k]["max_kl"] and ra[k]["top1_agreement"] == rb[k]["top1_agreement"]
            for k in ra
        ),
        "prompts_compared": len(ra),
    }
(OUT / "determinism.json").write_text(json.dumps(det, indent=1) + "\n")

print("wrote", OUT)
for p in sorted(OUT.iterdir()):
    print(" ", p.name, p.stat().st_size)
