"""TP2 MLP-only generation diagnostic: TP1 controls on each GPU, then TP2.

This script drives only the runtime module
(``hipengine.distributed.tp2_generate``) - it contains no engine logic. It is
handoff step 2 of ``docs/QWEN38-27B-GFX1100-TP2.md`` Packet 3: full-model
MLP-only TP2 with replicated attention/GDN, generating actual tokens on the
W7900 + RX 7900 XTX host, with the matched TP1 control run fresh on each
physical device and the teacher-forced full-logit production gate deciding
whether the sharded arithmetic stays inside the envelope.

No tensor-parallel speedup is claimed here. The measured walls are the
integrated trace of a diagnostic schedule (eager unfused route, Python-driven
staged exchange, sequential token prefill); they exist to attribute where the
schedule spends a token and to qualify the arithmetic, not to promote TP2.

Run:

    uv run python3 scripts/tp2_mlp_generate_e2e.py \
        --model /models/gguf/Qwen3.8-27B-Q4_K_M.gguf \
        --json benchmarks/results/2026-09-15-w7900-tp2-mlp-generate-e2e.json
"""

from __future__ import annotations

import argparse
import ctypes
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hipengine.loading.gguf import scan_gguf  # noqa: E402
from hipengine.loading.qwen35_gguf import build_qwen35_gguf_tensor_map  # noqa: E402
from hipengine.loading.qwen35_gguf_admission import (  # noqa: E402
    build_qwen35_gguf_role_manifest,
)
from hipengine.tokenization.gguf import Qwen35GGUFTokenizer  # noqa: E402
from hipengine.distributed.tp2_generate import (  # noqa: E402
    GenerationResult,
    MlpTP2GenerationSession,
)

#: Short prompts across the categories the engine's own suites use (code,
#: general English, general Japanese, mixed). Token counts stay small: this
#: is a diagnostic checkpoint, not a task qualification.
PROMPTS = (
    "Write a Python function that reverses a list.",
    "The weather today is",
    "日本の首都は",
    "Paris is the capital of",
)

#: One fixed teacher-forced token sequence for the full-logit gate, held
#: constant across arms so the metrics compare like rows.
TEACHER_FORCED_TOKENS = (
    9707, 198, 1115, 596, 13365, 311, 11202, 1226,
    25, 1879, 11, 662, 3290, 13, 5966, 2675,
)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted.astype(np.float64))
    return exp / exp.sum(axis=-1, keepdims=True)


def _kl_metrics(teacher: np.ndarray, student: np.ndarray) -> dict[str, float]:
    """Row-wise KL(P||Q) with P = teacher, plus top-1 agreement."""

    if teacher.shape != student.shape:
        raise ValueError(
            f"shape mismatch: teacher {teacher.shape} vs student {student.shape}"
        )
    p = _softmax(teacher)
    q = _softmax(student)
    kl = (p * (np.log(p + 1e-45) - np.log(q + 1e-45))).sum(axis=-1)
    top1_t = teacher.argmax(axis=-1)
    top1_s = student.argmax(axis=-1)
    return {
        "rows": int(teacher.shape[0]),
        "mean_kl": float(kl.mean()),
        "p95_kl": float(np.percentile(kl, 95)),
        "p99_kl": float(np.percentile(kl, 99)),
        "max_kl": float(kl.max()),
        "top1_agreement": float((top1_t == top1_s).mean()),
        "flipped_rows": int((top1_t != top1_s).sum()),
    }


#: The calibrated production envelope (docs/EXECUTION-PROFILES.md 6.1).
PRODUCTION_GATE = {
    "mean_kl": 1e-3,
    "p95_kl": 5e-3,
    "p99_kl": 2e-2,
    "max_kl": 5e-2,
    "top1_agreement": 0.99,
}


def _gate_passes(metrics: dict[str, float]) -> tuple[bool, list[str]]:
    failures = []
    for key, limit in PRODUCTION_GATE.items():
        value = metrics[key]
        if key == "top1_agreement":
            if value < limit:
                failures.append(f"{key} {value:.6f} < {limit}")
        elif value > limit:
            failures.append(f"{key} {value:.6g} > {limit}")
    return (not failures), failures


def _device_memory(session: MlpTP2GenerationSession) -> dict[str, Any]:
    lib = session.runtime.library
    free = ctypes.c_size_t()
    total = ctypes.c_size_t()
    out = {}
    for device in session.devices:
        lib.hipSetDevice(ctypes.c_int(int(device)))
        lib.hipMemGetInfo(ctypes.byref(free), ctypes.byref(total))
        out[str(device)] = {
            "free_gib": round(free.value / 2**30, 3),
            "total_gib": round(total.value / 2**30, 3),
        }
    return out


def _device_names(session: MlpTP2GenerationSession) -> dict[str, str]:
    return {
        str(device): session.runtime.device_get_name(int(device))
        for device in session.devices
    }


def _generation_record(
    result: GenerationResult,
    *,
    tokenizer: Qwen35GGUFTokenizer,
    prompt_text: str,
) -> dict[str, Any]:
    walls = np.array([t.total_s for t in result.step_traces], dtype=np.float64)
    decode = np.array(
        [t.total_s for t in result.step_traces if t.kind == "decode"], dtype=np.float64
    )
    stage_sums: dict[str, float] = {}
    for trace in result.step_traces:
        for name, value in trace.stages.items():
            stage_sums[name] = stage_sums.get(name, 0.0) + value
    return {
        "prompt": prompt_text,
        "prompt_tokens": list(result.prompt_token_ids),
        "generated_tokens": list(result.token_ids),
        "generated_text": tokenizer.decode(list(result.token_ids)),
        "finished_on_eos": result.finished_on_eos,
        "prefill_first_token_ms": round(float(walls[0]) * 1e3, 2),
        "decode_walls_ms": {
            "n": int(decode.size),
            "p50": round(float(np.percentile(decode, 50)) * 1e3, 3),
            "min": round(float(decode.min()) * 1e3, 3),
            "max": round(float(decode.max()) * 1e3, 3),
        },
        "stage_sums_s": {k: round(v, 4) for k, v in sorted(stage_sums.items())},
    }


def _exchange_summary(group: Any) -> dict[str, Any]:
    walls = np.array(group.exchange_walls_s, dtype=np.float64)
    return {
        "reductions": int(group.reductions),
        "wall_us": {
            "n": int(walls.size),
            "p50": round(float(np.percentile(walls, 50)) * 1e6, 2),
            "p95": round(float(np.percentile(walls, 95)) * 1e6, 2),
            "mean": round(float(walls.mean()) * 1e6, 2),
        },
    }


def run_arm(
    *,
    model: str,
    mode: str,
    devices: tuple[int, ...],
    max_new_tokens: int,
    eos_token_id: int,
    tokenizer: Qwen35GGUFTokenizer,
    label: str,
    driver: str = "python",
) -> tuple[dict[str, Any], np.ndarray]:
    """One fresh session: generations + teacher-forced logits.

    Returns ``(artifact_record, teacher_logits)``; the full logits stay in
    memory for the cross-arm gates and never enter the artifact.
    """

    started = time.perf_counter()
    session = MlpTP2GenerationSession(
        model, devices=devices, mode=mode, max_sequence_length=2048, driver=driver
    )
    built_s = time.perf_counter() - started
    record: dict[str, Any] = {
        "label": label,
        "mode": mode,
        "driver": driver if mode == "tp2" else None,
        "mlp_decode_variant": (
            session._shard_group.mlp_decode_variant
            if mode == "tp2" and session._shard_group is not None
            else None
        ),
        "devices": list(devices),
        "device_names": _device_names(session),
        "memory_after_load": _device_memory(session),
        "build_s": round(built_s, 2),
    }

    generations = []
    for prompt_text in PROMPTS:
        token_ids = tokenizer.encode(prompt_text)
        result = session.generate(
            token_ids, max_new_tokens=max_new_tokens, eos_token_id=eos_token_id
        )
        generations.append(
            _generation_record(result, tokenizer=tokenizer, prompt_text=prompt_text)
        )
    record["generations"] = generations

    teacher_logits = session.teacher_forced_logits(TEACHER_FORCED_TOKENS)

    if mode == "tp2" and session._shard_group is not None:
        first_generation_tokens = generations[0]["generated_tokens"]
        repeat = session.generate(
            tokenizer.encode(PROMPTS[0]),
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        )
        repeat_logits = session.teacher_forced_logits(TEACHER_FORCED_TOKENS)
        record["determinism"] = {
            "tokens_identical": list(repeat.token_ids) == list(first_generation_tokens),
            "logits_bit_identical": bool(np.array_equal(repeat_logits, teacher_logits)),
            "logits_max_abs_diff": round(
                float(np.abs(repeat_logits - teacher_logits).max()), 9
            ),
        }
        record["exchange"] = _exchange_summary(session._shard_group)
    session.close()
    return record, teacher_logits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
    parser.add_argument(
        "--json",
        default="benchmarks/results/2026-09-15-w7900-tp2-mlp-generate-e2e.json",
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--driver",
        default="compiled",
        choices=("python", "compiled"),
        help="the TP2 arm's staged-exchange transport: the compiled host driver "
        "(default; mapped pinned payload, no H2D return) or the Python staged route",
    )
    args = parser.parse_args(argv)

    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    info = scan_gguf(args.model)
    model_map = build_qwen35_gguf_tensor_map(info)
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(info)
    eos_token_id = int(tokenizer.eos_token_id)
    fingerprint = build_qwen35_gguf_role_manifest(model_map).fingerprint

    arm_specs = (
        ("tp1-device0-W7900", "tp1", (0,)),
        ("tp1-device1-RX7900XTX", "tp1", (1,)),
        ("tp2-W7900+RX7900XTX", "tp2", (0, 1)),
    )
    arms: list[dict[str, Any]] = []
    logits_by_label: dict[str, np.ndarray] = {}
    for label, mode, devices in arm_specs:
        print(f"=== arm {label} ({mode} on {devices}) ===", flush=True)
        record, teacher_logits = run_arm(
            model=args.model,
            mode=mode,
            devices=devices,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=eos_token_id,
            tokenizer=tokenizer,
            label=label,
            driver=args.driver,
        )
        arms.append(record)
        logits_by_label[label] = teacher_logits
        if record.get("determinism") is not None:
            print(
                f"determinism: tokens_identical={record['determinism']['tokens_identical']}, "
                f"logits_bit_identical={record['determinism']['logits_bit_identical']}",
                flush=True,
            )
        if record.get("exchange") is not None:
            print(
                f"exchange: {record['exchange']['reductions']} reductions, "
                f"p50 wall {record['exchange']['wall_us']['p50']} us",
                flush=True,
            )

    by_label = {arm["label"]: arm for arm in arms}

    gates: dict[str, Any] = {}

    def _gate(name: str, teacher_label: str, student_label: str) -> None:
        metrics = _kl_metrics(
            logits_by_label[teacher_label], logits_by_label[student_label]
        )
        passed, failures = _gate_passes(metrics)
        gates[name] = {
            "teacher": teacher_label,
            "student": student_label,
            "metrics": metrics,
            "passed": passed,
            "failures": failures,
        }
        status = "PASS" if passed else "FAIL"
        print(
            f"[{status}] {name}: mean KL {metrics['mean_kl']:.3e}, "
            f"max KL {metrics['max_kl']:.3e}, top-1 {metrics['top1_agreement']*100:.3f}%"
        )

    _gate("tp2_vs_tp1_teacher_device0", "tp1-device0-W7900", "tp2-W7900+RX7900XTX")
    _gate("tp2_vs_tp1_teacher_device1", "tp1-device1-RX7900XTX", "tp2-W7900+RX7900XTX")
    _gate("tp1_device0_vs_device1_control", "tp1-device0-W7900", "tp1-device1-RX7900XTX")

    def _decode_p50(arm: dict[str, Any]) -> float:
        values = [g["decode_walls_ms"]["p50"] for g in arm["generations"]]
        return round(float(np.median(values)), 3)

    comparison = {
        "decode_p50_ms": {
            "tp1_device0": _decode_p50(by_label["tp1-device0-W7900"]),
            "tp1_device1": _decode_p50(by_label["tp1-device1-RX7900XTX"]),
            "tp2": _decode_p50(by_label["tp2-W7900+RX7900XTX"]),
        },
        "reading": (
            "matched-composition decode walls on the diagnostic schedule; "
            "the TP2 wall includes the replicated attention/GDN cost and one "
            "staged reduction per layer"
        ),
    }

    artifact = {
        "schema_version": 1,
        "kind": "tp2-mlp-generate-e2e",
        "generated_at": started,
        "model": args.model,
        "model_fingerprint": fingerprint,
        "world_size": 2,
        "host": {
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "hardware_labels": {
            "device0": by_label["tp1-device0-W7900"]["device_names"]["0"],
            "device1": by_label["tp1-device1-RX7900XTX"]["device_names"]["1"],
            "note": (
                "two distinct physical cards on one host; earlier TP2 worklog "
                "entries that said 'two W7900s' were wrong - the artifact "
                "always recorded W7900 + RX 7900 XTX"
            ),
        },
        "sharding_scope": (
            "MLP-only TP2: replicated attention/GDN weights and compute on "
            "every rank (full replicated cost inside the walls), MLP gate/up "
            "column-sharded and down row-sharded, one staged bf16-partial "
            "reduction per layer per token, single residual add per rank"
        ),
        "no_speedup_claim": (
            "diagnostic schedule (eager unfused route, Python-driven staged "
            "exchange, sequential token prefill); walls attribute the "
            "schedule, they do not promote TP2"
        ),
        "production_gate": PRODUCTION_GATE,
        "gates": gates,
        "arms": arms,
        "comparison": comparison,
    }
    out_path = Path(args.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=1) + "\n")
    all_pass = all(g["passed"] for g in gates.values())
    print(f"artifact: {out_path} ({'all gates passed' if all_pass else 'GATE FAILURES'})")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
