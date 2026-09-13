#!/usr/bin/env python3
"""Dense vs DMS speed A/B on one GPU: same model, same prompt tokens.

For each arm this opens a fresh ``Qwen35GGUFFullStackRunner`` and one
``Qwen35GGUFResidentSession`` on the same validation-stream slice from the
DMS data manifest, prefills once, then decodes N steps with per-step wall
clock — the exact flow of ``scripts/qwen38_dms_concurrency_probe.py`` minus
the multi-session/cancellation machinery.

Arms:
- ``dense``: the standard uniform fixed-page BF16 KV route (no DMS kwargs).
- ``dms-bf16``: trained-DMS sessions with the BF16 compact backend.
- ``dms-int8``: trained-DMS sessions with the INT8 evaluation backend
  (offline evaluation only; changes attention numerics by design).

Diagnostics, not a retained benchmark: single session, one prompt per arm,
no correctness gate beyond finiteness. Same-host same-workload A/B only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _validation_stream(path: Path) -> list[int]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    sequences = sorted(
        (row for row in raw["sequences"] if str(row.get("split")) == "validation"),
        key=lambda row: str(row["sequence_id"]),
    )
    stream = [int(token) for row in sequences for token in row["token_ids"]]
    if not stream:
        raise ValueError("manifest has no validation tokens")
    return stream


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_arm(
    arm: str,
    *,
    model: Path,
    metadata: Path,
    prompt: list[int],
    steps: int,
    backend: str,
) -> dict:
    import numpy as np

    from hipengine.kvcache.dms import (
        create_dms_bf16_backend,
        create_dms_int8_evaluation_backend,
    )
    from hipengine.runtime.qwen35_gguf_runner import (
        Qwen35GGUFFullStackRunner,
        Qwen35GGUFResidentSession,
    )

    factory = {
        "dms-bf16": create_dms_bf16_backend,
        "dms-int8": create_dms_int8_evaluation_backend,
    }.get(arm)
    dms_kwargs: dict = {}
    if factory is not None:
        dms_kwargs = {
            "dms_metadata_path": metadata,
            "dms_backend_factory": factory,
            "dms_max_new_tokens": steps + 1,
        }

    started = time.perf_counter()
    runner = Qwen35GGUFFullStackRunner(model, backend=backend)
    loaded_at = time.perf_counter() - started

    max_len = len(prompt) + steps + 1
    session = Qwen35GGUFResidentSession(
        model,
        backend=backend,
        shared_runner=runner,
        max_sequence_length=max_len,
        use_wmma_prefill=True,
        use_gemv_decode=True,
        **dms_kwargs,
    )
    session.__enter__()
    try:
        prefill_started = time.perf_counter()
        session.prefill(
            prompt,
            use_bulk=True,
            bulk_attention_mode="bulk",
            return_logits=False,
            record_gpu_stage_timings=False,
        )
        prefill_seconds = time.perf_counter() - prefill_started

        current = int(prompt[-1])
        step_walls: list[float] = []
        tokens: list[int] = []
        finite = True
        for _ in range(steps):
            step_started = time.perf_counter()
            result = session.step(current, return_logits=True)
            step_walls.append(time.perf_counter() - step_started)
            current = int(result.token_id)
            tokens.append(current)
            finite = finite and bool(np.isfinite(result.logits).all())

        observability = None
        backend_obj = getattr(session, "_dms_backend", None)
        if backend_obj is not None:
            observability = backend_obj.observability_snapshot()
    finally:
        session.__exit__(None, None, None)

    step_arr = np.asarray(step_walls)
    return {
        "arm": arm,
        "load_seconds": round(loaded_at, 3),
        "prompt_tokens": len(prompt),
        "decode_steps": steps,
        "prefill_seconds": round(prefill_seconds, 4),
        "prefill_tok_s": round(len(prompt) / prefill_seconds, 1),
        "decode_step_ms_mean": round(float(step_arr.mean()) * 1000, 2),
        "decode_step_ms_median": round(float(np.median(step_arr)) * 1000, 2),
        "decode_tok_s": round(steps / float(step_arr.sum()), 2),
        "finite_logits_all_steps": finite,
        "first_output_tokens": tokens[:4],
        "dms_observability": observability,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path,
                        default=Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"))
    parser.add_argument("--metadata", type=Path,
                        default=Path("/models/dms/qwen38-27b-q4km-dms-w8192-local/dms_metadata.json"))
    parser.add_argument("--data-manifest", type=Path,
                        default=Path("/models/dms/xtx-manifests/qwen38-27b-q4km-xtx-capacity-manifest.json"))
    parser.add_argument("--prompt-tokens", type=int, default=16384)
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument("--arms", default="dense,dms-bf16,dms-int8")
    parser.add_argument("--backend", default="hip_gfx1100")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--single-arm", default=None,
        help="internal: run exactly one arm and write a partial JSON",
    )
    args = parser.parse_args()

    if args.single_arm:
        stream = _validation_stream(args.data_manifest)
        prompt = stream[: args.prompt_tokens]
        row = run_arm(
            args.single_arm,
            model=args.model,
            metadata=args.metadata,
            prompt=prompt,
            steps=args.decode_steps,
            backend=args.backend,
        )
        args.output.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
        print(
            f"[ab] arm={args.single_arm}: prefill {row['prefill_tok_s']} tok/s, "
            f"decode {row['decode_step_ms_median']} ms/step median",
            flush=True,
        )
        return 0

    stream = _validation_stream(args.data_manifest)
    if len(stream) < args.prompt_tokens:
        raise ValueError(
            f"validation stream has {len(stream)} tokens; need {args.prompt_tokens}"
        )
    prompt = stream[: args.prompt_tokens]
    prompt_sha = hashlib.sha256(
        np.asarray(prompt, dtype=np.int64).tobytes()
    ).hexdigest()

    # One fresh process per arm: a 22.7 GB runner's HIP allocations are not
    # reliably reclaimed in-process before the next load (the third arm OOMs
    # otherwise), and the capacity campaign's ladder uses fresh processes for
    # the same reason.
    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    results = []
    for arm in arms:
        part = args.output.with_suffix(f".{arm}.part.json")
        print(f"[ab] arm={arm} starting (fresh subprocess)", flush=True)
        completed = subprocess.run(
            (
                sys.executable, str(Path(__file__).resolve()),
                "--model", str(args.model),
                "--metadata", str(args.metadata),
                "--data-manifest", str(args.data_manifest),
                "--prompt-tokens", str(args.prompt_tokens),
                "--decode-steps", str(args.decode_steps),
                "--backend", args.backend,
                "--single-arm", arm,
                "--output", str(part),
            ),
            cwd=ROOT,
        )
        if completed.returncode != 0:
            print(f"[ab] arm={arm} FAILED rc={completed.returncode}", flush=True)
            results.append({"arm": arm, "status": "failed", "rc": completed.returncode})
            continue
        results.append(json.loads(part.read_text(encoding="utf-8")))

    out = {
        "schema_version": 1,
        "kind": "dense_vs_dms_speed_probe",
        "status": "diagnostic_not_a_benchmark",
        "host": socket_host(),
        "backend": args.backend,
        "gpu_note": "run with HIP_VISIBLE_DEVICES=0 (W7900) for the GPU0 lane",
        "model": {"path": str(args.model), "sha256": _sha256(args.model)},
        "dms_metadata": {"path": str(args.metadata), "sha256": _sha256(args.metadata)},
        "data_manifest": {"path": str(args.data_manifest), "sha256": _sha256(args.data_manifest)},
        "prompt_tokens": args.prompt_tokens,
        "prompt_sha256": prompt_sha,
        "decode_steps": args.decode_steps,
        "arms": results,
        "provenance": git_provenance(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"[ab] wrote {args.output}", flush=True)
    return 0


def socket_host() -> str:
    return socket.gethostname()


def git_provenance() -> dict:
    try:
        commit = subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
        ).strip()
        dirty = subprocess.check_output(
            ("git", "status", "--porcelain", "--untracked-files=no"),
            cwd=ROOT, text=True,
        ).strip()
        return {"commit": commit, "tracked_dirty": bool(dirty)}
    except Exception as exc:  # pragma: no cover
        return {"error": str(exc)}


if __name__ == "__main__":
    raise SystemExit(main())
