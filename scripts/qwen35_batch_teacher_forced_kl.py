#!/usr/bin/env python3
"""Teacher-forced per-step KL / top-1 correctness gate for native Qwen3.5/PARO c>N decode.

The free-running generated-token equal-prefix metric measures *chaotic near-tie
tracking* over many autoregressive decode steps: a sub-ulp per-step numerical
difference between the c>N batch path and the c=1 serial oracle compounds through
the KV cache until some logit near-tie flips, after which the two sequences diverge
entirely.  iter482 showed the c4 "blocker" is exactly this (FP non-associativity of
the batch context kernel vs the serial kernel), not a logic bug, so free-running
exact equality is the wrong correctness gate for c>N.

This tool implements the project correctness gate instead (see AGENTS.md:
``KL <= 0.05 AND top-1 agreement >= 90% vs reference``) under *teacher forcing*:

  1. Run the c=1 serial oracle autoregressively per row, capturing its full per-step
     next-token logits.  Its argmax sequence is the teacher sequence.
  2. Run the c>N native batch path force-fed the *same* teacher token each step, so
     the input prefix is identical and the sequences cannot chaotically diverge.
     Capture each row's per-step next-token logits via the identical serial
     final-norm + lm-head projection used by the c=1 oracle.
  3. Per (row, decode-step) compare next-token distributions: top-1 agreement and
     ``KL(softmax(c1) || softmax(batch))``.

This isolates *per-step* numerical correctness from chaotic sequence divergence.
It is correctness-only and never emits a throughput / performance claim.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.qwen35_batch_retained_bench import (  # noqa: E402
    DEFAULT_FIXTURE,
    DEFAULT_MODEL,
    _compiler_version,
    _load_prompt_slices,
)

# ---------------------------------------------------------------------------
# Pure metric core (no GPU; unit-testable)
# ---------------------------------------------------------------------------


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable log-softmax over the last axis (float64)."""
    arr = np.asarray(logits, dtype=np.float64)
    m = np.max(arr)
    shifted = arr - m
    return shifted - np.log(np.sum(np.exp(shifted)))


def kl_divergence_from_logits(p_logits: np.ndarray, q_logits: np.ndarray) -> float:
    """KL(P || Q) in nats, with P = softmax(p_logits), Q = softmax(q_logits).

    P is the reference (c=1 oracle); Q is the candidate (c>N batch path).
    """
    p_log = _log_softmax(p_logits)
    q_log = _log_softmax(q_logits)
    p = np.exp(p_log)
    return float(np.sum(p * (p_log - q_log)))


def top1_from_logits(logits: np.ndarray) -> int:
    return int(np.argmax(np.asarray(logits)))


def summarize_metrics(
    records: Sequence[dict[str, Any]],
    *,
    kl_threshold: float,
    top1_threshold: float,
) -> dict[str, Any]:
    """Aggregate per-(row, step) records into the gate verdict.

    Each record must contain ``row`` (int), ``kl`` (float), and ``top1_match`` (bool).
    Pass requires ``mean_kl <= kl_threshold`` AND ``top1_fraction >= top1_threshold``.
    ``max_kl`` and ``p95_kl`` are reported for context but are not the gate.
    """
    n = len(records)
    if n == 0:
        return {
            "passed": False,
            "reason": "no records",
            "n": 0,
            "kl_threshold": float(kl_threshold),
            "top1_threshold": float(top1_threshold),
        }
    kls = np.asarray([float(r["kl"]) for r in records], dtype=np.float64)
    matches = np.asarray([bool(r["top1_match"]) for r in records], dtype=bool)
    mean_kl = float(np.mean(kls))
    max_kl = float(np.max(kls))
    p95_kl = float(np.percentile(kls, 95))
    top1_fraction = float(np.mean(matches))
    rows = sorted({int(r["row"]) for r in records})
    per_row: dict[str, Any] = {}
    for row in rows:
        row_kls = np.asarray([float(r["kl"]) for r in records if int(r["row"]) == row], dtype=np.float64)
        row_match = np.asarray([bool(r["top1_match"]) for r in records if int(r["row"]) == row], dtype=bool)
        per_row[str(row)] = {
            "n": int(row_kls.size),
            "mean_kl": float(np.mean(row_kls)),
            "max_kl": float(np.max(row_kls)),
            "top1_fraction": float(np.mean(row_match)),
        }
    kl_ok = mean_kl <= kl_threshold
    top1_ok = top1_fraction >= top1_threshold
    return {
        "passed": bool(kl_ok and top1_ok),
        "kl_passed": bool(kl_ok),
        "top1_passed": bool(top1_ok),
        "n": int(n),
        "mean_kl": mean_kl,
        "max_kl": max_kl,
        "p95_kl": p95_kl,
        "top1_fraction": top1_fraction,
        "kl_threshold": float(kl_threshold),
        "top1_threshold": float(top1_threshold),
        "per_row": per_row,
    }


# ---------------------------------------------------------------------------
# GPU harness
# ---------------------------------------------------------------------------


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
        return True
    except OSError:
        return False


def _copy_full_logits(session, vocab_size: int) -> np.ndarray:
    from hipengine.core.memory import DeviceBuffer, copy_device_to_host, host_array_ptr

    out = np.empty((vocab_size,), dtype=np.float32)
    copy_device_to_host(
        host_array_ptr(out),
        DeviceBuffer(session.lm_logits.ptr, out.nbytes),
        runtime=session.runtime,
    )
    return out


def _run_c1_teacher(
    runner,
    prompts: list[list[int]],
    *,
    decode_tokens: int,
    max_sequence_length: int,
    max_layers: int,
    compiler_version: str | None,
    require_cached_build: bool,
) -> tuple[list[list[int]], list[list[int]], list[np.ndarray]]:
    """Run the c=1 serial oracle per row.

    Returns ``(teacher_inputs, c1_tokens, c1_logits)`` where for each row:
      - ``teacher_inputs[row][step]`` is the token fed at decode step ``step``
        (seed for step 0, c1 argmax of step ``step-1`` afterwards),
      - ``c1_tokens[row][step]`` is the c=1 argmax prediction at decode step,
      - ``c1_logits[row]`` is an ``(decode_tokens, vocab)`` float32 array.
    """
    from hipengine.core.dtype import DType
    from hipengine.kvcache import FixedPagedKVPolicy
    from hipengine.runtime.qwen35_paro_runner import Qwen35ParoResidentSession

    rows = len(prompts)
    vocab = int(runner.config.vocab_size) if hasattr(runner.config, "vocab_size") else None
    teacher_inputs: list[list[int]] = []
    c1_tokens: list[list[int]] = []
    c1_logits: list[np.ndarray] = []
    with Qwen35ParoResidentSession(
        runner,
        max_sequence_length=max_sequence_length,
        max_layers=max_layers,
        max_batch_size=1,
        compiler_version=compiler_version,
        require_cached_build=require_cached_build,
        kv_policy=FixedPagedKVPolicy(block_size=256, storage_dtype=DType.BF16),
    ) as session:
        vocab = int(session.vocab_size)
        for prompt in prompts:
            prefill = session.prefill_native(prompt, sample=True)
            if prefill is None:
                raise RuntimeError("c=1 prefill did not produce a seed token")
            seed = int(prefill.token_id)
            row_inputs = [seed]
            row_tokens: list[int] = []
            row_logits = np.empty((decode_tokens, vocab), dtype=np.float32)
            prev = seed
            for step in range(decode_tokens):
                position = len(prompt) + step
                session._set_token_embedding(prev, stream=0)
                session._set_position(position, stream=0)
                hidden = session._run_layers(position=position, stream=0)
                result = session._sample_from_hidden(hidden)
                row_logits[step] = _copy_full_logits(session, vocab)
                tok = int(result.token_id)
                row_tokens.append(tok)
                if step + 1 < decode_tokens:
                    row_inputs.append(tok)
                prev = tok
            teacher_inputs.append(row_inputs)
            c1_tokens.append(row_tokens)
            c1_logits.append(row_logits)
            # Reset sequence state (KV pages + linear/Mamba recurrent state) between
            # rows; the serial c=1 oracle reuses one slot-0 session, and the linear
            # recurrent state would otherwise carry over and corrupt later rows.
            session.reset()
    return teacher_inputs, c1_tokens, c1_logits


def _run_batch_teacher_forced(
    runner,
    prompts: list[list[int]],
    teacher_inputs: list[list[int]],
    *,
    decode_tokens: int,
    max_sequence_length: int,
    max_layers: int,
    compiler_version: str | None,
    require_cached_build: bool,
) -> list[np.ndarray]:
    """Run the native c>N batch path force-fed the teacher token sequence.

    Returns ``batch_logits`` where ``batch_logits[row]`` is an
    ``(decode_tokens, vocab)`` float32 array of the batch path's per-step
    next-token logits, captured through the identical serial lm-head projection.
    """
    from hipengine.core.dtype import DType
    from hipengine.core.tensor import Tensor
    from hipengine.generation import ResidentBatchScheduler
    from hipengine.kvcache import FixedPagedKVPolicy
    from hipengine.runtime.qwen35_paro_runner import Qwen35ParoResidentSession

    rows = len(prompts)
    hidden_size = int(runner.config.hidden_size)
    fp16_itemsize = 2
    batch_logits: list[np.ndarray] = []
    with Qwen35ParoResidentSession(
        runner,
        max_sequence_length=max_sequence_length,
        max_layers=max_layers,
        max_batch_size=rows,
        compiler_version=compiler_version,
        require_cached_build=require_cached_build,
        kv_policy=FixedPagedKVPolicy(block_size=256, storage_dtype=DType.BF16),
    ) as session:
        vocab = int(session.vocab_size)
        for _ in range(rows):
            batch_logits.append(np.empty((decode_tokens, vocab), dtype=np.float32))
        # Packed batch prefill (its own numerics); decode inputs are force-fed below.
        scheduler = ResidentBatchScheduler(capacity=rows)
        request_ids = [scheduler.submit(prompt, max_new_tokens=decode_tokens) for prompt in prompts]
        admitted = scheduler.admit_pending()
        if tuple(request_ids) != tuple(admitted):
            raise RuntimeError(f"unexpected admitted request ids {admitted!r}")
        slabs = scheduler.next_compact_prefill_slabs(
            chunk_size=max(len(prompt) for prompt in prompts),
            block_size=session.block_size,
        )
        if len(slabs) != 1:
            raise RuntimeError(f"expected one compact prefill slab, got {len(slabs)}")
        session.prefill_native_packed(slabs[0], sample=True)
        session.runtime.device_synchronize()
        slots = tuple(range(rows))
        for step in range(decode_tokens):
            positions = tuple(len(prompts[row]) + step for row in range(rows))
            inputs = [int(teacher_inputs[row][step]) for row in range(rows)]
            session._set_batch_token_embeddings(inputs, stream=0)
            session._set_batch_positions(positions, stream=0)
            hidden = session._run_layers_batch_decode(
                rows=rows,
                positions=positions,
                slots=slots,
                stream=0,
            )
            session.runtime.device_synchronize()
            for row in range(rows):
                row_view = Tensor.from_handle(
                    hidden.ptr + row * hidden_size * fp16_itemsize,
                    (1, hidden_size),
                    DType.FP16,
                    session.device,
                )
                session._sample_device_from_hidden(row_view, stream=0)
                session.runtime.device_synchronize()
                batch_logits[row][step] = _copy_full_logits(session, vocab)
    return batch_logits


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    from hipengine.runtime.qwen35_paro_runner import Qwen35ParoNextTokenRunner

    prompts = _load_prompt_slices(
        Path(args.fixture), prompt_length=args.prompt_length, batch_size=args.batch_size
    )
    compiler_version = _compiler_version(args.compiler_version_file)
    runner = Qwen35ParoNextTokenRunner(Path(args.model))

    teacher_inputs, c1_tokens, c1_logits = _run_c1_teacher(
        runner,
        prompts,
        decode_tokens=args.decode_tokens,
        max_sequence_length=args.max_sequence_length,
        max_layers=args.max_layers,
        compiler_version=compiler_version,
        require_cached_build=args.require_cached_build,
    )
    batch_logits = _run_batch_teacher_forced(
        runner,
        prompts,
        teacher_inputs,
        decode_tokens=args.decode_tokens,
        max_sequence_length=args.max_sequence_length,
        max_layers=args.max_layers,
        compiler_version=compiler_version,
        require_cached_build=args.require_cached_build,
    )

    records: list[dict[str, Any]] = []
    for row in range(len(prompts)):
        for step in range(args.decode_tokens):
            c1_lg = c1_logits[row][step]
            b_lg = batch_logits[row][step]
            c1_tok = int(c1_tokens[row][step])
            b_tok = top1_from_logits(b_lg)
            records.append(
                {
                    "row": row,
                    "step": step,
                    "c1_token": c1_tok,
                    "batch_token": b_tok,
                    "top1_match": bool(c1_tok == b_tok),
                    "kl": kl_divergence_from_logits(c1_lg, b_lg),
                }
            )

    summary = summarize_metrics(
        records, kl_threshold=args.kl_threshold, top1_threshold=args.top1_threshold
    )
    # First disagreement (for context only; not a chaos-limited equal-prefix metric).
    first_top1_mismatch = next(
        (
            {"row": r["row"], "step": r["step"], "c1_token": r["c1_token"], "batch_token": r["batch_token"], "kl": r["kl"]}
            for r in records
            if not r["top1_match"]
        ),
        None,
    )
    worst_kl = max(records, key=lambda r: r["kl"]) if records else None
    payload: dict[str, Any] = {
        "schema": 1,
        "kind": "teacher_forced_kl_top1_gate",
        "status": "passed" if summary.get("passed") else "failed",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "performance_claim": False,
        "gate": "KL(c1||batch) mean <= kl_threshold AND top-1 agreement >= top1_threshold (teacher-forced, per decode step)",
        "workload": {
            "model": str(args.model),
            "fixture": str(args.fixture),
            "prompt_length": int(args.prompt_length),
            "batch_size": int(args.batch_size),
            "decode_tokens": int(args.decode_tokens),
            "max_layers": int(args.max_layers),
            "max_sequence_length": int(args.max_sequence_length),
            "row_chunk_env": {
                "size": os.environ.get("HIPENGINE_QWEN35_BATCH_DECODE_FULL_ATTN_ROW_CHUNK_SIZE", ""),
                "layers": os.environ.get("HIPENGINE_QWEN35_BATCH_DECODE_FULL_ATTN_ROW_CHUNK_LAYERS", ""),
            },
            "native_path_note": "No row-chunk env set => pure native rows=N full-attention decode (the shippable native c-aware path).",
        },
        "summary": summary,
        "first_top1_mismatch": first_top1_mismatch,
        "worst_kl": None if worst_kl is None else {
            "row": worst_kl["row"], "step": worst_kl["step"], "kl": worst_kl["kl"],
            "top1_match": worst_kl["top1_match"],
        },
    }
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--fixture", default=DEFAULT_FIXTURE)
    parser.add_argument("--prompt-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--max-layers", type=int, default=40)
    parser.add_argument("--max-sequence-length", type=int, default=1024)
    parser.add_argument("--kl-threshold", type=float, default=0.05)
    parser.add_argument("--top1-threshold", type=float, default=0.90)
    parser.add_argument("--compiler-version-file", type=Path, default=None)
    parser.add_argument("--require-cached-build", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not _hip_available():
        print("libamdhip64.so not available; teacher-forced KL gate requires HIP/ROCm.", file=sys.stderr)
        return 3
    payload = run_gate(args)
    text = json.dumps(payload, indent=2, allow_nan=False)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text)
    print(text)
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
