"""Bounded diagnostic: opt-in rank-local bulk TP2 prefill vs the TP1 teacher.

Diagnostics only. Runs the experimental
``MlpTP2GenerationSession(bulk_prefill=True)`` route over the same trajectory
protocol the saved TP1 teacher used - prefill the prompt, then feed the saved
forced-decode tokens one at a time - and compares the full decode logits under
the unchanged production KL/top-1 envelope. This is not a benchmark, not a
speedup claim, and not a promotion certificate: it is the bounded GPU check
that the bulk-prefill wiring runs end to end, preserves the KV/GDN state into
the graph decode, and keeps the production numerical contract.

Usage::

    HIP_VISIBLE_DEVICES=0,1 HIP_PATH=/opt/rocm python3 \
        scripts/tp2_bulk_prefill_diagnostic.py \
        --reference /home/lhl/.cache/hipengine/tp2-d128-baseline/quality-tp1-d0.json \
        --json /home/lhl/.cache/hipengine/tp2-bulk-prefill-diagnostic.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shlex
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from tp2_mlp_generate_e2e import PRODUCTION_GATE  # noqa: E402
from tp2_teacher_coverage_broad import (  # noqa: E402
    CATEGORY_TOP1,
    _aggregate,
    _envelope_gate,
    _kl_rows,
)

MODEL = "/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"

# ``gguf_linear.resolve`` is wrapped so every resolution records the leaf it
# selected. The surface row names the rows=1 owner, but the launcher rewrites
# Q4/Q6 T16 dispatches to row- and shape-qualified leaves, so the log is what
# proves *which* kernel the route actually launched (and that an f32 request did
# not silently land on a bf16 store).
_RESOLVE_LOG: list | None = None


def _install_resolve_logger(module) -> None:
    original = getattr(module, "resolve", None)
    if original is None or getattr(original, "__wrapped__", None) is not None:
        return

    def logged_resolve(*args, **kwargs):
        log = _RESOLVE_LOG
        if log is not None:
            log.append({key: str(value) for key, value in kwargs.items()})
        return original(*args, **kwargs)

    logged_resolve.__wrapped__ = original
    module.resolve = logged_resolve


def _summarize_resolve_log(entries: list) -> dict:
    """Group the resolved leaves by (layer, quant, variant) with call counts."""

    counts: dict[tuple[str, str, str], int] = {}
    for entry in entries:
        key = (
            str(entry.get("layer", "")),
            str(entry.get("quant", "")),
            str(entry.get("variant", "")),
        )
        counts[key] = counts.get(key, 0) + 1
    return {
        "entries": len(entries),
        "leaves": [
            {"layer": layer, "quant": quant, "variant": variant, "calls": calls}
            for (layer, quant, variant), calls in sorted(counts.items())
        ],
    }


def _load_reference(path: Path) -> tuple[dict, list[np.ndarray]]:
    reference = json.loads(Path(path).read_text())
    arrays = [np.load(row["path"]) for row in reference["arrays"]]
    return reference, arrays


def _build_session(
    max_sequence_length: int,
    *,
    rows: int | None = None,
    schedule: str = "graphed",
    bulk: bool = True,
    decode_partial_dtype: str = "bf16",
    reduce_mode: str | None = None,
    t16_f16_rocblas_prefill: bool | None = None,
):
    from hipengine.distributed.tp2_generate import MlpTP2GenerationSession

    kwargs: dict = {}
    if bulk:
        kwargs["bulk_prefill"] = True
        kwargs["bulk_prefill_rows"] = (
            rows if rows is not None else max_sequence_length
        )
    if reduce_mode is not None:
        kwargs["reduce_mode"] = reduce_mode
    if t16_f16_rocblas_prefill is not None:
        kwargs["use_t16_f16_rocblas_prefill"] = t16_f16_rocblas_prefill
    return MlpTP2GenerationSession(
        MODEL,
        devices=(0, 1),
        mode="tp2",
        max_sequence_length=max_sequence_length,
        schedule=schedule,
        head_shard=True,
        decode_partial_dtype=decode_partial_dtype,
        **kwargs,
    )


def _trajectory(session, prompt, forced, *, bulk: bool) -> np.ndarray:
    """The saved teacher protocol: prefill the prompt, then forced decode."""

    session.reset()
    if bulk:
        session.bulk_prefill(prompt)
    else:
        for position, token in enumerate(prompt):
            session._forward_token(int(token), position, kind="prefill")
    rows: list[np.ndarray] = []
    start = len(prompt)
    for index, token in enumerate(forced):
        logits, _trace = session._forward_token(
            int(token), start + index, kind="decode"
        )
        rows.append(np.asarray(logits, dtype=np.float32).reshape(-1))
    return np.stack(rows)


def _score(teacher: np.ndarray, student: np.ndarray) -> dict:
    kl, top1 = _kl_rows(
        np.asarray(teacher, dtype=np.float64), np.asarray(student, dtype=np.float64)
    )
    summary = _aggregate(kl, top1)
    summary["gate"] = _envelope_gate(summary, top1_bar=CATEGORY_TOP1)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference",
        default="/home/lhl/.cache/hipengine/tp2-d128-baseline/quality-tp1-d0.json",
    )
    parser.add_argument("--prompt", default="mixed_ja_en_translate")
    parser.add_argument("--heldout", default="heldout_mixed_summary")
    parser.add_argument(
        "--smoke-rows",
        type=int,
        default=0,
        help="if >0, only score the first N decode rows as a fast smoke",
    )
    parser.add_argument("--max-sequence-length", type=int, default=200)
    parser.add_argument(
        "--bulk-capacity",
        type=int,
        default=0,
        help="bulk_prefill_rows capacity; 0 = max_sequence_length (default)",
    )
    parser.add_argument("--schedule", default="graphed", choices=("eager", "graphed"))
    parser.add_argument(
        "--decode-partial-dtype",
        default="bf16",
        choices=("bf16", "f32"),
        help=(
            "dtype of the single-row down partial before the staged reduction; "
            "f32 removes one bf16 rounding per rank"
        ),
    )
    parser.add_argument(
        "--reduce-mode",
        default=None,
        choices=("host", "device"),
        help=(
            "exchange reduction owner; default is the session's own default "
            "(device for graphed, host for eager). f32 partials need host"
        ),
    )
    parser.add_argument(
        "--pad-prompt-tokens",
        type=int,
        default=0,
        help=(
            "extend each prompt to this many tokens by repeating its last token; "
            "use >=512 to exercise the source-F16 prefill owner, which the "
            "suite's 52-64 token prompts do not reach"
        ),
    )
    parser.add_argument(
        "--t16-f16-rocblas-prefill",
        default=None,
        choices=("default", "on", "off"),
        help=(
            "source-F16 rocBLAS prefill owner on the TP2 attention path; "
            "'default' uses the session default, 'off' is the exact T16 fallback"
        ),
    )
    parser.add_argument(
        "--compare-serial",
        action="store_true",
        help="also run the token-serial TP2 prefill on the same session",
    )
    parser.add_argument(
        "--serial-only",
        action="store_true",
        help="run the plain token-serial TP2 route (no bulk prefill at all)",
    )
    parser.add_argument("--json", required=True)
    parser.add_argument(
        "--save-logits-dir",
        default="",
        help="optional directory to dump each prompt's raw logits as .npy",
    )
    args = parser.parse_args(argv)
    if args.t16_f16_rocblas_prefill is not None:
        # ``default`` must mean "omit the flag" - mapping it to True would make
        # an explicit `--t16-f16-rocblas-prefill default` silently override the
        # session's default-off decision, which is the opposite of its name.
        args.t16_f16_rocblas_prefill = {
            "default": None,
            "on": True,
            "off": False,
        }[args.t16_f16_rocblas_prefill]

    reference, arrays = _load_reference(Path(args.reference))
    ids = list(reference["suite"]["ids"])
    prompts = reference["suite"]["tokens"]
    forced_inputs = reference["forced_inputs"]
    by_id = {str(pid): index for index, pid in enumerate(ids)}
    for name in (args.prompt, args.heldout):
        if name not in by_id:
            raise SystemExit(f"prompt {name!r} not in reference suite {ids}")

    def assert_prompt_matches_reference(name: str, prompt: list[int]) -> None:
        """Refuse to score a student whose context differs from the teacher's.

        This check exists because its absence invalidated a whole gate run. The
        prompt was padded to reach a row count the source-F16 policy admits, but
        the teacher still held logits for the *unpadded* prompt, so the reported
        KL and top-1 compared different contexts rather than implementation
        drift. Padding is only legitimate when the reference was captured with
        the same padding, and that is what this asserts.
        """

        index = by_id[name]
        teacher_tokens = [int(t) for t in prompts[index]]
        if len(teacher_tokens) != len(prompt):
            raise SystemExit(
                f"prompt {name!r}: the student has {len(prompt)} tokens but the "
                f"reference teacher has {len(teacher_tokens)}. Scoring these would "
                "compare different contexts. Re-capture the teacher with a matching "
                "length (scripts/tp2_teacher_coverage_broad.py --pad-prompt-tokens)."
            )
        if teacher_tokens != prompt:
            first = next(
                (i for i, (a, b) in enumerate(zip(teacher_tokens, prompt)) if a != b),
                None,
            )
            raise SystemExit(
                f"prompt {name!r}: student tokens differ from the reference teacher "
                f"at position {first}. The teacher must cover the identical prompt."
            )

    def assert_trajectory_matches_reference(name: str, forced: list[int]) -> None:
        index = by_id[name]
        teacher_forced = [int(t) for t in forced_inputs[index]]
        if teacher_forced[: len(forced)] != forced:
            raise SystemExit(
                f"prompt {name!r}: the forced trajectory differs from the reference "
                "teacher's, so the compared decode positions are not the same run."
            )

    targets = [args.prompt, args.heldout]
    result: dict = {
        "kind": "tp2_bulk_prefill_diagnostic",
        "model": MODEL,
        "reference": str(args.reference),
        "reference_arm": reference.get("arm"),
        "command": " ".join(shlex.quote(part) for part in sys.argv),
        "host": platform.node(),
        "route": {
            "mode": "tp2",
            "schedule": args.schedule,
            "head_shard": True,
            "bulk_prefill": not args.serial_only,
            "bulk_prefill_rows": (
                args.bulk_capacity if args.bulk_capacity > 0 else args.max_sequence_length
            ),
            "decode_partial_dtype": args.decode_partial_dtype,
            "reduce_mode": args.reduce_mode or "session-default",
        },
        "protocol": "prefill(prompt) then forced decode (matches the teacher capture)",
        "smoke_rows": int(args.smoke_rows),
        "prompts": {},
    }

    session = None
    started = time.perf_counter()
    global _RESOLVE_LOG
    import hipengine.runtime.gguf_linear as gguf_linear

    _install_resolve_logger(gguf_linear)
    _RESOLVE_LOG = []
    try:
        bulk_capacity = (
            args.bulk_capacity
            if args.bulk_capacity > 0
            else args.max_sequence_length
        )
        session = _build_session(
            args.max_sequence_length,
            rows=bulk_capacity,
            schedule=args.schedule,
            bulk=not args.serial_only,
            decode_partial_dtype=args.decode_partial_dtype,
            reduce_mode=args.reduce_mode,
            t16_f16_rocblas_prefill=args.t16_f16_rocblas_prefill,
        )
        # Match the saved capture: build the decode schedule before running any
        # trajectory, so the decode replays the captured graphs and the bulk
        # prefill's state survives into them.
        session._ensure_graph_schedule()
        for prompt_id in targets:
            index = by_id[prompt_id]
            prompt = [int(t) for t in prompts[index]]
            if args.pad_prompt_tokens > len(prompt):
                # Extend the prompt so the pass reaches a row count the
                # source-F16 policy actually admits (its smallest is 512). The
                # suite prompts are 52-64 tokens, so without this the owner
                # falls back and the gate says nothing about it.
                #
                # This is *dispatch coverage*, not qualification: it is only
                # scoreable when the reference was captured with the same
                # padding, which assert_prompt_matches_reference enforces below.
                pad_id = int(prompt[-1])
                prompt = prompt + [pad_id] * (args.pad_prompt_tokens - len(prompt))
            forced = [int(t) for t in forced_inputs[index]]
            # Refuse to score a context the teacher never saw. This is the check
            # whose absence invalidated the first run of this gate.
            assert_prompt_matches_reference(prompt_id, prompt)
            assert_trajectory_matches_reference(prompt_id, forced)
            teacher = np.asarray(arrays[index])
            if args.smoke_rows > 0:
                forced = forced[: args.smoke_rows]
                teacher = teacher[: args.smoke_rows]
            t0 = time.perf_counter()
            logits = _trajectory(
                session, prompt, forced, bulk=not args.serial_only
            )
            elapsed = time.perf_counter() - t0
            logits = np.asarray(logits, dtype=np.float64)
            row: dict = {
                "category": reference["suite"]["categories"][index],
                "heldout": bool(reference["suite"]["heldout"][index]),
                "prompt_tokens": len(prompt),
                "decode_steps": int(logits.shape[0]),
                "vocab": int(logits.shape[1]),
                "seconds": elapsed,
                "finite": bool(np.isfinite(logits).all()),
            }
            if args.compare_serial or args.serial_only:
                session.bulk_prefill_enabled = False
                try:
                    serial = np.asarray(
                        _trajectory(session, prompt, forced, bulk=False),
                        dtype=np.float64,
                    )
                finally:
                    session.bulk_prefill_enabled = not args.serial_only
                row["serial_sha256"] = hashlib.sha256(
                    np.ascontiguousarray(serial.astype("<f4")).tobytes()
                ).hexdigest()
                row["bulk_vs_serial_metrics"] = _score(serial, logits)
                row["teacher_argmax"] = teacher.argmax(axis=-1).tolist()
                row["serial_argmax"] = serial.argmax(axis=-1).tolist()
                row["bulk_argmax"] = logits.argmax(axis=-1).tolist()
            if logits.shape != teacher.shape:
                row["status"] = "shape_mismatch"
                row["teacher_shape"] = list(teacher.shape)
            else:
                row["metrics"] = _score(teacher, logits)
                row["status"] = (
                    "pass" if row["metrics"]["gate"]["passed"] else "fail"
                )
                row["logits_sha256"] = hashlib.sha256(
                    np.ascontiguousarray(logits.astype("<f4")).tobytes()
                ).hexdigest()
                if args.save_logits_dir:
                    out_dir = Path(args.save_logits_dir)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    np.save(out_dir / f"{prompt_id}.bulk.npy", logits.astype("<f4"))
                    if args.compare_serial or args.serial_only:
                        np.save(
                            out_dir / f"{prompt_id}.serial.npy",
                            serial.astype("<f4"),
                        )
            result["prompts"][prompt_id] = row
            print(json.dumps({prompt_id: row}, sort_keys=True), flush=True)
    finally:
        if session is not None:
            session.close()
        result["resolve_log"] = _summarize_resolve_log(_RESOLVE_LOG or [])
        _RESOLVE_LOG = None
    result["seconds"] = time.perf_counter() - started
    result["production_gate"] = PRODUCTION_GATE
    result["all_passed"] = all(
        row.get("status") == "pass" for row in result["prompts"].values()
    )
    Path(args.json).write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {args.json} all_passed={result['all_passed']}", flush=True)
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
