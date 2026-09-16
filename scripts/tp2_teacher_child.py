"""Fresh-process, main-thread TP2/TP1 teacher-forced control child.

One process builds exactly one ``MlpTP2GenerationSession`` and executes every
teacher-forced call on the process's main thread, so a hung device wait can be
attributed to a single session with a single owning thread. The parent
(``tp2_teacher_bisect_parent.py``) enforces the deadline and reaps this
process; this child never closes the session while a call is blocked.

Output is line-oriented and flushed after every line so the parent can retain
partial progress on a timeout:

    CHILD_TOKENS [<exact token ids used>]
    CHILD_BUILD_OK elapsed_s=<s> mode=... schedule=... devices=...
    CHILD_EXEC_START calls=<k> positions=<n> reset_between=<bool>
    CHILD_CALL call=<i> positions=<n> wall_s=<s> finite=<bool> argmax_last=<id>
    CHILD_OK calls=<k> positions=<n> total_wall_s=<s>
    CHILD_CLEANUP_OK

Fail-closed behavior:

* ``--calls`` must be >= 1; a zero-call run is rejected, never reported OK;
* every returned logits tensor must have shape ``(positions, vocab_size)`` and
  be finite, or the child fails;
* teardown failure is reported (``CHILD_CLEANUP_FAIL``) and makes the exit
  code non-zero even when the calls themselves succeeded.

Usage::

    python scripts/tp2_teacher_child.py --positions 16 --devices 1 \
        --mode tp1 --schedule eager --calls 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

MODEL = "/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"

#: The fixed canonical sequence every control case is a prefix of. It is a
#: chat-templated code request, so prefixes are ordinary prefill contexts.
CANONICAL_PROMPT = (
    "<|im_start|>user\nWrite a Python function that reverses a list and "
    "explain its complexity in detail with examples and edge cases "
    "considered.<|im_end|>\n<|im_start|>assistant\n"
)


def canonical_tokens() -> tuple[int, ...]:
    """Encode :data:`CANONICAL_PROMPT` once, deterministically."""

    import hipengine.loading as _loading
    from hipengine.tokenization.gguf import Qwen35GGUFTokenizer

    tok = Qwen35GGUFTokenizer.from_gguf_info(_loading.load_gguf_index(MODEL))
    return tuple(int(t) for t in tok.encode(CANONICAL_PROMPT))


def _session_factory(
    model: str, *, devices: tuple[int, ...], mode: str, schedule: str | None
) -> object:
    """Construct one session; a seam for mocked tests (no GPU needed)."""

    from hipengine.distributed.tp2_generate import MlpTP2GenerationSession

    return MlpTP2GenerationSession(model, devices=devices, mode=mode, schedule=schedule)


def validate_logits(
    logits: np.ndarray, *, positions: int, vocab_size: int
) -> tuple[bool, str]:
    """Return ``(ok, detail)``; enforce full shape and finiteness."""

    if logits.ndim != 2:
        return False, f"ndim={logits.ndim} expected 2"
    if logits.shape[0] != positions:
        return False, f"rows={logits.shape[0]} expected {positions}"
    if logits.shape[1] != vocab_size:
        return False, f"cols={logits.shape[1]} expected {vocab_size}"
    if not bool(np.isfinite(logits).all()):
        return False, f"non-finite logits ({logits_stats(logits)})"
    return True, ""


def logits_stats(logits: np.ndarray) -> str:
    """A compact finite/nan/inf summary for failure evidence."""

    finite_mask = np.isfinite(logits)
    finite_values = logits[finite_mask]
    return (
        f"shape={list(logits.shape)} finite={int(finite_mask.sum())}/{logits.size} "
        f"nan={int(np.isnan(logits).sum())} inf={int(np.isinf(logits).sum())} "
        f"min={float(finite_values.min()) if finite_values.size else float('nan'):.4g} "
        f"max={float(finite_values.max()) if finite_values.size else float('nan'):.4g}"
    )


def _emit(line: str) -> None:
    print(line, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", type=int, required=True)
    parser.add_argument("--devices", required=True, help="comma-separated device ids")
    parser.add_argument("--mode", choices=("tp1", "tp2"), required=True)
    parser.add_argument("--schedule", choices=("eager", "graphed"), default=None)
    parser.add_argument("--calls", type=int, default=1)
    parser.add_argument(
        "--reset-between",
        action="store_true",
        help="call session.reset() between calls (reset/reuse boundary check)",
    )
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    devices = tuple(int(d) for d in args.devices.split(",") if d != "")
    if args.mode == "tp2" and len(devices) != 2:
        _emit(f"CHILD_FAIL reason=bad-devices detail=tp2 needs two devices got={devices}")
        return 2
    if args.mode == "tp1" and len(devices) != 1:
        _emit(f"CHILD_FAIL reason=bad-devices detail=tp1 needs one device got={devices}")
        return 2
    if args.positions < 1:
        _emit("CHILD_FAIL reason=bad-positions")
        return 2
    if args.calls < 1:
        _emit(f"CHILD_FAIL reason=bad-calls detail=calls must be >=1 got={args.calls}")
        return 2

    full = canonical_tokens()
    if args.positions > len(full):
        _emit(
            f"CHILD_FAIL reason=prefix-too-short requested={args.positions} "
            f"available={len(full)}"
        )
        return 2
    tokens = full[: args.positions]
    _emit(f"CHILD_TOKENS {list(tokens)}")

    session: object | None = None
    closed = False
    cleanup_error: str | None = None
    record: dict[str, object] = {
        "positions": args.positions,
        "devices": list(devices),
        "mode": args.mode,
        "schedule": args.schedule,
        "calls": args.calls,
        "reset_between": bool(args.reset_between),
        "tokens": list(tokens),
    }

    def _cleanup() -> None:
        nonlocal closed, cleanup_error
        if closed or session is None:
            return
        closed = True
        try:
            session.close()
        except Exception as error:  # noqa: BLE001 - record, fail closed
            cleanup_error = f"{type(error).__name__}: {error}"

    status = "ok"
    exit_code = 0
    try:
        build0 = time.perf_counter()
        session = _session_factory(
            MODEL, devices=devices, mode=args.mode, schedule=args.schedule
        )
        build_s = time.perf_counter() - build0
        record["build_s"] = build_s
        vocab_size = int(session.vocab_size)  # type: ignore[attr-defined]
        _emit(
            f"CHILD_BUILD_OK elapsed_s={build_s:.3f} mode={args.mode} "
            f"schedule={session.schedule} devices={list(devices)} "  # type: ignore[attr-defined]
            f"vocab_size={vocab_size}"
        )
        _emit(
            f"CHILD_EXEC_START calls={args.calls} positions={args.positions} "
            f"reset_between={bool(args.reset_between)}"
        )

        call_records: list[dict[str, object]] = []
        exec0 = time.perf_counter()
        for call in range(args.calls):
            if call > 0 and args.reset_between:
                session.reset()  # type: ignore[attr-defined]
            c0 = time.perf_counter()
            logits = np.asarray(
                session.teacher_forced_logits(tokens), dtype=np.float32  # type: ignore[attr-defined]
            )
            wall = time.perf_counter() - c0
            ok, detail = validate_logits(
                logits, positions=args.positions, vocab_size=vocab_size
            )
            _emit(f"CHILD_CALL_STATS call={call} {logits_stats(logits)}")
            if not ok:
                status = "invalid_logits"
                exit_code = 1
                record["calls_detail"] = call_records
                _emit(f"CHILD_FAIL reason=invalid-logits call={call} detail={detail}")
                break
            argmax_last = int(np.argmax(logits[-1]))
            call_records.append(
                {
                    "call": call,
                    "wall_s": wall,
                    "finite": True,
                    "argmax_last": argmax_last,
                    "shape": list(logits.shape),
                }
            )
            _emit(
                f"CHILD_CALL call={call} positions={args.positions} "
                f"wall_s={wall:.4f} finite=True argmax_last={argmax_last}"
            )
        else:
            total_wall = time.perf_counter() - exec0
            record["calls_detail"] = call_records
            record["exec_s"] = total_wall
            _emit(
                f"CHILD_OK calls={args.calls} positions={args.positions} "
                f"total_wall_s={total_wall:.4f}"
            )
    except Exception as error:  # noqa: BLE001 - surface, do not swallow
        status = "error"
        exit_code = 1
        record["error"] = f"{type(error).__name__}: {error}"
        _emit(f"CHILD_FAIL reason=exception detail={type(error).__name__}: {error}")
    finally:
        # Only reached once the calls have returned (or raised). A blocked call
        # is killed by the parent, never closed from here.
        _cleanup()

    if cleanup_error is not None:
        status = "cleanup_failed"
        exit_code = 1
        record["cleanup_error"] = cleanup_error
        _emit(f"CHILD_CLEANUP_FAIL detail={cleanup_error}")
    elif session is not None:
        _emit("CHILD_CLEANUP_OK")

    record["status"] = status
    record["exit_code"] = exit_code
    _write_json(args.json, record)
    _emit(f"CHILD_EXITING code={exit_code} status={status}")
    return exit_code


def _write_json(path: str | None, record: dict[str, object]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    # A poisoned/errored device can leave HIP runtime teardown hanging after the
    # session has closed (observed on device 1). The session's cleanup already
    # ran, so bypass interpreter shutdown and exit with the logical code; the
    # parent must see a definite return code rather than a timeout.
    _code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_code)
