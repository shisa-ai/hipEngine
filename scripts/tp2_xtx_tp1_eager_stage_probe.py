#!/usr/bin/env python3
"""Minimal single-prompt XTX eager prefill/step/sampler stage probe.

Localizes the first bad stage of the optimized resident TP1 route
(``Qwen35GGUFResidentSession``) on the RX 7900 XTX by running one short prompt
eagerly with a synchronized guard and a flushed stage log after every step.

The CLI default ``--use-gemv-decode`` is True, so the failing resident route runs
with the ``pack8_gemv_decode_*`` family active; this probe can toggle it to
separate "device-1 fault" from "GEMV-decode-family fault".

Each stage prints ``STAGE <name> START`` before any device work and
``STAGE <name> OK`` after a ``device_synchronize``, so the last printed line in
the log identifies a hang. Outputs are checked for sentinel/unwritten values
(``INT64_MAX`` token id, non-finite logits, all-zero logits) and every failure
is recorded with its exact stage rather than raised past the log.

This is a bounded diagnostic: one prompt, one session, no reset. Do not treat
an idle-healthy card as a certified healthy context.

Usage:
  HIP_VISIBLE_DEVICES=1 python3 scripts/tp2_xtx_tp1_eager_stage_probe.py \
      --model /models/gguf/Qwen3.8-27B-Q4_K_M.gguf --json /tmp/xtx-eager.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL = Path("/models/gguf/Qwen3.8-27B-Q4_K_M.gguf")
PROMPT = "Write one line of Python that prints hello."


def _log(message: str) -> None:
    print(message, flush=True)


def _summarize_logits(logits: np.ndarray, vocab_size: int) -> dict[str, object]:
    if logits is None:
        return {"present": False}
    flat = np.asarray(logits).reshape(-1)
    finite = bool(np.isfinite(flat).all())
    return {
        "present": True,
        "shape": list(np.asarray(logits).shape),
        "finite": finite,
        "all_zero": bool(np.all(flat == 0.0)),
        "nonzero": int(np.count_nonzero(flat)),
        "argmax": int(np.argmax(flat)) if flat.size else None,
        "min": float(flat.min()) if flat.size else None,
        "max": float(flat.max()) if flat.size else None,
        "sentinel_fraction": float(np.mean(flat == 0x7BFF)) if flat.size else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--use-gemv-decode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--with-graph", action="store_true", help="also probe graph capture/replay")
    parser.add_argument("--with-serial", action="store_true", help="also probe the use_bulk=False path")
    parser.add_argument("--os-exit", action="store_true", help="skip destructors (diagnostic only)")
    args = parser.parse_args(argv)

    # Match the working true-AR harness default: decode repack is on unless the
    # caller explicitly disables it. Without this the resident session's decode
    # path diverges and the lm-head logits come back NaN on a healthy device.
    os.environ.setdefault("HIPENGINE_GGUF_DECODE_REPACK", "1")

    artifact: dict[str, object] = {
        "kind": "tp2_xtx_tp1_eager_stage_probe",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "env": {
            "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
            "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES"),
        },
        "use_gemv_decode": bool(args.use_gemv_decode),
        "stages": [],
        "first_bad_stage": None,
    }
    stages: list[dict[str, object]] = artifact["stages"]  # type: ignore[assignment]

    def record(name: str, payload: dict[str, object]) -> None:
        entry = {"stage": name, "ms": payload.pop("ms", None), **payload}
        stages.append(entry)
        if entry.get("ok") is False and artifact["first_bad_stage"] is None:
            artifact["first_bad_stage"] = name
        _log(f"STAGE {name} {'OK' if entry.get('ok') else 'FAIL'} {entry}")

    from hipengine.loading.gguf import scan_gguf
    from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession
    from hipengine.tokenization.gguf import Qwen35GGUFTokenizer
    from scripts.gguf_mtp_bench import build_chat_prompt

    _log("STAGE build START")
    info = scan_gguf(args.model)
    tokenizer = Qwen35GGUFTokenizer.from_gguf_info(info)
    tokens = build_chat_prompt(tokenizer, args.prompt)
    build_start = time.perf_counter()
    session = Qwen35GGUFResidentSession(
        args.model,
        max_sequence_length=max(512, len(tokens) + 8),
        use_wmma_prefill=True,
        use_gemv_decode=bool(args.use_gemv_decode),
    )
    runtime = session.runtime
    vocab_size = int(session.runner.vocab_size)
    device_info = runtime.device_info(0)
    record(
        "build",
        {
            "ok": True,
            "ms": 1000.0 * (time.perf_counter() - build_start),
            "device": {
                "name": device_info.name,
                "uuid": device_info.uuid,
                "pci_bus_id": device_info.pci_bus_id,
            },
            "prompt_tokens": len(tokens),
            "vocab_size": vocab_size,
        },
    )

    def guard(stage: str, fn) -> None:  # type: ignore[no-untyped-def]
        _log(f"STAGE {stage} START")
        start = time.perf_counter()
        try:
            payload = fn()
            runtime.device_synchronize()
            record(stage, {"ok": True, "ms": 1000.0 * (time.perf_counter() - start), **payload})
        except Exception as error:  # noqa: BLE001 - record the exact stage and continue
            record(
                stage,
                {
                    "ok": False,
                    "ms": 1000.0 * (time.perf_counter() - start),
                    "error": f"{type(error).__name__}: {error}",
                },
            )

    def prefill_auto():
        session.reset()
        result = session.prefill(tokens, use_bulk=None, return_logits=True)
        token = int(result.token_id)
        return {
            "token_id": token,
            "token_in_range": 0 <= token < vocab_size,
            "logits": _summarize_logits(result.logits, vocab_size),
        }

    def eager_step():
        session.reset()
        first = session.prefill(tokens, use_bulk=None, return_logits=False)
        result = session.step(int(first.token_id), return_logits=True)
        token = int(result.token_id)
        return {
            "prefill_token_id": int(first.token_id),
            "step_token_id": token,
            "token_in_range": 0 <= token < vocab_size,
            "logits": _summarize_logits(result.logits, vocab_size),
        }

    guard("prefill-auto", prefill_auto)
    guard("eager-step", eager_step)

    if args.with_serial:
        def prefill_serial():
            session.reset()
            result = session.prefill(tokens, use_bulk=False, return_logits=True)
            token = int(result.token_id)
            return {
                "token_id": token,
                "token_in_range": 0 <= token < vocab_size,
                "logits": _summarize_logits(result.logits, vocab_size),
            }

        guard("prefill-serial", prefill_serial)

    if args.with_graph:
        def graph_roundtrip():
            session.reset()
            session.prefill(tokens, use_bulk=None, return_logits=False)
            graph = session.capture_decode_graph(
                position=int(session.position),
                steps_per_replay=1,
                max_replay_steps=2,
                attention_max_context_len=int(session.position) + 2,
            )
            try:
                graph.replay(1)
                result = graph.read_sample(return_logits=True)
                token = int(result.token_id)
                return {
                    "token_id": token,
                    "token_in_range": 0 <= token < vocab_size,
                    "logits": _summarize_logits(result.logits, vocab_size),
                }
            finally:
                graph.close()

        guard("graph-roundtrip", graph_roundtrip)

    _log("STAGE teardown START")
    session.close()
    record("teardown", {"ok": True})
    if args.json:
        args.json.write_text(json.dumps(artifact, indent=1) + "\n")
    _log(f"first_bad_stage={artifact['first_bad_stage']}")
    if args.os_exit:
        # Diagnostic only: skipping destructors must never count as teardown
        # qualification.
        os._exit(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
