#!/usr/bin/env python3
"""Save and restore Gemma 4 KV state across processes, with an exactness gate.

Three modes, one state file:

* ``save`` runs the prefill once and writes the KV state -- planes, position,
  last-row logits and the campaign identity (prompt-id hash, artifact head
  hash, context, engine commit) -- so a later run can skip that prefill.
* ``restore`` loads the state into a fresh runner and decodes from it,
  reporting decode throughput and the restore wall time. This is the
  iteration harness: a 262015-token context is minutes of load instead of a
  76-minute prefill.
* ``verify`` does save and restore in one process on one runner, then asserts
  the restored decode's greedy token IDs are exactly the fresh decode's. The
  runner-level cross-runner parity is covered by
  ``tests/test_gpu_gemma4_kv_restore.py``; this mode is the at-scale check of
  the same contract, where the file roundtrip is the risky part.

The prompt ids come from the frozen campaign corpus via
``scripts.gemma4_campaign_bench.exact_prompt_ids``, and the identity hashes
use the campaign bench's own formulas, so a saved state pairs with the
ladder's rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.gemma4_campaign_bench import exact_prompt_ids, resolve_artifact


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.strip()
    except Exception:
        return ""


def _sha256_head(path: Path, limit: int = 64 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(limit))
    return digest.hexdigest()


def _identity(args: argparse.Namespace, prompt_ids: list[int]) -> dict:
    return {
        "prompt_ids_sha256": hashlib.sha256(
            b"".join(int(t).to_bytes(4, "little") for t in prompt_ids)
        ).hexdigest(),
        "artifact_head_sha256_64MiB": _sha256_head(Path(args.artifact)),
        "context": int(args.context),
        "output_tokens": int(args.output),
        "engine_commit": _git_commit(),
    }


def _resolve_runner(args: argparse.Namespace):
    import hipengine

    llm = hipengine.LLM(model=str(args.artifact), max_sequence_length=int(args.context))
    generator = llm._get_text_generator()
    generator.context_length = int(args.context)
    runner = generator._ensure_runner()
    return generator, runner


def _decode(runner, first_token: int, count: int) -> tuple[list[int], float]:
    import numpy as np

    generated = [int(first_token)]
    decode_s = 0.0
    for _ in range(count - 1):
        started = time.perf_counter()
        logits = runner.forward([generated[-1]])
        import hipengine.core.hip as hip

        hip.get_hip_runtime().device_synchronize()
        decode_s += time.perf_counter() - started
        generated.append(int(np.argmax(logits)))
    return generated, decode_s


def _fresh_phase(runner, generator, prompt_ids: list[int], args):
    import numpy as np

    runner.reset()
    started = time.perf_counter()
    logits = runner.forward(prompt_ids)
    import hipengine.core.hip as hip

    hip.get_hip_runtime().device_synchronize()
    prefill_s = time.perf_counter() - started
    first = int(np.argmax(logits))
    return logits, first, prefill_s


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=resolve_artifact())
    parser.add_argument("--prompt", type=int, required=True, help="exact prompt tokens")
    parser.add_argument("--output", type=int, default=128, help="decode tokens incl. the first")
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--state", type=Path, required=True, help="KV state file")
    parser.add_argument("--mode", choices=("save", "restore", "verify"), required=True)
    parser.add_argument("--expect-gpu", default="8060S")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    import hipengine.core.hip as hip

    hip.get_hip_runtime()
    import ctypes

    library = ctypes.CDLL("libamdhip64.so")
    buffer = ctypes.create_string_buffer(256)
    library.hipDeviceGetName(buffer, len(buffer), 0)
    device = buffer.value.decode("utf-8", errors="replace")
    print(f"[kv_restore_bench] device0={device}", flush=True)
    if args.expect_gpu and args.expect_gpu not in device:
        print(
            f"ERROR: logical HIP device 0 is {device!r}, expected it to contain "
            f"{args.expect_gpu!r}", file=sys.stderr,
        )
        return 2

    if args.prompt + args.output > args.context:
        parser.error(
            f"prompt ({args.prompt}) + output ({args.output}) exceeds context {args.context}"
        )

    generator, runner = _resolve_runner(args)
    prompt_ids = exact_prompt_ids(generator.tokenize, args.prompt)
    identity = _identity(args, prompt_ids)

    result = {
        "schema": 1,
        "kind": "gemma4_kv_restore_bench",
        "mode": args.mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompt_tokens": args.prompt,
        "output_tokens": args.output,
        "context": args.context,
        "state": str(args.state),
        "identity": identity,
    }

    if args.mode == "save":
        logits, first, prefill_s = _fresh_phase(runner, generator, prompt_ids, args)
        started = time.perf_counter()
        runner.save_kv_state(args.state, identity=identity, logits=logits)
        save_s = time.perf_counter() - started
        ids, decode_s = _decode(runner, first, args.output)
        result.update(
            prefill_s=prefill_s, save_s=save_s, decode_s=decode_s,
            decode_tps=(args.output - 1) / decode_s if decode_s > 0 else None,
            generated_token_ids=ids,
        )
        print(
            f"[kv_restore_bench] prefill {prefill_s:.2f}s, save {save_s:.2f}s, "
            f"decode {decode_s:.2f}s ({(args.output - 1) / max(decode_s, 1e-9):.2f} tok/s)",
            flush=True,
        )
    elif args.mode == "verify":
        logits, first, prefill_s = _fresh_phase(runner, generator, prompt_ids, args)
        started = time.perf_counter()
        runner.save_kv_state(args.state, identity=identity, logits=logits)
        save_s = time.perf_counter() - started
        fresh_ids, fresh_decode_s = _decode(runner, first, args.output)

        started = time.perf_counter()
        saved_logits = runner.restore_kv_state(args.state, identity=identity)
        restore_s = time.perf_counter() - started
        assert saved_logits is not None and saved_logits.tobytes() == (
            logits.tobytes()
        ), "restored logits differ from the fresh prefill's last row"
        restored_first = int(saved_logits.argmax())
        assert restored_first == first, "restored first token differs"
        restored_ids, restored_decode_s = _decode(runner, restored_first, args.output)
        assert restored_ids == fresh_ids, (
            f"restored decode diverged at index "
            f"{next(i for i, (a, b) in enumerate(zip(restored_ids, fresh_ids)) if a != b)}"
        )
        result.update(
            prefill_s=prefill_s, save_s=save_s, fresh_decode_s=fresh_decode_s,
            restore_s=restore_s, restored_decode_s=restored_decode_s,
            generated_token_ids=fresh_ids,
            exact_ids_equal=True,
        )
        print(
            f"[kv_restore_bench] prefill {prefill_s:.2f}s, save {save_s:.2f}s, "
            f"restore {restore_s:.2f}s, decode fresh {fresh_decode_s:.2f}s / "
            f"restored {restored_decode_s:.2f}s, exact_ids_equal=True",
            flush=True,
        )
    else:
        started = time.perf_counter()
        saved_logits = runner.restore_kv_state(args.state, identity=identity)
        restore_s = time.perf_counter() - started
        if saved_logits is None:
            print("ERROR: the state file carries no logits", file=sys.stderr)
            return 1
        first = int(saved_logits.argmax())
        ids, decode_s = _decode(runner, first, args.output)
        result.update(
            restore_s=restore_s, decode_s=decode_s,
            decode_tps=(args.output - 1) / decode_s if decode_s > 0 else None,
            generated_token_ids=ids,
        )
        print(
            f"[kv_restore_bench] restore {restore_s:.2f}s, decode {decode_s:.2f}s "
            f"({(args.output - 1) / max(decode_s, 1e-9):.2f} tok/s)",
            flush=True,
        )

    runner.close()
    result["status"] = "ok"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(f"artifact={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
