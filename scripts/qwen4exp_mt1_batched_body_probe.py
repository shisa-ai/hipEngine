#!/usr/bin/env python3
"""MT-1 batched verifier body vs serial c1 body on the real Qwen4Exp target.

Composes the rows>1 prefill layer owners into ``verify_target_block_batched_body``
and compares it, row by row, against ``verify_target_block_serial_exact`` (the
serial c1 decode oracle). This is the complete-pass measurement MT-1 asks for.

Usage:
    LD_LIBRARY_PATH=<rocm devel lib> PATH=<rocm bin>:$PATH \
      python3 scripts/qwen4exp_mt1_batched_body_probe.py --rows 6 --output <json>
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_MODEL = pathlib.Path(
    "/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL"
)


def _compose_batched_body(runner, tokens):
    """Compose the rows>1 prefill owners into a non-resetting verifier body.

    Mirrors the production verifier's state ownership: it does not reset, so the
    caller can wrap it in ``begin_device_transaction``. It uses the same rows>1
    layer owners the prefill chunk uses (QSA prefill / GDN / PLE) and scores all
    rows with one batched head.
    """

    import numpy as np

    from hipengine.core.memory import copy_device_to_host, host_array_ptr
    from hipengine.core.dtype import DType
    from hipengine.runtime.gguf_linear import (
        GGUF_ACTIVATION_F32,
        GGUF_OUTPUT_F32,
        launch_gguf_linear,
    )
    from hipengine.runtime.qwen4_exp_runner import run_qwen4_exp_gr_read

    cfg = runner.config
    count = len(tokens)
    output = runner.target_verify_output(count)
    last_residual_ptr, hidden_rows = runner._prefill_chunk(  # noqa: SLF001 - diagnostic
        tuple(tokens), capture_hidden_seeds=True
    )
    row_bytes = cfg.residual_width * DType.BF16.itemsize
    base_ptr = int(last_residual_ptr) - (count - 1) * row_bytes
    head = run_qwen4_exp_gr_read(
        base_ptr,
        runner.resident.weight("root.head_hc_norm").allocation("raw").tensor.ptr,
        runner.resident.weight("root.head_hc_down"),
        runner.resident.weight("root.head_hc_up"),
        None,
        output.head_scratch,
        rows=count,
        branches=cfg.residual_branch_count,
        hidden=cfg.hidden_size,
        low_rank=cfg.residual_low_rank,
        runtime=runner.runtime,
    )
    launch_gguf_linear(
        runner.resident.weight("root.lm_head"),
        head.mixed.ptr,
        output.logits_rows.ptr,
        count,
        cfg.hidden_size,
        cfg.vocab_size,
        activation_dtype=GGUF_ACTIVATION_F32,
        output_dtype=GGUF_OUTPUT_F32,
        runtime=runner.runtime,
    )
    logits = np.empty((count, cfg.vocab_size), dtype=np.float32)
    copy_device_to_host(
        host_array_ptr(logits), output.logits_rows, logits.nbytes, runtime=runner.runtime
    )
    assert hidden_rows is not None
    return np.argmax(logits, axis=1), logits, hidden_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=pathlib.Path, default=DEFAULT_MODEL)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument(
        "--widths",
        default="2,4,6,8",
        help="Comma-separated verifier widths to compare (each is a fresh reset).",
    )
    parser.add_argument("--output", type=pathlib.Path, default=None)
    args = parser.parse_args()

    import numpy as np

    os.environ.setdefault("HIPENGINE_HIP_ARCH", "gfx1151")

    from hipengine.generation.qwen4_exp_gguf import Qwen4ExpGGUFTextGenerator
    from hipengine.generation.qwen4_exp_profiles import (
        QWEN4_EXP_BACKEND,
        QWEN4_EXP_MODEL,
        QWEN4_EXP_QUANTS,
        register_qwen4_exp_gfx1151_profiles,
    )
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.loading.gguf import discover_gguf_files, load_gguf_index
    from hipengine.models.registry import resolve_model
    from hipengine.execution_profiles import ExecutionProfile, resolve_runtime_profile

    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()
    index = load_gguf_index(discover_gguf_files(args.model_root)[0])
    plugin = resolve_model(index.architecture or "")
    resolved = resolve_runtime_profile(
        model=QWEN4_EXP_MODEL,
        backend=QWEN4_EXP_BACKEND,
        quant=QWEN4_EXP_QUANTS[1],
        profile=ExecutionProfile("production"),
    )

    def factory():
        return Qwen4ExpGGUFTextGenerator(
            model_path=args.model_root,
            weight_index=index,
            model_plugin=plugin,
            backend="hip_gfx1151",
            max_sequence_length=256,
            prefill_chunk_size=512,
        )

    generator = resolved.construct_generator(factory)
    payload: dict = {
        "schema": 1,
        "probe": "qwen4exp_mt1_batched_verifier_body",
        "model": str(args.model_root),
        "widths": [],
    }
    try:
        runner = generator.runner
        for rows in [int(value) for value in str(args.widths).split(",") if value]:
            tokens = list(range(1, rows + 1))
            runner.reset()
            serial = runner.verify_target_block_serial_exact(
                tokens, capture_logits=True, capture_hidden_seeds=True
            )
            runner.reset()
            batched_tokens, batched_logits, batched_hidden = _compose_batched_body(
                runner, tokens
            )
            serial_hidden = np.asarray(serial.hidden_seeds, dtype=np.float32)
            batched_hidden = np.asarray(batched_hidden, dtype=np.float32)
            serial_logits = np.asarray(serial.logits, dtype=np.float32)
            batched_logits = np.asarray(batched_logits, dtype=np.float32)
            per_row_hidden = np.max(
                np.abs(serial_hidden - batched_hidden), axis=1
            )
            per_row_logits = np.max(
                np.abs(serial_logits - batched_logits), axis=1
            )
            payload["widths"].append(
                {
                    "rows": rows,
                    "serial_token_ids": list(serial.token_ids),
                    "batched_token_ids": [int(value) for value in batched_tokens],
                    "top1_agreement": float(
                        np.mean(
                            np.asarray(serial.token_ids) == np.asarray(batched_tokens)
                        )
                    ),
                    "hidden_bit_exact": bool(
                        np.array_equal(serial_hidden, batched_hidden)
                    ),
                    "hidden_max_abs_delta": float(
                        np.max(np.abs(serial_hidden - batched_hidden))
                    ),
                    "hidden_per_row_max_abs_delta": [
                        float(value) for value in per_row_hidden
                    ],
                    "logits_max_abs_delta": float(
                        np.max(np.abs(serial_logits - batched_logits))
                    ),
                    "logits_per_row_max_abs_delta": [
                        float(value) for value in per_row_logits
                    ],
                }
            )
        payload["status"] = "measured"
    except BaseException as exc:  # noqa: BLE001 - diagnostic probe
        payload["status"] = "error"
        payload["error"] = repr(exc)
        raise
    finally:
        generator.close()
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(text + "\n")
    print(text)
    return 0 if payload["status"] == "measured" else 1


if __name__ == "__main__":
    raise SystemExit(main())
