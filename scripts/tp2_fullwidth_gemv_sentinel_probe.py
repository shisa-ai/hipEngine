"""Reproducible sentinel probe: does the full-width decode GEMV write outputs?

The TP1 full-width MLP on device 1 (RX 7900 XTX) returns all-NaN logits. A
layer-0 probe showed ``ffn_gate``/``ffn_up`` contain stale non-finite values
after ``launch_gguf_linear(..., use_gemv_decode=True)`` at
``out_features=17408``. This probe fills both output buffers with a distinctive
finite bf16 sentinel (``0x7BFF``), runs one teacher-forced token, and counts how
many sentinel words remain: a fully written output leaves none, a no-op leaves
all of them. It also reports whether the final logits are finite.

This is a diagnostic, not a correctness gate. Run it on each device:

    python scripts/tp2_fullwidth_gemv_sentinel_probe.py --device 0
    python scripts/tp2_fullwidth_gemv_sentinel_probe.py --device 1

The ``os._exit`` in ``__main__`` is diagnostic-only: it avoids a device-1 HIP
runtime teardown hang and must never be read as evidence that teardown is
qualified.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

MODEL = "/models/gguf/Qwen3.8-27B-Q4_K_M.gguf"
#: A distinctive finite bf16 bit pattern (0x7BFF is exponent 0xF7, mantissa 0x7F:
#: a large finite value, not the FP16 65504 that the same 16-bit word encodes).
#: The probe compares raw uint16 words, so the numeric value is irrelevant; it
#: only has to be unlikely to appear as a real GEMV output.
SENTINEL_BF16 = 0x7BFF


def classify(device: int, gate_unwritten: int, up_unwritten: int, total: int) -> str:
    """One-line verdict for a probe result (pure, unit-tested)."""

    if gate_unwritten == 0 and up_unwritten == 0:
        return f"device {device}: WRITTEN (gate 0/{total}, up 0/{total})"
    if gate_unwritten == total and up_unwritten == total:
        return f"device {device}: NO-OP (gate {total}/{total}, up {total}/{total} unwritten)"
    return (
        f"device {device}: PARTIAL (gate {gate_unwritten}/{total}, "
        f"up {up_unwritten}/{total} unwritten)"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--token", type=int, default=248045)
    args = parser.parse_args(argv)

    from hipengine.core.device import scoped_current_device
    from hipengine.core.runtime import MemcpyKind
    from hipengine.distributed.tp2_generate import MlpTP2GenerationSession

    device = int(args.device)
    session = MlpTP2GenerationSession(MODEL, devices=(device,), mode="tp1", schedule="eager")
    ffn = int(session._runners[device].ffn_size)
    orig = session._local_mlp

    def bf16(words: np.ndarray) -> np.ndarray:
        return (words.astype(np.uint32) << 16).view(np.float32)

    def read(ptr: int, count: int) -> np.ndarray:
        out = np.empty(count, dtype=np.uint16)
        with scoped_current_device(session.runtime, device):
            session.runtime.memcpy(
                out.ctypes.data, ptr, count * 2, MemcpyKind.DEVICE_TO_HOST
            )
        return out

    def wrapped(dev: int, layer_id: int) -> int:
        buffers = session._step_buffers[dev]
        if layer_id == args.layer:
            sentinel = np.full(ffn, SENTINEL_BF16, dtype=np.uint16)
            for slot in ("mlp_gate", "mlp_up"):
                with scoped_current_device(session.runtime, device):
                    session.runtime.memcpy(
                        buffers[slot].ptr,
                        sentinel.ctypes.data,
                        ffn * 2,
                        MemcpyKind.HOST_TO_DEVICE,
                    )
        result = orig(dev, layer_id)
        if layer_id == args.layer:
            gate = read(buffers["mlp_gate"].ptr, ffn)
            up = read(buffers["mlp_up"].ptr, ffn)
            wrapped.gate_unwritten = int((gate == SENTINEL_BF16).sum())
            wrapped.up_unwritten = int((up == SENTINEL_BF16).sum())
        return result

    wrapped.gate_unwritten = -1
    wrapped.up_unwritten = -1
    session._local_mlp = wrapped
    logits = None
    try:
        logits = np.asarray(session.teacher_forced_logits((args.token,)), dtype=np.float32)
    finally:
        session.close()
    finite = int(np.isfinite(logits).sum()) if logits is not None else 0
    total = logits.size if logits is not None else 0
    print(classify(device, wrapped.gate_unwritten, wrapped.up_unwritten, ffn), flush=True)
    print(f"device {device}: logits finite={finite}/{total}", flush=True)
    return 0


if __name__ == "__main__":
    # Device-1 teardown can hang after a failed forward; exit deterministically.
    _code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_code)
