"""One-off check: the runtime's paired head against the serial head on the same rows.

The kernel gate proves `dense_gemv_bf16_f32_out_rowtile2` is bit-identical per row, and
the matched timing run shows the trajectory digest is unchanged. This checks the runtime
wiring itself: it loads the AR, drives a short two-branch prefill, asks for both
branches' logits through the paired path, then recomputes each branch's logits with the
single-row kernel on the same normalized row and requires exact equality.

    python3 scripts/yue2_ar_paired_head_check.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hipengine.core.memory import (  # noqa: E402
    copy_device_to_host,
    host_array_ptr,
)
from hipengine.kernels.hip_gfx1100.linear import dense_gemv  # noqa: E402
from hipengine.loading.yue2 import load_yue2_weights  # noqa: E402
from hipengine.runtime.yue2_ar import Yue2ArRuntime  # noqa: E402

POSITIVE = [9707, 1879, 11, 220, 16, 13, 220, 18]
NEGATIVE = [9707, 1879, 11, 220, 16, 13]


def main() -> int:
    import glob
    import os

    model_dir = os.environ.get("YUE2_MODEL_DIR") or sorted(
        glob.glob(str(Path.home() / ".cache/huggingface/hub/models--m-a-p--YuE2-3B/snapshots/*"))
    )[0]
    started = time.perf_counter()
    weights = load_yue2_weights(model_dir)
    runtime = Yue2ArRuntime(weights, branches=2, max_context=2056)
    print(f"loaded in {time.perf_counter() - started:.1f} s")

    for branch, tokens in ((0, POSITIVE), (1, NEGATIVE)):
        rows = np.stack([runtime.embed_row(token) for token in tokens])
        runtime.prefill_host_rows(rows, branch=branch)

    paired = [runtime.logits(branch, as_bf16=False) for branch in range(2)]

    vocab = runtime.spec.vocab_size
    hidden = runtime.spec.hidden_size
    worst = 0.0
    for branch in range(2):
        runtime.kernels.vv_rmsnorm_bf16(
            runtime._hidden[branch].ptr, runtime.final_ln.ptr, runtime._normed.ptr, 1, hidden,
            runtime.spec.rms_norm_eps, library=runtime.library, runtime=runtime.runtime,
        )
        dense_gemv.dense_gemv_bf16_f32_out(
            runtime._normed.ptr, runtime.lm_head.ptr, runtime._logits_f32.ptr, 1, hidden, vocab,
        )
        serial = np.empty(vocab, dtype=np.float32)
        copy_device_to_host(host_array_ptr(serial), runtime._logits_f32, vocab * 4)
        diff = float(np.abs(serial - paired[branch]).max())
        worst = max(worst, diff)
        print(f"branch {branch}: max abs difference {diff}, "
              f"argmax paired {int(paired[branch].argmax())} serial {int(serial.argmax())}, "
              f"rows differ between branches: {not np.array_equal(paired[0], paired[1])}")

    print()
    if worst == 0.0:
        print("PASS: the paired head is bit-identical to the serial head on both branches")
    else:
        print(f"FAIL: paired head differs from the serial head by {worst}")
    runtime.close()
    print(json.dumps({"worst_max_abs": worst, "exact": worst == 0.0}))
    return 0 if worst == 0.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
