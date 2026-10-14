# Qwen4Exp prefill: QSA selection, attention gate and residual write fusion (2026-10-14)

`qwen4_exp_gguf` / `hip_gfx1151` / `gguf_ud_q4_k_xl`, `production` profile,
Radeon 8060S / gfx1151, host `gfx1151` (machine
`55ea6c509d0b49eea8de7094a1023668`), BF16 KV, chunk 4096, greedy decode with
compact capture, warm PLE cache, tracked-clean `2dd109247`, measured
2026-10-14 UTC. Canonical 12-case protocol and the 1024 / 8192 / 16384
fixture, one warmup and three repetitions each.

| Prompt tokens | Prefill tok/s before | Prefill tok/s now | Decode tok/s before | Decode tok/s now |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 1037.2 | 1036.5 | 23.55 | 23.66 |
| 1024 | 1314.2 | **1323.8** | 22.60 | **22.69** |
| 4096 | 1615.5 | **1646.3** | 23.12 | **23.23** |
| 8192 | 1570.4 | **1621.5** | 23.02 | **23.18** |
| 16384 | 1516.7 | **1576.1** | 22.91 | **23.04** |

"Before" is `../2026-10-14-qwen4exp-final/` on the same host and protocol.
Every change is bit-identical: `scripts/qwen4exp_rate_regression_check.py
--require-identical-output` passes on both fixtures with identical output
tokens in all 24 cases (`regression-check.txt`). The numerical and
long-prompt gates of the plan therefore carry over unchanged
(`../2026-10-14-qwen4exp-decode-batch/gate-c.json`,
`../2026-10-14-qwen4exp-long-context-gate/`).

Against gufo `v0.12.0` on the same host (2026-10-12): 0.95x / 0.89x /
0.95x prefill at 512 / 1024 / 4096 tokens.

## What changed

Same-process numbers are prefill tok/s with the change off and on in one
process (decode tok/s for the PLE change).

| Change | Same-process |
| --- | ---: |
| QSA prefill rows score only the blocks they can see, from a query tile staged once with conflict-free key rows; with the WMMA attention the top-k writes the selection mask directly | `code-p4096` 1612.6 -> 1629.4 |
| Attention gate read in place from the q projection, with the F16 image of the gated context written for the output projection | `code-p4096` 1631.4 -> 1634.5 |
| Residual write fused into the following hyper-connection norm (after the token mixer and, across the layer boundary, after the MoE), chunks of 1024 rows or more | `code-p4096` 1634.4 -> 1646.3 |
| Decode PLE row staged on the host while layer 0 runs, copied asynchronously from the pinned ring | decode 23.52 -> 23.67 |

Not kept: a split norm/RoPE kernel that computes the rotary table once per
row instead of once per head (1628.0 against 1630.2, inside noise; the kernel
is bound by memory traffic, not by the transcendentals).

Kernel time per `code-p4096` prefill before these changes (`rocprofv3`,
production, HEAD `492186123`): 2493 ms against gufo's 2373. The largest
families left against gufo: the GDN conv (54.8 ms; gufo folds the conv into
the input-projection GEMM epilogue and runs an 11.4 ms boundary kernel) and
the GDN norm/gate epilogue (57.7 against 44.4 ms).

## Files

- `hipengine-production.json`, `hipengine-production-p1024-p8192-p16384.json`:
  the new regression baselines (`benchmarks/HARNESSES.md`).
- `regression-check.txt`.
