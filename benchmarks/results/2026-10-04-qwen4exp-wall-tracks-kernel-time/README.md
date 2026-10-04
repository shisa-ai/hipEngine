# The decode wall tracks kernel time 1:1, and one graph launch costs 0.0004 ms

Two results that redirect the decode side of the campaign. **No engine change and
no default changed.**

## 1. A kernel-time saving reproduces end-to-end in the wall

`DE-3`'s shape-gated narrow-projection block size is worth a measured
**-1.416 ms/step of decode kernel time** (its own per-kernel join, narrow family
24.005 -> 22.590). Its own row could not claim an end-to-end number. This
measures one.

Interleaved A/B/A/B at `code-p512`, chunk 1024, one process per arm, 1 warmup +
3 repetitions, production profile, warm PLE, BF16 KV:

| `HIPENGINE_GGUF_PACK8_GEMV_WIDE` | decode tok/s | wall ms/step |
| ---: | ---: | ---: |
| 1 (default) | 20.983 (20.981 / 20.986) | 47.657 |
| 0 (lever off) | 20.325 (20.322 / 20.327) | 49.201 |

**Turning the lever off costs 1.545 ms/step of wall time, against the 1.416
ms/step of kernel time it was predicted to cost - 109% of the prediction, well
inside this design's resolution.** Every arm is separated and the within-arm
spread is 0.02%.

**This refutes `DE-11`'s "a host-only or device-only change is capped at roughly
half its nominal saving".** The wall behaves as `kernel_sum + remainder` with a
fixed remainder, not as `max(host, device)`. **A millisecond removed from the
decode kernels is a millisecond removed from the decode wall.** Kernel
optimization is therefore a direct decode lever, which is the opposite of the
conclusion the decode-attribution thread had been converging on.

**What this does not resolve.** The A/B is consistent with `DE-11`'s kernel sum
(`46.19` at p512, remainder `1.46`) and with the correction in worklog
`20261004T100141` (kernel `37.53`, remainder `10.13`) - both predict the same
lever-off wall to within 0.13 ms, because the two readings differ only in the
assumed kernel sum, not in the delta. **A same-session kernel sum is still
required to settle the remainder's magnitude.**

## 2. One graph launch costs 0.0004 ms

The decode loop calls `_run_decode_layer_graphed` once per layer, so a step
issues **48 graph launches**, and 48 launches at ~0.2 ms would have reproduced
`DE-11`'s remainder exactly. Measured with
`scripts/qwen4exp_graph_launch_cost_probe.py` (N=2000 per arm, 5 runs, one sync
per run, `HipRuntime.load()`, nonblocking stream):

| arm | median ms/launch |
| --- | ---: |
| empty graph launch | **0.0004** |
| one-memset graph launch | 0.0037 |
| one-memset eager | 0.0021 |

**48 launches are 0.018 ms/step; the graph wrapper's own cost over eager is
0.0016 ms, or 0.079 ms/step at 48 launches.** Both are two to three orders of
magnitude below what the hypothesis needed. Per-launch transition cost is not the
remainder, and `DE-6`'s PM4 / retained command-list row is not the explanation
through that path.

**Limit:** this bounds the **launch** cost, not intra-graph scheduling. A real
per-layer MoE graph holds ~68 kernels.

## Artifacts

| file | what |
| --- | --- |
| `pass{1,2}-wide{0,1}.json`, `pass3-wide1.json` | the five interleaved arms |

## Limits

One host (gfx1151 Strix Halo, Radeon 8060S, not the gfx1100 default hardware),
one model (`unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL`, UD-Q4_K_XL), one profile,
one case (`code-p512`), one chunk size (1024), two arms, 3 repetitions each.
`HIPENGINE_GGUF_PACK8_GEMV_WIDE` is a T2 arithmetic change that has already passed
the production envelope (`DE-3`); this measures its speed, not its numerics. The
graph probe measures an empty and a one-op graph, not the engine's own captures.
