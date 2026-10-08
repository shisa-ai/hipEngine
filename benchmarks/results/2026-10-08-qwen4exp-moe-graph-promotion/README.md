# Per-layer MoE decode graph validated against the eager path

This packet records the first complete-engine A/B of the existing per-layer MoE
decode graph (`MoeGraphCache`), the gfx1151 production default since its
introduction (capability `QWEN4_EXP_MOE_GRAPH=True`). One model residency, all
12 canonical cases, production compact-capture path, balanced arm order per
repetition; the before-arm forces the eager fallback through the cache's own
enabled flag, the after-arm captures and replays. Engagement is verified from
the cache's own stats (48 captures, 18,432 replays per case-arm = 48 layers x
128 steps x 3 repetitions; zero rejects), and the cache's self-validating
parity guarantees the replay is bit-identical to eager.

## Result

| Metric | Eager | Graph replay | Ratio |
| --- | ---: | ---: | ---: |
| Prefill | 340.784 tok/s | 340.785 tok/s | **1.0000x** |
| Decode | 16.328 tok/s | **16.846 tok/s** | **1.0317x** |
| Aggregate client wall | 480.532 s | **471.854 s** | **0.9819x** |

All 12 trajectories within-mode deterministic and cross-mode exact; zero
allocation leaks after close. Per-case decode deltas: +1.4% to +6.8% on nine
cases, -0.5% on two p512 cases and -3.4% on one p4096 case (within the
same-residency instrument's arm noise; the other three p4096 cases are
+2.3/+5.5/+6.8%).

## Interpretation

Because the graph is already the production default, every canonical rate
this campaign has measured includes this benefit; the A/B is the missing
qualification evidence for that default, not a promotion. Two model
corrections follow for the decode-depth campaign:

- The earlier rocprof stall decomposition describes the **eager fallback path**
  (HIP graph capture under the profiler falls back; the cache rejects nothing,
  it simply runs eager). Production decode is ~3.2% better than that traced
  model at the aggregate.
- The remaining p4096 stall mass therefore sits **outside** the MoE graph
  envelope: the dense projection chain (not yet graph-captured), the attention
  and GR sections, and the host preparation before each per-layer graph
  launch. The committed segment-graph plan targets exactly those seams.

## Reproduction

```bash
export ROCMENV=/home/lhl/miniforge3/envs/therock10-staging-20260828
export ROCMLIB=$ROCMENV/lib/python3.12/site-packages/_rocm_sdk_devel
export PATH="$ROCMENV/bin:$ROCMLIB/bin:$PATH"
export LD_LIBRARY_PATH="$ROCMLIB/lib:$ROCMLIB/lib64:$ROCMLIB/lib/llvm/lib"
export HIPENGINE_HIP_ARCH=gfx1151 PYTHONPATH=/home/lhl/hipEngine
uv run python -u scripts/qwen4exp_moe_graph_engine_ab.py \
  --model-root /home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --compiler-version-file /tmp/hipcc-version-qwen4exp.txt \
  --output benchmarks/results/2026-10-08-qwen4exp-moe-graph-promotion/engine-ab.json
```

Machine-readable record: [engine-ab.json](engine-ab.json).
