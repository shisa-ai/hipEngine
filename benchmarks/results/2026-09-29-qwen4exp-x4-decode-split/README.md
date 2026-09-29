# X-4: the current-default decode device split, by role

X-4's harness captured the decode transition and then said, in its own `decision.next`:

> "analyze marker-scoped kernel/API/role owners"

**That analysis had never been run**, and six `DE-*` rows are parked "after X-4" waiting for
a device split to rank against. This packet is that analysis.

## Method

`scripts/qwen4exp_context_decode_profile.py --profile --role-markers` captures one
`code-p4096` decode transition at live count 4097, with snapshot restore before each of three
repetitions, wrapped in `rocprofv3 --runtime-trace --kernel-trace --marker-trace
--kernel-rename`.

`scripts/qwen4exp_x4_decode_role_attribution.py` then attributes every decode kernel to its
semantic role with the join the established decoder profiler uses: step windows select the
measured transitions, each **HIP launch call** is joined to the **innermost** role range
containing it on the same host thread, and kernels map to roles by `correlation_id`.

### Graph-replayed kernels needed a second route, and it is exact

The correlation-id join reached **3,516** of 5,313 kernels and missed **1,797**. The misses
are not a mystery: **0 of the 1,797 unattributed correlation ids appear in the HIP launch
trace at all**, because this engine replays them from captured HIP graphs (192
`hipGraphLaunch` in the capture). A graph-replayed kernel has no launch API row to join.

The fix is not a heuristic. `rocprofv3 --kernel-rename` renames each kernel with its
enclosing roctx region, so those kernels arrive **already named**
`qwen4exp_role:moe:layers.N.expert_gate`. Reading the role back out of the name recovers an
identity **the profiler assigned**, and takes attribution to **5,313 of 5,313 - 100%**.

That is the reusable part of this packet: **in a graph-replaying engine, role attribution
must not rely on the correlation-id join alone.**

## Result: decode kernel time by family

**54.218 ms/step** of kernel time over three step windows; 1,771 kernels/step across 740
distinct roles.

| family | ms/step | share | kernels/step |
|---|---:|---:|---:|
| **`moe:expert_gate`** | **24.550** | **45.28%** | 583.0 |
| `linear:attn_qkv` | 4.658 | 8.59% | 36.0 |
| `linear:root.lm_head` | 2.980 | 5.50% | 1.0 |
| `linear:attn_gate` | 2.890 | 5.33% | 36.0 |
| `linear:ssm_out` | 2.842 | 5.24% | 36.0 |
| `qsa_decode:attn_q` | 2.621 | 4.83% | 120.0 |
| `gdn:attn_qkv` | 2.401 | 4.43% | 72.0 |
| `linear:attn_q` | 1.815 | 3.35% | 12.0 |
| `linear:hc_ffn_down` | 1.771 | 3.27% | 48.0 |
| `linear:hc_attn_down` | 1.768 | 3.26% | 48.0 |
| `linear:hc_attn_up` | 0.998 | 1.84% | 48.0 |
| `linear:hc_ffn_up` | 0.995 | 1.83% | 48.0 |
| `linear:attn_output` | 0.947 | 1.75% | 12.0 |
| `decode_boundary` | 0.508 | 0.94% | 151.0 |
| `gr_read:hc_ffn_down` | 0.475 | 0.88% | 144.0 |
| `gr_read:hc_attn_down` | 0.450 | 0.83% | 144.0 |

**The MoE expert gate is 45.28% of decode kernel time** - more than every other family
combined below it, and it is a per-expert GEMM family spread over 48 layers rather than one
large kernel. The individual roles are flat across layers: the largest single one is
`moe:layers.2.expert_gate` at 0.724 ms/step (1.33%), and the rest cluster at 0.5-0.6.

**`linear:root.lm_head` is 5.50% in a single kernel per step** - the second largest single
role in the whole split is one output projection.

## Decode is not kernel-bound

The unprofiled wall for the same transition is **median 153.0 ms** (mean 162.7, spread
1.30x over three repetitions, `token_exact` and `state_exact` both true). Against 54.2 ms of
kernel time that is **~35% kernel and ~65% host/launch overhead**, with 48 graph launches
carrying 1,771 kernels.

This is the number the `DE-5`/`DE-6` runtime rows were missing, and it says the runtime side
is larger than the device side for a decode step.

## Limits

- **Profiled capture.** Per-kernel times are diagnostic; the profiled arm's own wall is
  unreliable (spread **4.12x**, min 58.9 ms to max 242.7 ms), so the kernel/wall ratio uses
  the **unprofiled** wall. The unprofiled spread is still 1.30x, so treat ~35% as
  approximate.
- **One case** (`code-p4096`), **one live count** (4097), one model, one host, one execution
  profile, three repetitions.
- The wall is the measured transition only - the snapshot restore happens **before** each
  measured transition, per the harness protocol.
- Role attribution uses the innermost-window join plus the kernel-rename fallback. Kernels
  that were neither launched inside a role range nor renamed are reported as `unattributed`
  rather than assigned; there are **none** here, but the fallback is only as good as the
  roctx coverage.
- **No performance claim and no promotion.** No default changed.
