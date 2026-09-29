# QA-5 re-rank: the named kernel does not run, and the running one is at 100% occupancy

QA-5 says: *"Full-attention context kernel, further - 25% occupancy, LDS-limited in the
launch carrying 81% of its time; two bit-exact levers already taken."* Its stated next
step is **"Re-read its share in X-3 before another lever."** This does that re-read, and
the answer is **no**.

## 1. The named kernel does not appear in the current default capture

X-3's attribution (`2026-09-17-qwen4exp-per-role-cost/role-analysis.json`, window
22,698.6 ms, 9,652 kernel rows) contains **zero** entries for
`qwen35_paged_full_attn_prefill_gqa_gate_fp16_kernel` - the kernel the 25% occupancy
figure describes. Its occupancy is real and I reproduced it (VGPR=256, 1,934
instructions, **8 waves/CU = 25.0%**), but it is not running at `code-p4096`.

## 2. The kernel that does carry the time runs at 100% occupancy

The whole full-attention family in that capture is **one kernel**:

| kernel | ms | share of window | rows |
|---|---:|---:|---:|
| `qwen35_paged_full_attn_decode_context_tensor_batch_kernel` | 801.566 | **3.531%** | 36 |

Measured occupancy for that kernel: **VGPR=18, SGPR=53, 554 instructions, 32 waves/CU =
100.0%**. There is no occupancy headroom to recover in the kernel that is actually
executing.

So QA-5's premise is stale on both counts: the LDS-limited 25% kernel is not in the hot
path, and the hot-path full-attention kernel is already at full occupancy.

## 3. Where the LDS limit *would* bind, if that kernel ran

The launch computes `shared = (max_context_len + threads + head_dim) * 4` with
`threads = 32` when `max_context_len <= 1024`, else `64`; `head_dim = 256` (read from the
model's `attention_key_length`, not assumed). Against a 64 KiB LDS per CU and the
measured 256-VGPR register limit (8 waves/CU):

| context | threads | LDS bytes | blocks/CU | waves/CU | occupancy | binding |
|---:|---:|---:|---:|---:|---:|---|
| 512 | 32 | 3,200 | 20 | 8 | 25.0% | registers |
| 1024 | 32 | 5,248 | 12 | 8 | 25.0% | registers |
| 2048 | 64 | 9,472 | 6 | 8 | 25.0% | registers |
| 4096 | 64 | 17,664 | 3 | 6 | **18.8%** | **LDS** |
| 8192 | 64 | 34,048 | 1 | 2 | **6.2%** | **LDS** |
| 16064+ | 64 | >65,536 | - | - | - | global score scratch |

**The binding resource changes with context length.** "25% occupancy, LDS-limited"
conflates two regimes: below the crossover registers bind (25%); above it LDS does
(18.8% at 4096, 6.2% at 8192). A single figure is correct for neither regime above 2048.

## 4. The 64 KiB overflow is already guarded - no bug

`hipengine/kernels/hip_gfx1100/attention/paged_attn_decode.hip:3796`:

```cpp
use_global_scores = global_score_workspace or (max_context_len + threads + head_dim) * 4 > 65536
```

This is exactly the threshold derived above (switching at context 16,064 for this shape),
and it matches the documented behaviour in `docs/KERNELS.md`: *"Native BF16-gated prefill
uses owned global score scratch when its context-sized shared allocation would exceed
64 KiB."* I initially read the >64 KiB row as a possible launch failure; checking the
guard before reporting it showed the case is handled by design.

## Verdict

**Another occupancy/LDS lever on this kernel is not justified.** The share is 3.531%,
the kernel named in the row does not run in the current default, and the kernel that does
is at 100% occupancy. QA-5 is closed as a stale premise.

## Limits

- One capture (`code-p4096`), one default, one shape. A different workload could put the
  prefill GQA gate kernel back in the hot path.
- The occupancy model is arithmetic from the launch code plus the measured VGPR count; no
  counter-based occupancy measurement (`rocprofv3` `OccupancyPercent`) was taken.
- Share is measured, not modelled; but "81% of its time" from the row refers to a
  different (prefill-only microbench) measurement and was **not** reproduced here.
- The full-attn family total (801.566 ms) is a sum over one kernel; it is not an
  end-to-end attribution.
