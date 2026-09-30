# RT-4 — bounded KV reset at native capacity

**Status:** implemented, unit-tested GREEN, measured; **default-off** pending the
exactness gate.  Supersedes the `29.182 ms / 6,144 MiB` figures in the RT-4 row.

**Date:** 2026-09-30
**Host:** local workstation, AMD Radeon Pro W7900 (gfx1100/RDNA3), ROCm from the
`vllm` conda env's `_rocm_sdk_devel` tree.
**Model:** `unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL` (UD-Q4_K_XL).
**Case:** `code-p1024` — 1024-token prompt, 64 decode transitions.

## What was measured

`scripts/qwen4exp_de5_reset_split.py --capacity 131072`, run twice with only
`HIPENGINE_QWEN4_EXP_BOUNDED_KV_RESET` differing.

| | flag=0 (full clear) | flag=1 (bounded) |
| --- | --- | --- |
| warmup `reset_ms` | 15.689 | **1.133** |
| rep1 `reset_ms` | 15.519 | **1.741** |
| `state_zero` | 0.549 / 0.555 | 0.550 / 0.568 |
| `index_resets_total` | 0.017 / 0.041 | 0.017 / 0.041 |
| decode last-eighth | 51.800 / 49.662 | 51.983 / 49.734 |

**The reset falls from 15.5 ms to 1.13-1.74 ms — a mean 14.17 ms saved per
request, 11.0x.** What remains is `state_zero` (0.55 ms) plus the index resets;
the KV clearing itself becomes effectively free because the request wrote
1024 of 131072 positions.

**Decode is unaffected in both regimes** (first-eighth 192.26/143.87 vs
194.99/143.81; last-eighth 51.80/49.66 vs 51.98/49.73). This is the expected
signature of a change that only skips zeroing bytes that are already zero.

## The traffic model is validated

The probe reports `kv_bytes_total = 3,221,225,472 B` (3,072 MiB) across 12
attention states, 12 key + 12 value caches of 134,217,728 B each, and predicts
`kv_memset_ms_at_220GBs = 14.64 ms`. The measured full clear is 15.17-15.33 ms,
**within 5% of prediction**, which independently confirms both the byte total
and RT-4's 220 GB/s memset rate.

## Correction to the RT-4 row

The row records **6,144 MiB / 29.182 ms** at 128K. Both are **2x too large**.

The row's own footprint formula — `12 QSA layers x 2 kv heads x 256 key length x
2 x 2B bf16` = **24,576 B/token** — gives `131072 x 24,576 = 3,221,225,472 B =
3,072 MiB`, not 6,144 MiB. This probe reaches the same 3,072 MiB from the
allocated buffers (`12 x 128 MiB x 2`) with no reference to the row, and measures
15.5 ms against the row's 29.182 ms. Halving the row's time gives 14.59 ms, which
matches this run's 14.64 ms prediction.

**The row's per-token footprint, bandwidth and waste-fraction conclusions are
unaffected** — 99.6% waste at 128K for a 512-token request holds. Only the
absolute byte and millisecond figures were doubled.

## The change

`reset()` currently memsets the full allocated capacity of every K/V cache. The
bound lives on the attention state:

- `written_cache_bytes()` returns `round_up(position, block_size) x bytes_per_position`,
  computed from the host-resident `position_host` — no device read.
- `clear_written_cache()` memsets only `[0, bound)`.

**Soundness:** `allocate()` memsets the whole cache, and KV writes only ever land
below `position` (the block table is the identity `arange(blocks)` and is never
mutated), so every byte at or above the bound is already zero. Clearing it is
pure waste. This is why the change is classified **T1 (exact)** rather than a
reassociation.

## Unit evidence

`tests/test_gpu_qwen4exp_kv_reset_bounded.py` — 5 tests, written RED first
(`AttributeError` before implementation), now GREEN.

The two load-bearing tests are:

- **`test_bounded_clear_zeroes_the_written_range_and_leaves_the_rest`** — poisons
  the buffer, then proves the clear is bounded rather than a full memset wearing
  a new name.
- **`test_shorter_request_after_longer_sees_no_stale_bytes`** — the classic
  hazard: a long request writes deep, a shorter one follows, and the shorter one
  must not observe stale K/V. This is the case the RT-4 row names explicitly.

Also covered: block rounding, and equality with a full clear over the written
range.

**A real API trap was found while writing these tests and is worth remembering:**
`copy_host_to_device(buffer, host_ptr)` defaults `nbytes` to **the destination
buffer's size**, not the host array's. Copying a short marker without an explicit
size read past the end of the NumPy array; the pointer-shaped garbage in the
first failure output was adjacent heap. `copy_host_to_device` and `memset` are
also asynchronous, so tests must synchronize explicitly.

## Not yet done

- **The exactness gate.** `scripts/qwen4exp_layer2_profile_gate.py --candidate
  bounded_kv_reset` (T1) must show identical strict/candidate state and logits
  hashes. The gate refuses to run on a dirty tree, so it follows this commit.
- **Rollback, cancellation and c2** behavior named by the row is not exercised
  here. The bound is derived from `position`, which `reset()` sets to 0, so a
  cancelled or rolled-back request re-derives a smaller bound on its next
  prefill — but that is an argument, not a measurement.
