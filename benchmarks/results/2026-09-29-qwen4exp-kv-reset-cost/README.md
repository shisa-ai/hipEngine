# RT-4: attributing the capacity-sensitive KV reset cost

**The reset is linear in capacity at ~220 GB/s, and at 128K capacity it clears
6 GiB in 29.182 ms - of which 99.6% is storage a 512-token request never
touched.**

RT-4 says "measure before changing clearing contract". This is that measurement.

## What is being measured

`Qwen4ExpGGUFResidentModelRunner.reset` zeroes the **full** key and value caches
for every QSA attention state:

```python
self.runtime.memset(attention.key_cache.ptr, 0, attention.key_cache.nbytes)
self.runtime.memset(attention.value_cache.ptr, 0, attention.value_cache.nbytes)
```

The caches are sized by **capacity**, not by what a request wrote, so this is
`O(capacity)`. For Qwen3.8-Flash-Next UD-Q4_K_XL the per-token footprint is
**24,576 bytes** (12 QSA layers x 2 kv heads x 256 key length x 2 (key+value) x
2 bytes bf16), read from the GGUF metadata rather than assumed.

`scripts/qwen4exp_kv_reset_cost.py` allocates only the KV-sized buffers and times
exactly that memset pair over a capacity ladder. It does not load the model.

## Result

| capacity | cleared per reset | median | effective | wasted at 512 written |
|---:|---:|---:|---:|---:|
| 1,024 | 48.0 MiB | 0.238 ms | 211.6 GB/s | 50.0% |
| 4,096 | 192.0 MiB | 0.885 ms | 227.6 GB/s | 87.5% |
| 16,384 | 768.0 MiB | 3.709 ms | 217.1 GB/s | 96.9% |
| 65,536 | 3,072.0 MiB | 14.619 ms | 220.3 GB/s | 99.2% |
| 131,072 | 6,144.0 MiB | **29.182 ms** | 220.8 GB/s | **99.6%** |

Median of 7 repeats after 2 warmups, each timed with a device synchronize on both
sides.

## The two findings

**1. It is already bandwidth-bound, so the memset itself has no headroom.**
Effective bandwidth is flat at 211-228 GB/s across a 128x range of sizes. That is
~86% of this APU's ~256 GB/s LPDDR5X peak, and it is consistent enough across the
ladder to be a memory-bandwidth limit rather than an overhead artifact. No
tuning of the memset makes it meaningfully faster; **the only lever is clearing
fewer bytes.**

**2. The waste is nearly total at realistic capacities.** A 512-token request at
128K capacity pays 29.182 ms to clear 6 GiB, of which it wrote **24 MiB
(0.4%)**. Even at 16K capacity the waste is 96.9%.

## What this does and does not establish

- **Establishes:** the reset's dominant cost is the capacity-sized K/V memset; its
  size, scaling, bandwidth and waste fraction are now measured rather than
  estimated. It is a real, quantified cost at long-context capacities.
- **Does not establish a saving.** This measures the current contract; it makes no
  claim about what an alternative contract would cost. Clearing only previously
  written ranges, or a validated overwrite-before-read, must still satisfy RT-4's
  conditions: exact visibility/ownership, preserved recurrent initialization,
  poison-tested unused storage, and correct behavior for shorter-after-longer
  requests, sparse tails, rollback, cancellation and c2.
- **Not an end-to-end claim.** `performance_claim: false`. The boundary excludes
  the recurrent state zero, the index resets and the position writes, which are
  the other terms in `reset()`.

## Note on the historical figure

The ledger records "native-capacity short prefill historically lost 0.24-0.74%".
That is a *complete-request* percentage at whatever capacity that test used; it
is not comparable to a millisecond figure at a different capacity, and this
packet does not restate it as one. The ladder above is the shape of the curve.

## Limits

- One host (gfx1151 APU, unified memory).
- One model's per-token KV footprint (24,576 B/token); the byte figures scale
  with it.
- The memset only; not a full `reset()`.
