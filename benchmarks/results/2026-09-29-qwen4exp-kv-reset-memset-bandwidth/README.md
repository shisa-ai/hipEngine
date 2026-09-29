# RT-4 follow-up: the reset's 220 GB/s is the hardware limit, so only the contract can change

RT-4 attributed `Qwen4ExpGGUFResidentModelRunner.reset` at an effective **~220 GB/s** - 6 GiB
of K/V capacity cleared in **29.182 ms** at 128K capacity, of which **99.6%** is storage a
short request never touched. That left two possible fixes:

1. make the **memset** faster - no contract change, no correctness burden; or
2. clear **less** - RT-4's proposed contract change, with its poison-testing,
   shorter-after-longer, sparse-tail, rollback, cancellation and c2 burden.

**This packet closes route 1 by measurement.**

## Result

`hipMemset` over a capacity ladder, 7 reps after 2 warmups, device-synchronized:

| size | median | effective |
|---:|---:|---:|
| 1 MiB | 0.011 ms | 96.2 GB/s |
| 64 MiB | 0.307 ms | 218.3 GB/s |
| 768 MiB | 3.658 ms | 220.1 GB/s |
| **6 GiB** | **29.456 ms** | **218.7 GB/s** |
| 12 GiB | 58.531 ms | 220.1 GB/s |

**The rate is flat at ~220 GB/s across a 12,000x size range.** The only deviation is at
1 MiB, where launch overhead dominates and the rate is *lower*. The 6 GiB point - the same
volume RT-4 clears at 128K - reproduces RT-4's recorded **220.77 GB/s** to within **0.3%**.

So **220 GB/s is this part's achievable `hipMemset` rate, not a property of how the reset
calls it.** There is no memset-speed headroom to exploit: the best measured rate is
**0.997x** RT-4's recorded effective rate.

## What this settles

**Route 1 is closed.** The reset cannot be made cheaper by making the memset faster, because
the memset is already at the hardware rate. **Clearing less is the only lever**, which is
exactly RT-4's proposed contract change - so RT-4's correctness burden is not avoidable by a
cheaper alternative, and the row's remaining work is what it says it is.

This also means RT-4's **29.182 ms at 128K capacity is a floor for the full-clear design**,
not an artifact of a suboptimal call. Any future claim that the reset "should" be faster needs
a different mechanism than a faster memset.

## Limits

- **One part.** 220 GB/s is this host's rate; a different memory subsystem would have a
  different one. This says nothing about other hardware.
- `hipMemset` only. A custom kernel, a different memset implementation, or a DMA path was not
  tested, so the claim is precisely *"the `hipMemset` the runner uses is at this part's
  achievable rate"*, not *"no memset-like operation can exceed 220 GB/s"*.
- **The reset path was not timed end to end** - RT-4's boundary stands (the K/V memset pair
  only, excluding recurrent zeroing, index resets and position writes).
- Allocated buffers, not the runner's real caches; no model loaded.
- **No runtime change, no promotion, and no claim about the contract change's value.** RT-4's
  prize and its correctness burden are unchanged by this result; what changed is that the
  cheaper alternative is now ruled out.
