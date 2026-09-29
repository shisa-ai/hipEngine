# DP-3: split-F16-operand tile cost, measured before any scope work

`DP-3` proposes a split-F16-operand variant: represent each f32 operand as an
f16 hi/lo pair and issue the three products `Ah*Bh + Ah*Bl + Al*Bh` per tile,
recovering roughly f32 precision at three WMMA issues where the promoted kernel
issues one.

The ledger's named next action is "microbenchmark against `dense_wide256` before
any scope work". This is that microbenchmark, and it rejects the scheme.

## What was measured

`WMMA_REP` was added as a template parameter **defaulting to 1**, so every
existing instantiation is unchanged, and `dense_wide256_split3` issues the same
product three times on identical operands.

That isolates the **WMMA issue cost** of the split scheme and nothing else. Its
result is the accumulator summed three times over - **not** a precision
improvement and **not** a candidate route. Only its timing is meaningful.

## Results, same host

`gfx1151`, real Q8_0 weights from unsloth Qwen3.8-Flash-Next `UD-Q4_K_XL`,
seeded standard-normal f32 activations, 1024 rows, 20 event-timed repetitions,
3 warmup launches, medians.

| role | `f32in` (1 WMMA) | `f16in`+cast | `split3` (3 WMMA) | ratio 3/1 |
|---|---:|---:|---:|---:|
| attn_qkv | 2.6130 | 1.4263 | 4.5880 | 1.756 |
| attn_q | 2.8149 | 1.7834 | 5.1905 | 1.844 |
| attn_gate | 1.5622 | 0.9733 | 2.7919 | 1.787 |
| ssm_out | 1.7431 | 1.1360 | 2.9045 | 1.666 |
| shared_gate | 0.2307 | 0.1938 | 0.4699 | 2.037 |

**Three products cost 1.787x, not 3x.** The marginal product costs 0.996 ms
against a 2.6007 ms baseline, so WMMA issue is ~38% of tile time and ~62% is
memory/LDS. Issue cost is **not** prohibitive, and the narrow `shared_gate`
geometry (2.037) is the only one that scales close to linearly.

## The number the decision turns on

`f16in` is the promoted route, so a split would be built on it - and it must
still beat the `f32in` route it would replace. Borrowing an `f16in` figure from
the DP-7 artifact would be a cross-machine comparison, so it was timed in the
same process, with the cast launch inside the timed region exactly as the
promoted route pays it.

| role | `f16in` x 1.787 | vs `f32in` |
|---|---:|---:|
| attn_qkv | 2.549 | **0.958** |
| attn_q | 3.288 | 1.168 |
| attn_gate | 1.739 | 1.113 |
| ssm_out | 1.893 | 1.086 |
| shared_gate | 0.331 | 1.711 |

**Median 1.113.** A split on the promoted route is *slower* than the `f32in`
route it would replace on four of five shapes, and by 71% on `shared_gate`.

## Verdict: rejected on cost

This is a **lower bound**. A real split additionally pays doubled LDS operand
bytes, the hi/lo splitting arithmetic and extra registers - none of which the
probe issues. So the true cost is above these numbers, and the median is already
above 1.0.

The one shape that came in under is `attn_qkv` at 0.958, and that is a 4.2%
margin against overheads that are certainly larger than 4.2%. It does not
support building the split.

**Reversal condition, stated so this is falsifiable rather than merely negative:**
a real split measuring below 1.0 on the shapes that carry the dense prefill time
would overturn this. The probe says the expected value of finding that is low,
which is what a pre-scope microbenchmark is for.

## Not claimed

- No correctness gate. `split3`'s numerics are the accumulator summed three
  times over, not a precision improvement.
- No production route, no promotion, no dispatch change. `split3` is registered
  only so the probe can reach it through the same wrapper path as the real
  variants.
- Kernel microbenchmark, not engine throughput.
- The `f32in` vs `f16in` gap here is the promotion rationale, not a new finding.
