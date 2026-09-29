# MR-1 priced: the reorder-free share is 0.0145%, 114x below the rejected lever

MR-1 ("Router logits producer, 617.73 ms, reduction-structure-bound") records that the
warp-per-token reduction wins 2.16x and +1.65% end to end but **fails top-1** at the
amplification node. Its stated next step is:

> First try structures preserving the existing reduction: fold the shared-expert gate row
> into the router GEMM (gufo computes both in one GEMM), or remove a pass. **Price the
> reorder-free share first.**

This prices it. **The reorder-free share is 0.0145% of the window.**

## What the "shared-expert gate row" actually is

Read from the GGUF, not assumed:

| tensor | shape | type | what it is |
|---|---|---|---|
| `blk.N.ffn_gate_inp.weight` | (512, 2560) | F32 | the router GEMM |
| `blk.N.ffn_gate_inp_shexp.weight` | **(2560,)** | F32 | **a vector - one output row** |
| `blk.N.ffn_gate_shexp.weight` | (640, 2560) | Q8_0 | the shared expert's own gate projection (a different tensor) |

The "row" MR-1 wants folded is a **1x2560 vector**, so folding it produces **one extra
output row** in the router GEMM.

## The price

From X-3's attribution (window 22,698.6 ms):

| item | kernel | ms | share |
|---|---|---:|---:|
| router logits producer | `qwen35_router_logits_f32_token_tile_dense_exact_kernel` | 617.805 | 2.722% |
| **the shared-expert gate row** | `dense_gemv_out_kernel` | **3.286** | **0.0145%** |

The 617.805 ms matches MR-1's stated 617.73 ms, and the 3.286 ms is the entire
`linear:layers.*.shared_expert_gate` role across all 48 layers.

**So folding the row into the router GEMM can save at most 3.286 ms = 0.0145%**, and that
is an **over-estimate**: the multiply-accumulate still happens, just inside the router's
kernel. The genuine saving is the launch and one activation read, which is smaller still.

## Verdict

The reorder-free share is **114x smaller** than the rejected reordered lever's 1.65%. It
cannot recover that win. MR-1's pricing step is answered: **do not build the reorder-free
router restructuring on the expectation of a meaningful end-to-end gain.**

This does not re-open the rejected warp reduction. MR-1 already records that only the
tested warp reduction is rejected and that other reordered implementations would need
their own production gate; that remains true. What is now closed is the *reorder-free*
branch, which MR-1 asked to price first precisely so this could be decided cheaply.

## Limits

- **Attribution, not an experiment.** No kernel was written or run; these are measured
  shares from an existing capture plus tensor shapes from the GGUF.
- One capture (`code-p4096`), one model, one default. A different workload or shape could
  change the share.
- The 3.286 ms is the *whole* role's time; the foldable part is at most that.
- "Remove a pass" was not separately priced. If it means removing a launch, the launch
  cost is on the order of microseconds per layer against a 22.7 s window.
