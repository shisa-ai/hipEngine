# DE-8: the 2.98 ms head is at its own family's read rate — no lever at c=1

Date: 2026-10-01
Row: `DE-8` (Head) in `docs/QWEN4EXP-OPTIMIZATION-LEDGER.md`
Class: **re-derivation from existing artifacts — no new run**
Source: `benchmarks/results/2026-09-29-qwen4exp-x4-decode-split/` (`attribution.json` + the role-marker join)
Host: AMD Radeon 8060S (integrated, **gfx1151**), model `unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL`,
production profile, case `code-p4096` at live count 4097, device-argmax boundary

## Question

DE-8 records the historical `lm_head` cost of 3.25 ms/token and asks for it to be priced:
*"Price GEMV versus argmax in `X-4`; only screen weight reuse or stable k=1 reduction if material."*
`X-4` has since priced it — the head is `linear:root.lm_head`, **2.980 ms/step, one kernel per step,
5.50% of decode kernel time and 6.04% of the 49.3 ms wall** — which makes it the second largest single
role in the whole decode split. This is the screen that follows.

## What the head reads

`hipengine/loading/qwen4_exp_mtp_gguf.py` maps `root.lm_head` to **`output.weight`**, which in this GGUF is

| tensor | type | shape | bytes |
| --- | --- | --- | --- |
| `output.weight` | **Q8_0** | 248320 × 2560 | **675,430,400** |

A dense GEMV at k=1 must read all of it once per step, so the head is a **675.4 MB read per token**, and
its 2.980 ms is **226.7 GB/s**. The launch geometry agrees that the read is not obviously redundant:
grid 3,973,120 = 248,320 rows × 16 blocks/row, workgroup 128, VGPR 48, LDS 512, scratch 0.

## The comparison that decides it

Four sibling decode GEMV families in the **same capture, on the same host, in the same kernel family**
(Q8_0 weights, one kernel per tensor, k=1), with their byte traffic computed from the same GGUF header:

| family | MB/step | ms/step | GB/s |
| --- | ---: | ---: | ---: |
| **`linear:root.lm_head`** | **675.4** | **2.980** | **226.7** |
| `linear:attn_qkv` (36 kernels) | 1,002.7 | 4.658 | 215.3 |
| `linear:attn_gate` (36) | 601.6 | 2.890 | 208.2 |
| `linear:ssm_out` (36) | 601.6 | 2.842 | 211.7 |
| `linear:attn_output` (12) | 200.5 | 0.947 | 211.8 |

Sibling range **208.2-215.3 GB/s**, median **211.7**. The head is at **1.071x the sibling median** and
**1.053x the fastest sibling**: it is the **fastest of the five**, not a laggard. At the sibling median
rate it would cost **3.190 ms** and at the fastest sibling rate **3.137 ms** — both *slower* than the
2.980 ms it actually takes.

## Conclusion

**The head is at the achievable read rate of its own kernel family, and there is no head-specific
inefficiency to fix.** The 2.98 ms is 675.4 MB of Q8_0 weights read once per token at a rate that no
sibling family in the same capture beats. DE-8's named screens therefore have no target:

* **Weight reuse** — there is nothing to reuse at c=1: the read is per-token and the weights are already
  read exactly once.
* **Stable k=1 reduction** — the reduction is not the cost; the traffic is. A hypothetical perfect
  reduction at the fastest sibling rate would still be slower than what ships.
* **The row's one real lever is forbidden by the row**: reading fewer bytes means pruning the vocabulary,
  and DE-8's own constraint is *"Preserve the full target vocabulary; target-head structure and final
  top-1 are distinct from draft vocabulary pruning."*
* **The remaining structural option is batching, not a c=1 kernel change**: 675.4 MB amortized over N
  rows is 675.4/N MB per token. That is `DE-10`'s lane (*"only if weight rereads dominate, screen one
  row-batched dense/expert owner"*), and it is where this row's cost actually lives.

**Row status: CLOSED — priced, screened, no lever within the row's constraints.** No performance claim,
no promotion, no default changed.

## Limits

**No new measurement was taken** — every figure comes from the 2026-09-29 `X-4` decode capture, whose
own limits apply (one host, one model, one quant, one profile, one case at one live count). The sibling
comparison bounds the head against **its own kernel family on this host**, which is what the screen
needs; it does **not** establish the hardware's true read roof, so it cannot show whether the *family as
a whole* has headroom against a faster possible rate — that is a different question belonging to the
dense-GEMV rows (`X-1`, `X-2`), and nothing here answers it. The `decode_boundary` family (0.508 ms/step)
contains the final logits and argmax work; it is **not** decomposed per head here, so the "GEMV versus
argmax" split is reported as 2.980 ms of GEMV against a 0.508 ms boundary family that also covers
non-head boundary work. The c=1 path only: nothing in this screen speaks to c>N, where the arithmetic
changes.

## Files

* `artifact.json` — the table above with byte counts, rates, ratios, and sources.
* Reused: `benchmarks/results/2026-09-29-qwen4exp-x4-decode-split/{attribution.json,rocprof/x4decode_marker_api_trace.csv}`.
