# RT-1 blocker (a): short and long requests on the SAME prepared owners

`docs/REFACTOR.md` names four requirements for enabling automatic prefill-workspace
selection. Three were already cleared by measurement. The fourth was a measurement rather
than a qualification:

> "measure short/long requests on the same prepared owners"

Every prior measurement, including RT-1's and RT-3's, used **one process per arm**, so the
selection path itself had never been exercised end-to-end on a prepared pair. This packet
does that, and the result removes the requirement rather than satisfying it.

## Method

One runner, two prepared owners, one process. Base owner **4096** (it sets the staging
capacity), extra owner **1024** (the selector may only choose owners at or below the base, so
the small owner must be the extra). The selector is configured through
`_configure_prefill_workspace_selection([base, extra], reserve_bytes=4 GiB)`, and the owner
that served each request is **recorded by a probe inside `_prefill_chunked_impl`** - the
function the selector calls *after* swapping the workspace fields - so the served owner is
observed, not inferred from timing.

Then the same short prompt is re-measured with the selector turned **off**, which forces it
onto the large base owner. That A/B is the decisive comparison: it isolates exactly what the
shape policy buys a short request, on the same prepared owners, in the same process.

Host `gfx1151` (`55ea6c509d0b49eea8de7094a1023668`), UD-Q4_K_XL, production profile,
`code-p512` / `code-p4096`, `max_sequence_length` 4352, 8 decode steps, 5 reps per arm after
3 warmups.

## Result

The selection path works end to end, and it selects correctly for both shapes:

| request | tokens | owner served | median | spread | logits |
|---|---:|---:|---:|---:|---|
| short, selector **on** | 512 | **1024** | 3.2068 s | 1.99% | `fefecd92...` |
| short, selector **off** | 512 | **4096** | 3.1730 s | 0.73% | `fefecd92...` |
| long | 4096 | **4096** | 15.9833 s | 0.47% | `2cb5f393...` |

**The shape policy buys the short request nothing.** Forcing it onto the small 1024 owner is
**1.06% slower**, and that difference is **0.017x the within-arm spread** - two orders of
magnitude inside the noise. The honest statement is *no measurable benefit*, not *the small
owner is worse*.

**And the policy is arithmetic-neutral here**: the short request's `logits_sha256` is
**identical** on the 1024 and the 4096 owner (`fefecd92...`), consistent with RT-1's
12-of-12 chunk invariance.

**The extra owner costs 1.555 GiB** and returns no measured benefit.

## Why the premise does not hold

`REFACTOR.md` justifies the shape policy as *"preserving the smaller owner's short costs"*.
That premise assumes the smaller workspace makes a short request cheaper. It does not, and
the reason is in the code: the workspace is **scratch**. Its row count is
`min(chunk_size, max_sequence_length)`, and the kernels' work scales with the **actual token
count**, not with the size of the scratch they are handed. A 512-token prefill does 512
tokens of work whichever owner supplies the buffers; the larger owner just has more room
allocated around it.

This is the same mechanism RT-1 already found from the other direction: for any prompt at or
below the smaller chunk, the two chunk sizes were byte-for-byte identical in time and memory
because both resolve to the same `min(chunk, max_seq_len)`. Here the owners genuinely differ
(1024 vs 4096 rows against a 4352 sequence) and the short request is still unaffected - which
is the stronger version of the same conclusion.

## What this changes

**It removes a blocker instead of clearing it.** The shape policy was the reason a blanket
chunk increase could not be promoted: RT-3's short-request objection said the smaller owner's
short costs had to be preserved. If short requests are unaffected by owner size, there is
nothing to preserve, and **the simpler single-owner configuration is at least as good as the
two-owner policy** - without the extra 1.555 GiB, without the borrow/restore path, and without
a selector that has to be qualified.

So RT-1's promotion does not need a shape policy. What remains is the memory-admission
question for a single larger owner, which is a capacity question rather than a policy one.

## Limits

- **One short case and one long case.** `code-p512` and `code-p4096` only; the other 10
  canonical cases were not run through the selection path.
- **The 1.06% is inside the noise**, so it supports "no measurable benefit" and nothing
  stronger. The short arm's spread (1.99%) exceeds the effect.
- One host, one model, one quant, one execution profile. GTT is 120 GB here and the base
  owner peaks near 89 GB, so the two-owner configuration is near this host's ceiling - a
  lower-memory host could not prepare both owners at all.
- **The selector is enabled only inside this process.** No runtime default changed, no
  production wiring added.
- **No promotion claim.** This is diagnostic evidence about the shape policy's value.
