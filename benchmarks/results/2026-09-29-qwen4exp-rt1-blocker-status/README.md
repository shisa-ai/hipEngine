# RT-1 promotion status: the blocker set is now precise and much smaller

RT-1 (prefill chunk 4096) is a measured, exact, non-regressive win: **-4.28% to -4.94% at
p4096**, indistinguishable at p1024, and **bit-identical logits on 12 of 12 canonical cases**.
`AGENTS.md` says such wins are promoted "unless there is a concrete blocker recorded in a
durable worklog entry".

**There is one, and it is recorded**: `docs/REFACTOR.md` §"Prefill-Only Workspace Ownership"
and §"Qualified Chunk Tradeoffs" name what enabling automatic selection requires. This packet
records which of those requirements are now **cleared** by this session's measurements and
which remain.

## The mechanism already exists, is tested, and is deliberately unwired

This matters because it changes the shape of the remaining work from "build a policy" to
"enable and qualify a policy".

| piece | state |
|---|---|
| `Qwen4ExpPrefillWorkspace` | owns prefill scratch/inputs independently |
| `_allocate_extra_prefill_workspace(chunk_size)` | prepares an inactive, runner-owned owner |
| `_configure_prefill_workspace_selection(workspaces, *, reserve_bytes)` | selects among prepared owners; allocates nothing; restores borrowed fields on failure |
| per-request selection | `next((w for w in choices if len(tokens) <= w.prefill_chunk_size), choices[-1])` - smallest owner that fits, else the largest |
| unit coverage | `tests/test_unit_qwen4exp_prefill_owner.py` covers ordering, duplicates, foreign owners, staging capacity and the reserve check |
| memory accounting | `scripts/qwen4exp_owned_workspace_check.py` verifies the extra owner allocates exactly its prefill-only budget and that prefill preparation does not replace decode state |
| **production wiring** | **absent - `_configure_prefill_workspace_selection` has no caller outside tests** |

So the shape-based policy RT-1/RT-3 call for is **implemented and unit-tested**; what is
missing is enabling it and qualifying it. `REFACTOR.md` says so explicitly: *"a private,
default-disabled selector"*, and the check script's own report reads *"automatic selection is
not enabled"*.

## Cleared this session

| requirement (from `REFACTOR.md`) | status | evidence |
|---|---|---|
| numerical qualification at the new chunk | **CLEARED** | X-6 chunk-4096 gate passes **bit-identically** to chunk 1024 on all 12 cases, all five KL metrics to the last digit, 12 trajectory hashes, 3 repeats |
| "preserve strict setup" / arithmetic neutrality | **CLEARED** | chunk 1024 vs 4096 publishes **identical `logits_sha256` on 12 of 12 cases** |
| "short complete requests include small losses" / "do not average the short losses away" | **CLEARED** | p512 chunk-4096 cost is **+0.46% mean, smaller than the within-arm spread** - indistinguishable, not a loss; p1024 is -0.39% |
| chunk-4096 accounting | already recorded as repaired in `REFACTOR.md` | `2026-09-15-chunk4096-accounting` |

## Still open

1. **"measure short/long requests on the same prepared owners"** - the specific gap. All
   measurements so far, including RT-1's and RT-3's, used **one process per arm**, i.e.
   separately prepared owners. The selection path itself has never been exercised
   end-to-end on a prepared pair. This is a bounded measurement and the next concrete step.
2. **"qualify full state/lifetime and c2 behavior"** on the selection path.
3. **"account for all owners and host staging in public construction"** - the reserve check
   in the selector is explicitly *not* public memory admission.
4. **Allocation and lifetime gates** for the selector.

## Verdict

**Not promoted, and the reason is a recorded concrete blocker, not an unmeasured risk.**
The blocker set is now four named items, down from an open-ended "short-request costs keep
1024 as the default" - and the three largest (numerics at the new chunk, arithmetic
neutrality, short-request cost) are cleared with measurements rather than arguments.

The next step is item 1: run short and long requests through the selection path on the same
prepared owners and compare against the single-owner baselines.

## Limits

- **Status synthesis, not a new measurement.** Every figure quoted here is from an existing
  packet in this repository; nothing was re-run for this packet.
- One model, one quant, one host, one execution profile.
- The "cleared" entries are cleared *for the evidence scope measured* (canonical 12 cases,
  one prompt family). Heldout categories and other routes' gates are not covered by them.
- No performance claim and no promotion claim.
