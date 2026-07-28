# Redline Integration and GPU-Fault Dossier

This document is the durable diagnosis record for hipEngine's experimental
[Redline](https://github.com/warpfront/redline/) retained-PM4 integration. It
separates accepted performance evidence, genuine GPU faults, ordinary harness
failures, established facts, and untested hypotheses.

Redline remains **unvendored, explicit, and default-off**. Process isolation is
permitted to recover benchmark rows, but it is not an acceptable runtime fix.

## Current verdict

- Same-HSACO W7900 micro transport is accepted: 240/240 rows pass correctness;
  Redline beats HIP on 239 rows at median 2.792x and Vulkan on 208 rows at
  median 1.696x.
- The narrow production-sized Qwen3.6 GGUF p512/d128 graph diagnostic is also
  accepted: strict retained PM4 is bit-identical and improves decode
  92.812 -> 100.357 tok/s (+8.129%). A one-graph persistent soak passes 512 PM4
  launches with 0.095% range/median.
- Package-default integration is blocked by cross-device lifecycle failure. The
  W7900 reports an address-zero VM fault at row 49; the RX 7900 XTX reaches row
  173, times out on a Redline HSA completion signal, and then loses its
  HIP/Vulkan process context.
- Ownership is not yet isolated between Redline's queue/IB lifetime,
  Hipfire's adapter lifetime, and ROCr. The available evidence points more
  strongly to create/drop churn than replay count or packed-dot math.

Machine-readable evidence:

- [`benchmarks/results/2026-07-28-gfx1100-redline-transport-spike.json`](../benchmarks/results/2026-07-28-gfx1100-redline-transport-spike.json)
- [`benchmarks/results/2026-07-28-gfx1100-redline-gguf-graph-spike.json`](../benchmarks/results/2026-07-28-gfx1100-redline-gguf-graph-spike.json)
- [`benchmarks/results/2026-07-28-gfx1100-redline-rx7900xtx-lifecycle-diagnostic.json`](../benchmarks/results/2026-07-28-gfx1100-redline-rx7900xtx-lifecycle-diagnostic.json)

## Pinned environment

| Component | Identity |
| --- | --- |
| GPU | AMD Radeon Pro W7900, gfx1100, 96 CUs, GPU ordinal 0 |
| ROCm | TheRock HIP 7.15.0-0000000 |
| LLVM | AMD clang 23.0.0git `aa451e1f...e96` + TheRock patch |
| Redline | clean `33683f3d4f302a6c56bcc7a4c33ab8be3262dd2e` |
| Hipfire bridge | `455ffb9dfd6a5712889b504737f88fbbe87d3efe` |
| Same-HSACO binary | SHA-256 `655141e2e5eef7a1a31f08a9da7b6fb19cdc114b61a79152331ac1ee0a72a291` |
| Common controls | `HIP_VISIBLE_DEVICES=0`, `ROCR_VISIBLE_DEVICES=0`, gfx1100, Radiowave-tuned wave policy, default scheduler, certified-VMEM RMW, auto Redline queues, 3 warmups, 7 samples |

The primary table above describes GPU0. GPU1 is an AMD Radeon RX 7900 XTX,
gfx1100, physical PCI `0000:10:00.0`, unique ID `0xcc4d02090dc9c3ff`, card
model `0x744c`, revision `0xc8`. Its cross-device run uses the same binary and
TheRock stack.

The raw micro root is
`/tmp/hipengine-redline-w7900-20260728`; the raw graph root is
`/tmp/hipengine-redline-graph-gate-20260728`. Those paths are disposable. The
compact artifacts retain their commands and hashes.

## Genuine GPU fault: canonical reproduction

The one-process same-HSACO command is:

```bash
HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 HIPFIRE_BENCH_ARCH=gfx1100 \
  /home/lhl/redline/target/hipengine-w7900-gfx1100/release/hipfire-6409-bench \
  --matrix hipengine --backends redline,hip,vulkan \
  --wave-policy radiowave --redline-queues auto \
  --scheduler-profile default --redline-rmw radiowave-vmem \
  --warmups 3 --samples 7 \
  --out /tmp/hipengine-redline-w7900-20260728/hipfire-same-hsaco/results.json
```

Observed behavior:

1. Rows 1-48 pass every selected backend's CPU oracle. They cover serial
   dispatch-grid (8 rows), geometry (8), reduction (24), and memory/waitcnt (8).
2. Row 49 is the first packed-dot row:
   `serial_latency/packed-dot/variant=q8_signed,groups=16,n=32768,body=64,wg=64`.
3. Its backend order is `redline -> hip -> vulkan`; no Redline timing or
   correctness row is emitted before the process aborts.
4. ROCr reports:

   ```text
   Memory access fault by GPU node-1 (...) on address (nil).
   Reason: Page not present or supervisor privilege.
   ```

5. The process exits through `SIGABRT`/exit 134 and never writes a complete
   results artifact.

Raw hashes:

- stdout: `bb6d1c0d45ece5a856f49237fac61c6f493e05ba6ce9cd249bedd83daf0ecf13`
- stderr: `8859d2d1883cc930568c1f2f5c7de449441cca0721dd12278f790f435039e73e`

This is a real GPU VM fault, not a correctness mismatch, timeout, comparator
rejection, or Python exception. The same signature was observed in the separate
source-matched orchestration below, but exact-command repeatability and a fixed
failure threshold have not yet been established.

## Controls already run

| Test | Result | What it establishes |
| --- | --- | --- |
| Fresh-process packed-dot, same binary/HSACO and controls, `--filter packed-dot` | 16/16 rows pass; Redline wins 14/16; empty stderr | Packed-dot code, arguments, launch shapes, and CPU oracle are valid from fresh state. Result SHA-256 `1c81b3bc...a3c`. |
| Ten family-isolated processes | Every process exits 0; deduplicated union passes 240/240 | Every family can complete under the same stack. Process isolation avoids the trigger but is only a benchmark workaround. Status SHA-256 `8c0a8943...7e36`. |
| Separate source-matched matrix runner using three warmups | A fresh family child faults entering serial q8 packed-dot after earlier matrix subprocesses; raw log SHA-256 `fef5b843...be74` | The same address-zero signature occurs in a second orchestration, but its process/code-object lifecycle differs from the canonical same-HSACO run. It is supporting evidence, not the attribution denominator or proof of the same trigger. |
| Independent simple families | Complete | Independent geometry/memory/VOPD/sampler work is not categorically broken. |
| Independent reduction families | Complete | Independent reduction and two-stage paths are not categorically broken. |
| Fresh independent tapes with safe preheat | Complete | The independent-tape setup itself is viable when started in a fresh process. |
| Strict two-kernel Redline hipGraph test | Four fresh-input BF16-bit-exact PM4 replays pass | Basic capture/instantiate/replay/destroy works. |
| Qwen3.6 p512/d128 ABBA | Two fresh Redline processes, 256 total strict PM4 launches, exact logits/IDs, no fallback | A production-sized 627-node graph is valid and faster in a narrow process lifecycle. |
| One-session Qwen persistent soak | One graph, four reset/prefill/warmup/d128 runs, 512 PM4 launches, exact output and memory recovery | Hundreds of replays and roughly 321k captured kernel dispatches on one retained graph do not reproduce the fault. |

### Background-run ledger

The `bg-*` labels are session-local, but recording them prevents ordinary
harness failures from being confused with the GPU fault.

| Runs | Outcome | Classification |
| --- | --- | --- |
| `bg-1` | Redline release build and 6/6 Rust tests pass; follow-up hash step expects a nonexistent library filename | Post-build harness/path failure. No GPU fault. |
| `bg-2` | Direct dispatch control completes 16/16 | Successful retained-PM4 control. |
| `bg-3`, `bg-4` | `three-backend hardware identities do not match` | Harness validation failure caused by RADV's parenthesized device suffix; fixed by normalization. Not a GPU fault. |
| `bg-5`, `bg-6`, `bg-7` | Missing measured code-object sidecar | Harness discovery failure; fixed by searching the actual hipEngine JIT cache. Not a GPU fault. |
| `bg-8`, `bg-9`, `bg-10` | Independent Redline/simple, native-peer, and reduction work completes | Successful component evidence used by the final matrix. |
| `bg-11` | Source-matched matrix orchestration reaches an address-zero GPU page fault in its serial q8 packed-dot family child after earlier matrix subprocesses | Genuine supporting reproduction of the signature, but not proof of the canonical same-process trigger. |
| `bg-12` | Fresh independent tapes complete | Successful isolation control. |
| `bg-13` | Canonical same-HSACO full process aborts at row 49 | Genuine primary GPU-fault reproduction. |
| `bg-14` | Fresh packed-dot process passes 16/16 | Kernel/family isolation control. |
| `bg-15` | All ten family processes exit 0 | Final partitioned 240/240 evidence. |

A separate Python-feature Redline build failure (`build.rs` requests unavailable
mold) and the immediately repaired TheRock `ld.lld` truncation incident were
host-toolchain failures. The native HIP graph smoke passed after exact package
restoration and before Redline graph measurement; neither is evidence for this
GPU VM fault.

## RX 7900 XTX cross-device result

GPU1 had no prior retained Redline lifecycle result. Run the same 240-row binary
on the RX 7900 XTX with the pinned TheRock libraries. Because ROCr visibility
remaps physical GPU1 to visible ordinal 0, the correct filter pair is
`ROCR_VISIBLE_DEVICES=1 HIP_VISIBLE_DEVICES=0`; here `ROCM_ROOT` is
`/home/lhl/mambaforge/envs/therock/lib/python3.12/site-packages/_rocm_sdk_devel`:

```bash
PATH="$ROCM_ROOT/bin:$ROCM_ROOT/lib/llvm/bin:$PATH" \
LD_LIBRARY_PATH="$ROCM_ROOT/lib:$ROCM_ROOT/lib/llvm/lib:$LD_LIBRARY_PATH" \
ROCR_VISIBLE_DEVICES=1 HIP_VISIBLE_DEVICES=0 HIPFIRE_BENCH_ARCH=gfx1100 \
  /home/lhl/redline/target/hipengine-w7900-gfx1100/release/hipfire-6409-bench \
  --matrix hipengine --backends redline,hip,vulkan \
  --wave-policy radiowave --redline-queues auto \
  --scheduler-profile default --redline-rmw radiowave-vmem \
  --warmups 3 --samples 7 \
  --out /tmp/hipengine-redline-rx7900xtx-20260728/hipfire-same-hsaco-therock/results.json
```

The answer is **yes, GPU1 also fails**, but not identically:

- rows 1-172 pass correctness for all three backends;
- row 173 is independent packed-dot q6-zero, wg64, backend order
  `hip -> vulkan -> redline`; HIP and Vulkan pass before Redline times out on
  HSA signal `0x7f3de55f8b00` after 5 seconds;
- row 174 reports Vulkan device loss and HIP/Redline `HipError(719)`; the
  remaining 68 rows are rejected;
- stderr says RADV's command stream was cancelled because the context was lost
  and labels its context innocent;
- there is no userspace address-zero message. Kernel-log correlation is absent
  because this user cannot read `dmesg`;
- the harness returns 0 after recording errors, so its exit code is not a pass;
- a fresh post-fault process successfully performs HIP init, 4-KiB allocation,
  synchronization, and free on the RX 7900 XTX.

The incomplete 172-row prefix has **no performance-claim status**. It confirms
that broad lifecycle/context stability is not W7900-specific, while the
different row and signature mean a shared exact root cause is not yet proven.
Raw result/stdout/stderr hashes are `e2216c2f...05f1`, `3b55f6c8...a133`, and
`1e69d736...577`.

Two setup attempts launched no GPU work and are not faults: applying both
visibility filters as ordinal 1 hides the ROCr-remapped sole device, and omitting
the pinned `LD_LIBRARY_PATH` loads a system ROCr missing
`hsa_amd_counted_queue_acquire`.

## Lifetime audit of the failing path

The audit uses clean Redline `33683f3d`:

- `examples/hipfire-6409/src/redline_backend.rs` — `RedlineBackend::new()` and
  `measure()`;
- `crates/redline-dispatch/src/aql/replay.rs` — `SingleQueuePm4Ib` construction,
  replay, field order, and `Drop`;
- `crates/redline-rocr/src/runtime.rs` — `AqlQueue`, `CompletionSignal`,
  `KernargBuffer`, and their teardown.

At that pinned source, `RedlineBackend` keeps these objects for the process:

- one initialized public-ROCr `Runtime` and selected `GpuDevice`;
- wave32/wave64 `Executable`s for every scheduler profile, loaded at backend
  construction rather than per row;
- one discovered `KernargPool`.

Each serial `measure()` creates and later drops:

- HIP input/output buffers;
- one or more kernarg buffers;
- one unprofiled single-queue ownership/acquire IB;
- one profiled single-queue retained IB, including a completion signal,
  indirect-command buffer, and 16-byte GPU timestamp buffer.

After output readback, the explicit order is profiled IB, ownership IB,
kernargs, then HIP buffers. `SingleQueuePm4Ib` declares its queue owner before
completion/indirect/timestamp pointees, so normal Rust field drop destroys the
queue before those buffers. `AqlQueue::drop()` attempts
`hsa_queue_inactivate()` and then `hsa_queue_destroy()`, but cleanup errors are
not surfaced to the benchmark.

For the 48 successful serial rows, source/protocol arithmetic implies:

- 96 single-queue retained-IB objects created and dropped (ownership + profiled);
- 960 successful queue submissions (two per warmup/sample, 10 repetitions/row);
- about 53,560 successful kernel dispatches inside the profiled IBs.

These counts are inferred from pinned source and the completed row protocol;
they were not emitted as runtime counters. The first failing row begins the
97th/98th queue-owner creation pair, but current logs do not say whether the
fault occurs in its ownership IB, timestamp prologue, first kernel dispatch, or
timestamp epilogue.

The Qwen persistent control executes more kernel dispatches while reusing one
graph. Combined with the eager family controls, this makes **queue/IB/signal or
GPU-visible allocation create/drop churn** the leading hypothesis. It does not
prove that Redline core owns the bug; ROCr queue destruction or HIP/ROCr
interoperation may be involved.

## What is ruled out, and what is not

### Ruled out or strongly disfavored

- **Packed-dot math or deterministic bad HSACO:** fresh packed-dot passes 16/16
  with the same binary and code-object policy.
- **A universal packed-dot launch-shape defect:** all four variants and both
  workgroup sizes pass in both timing modes when isolated.
- **Simple lack of warmup:** three warmups are present in the canonical repro;
  the separate preheated run also faults.
- **Replay-count exhaustion alone:** one Qwen graph survives 512 launches and
  substantially more captured kernel dispatches.
- **Per-row executable/module accumulation in the canonical Hipfire backend:**
  executables are loaded up front and retained, not loaded for each row.

### Still open

1. Queue create/inactivate/destroy churn or a silent queue-destroy failure.
2. Completion-signal reuse/destruction ordering inside ROCr.
3. Reuse of freed kernarg, indirect-IB, or timestamp virtual addresses before
   ROCr/CP has fully stopped touching them.
4. The separate ownership queue plus profiled queue topology.
5. GPU timestamp `COPY_DATA`/`RELEASE_MEM` prologue or epilogue.
6. HIP reset/allocation interoperation with direct public-ROCr submission.
7. Vulkan coexistence or backend-order state, despite Redline being first on the
   failing row.
8. A specific predecessor family poisoning later state rather than a pure
   create/drop threshold.
9. Redline adapter behavior versus an underlying ROCr or amdgpu driver defect.

## Next diagnostic matrix

Do not rerun the full 240-row benchmark to diagnose this. Build a minimal
upstreamable stress reproducer and run the following reductions in order:

1. **Repeat one known-good tiny retained IB with recreate-per-cycle.** Run at
   least 128 create/replay/wait/drop cycles and report queue IDs, destroy status,
   completion handles, indirect/timestamp/kernarg addresses, and the exact
   failing cycle. This tests churn without family changes.
2. **Reuse versus recreate.** Hold one queue/IB for the same number of replays,
   then recreate every cycle. A recreate-only fault sharply isolates lifetime
   handling.
3. **Create/drop without submit.** If this fails, queue lifecycle is sufficient;
   if it passes, execution/completion is required.
4. **Unprofiled versus profiled.** Remove GPU timestamp buffers and timestamp PM4
   while preserving the kernel tape. This isolates timestamp resources.
5. **One queue versus separate ownership/profile queues.** Fold the ownership
   acquire and dispatch into one retained queue as a diagnostic only. This is
   not a production workaround unless correctness and ordering are proven.
6. **Backend subtraction.** Run Redline-only; then add HIP reset/allocation;
   then add Vulkan construction/execution. This identifies interop requirements.
7. **Prefix/order bisection.** Run `packed-dot` after each predecessor family and
   after repeated copies of one family. Distinguish poison-family from count
   threshold.
8. **Teardown instrumentation.** Fail loudly on every non-success
   `hsa_queue_inactivate`, `hsa_queue_destroy`, signal destroy, and memory free;
   sample queue read/write indices before destruction; never silently continue
   after an unproven teardown in the reproducer.
9. **Fault correlation.** Retain the kernel driver's VM-fault record with PID /
   PASID / VMID and the userspace resource-address ledger. Avoid privileged
   collection commands in the benchmark itself; record the host command and
   permissions separately.
10. **Stress the fix on both discrete cards.** Require at least 1,000
    create/replay/drop cycles, the original one-process 240-row matrix on W7900
    and RX 7900 XTX, the strict graph tests, and the Qwen persistent soak before
    calling the lifecycle issue fixed.

The reproducer should use pinned Redline public APIs and live in hipEngine or an
upstream PR branch; do not edit the external `/home/lhl/redline` checkout as
part of hipEngine work. Any temporary patch must be explicit, hashed, and
reported in provenance.

## Promotion policy

Redline cannot become a package default until all of the following are true:

1. the minimal reproducer has an owned root cause and upstream fix;
2. the create/drop stress and original broad process pass without isolation;
3. strict PM4 proof detects zero fallback;
4. natural-prompt/category and 4K-context correctness/performance pass;
5. concurrency, cancellation, server shutdown, and memory-recovery gates pass.

A page fault is fail-stop. No retry, subprocess restart, native-shadow fallback,
or family-specific process split may be presented as a production fix.
