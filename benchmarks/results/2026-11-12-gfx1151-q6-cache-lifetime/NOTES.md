# Q6 prefill scratch lifetime repair

The Q6 full-GEMM prefill route now owns temporary device storage and hipBLASLt
handles through a resident-session workspace, instead of module-global
shape-only dictionaries. Runtime identity, allocation device and stream are
part of the binding. A session cannot change devices after binding. Closing an
owner synchronizes its streams before releasing buffers and descriptors;
closing one model does not release another model's storage. Direct kernel
calls without an owner use temporary storage retired before returning.

The F16 staging dispatcher passes the workspace only to a registered callable
that declares its owner type. Qwen resident sessions, independent slot views,
and Muse Glimmer sessions supply owners. Arithmetic and the existing
per-shape timing-based algorithm selection are unchanged. No flag is added.

## Public lifecycle evidence

Physical host `gfx1151`, Radeon 8060S / Ryzen AI MAX+ 395, Qwen3.8-27B
`UD-Q4_K_M`, production profile, BF16 KV, MTP budget 3, prefix cache off,
capacity 4096, 2053 neutral context tokens and 32 greedy outputs with ignored
EOS. Full model fingerprint, compiler, profile manifest, baseline commit and
candidate source hashes are in `summary.json`.

- The prior diagnostic left 727,711,744 tracked bytes (694 MiB) after close.
  The repaired public request leaves zero, including after garbage collection.
- Generated IDs and non-timing diagnostic controls match the prior request.
- Two model instances coexist in one process. After closing the first, the
  second preserves its outputs and owns live scratch. After both close, tracked
  allocations return to zero. Opening and closing another model also returns
  them to zero and preserves the same output IDs.
- This is tracked allocation evidence, not a process-wide or driver-memory
  measurement. The coexisting sessions generate sequentially, not concurrently.
  Separate GPU streams/owners and their cleanup are also covered by the focused
  GPU test. No throughput claim is made.

## Reproduction

Run from the repository root. The wrapper selects the local TheRock environment
at `~/miniforge3/envs/therock`. The public lifecycle driver is archived unchanged
from its executed temporary copy; it reads the prior local warm-request JSON
for its baseline-ID assertion.

```bash
P=benchmarks/results/2026-11-12-gfx1151-qwen38-mixed-pair-batch
D=benchmarks/results/2026-11-12-gfx1151-q6-cache-lifetime
bash "$P/run.sh" benchmarks/results/2026-11-12-gfx1151-qwen38-dense-followup/attribution.py \
  --prompt-tokens 2053 --out /tmp/qwen38-cache-owner-long.json
bash "$P/run.sh" "$D/public_lifecycle.py"
bash "$P/run.sh" -m pytest tests/test_gpu_q6_fullgemm_workspace.py \
  tests/test_live_muse_glimmer_prefix_cache.py::test_resumed_prefill_matches_fresh_prefill -q
uv run pytest tests/test_unit_q6_fullgemm_workspace.py \
  tests/test_unit_int8_mtp_teardown.py tests/test_unit_qwen35_gguf_state_ownership.py \
  tests/test_unit_gguf_linear_dispatch_cache.py tests/test_unit_gguf_prefill_route_scopes.py -q
uv run pytest -q
```

The CPU tests reproduce the original direct-call leak and cover allocation and
plan failure cleanup, failed synchronization with later retirement, close
idempotence, runtime/stream isolation, device mismatch rejection, descriptor
release errors, dispatch keyword compatibility and independent slot-view
ownership. The GPU test compares against a CPU Q6 reference, checks repeat
output equality after another owner closes, and asserts allocation balance.
The full CPU tier passed; focused final checks cover the subsequent input/device
validation additions without rerunning unrelated tests. The live Muse Glimmer
prefix-resume comparison also passes with the shared staging owner.
