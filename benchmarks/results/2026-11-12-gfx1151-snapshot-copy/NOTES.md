# gfx1151 rollback snapshot copies

The gfx1151 registry now exposes the existing
`linear_state_pair_copy/f32/chunked_i32` implementation. Initial-state capture
and rollback use device pointer tables instead of per-state copies when layer
indices and family sizes match. Other layouts keep the memcpy fallback.
Producer-folded capture and serial/native selected-row commit are unchanged.
There is no arithmetic change or new flag.

## Protocol and results

Physical host `gfx1151`, Radeon 8060S / Ryzen AI MAX+ 395, Qwen3.8-27B
`UD-Q4_K_M`, production profile, BF16 KV, MTP budget 3, prefix cache off,
one request, capacity 2048 and 64 greedy outputs with ignored EOS. The model
fingerprint, compiler, source hashes and profile manifest are in `summary.json`.

The same-owner public-library comparison runs the canonical ten prompts plus
eight category-heldouts in loop/batch/batch/loop order after a discarded warmup.
It changes only `_StateJournal._copy_initial_state`: the loop arm returns false
to use the existing memcpy path; the batch arm calls the shipping implementation.
The snapshot operation is outside the verifier graph. Both arms use the same
resident model, allocations and captured verifier graphs.

| Same-owner execution | Mean suite wall |
| --- | ---: |
| Per-state copy loop | 63.790 s |
| Pointer-table copy | 62.838 s |

Whole-request wall is 1.49% lower in this diagnostic, including prompt processing.
Every generated ID and checked acceptance/fallback control matches across arms
and repeats. This is not a decode-only rate or a fresh-session performance claim.

The earlier fresh-owner AR/loop/batch/batch/loop comparison failed control
repeatability on `heldout_general_ja_speculative`, with all generated IDs equal.
Focused fresh-owner repeats reproduce 19 cycles / 44 accepted drafts and
18 / 45 in the **loop baseline too**; one batch repeat records 20 / 43.
Those failed comparisons remain in `public_ab.json` and `focused.json`.
The cause of acceptance variation is unresolved. The same-owner result does not
close that issue or replace either failed comparison.

A separate public live-byte oracle compares every batched snapshot with resident
state immediately after copying, including repeated held-out requests. All bytes
match. Its synchronized readbacks distort timings; `byte-oracle.json` is
correctness evidence only. Unit and GPU fixtures also cover restore, serial
rows, incompatible layouts, alignment/tails and allocation retirement.

Short/long public attribution (15/2053 input tokens, 32 outputs, capacity 4096)
matches prior output IDs and non-timing MTP controls. Per-state initial snapshot
calls fall from 1152/768 to zero; the pointer-table route replaces them. Both
requests leave zero tracked allocations after close. The live 1024 boundary/reset
regression passes. No W7900 timing or cross-host transfer is claimed.

## Reproduction

Run from the repository root; the wrapper selects the local TheRock environment
at `~/miniforge3/envs/therock`.

```bash
P=benchmarks/results/2026-11-12-gfx1151-qwen38-mixed-pair-batch
D=benchmarks/results/2026-11-12-gfx1151-snapshot-copy
bash "$P/run.sh" "$D/public_ab.py" --out /tmp/qwen38-snapshot-public-ab.json
bash "$P/run.sh" "$D/focused.py" --out /tmp/qwen38-snapshot-focused.json
bash "$P/run.sh" "$D/same_owner.py" --out /tmp/qwen38-snapshot-same-owner.json
bash "$P/run.sh" "$D/same_owner.py" --check-bytes --out /tmp/qwen38-snapshot-byte-oracle.json
bash "$P/run.sh" -m pytest tests/test_gpu_gguf_state_journal_copy.py tests/test_gpu_linear_state_commit_chunked.py -q
bash "$P/run.sh" -m pytest tests/test_live_qwen35_gguf_mtp_e2e.py::test_dense_q4_k_m_mtp_crosses_1024_via_prelaunch_eager_fallback -q
uv run pytest tests/test_unit_qwen35_gguf_linear_state_snapshot.py tests/test_unit_qwen35_gguf_linear_state_commit.py tests/test_unit_gfx1151_backend.py tests/test_unit_qwen35_gguf_mtp_session.py -q
python3 scripts/docs/check_all.py
python3 audit/audit.py check
```

The first two commands intentionally assert control repeatability and may fail
with the observed baseline issue. The archived drivers preserve their executed
form; `mixed_plan_counts` in the fresh-owner JSON is a legacy field name for
snapshot capture/restore calls, not mixed projection calls.
