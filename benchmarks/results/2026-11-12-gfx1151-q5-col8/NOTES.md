# Q5 T16 wide verifier column tiles

The gfx1151 Q5 T16 rowtile selector now uses the existing eight-column owner
at K/N 5120/12288 and 5120/17408. Previously selected shapes are unchanged.
Narrow shapes keep the four-column fallback. This changes only backend shape
selection, not kernel arithmetic, scalar decode, input admission or peer backends.

## Protocol and results

Physical host `gfx1151`, Radeon 8060S / Ryzen AI MAX+ 395, Qwen3.8-27B
`UD-Q4_K_M`, production profile, BF16 KV, MTP budget 3, prefix cache off,
capacity 2048, one request and 64 greedy outputs with ignored EOS. Model
fingerprint, compiler, profile manifest and source hashes are in `summary.json`.
Baseline source is `3d228fecc`, including the preceding IQ4 tile improvement.

Public `LLM.generate_detailed()` runs fresh sessions in true AR,
parent/candidate/candidate/parent order, with a discarded warmup per session,
over the canonical ten category prompts plus eight held-outs. The parent arm
restores the previous shape set before constructing the model and its graphs.
The candidate calls the shipping selection unchanged.

| Q5 shape selection | Mean whole-request suite wall |
| --- | ---: |
| Previous shape set | 63.453 s |
| Added wide col8 shapes | 62.700 s |

Wall is 1.19% lower, with every category improving. All parent/candidate output
IDs, acceptance counts, fallback reasons and repeats match. Call records confirm
the added wide shapes reach col8. Every session returns tracked allocations to
zero on close. These timings include prompt processing, not just decode. No
AR-exactness, task-quality or cross-host performance claim is made. The separate
acceptance-repeatability issue from the snapshot experiment remains open.

Hardware counters collected before this unit showed high Q5 occupancy; unlike
IQ4, reducing register pressure was not the obvious first lever. The existing
col4/col8 screen checks exact output bytes before comparing HIP-event timings:
rows 2/3/4/8, seven dense shapes, three warmups, eight samples of ten calls with
alternating order. Both added shapes improve at every tested width. Narrow
K/N 5120/1024 regresses and 5120/6144 is mixed, so their selection is unchanged.
This screen uses synthetic weights and is kernel-only evidence, not model timing.
[Counter protocol and aggregates](../2026-11-12-gfx1151-iq4-col4/NOTES.md).

## Validation and reproduction

The selection test fails before the change and passes afterwards. Backend and
capability-boundary tests pass. Existing GPU col4/col8 parity tests and
selected-production plus CPU-reference tests pass. The live 1024-boundary/reset
regression passes. No kernel source or arithmetic changed.

Run from the repository root. The wrapper selects `~/miniforge3/envs/therock`.
Archived drivers preserve the executed temporary scripts; `loop` denotes the old
shape selection, `batch` the candidate, and `mixed_plan_counts` records col8
calls by `(rows, K, N)` rather than mixed-pair calls.

```bash
D=benchmarks/results/2026-11-12-gfx1151-q5-col8
P=benchmarks/results/2026-11-12-gfx1151-qwen38-mixed-pair-batch
bash "$P/run.sh" "$D/screen.py"
bash "$P/run.sh" "$D/public_ab.py" --out /tmp/qwen38-q5-public-ab.json
bash "$P/run.sh" -m pytest tests/test_gpu_gguf_t16_selected_gemv_decode.py::test_q5_t16_dense_rowtile_col8_matches_col4_bits tests/test_gpu_gguf_t16_selected_gemv_decode.py::test_q5_t16_dense_decode_and_rowtile_match_selected_production_bits -q
bash "$P/run.sh" -m pytest tests/test_live_qwen35_gguf_mtp_e2e.py::test_dense_q4_k_m_mtp_crosses_1024_via_prelaunch_eager_fallback -q
uv run pytest tests/test_unit_gfx1151_backend.py tests/test_unit_gfx1151_policy_capability_gguf_boundary.py -q
python3 scripts/docs/check_all.py
python3 audit/audit.py check
```
