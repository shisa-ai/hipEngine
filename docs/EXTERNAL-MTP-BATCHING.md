# External MTP Verification-Batching Survey (X1)

Purpose: test the premise behind
[`QWEN38-GFX1151-SCALING-CAMPAIGN.md`](QWEN38-GFX1151-SCALING-CAMPAIGN.md)
diagnosis A2 — whether production speculative engines flatten draft
verification for all in-flight requests into one target pass, or issue
sequential per-request/per-group complete cycles like our width-4 partition.
Reading pass only: no vendoring, no code port, no perf claim.

All local pins were read at their checked-out commits on 2026-08-30 under
`/home/lhl/.local/state/hipengine-external-survey/repos/`. vLLM and SGLang are
not pinned locally; they were read from commit-pinned upstream URLs fetched
the same day.

## Table

| Engine (commit) | Verification batch dimension | Model passes per decode cycle | Width caps |
| --- | --- | --- | --- |
| llama.cpp mainline `4e97ac86ebe2c4cb8212d98d2641ad6768810896` | **Flattened across slots.** One shared target batch; each speculating slot adds its sampled token plus all of its draft tokens as `1 + n_draft` rows of one `llama_batch` (`tools/server/server-context.cpp:504-541` `handle_last_sampled_token`); all batchable slots decode in a single target pass | Draft: one `common_speculative_draft()` per cycle, itself batched — draft-simple adds one row per drafting sequence into a single draft-ctx decode per draft step and loops steps over all sequences together (`common/speculative.cpp:282-370`; MTP impl at `:1364`). Target: one verify forward. Plus per-slot KV checkpoints (memory copies, not model passes) | Total token budget `n_batch` (`--batch-size`; server batch sized `max(n_batch, n_parallel)` at `server-context.cpp:1343-1348`; `add()` refuses past the cap and remaining slots are skipped, `:3085-3090`). Slot count = `n_parallel`. Per-slot draft length = context/sampling headroom (`get_n_draft_max` `:483-501`), not a fixed small width. No fixed per-group width cap |
| llama.cpp-mike `152d337fadb93c2a099653c4072d5512c92c5bfd` | Same mechanism — `handle_last_sampled_token` byte-comparable (sampled + drafts into the shared batch) | Same | Same `n_batch` token-budget cap |
| laurent (Vulkan, DFlash drafter) `c28d538df5c02643e701a8004db84dbf1bb0ffb2` | Same shared-batch flattening (`spec_draft`/`spec_i_batch` present, same structure) | Same | Same |
| nathan (Vulkan) `0eb528051a56f34567312ce63ab4e14a3fc71d89` | Same shared-batch flattening | Same | Same |
| q38rocm (ROCmFPX wrapper) `5d0977403b0dac778598b1af499bf178b46c0b35` | Wrapper repo (prebuilt `engine/bin`); engine is the ROCmFPX llama.cpp fork pinned at `0fc9568e07ccc8553010864cb8db1957e629cbfa` (README.md:20). Server mechanism is the llama.cpp shared-batch path; its `patches/mtp-prompt-cache-fix.patch` edits `tools/server/server-context.cpp` checkpoint restore only, not batching | Same | Same llama.cpp caps; its `speed` profile runs `BATCH_SIZE=2048`, `UBATCH_SIZE=1024` (`run_server.sh`) |
| vLLM V1 `8c51b92654100aa1d698aeef862cad09c8cc5df8` ([blob](https://github.com/vllm-project/vllm/blob/8c51b92654100aa1d698aeef862cad09c8cc5df8)) | **Flattened across requests.** Every scheduled request carries `1 + num_spec_tokens` query tokens in one model forward; `_calc_spec_decode_metadata` lays out `num_draft_tokens + 1` sampled rows per request over the concatenated `cu_num_scheduled_tokens` of all requests (`vllm/v1/worker/gpu_model_runner.py:2876-2900`, `uniform_decode_query_len = 1 + num_spec_tokens` at `:914`) | Draft: `num_speculative_tokens` batched draft forwards per cycle, each covering the **entire request batch** in one pass (`for token_index in range(self.num_speculative_tokens - 1)` over `batch_size` rows, `vllm/v1/spec_decode/llm_base_proposer.py:510,690`). Target: one verify forward | Per-request draft length = `num_speculative_tokens`; request count = `max_num_seqs`; total rows bounded by the `max_num_batched_tokens` scheduler budget; CUDA-graph padded shapes |
| SGLang EAGLE v2 `e51a3ae65e3401b21de860c90a64102133d2d6a6` ([blob](https://github.com/sgl-project/sglang/blob/e51a3ae65e3401b21de860c90a64102133d2d6a6)) | **Flattened tree across requests.** One target verify forward over `bs × num_draft_tokens` tree tokens for the whole `ScheduleBatch` (`build_eagle_verify_input` `python/sglang/srt/speculative/eagle_worker_common.py:316`, `run_eagle_verify` "Batch 1: Target verify" `:461,:494`) | Draft: `speculative_num_steps` batched draft forwards, whole batch each step (`eagle_worker_v2.py:510,529`); target: one verify forward; plus one draft-extend forward after verify to re-sync draft KV (`_draft_extend_for_decode` `:907`) | `speculative_num_steps`, `speculative_eagle_topk`, `speculative_num_draft_tokens` per request; `max_running_requests`; CUDA-graph `max_bs` |

## Findings

1. **Nobody runs sequential complete cycles per request group.** All seven
   surveyed engines flatten the draft verification of every in-flight request
   into one target forward whose token dimension is
   `sum_requests (draft_i + 1)` (llama.cpp/vLLM) or `bs × draft_tree`
   (SGLang). Our C5-C8 two-sequential-4-subgroup decomposition
   (A2 / RF-OI5) has no analogue in any surveyed engine.
2. **Caps are token budgets, not widths.** The width cap in every engine is
   the generic decode-batch token budget (`n_batch`, default 2048;
   `max_num_batched_tokens`) shared with non-speculative decode, plus a slot
   count (`n_parallel` / `max_num_seqs` / `max_running_requests`). None
   hardcodes a small per-cycle request width like our 4.
3. **The draft pass is also batched.** llama.cpp drafts all drafting slots
   per step in one draft-ctx decode; vLLM/SGLang draft the whole batch each
   step. So "passes per cycle" is `draft_steps + 1 target` for the whole
   batch, not `(draft_steps + 1) × groups`. Our per-subgroup loop pays the
   proposal side once per subgroup.
4. **Implication for M1.** A single wide verify group over all due requests
   is the industry-standard shape, and its expected benefit is exactly the
   A2 signature: amortize one target weight sweep (and one proposal) over
   all rows instead of twice. The surveyed caps support bounding by rows
   (C8 × D24+8 ≈ 200 rows stays under a 2048-token budget), which is what
   M1's capability-owned bound should express.

Caveat: this is a batch-dimension reading, not an acceptance-rate or
scheduler-economics equivalence claim. Rejection handling (checkpoint
rollback in llama.cpp, `num_rejected_tokens` in vLLM, tree-sampling in
SGLang) differs and is out of scope for X1.
