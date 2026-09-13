# MODEL-SURYA.md — Surya OCR 2 on hipEngine

Status: **implemented — CPU reference and gfx1151 HIP lane** (2026-09-11). The
per-model architecture and bring-up plan below is followed by an
implementation-status section. Torch-free preprocessing, vision tower, text
decoder, and the public OCR API run end to end; the HIP lane executes the
vision tower and text decoder on gfx1151.

### Implementation status (2026-09-11)

| Area | State | Evidence |
| --- | --- | --- |
| Model/weight/tokenizer contracts | done | `hipengine/models/surya.py`, `hipengine/loading/surya.py`, `tests/test_surya_model_contract.py` |
| Torch oracle fixtures | done | `scripts/surya_oracle_torch.py`, `tests/fixtures/surya/*.npz`, `tests/test_surya_oracle_fixtures.py` |
| CPU-reference text decoder (GDN state, causal cache, mRoPE) | done | `hipengine/kernels/cpu_reference/surya.py`, `tests/test_surya_text_decoder.py` |
| CPU-reference vision tower + merger | done | `tests/test_surya_vision.py` |
| Public OCR API, greedy oracle parity | done | `hipengine/loading/surya.py:run_surya_ocr`, `tests/test_surya_e2e.py`, `tests/test_surya_public_api.py` |
| HIP text decoder on gfx1151 | done | `hipengine/runtime/surya.py`, `tests/test_surya_gpu.py` |
| HIP vision tower on gfx1151 | done | `tests/test_surya_gpu.py::test_gpu_vision_matches_oracle_features` |
| Registered HIP generator (`surya_ocr2`/`hip_gfx1151`/fp32) | done | `hipengine/generation/surya_gpu.py` |
| Request/resource contracts (greedy-only, capacity, EOS) | done | `hipengine/generation/surya_contract.py`, `tests/test_surya_generation_contract.py` |
| GPU `rocprofv3` kernel-trace evidence | done | `benchmarks/results/2026-09-11-gfx1151-surya-kernel-trace.json` |
| Multi-prompt `generate` isolation | done | `tests/test_surya_gpu.py::test_gpu_generator_multi_prompt_isolation` |
| Resize / grid-geometry branches | done | `tests/test_surya_gpu.py::test_gpu_vision_geometry_sweep_matches_cpu_reference` (4 cases; 3:1 aspect ratios measured but uncommitted) |
| Rectangular-page coverage | done (vision + isolation) | `tests/test_surya_gpu.py` rect vision vs `oracle_rect` `vision_merged` |
| Full-page coverage (1024x1024, real text) | done | `tests/fixtures/surya/page_full.png`, `scripts/surya_oracle_greedy.py`, `tests/test_surya_gpu.py::test_gpu_full_page_layout_matches_oracle` |
| OCR output quality check (labels + bboxes vs drawn geometry) | done | `tests/test_surya_gpu.py::test_gpu_full_page_layout_matches_oracle` second half |
| Greedy reference fixtures are reproducible | done | `scripts/surya_oracle_greedy.py --case all` regenerates `oracle_greedy.json`, `oracle_fullpage_greedy.json` and `oracle_corpus.json` byte-for-byte |
| Multi-page held-out OCR corpus | done | `page_columns.png` (two-column) and `page_list.png` (numbered list) via `tests/test_surya_gpu.py::test_gpu_ocr_corpus_matches_oracle` |
| 300-DPI A4 page fixture and its transcription measurement | done | `scripts/surya_bench_pages.py:make_page_a4`, gated by `tests/test_surya_transcription.py::test_transcription_meets_ground_truth[a4]` |
| OpenAI-compatible HTTP serving (`hipserver`) | done | `hipengine/server/multimodal.py`, `scripts/surya_http_e2e.py`, `tests/test_surya_server_multimodal.py` |
| Registry migration | pending | tracked in `docs/REFACTOR.md` |
| `KVLiveSpans` KV ABI | done | `surya_scatter_kv_f32_spans`, `surya_full_attn_decode_f32_spans`; `tests/test_surya_kv_spans.py`, `scripts/surya_kv_spans_bench.py` |
| Quantized (GGUF) Surya decoder | pending | safetensors fp32 is the implementation target |
| MTP / speculative decoding | pending | 15 MTP tensors inventoried, unused |

Correctness gates in place: CPU reference matches the torch fp32 oracle on
greedy token IDs (`tests/fixtures/surya/oracle_greedy.json`), the HIP vision
tower sits inside the CPU-reference noise band against `vision_merged`, and the
full HIP pipeline (GPU vision features + GPU prefill/decode) reproduces the
oracle greedy IDs exactly. On the full-size page the torch fp32 oracle, the
NumPy CPU reference, and the gfx1151 HIP lane all produce the same 78-token
layout-JSON output for `page_full.png` (grid 1x64x64, 1024 merged visual
tokens). All HIP tests carry a ROCm-availability guard.

The full-page gate also checks the decoded text is the *right answer* for the
page that was drawn, not merely that the lanes agree: the output parses as
JSON, the label sequence is `Section-Header, Text, Text, Caption, Table`, every
bbox is in range and ordered, and the header/caption/table boxes match the
drawn geometry (heading at y=50, caption at y=458, table below it).

Two held-out layouts cover structure the single-column fixtures cannot
produce. `page_columns.png` (512x512) has a spanning title and two prose
columns; the oracle keeps them apart as a left-column pair (x 65-419) and a
right-column pair (x 543-897), so column detection is exercised rather than
just text finding. `page_list.png` (512x512) decodes to
`Section-Header` + `List-Group` + `Text`, and is the only fixture that reaches
the list label. Both are 1x32x32 grids and both reach a natural EOS (94 and 46
tokens).

### Transcription acceptance (2026-09-12)

Agreeing with a captured oracle is regression coverage, not transcription
qualification. Surya is prompt-driven: the wording *is* the task, and the
ad-hoc prompt `"Transcribe this page."` used by the early fixtures is not one
of the checkpoint's training-time prompts. Its continuation is layout JSON or a
degenerate repeated `<ul><li>` run, so a lane could pass by reproducing
garbage.

`hipengine/generation/surya_protocol.py` pins the real contract
(`FULL_PAGE_HTML_PROMPT`, `LAYOUT_JSON_PROMPT`, `BLOCK_HTML_PROMPT`) and parses
full-page output with `parse_full_page_html` / `extract_text` / `extract_tables`
(stdlib `html.parser` only, no torch and no third-party HTML library).
`scripts/surya_transcription_score.py` scores a page against the text actually
drawn on it — reading order, line recall and omissions, line exact rate,
character error rate, table shape/header/cells, and truncation — and
`scripts/surya_transcription_report.py` emits the measurement artifact.

Measured fp32 on gfx1151 with `FULL_PAGE_HTML_PROMPT`, one page per scope:

| page | tokens | finish | recall | exact | CER | order violations | table cells |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| ja | 287 | eos | 1.000 | 1.000 | 0.0000 | 0 | — |
| mixed | 341 | eos | 1.000 | 1.000 | 0.0000 | 0 | — |
| dense | 1054 | eos | 1.000 | 1.000 | 0.0000 | 0 | — |
| table | 448 | eos | 1.000 | 1.000 | 0.0000 | 0 | 1.000 (32/32) |
| blank | 12 | eos | 1.000 | 1.000 | 0.0000 | 0 | — |
| scan | 616 | eos | 1.000 | 1.000 | 0.0000 | 0 | — |
| long | 1804 | eos | 1.000 | 1.000 | 0.0000 | 0 | — |

The gate is `tests/test_surya_transcription.py` (ROCm- and checkpoint-gated):
exact id parity with a captured torch fp32 protocol oracle on six pages, a
bounded coordinate drift on the seventh, plus the ground-truth bars above, plus
a starved-budget case that must report truncation and omissions rather than
pass a prefix.

**Parity caveat: the degraded scan's bbox digits.** Six of the seven pages
reproduce the oracle's greedy ids exactly. On the degraded scan the HIP lane
differs on 21 of 616 ids (3.4%), all of them bbox coordinate digits: the parsed
labels and texts are identical and the worst coordinate delta is 4 of 1000.
This is intrinsic fragility of that page rather than a HIP-specific defect —
torch bf16 against torch fp32 differs on 15 ids there and even changes the
decoded text, and the scan carries every HIP-versus-torch row above the review
bar in the numerical gate, with teacher probability margins of 0.002-0.099 and
top-5 overlap 3-5. The test therefore splits into `EXACT_ID_CASES` (full id
equality) and `COORDINATE_CASES` (equal finish reason, under 5% coordinate-id
diff, identical labels and texts, worst bbox delta under 8 of 1000).

**Fixture defect found and fixed (2026-09-13).** Four of the seven bench pages
drew text past the canvas, so their intended text was not what was in the
image: `ja` body lines were 576-663 px wide on a 512 px page, `mixed` Latin
lines 503/506 px against 480 usable, `scan` up to ~531 px against 476, and
`page_long` drew six 224 px blocks from y=168 so blocks 5 and 6 fell off a
1024 px canvas. Scored against the intended text those pages read as CER 0.220
(`ja`) and 0.039 (`mixed`), and the apparent hallucination `line.` ->
`literature` was the model completing a clipped word. The pages are now drawn
through a fit assertion that raises instead of clipping, and the four were
re-cut onto canvases that hold them (1024x1024 for `ja`/`mixed`/`scan`,
1024x1600 for `long`); `page_ja.png` and its siblings are the pages the
acceptance test and the benchmark suite both use, with no separate variant.
The vision grids grew with the canvas (`ja`/`mixed`/`scan` 32x32 -> 64x64,
`long` 64x64 -> 100x64), so `oracle_bench.json` and the lane comparison were
re-captured. The one measurement that did not change is the truncated
`scan` continuation: the old page's clipped text drove torch fp32 to 504 ids
against a 512 cap, while the re-cut page reaches a natural EOS at 158.

#### The lane comparison (2026-09-13)

The re-cut suite is 12 pages x 2 lanes, `--split all --runs 3`, all 24 rows
PASS: hipEngine decodes at 54.6-57.0 tok/s against the torch fp32 lane's
24.6-28.1, a median 2.19x (2.03x-2.23x on every page), and end-to-end is
0.20-8.85 s against 0.29-14.38 s. The four re-cut pages cost hipEngine more
end-to-end time than the clipped ones did (`ja` 2.71 -> 5.75 s, `long`
5.67 -> 8.85 s, `mixed` 5.13 -> 6.76 s) because their vision grids are 4x
bigger, while decode is unchanged at ~55 tok/s.

**The scan page cannot carry an exact-id gate under this prompt.** The bench
suite's ad-hoc prompt `"Transcribe this page."` is not one of the checkpoint's
training-time prompts, and on the re-cut degraded scan both lanes free-run into
the same rigid 10-box layout template: identical labels, identical `count`
values, a fixed line pitch. Only the bbox digits differ, and neither lane's
digits are a reading of the page — the lines are drawn 470-554 of 1000 wide,
torch reports 474-566, and hipEngine a flat 564. Gating those digits would
measure which arbitrary continuation the sampler picks, so `scan` declares
`parity="structure"`: the gate is the layout skeleton (box count, label
sequence, reading order, per-box `count`), and the coordinate drift is recorded
as `bbox_max_delta` (90 of 1000) instead of gated. The divergence is not a HIP
defect and not a regression: it reproduces at the pre-KV-spans commit
`b1e241d57` with the same first divergence (id 7 of 158), and under the
checkpoint's real prompt on the same image both lanes read the page and agree
within 2 of 1000 with identical labels and text. The other eleven cases keep
the exact-id gate.

**Each lane is measured in its own process.** Both lanes in one process corrupt
the tail of the hipEngine lane on this suite: the same 12 cases measured with
`--lanes hipengine_gpu` alone pass every gate, while `--lanes
hipengine_gpu,torch_cuda` returns the last three hipEngine rows (`full`,
`columns`, `list`) degenerate from the first generated token. Every stage time
is unchanged — `full`'s e2e is 3.663 s against 3.344 s in isolation and only
because it ran 96 tokens instead of 78, `init_s` is identical to 0.04 s — so the
device was healthy and fast and the logits were wrong, which puts this in the
class tracked as task #80 rather than in the pages or the harness. The retained
comparison is therefore one run per lane merged with `--merge`; a combined run
is reproducible with
`HIPENGINE_HIP_ARCH=gfx1151 python3 scripts/surya_perf_compare.py --split all --runs 3 --lanes hipengine_gpu,torch_cuda`.

#### The 300-DPI A4 page (2026-09-12)

The seven pages above are 512-1600 px on a side. `page_a4.png` is the real
page-scale case: 2480x3508 at 300 DPI, which `smart_resize` rounds to 2496x3520
— exactly the 220x156 patch grid and 8580 merged image tokens the page-scale
memory plan is written against.

Measured fp32 on gfx1151 with `FULL_PAGE_HTML_PROMPT`,
`scripts/surya_transcription_report.py`, artifact
`benchmarks/results/2026-09-12-gfx1151-surya-transcription-acceptance.json`:

| metric | value | gate |
| --- | ---: | --- |
| `line_recall` | 1.0000 (12/12) | 1.0000 |
| `line_exact_rate` | 1.0000 | 0.9500 |
| `cer` | 0.0000 | 0.0100 |
| reading-order violations | 0 | 0 |
| `table_cell_accuracy` | 1.0000 (32/32) | 1.0000 |
| finish | eos, 2108 tokens, not truncated | eos |

Stages: vision 47.7 s, prefill 8.0 s, decode 58.4 s (2108 steps, 27.7 ms/step,
36.1 tok/s), 114.3 s total.

The 8580-image-token prefill is verified rather than inferred from the grid
arithmetic: `render_chat_prompt` — the same call the generator makes — returns
8701 `input_ids` for this page, of which exactly 8580 are contiguous image pads
(121 text tokens, then the image span), and that list is what the prefill
receives. `tests/test_surya_transcription_fixtures.py` pins the count, the
contiguity, and the absence of any other image flag. The serving path reports
the same 8701 as `usage.prompt_tokens`, and its HTTP response reproduces this
row exactly (see "HTTP serving" below).

**The first measurement of this page read as a failure, and the failure was in
the ground truth.** Scored per drawn line it measured recall 0.2414, CER 0.8901,
and one reading-order violation. The model output was in fact exact: it returned
each paragraph as one block, which is correct, and the ground truth had split
each paragraph into the 4-5 physical lines the renderer wrapped it onto. Every
one of the 12 text units matches at normalized similarity 1.0 and the 22
"omitted" lines are the wrapped lines of 5 paragraphs the model transcribed
verbatim. The A4 ground truth is therefore paragraph-level, because the body is
wrapped prose; the other seven pages draw one logical unit per physical line, so
for them the two granularities coincide. `_a4_paragraphs()` builds the units
from the same constants the page is drawn from, and
`tests/test_surya_transcription_fixtures.py` pins that a wrapped line is not its
own unit. This is the second defect of that shape found in this suite, after the
four bench pages that draw text past their own canvas.

The page is in the ground-truth and reading-order gates (`QUALITY_CASES`) but
not the implementation-parity gates, because no torch fp32 reference has been
captured for it.

### Numerical gate (2026-09-12)

The lane's arithmetic gate was exact greedy-id equality: all-or-nothing, and
unable to qualify a reassociation or report how much drift a route introduced.
`scripts/surya_numerical_gate.py` measures the project's declared production
envelope instead (`docs/EXECUTION-PROFILES.md` section 6): mean/p95/p99/max row
KL of a candidate against the teacher over teacher-forced full-vocabulary rows,
top-1 agreement globally and per page, and the BF16-relative comparison. The
teacher's own greedy chain is forced into every arm; vision features stay
arm-specific, because the vision-attention tiling is the arithmetic under test.

The 512 MiB default budget is a no-op at these page sizes (one tile covers all
queries), so the measurement forces the reassociation with a 12 MiB budget
(4-10 tiles per page, including partial tiles). Across 4562 teacher-forced rows
spanning all seven pages the tiled and dense paths are **bit-identical**:
mean/p95/p99/max KL `0.000e+00`, top-1 agreement `100%`, no row above the
`2e-2` review bar. The tiling partitions queries rather than keys and reduces
over the same full key range, so it is arithmetic-neutral rather than a
reassociation — a stronger statement than the earlier "agree within 1e-4".
Artifact: [numerical gate](benchmarks/results/2026-09-12-gfx1151-surya-numerical-gate.json).

The BF16-relative block of the same artifact is a **cross-implementation**
diagnostic, not the binding production-versus-strict comparison: the teacher is
torch fp32 and the candidates are hipEngine's GEMM routes, whose reduction
orders differ from torch's. Recorded for context (4562 rows):

| arm | mean KL | p95 | p99 | max | top-1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| hipEngine dense | 7.157e-04 | 1.947e-03 | 1.988e-01 | 7.706e-01 | 0.99868 |
| hipEngine tiled | 7.157e-04 | 1.947e-03 | 1.988e-01 | 7.706e-01 | 0.99868 |
| torch bf16 | 2.429e-04 | 1.114e-03 | 1.219e-02 | 3.386e-01 | 0.99605 |

The two hipEngine arms are identical to the last digit, confirming the tiling is
arithmetic-neutral end to end. The HIP route's top-1 is higher than bf16's while
its KL tail is larger, and its tail is concentrated on the degraded scan where
the teacher distribution is flat. Every row above the review bar records top-5
overlap, the teacher's probability and logit margin, and the flip flag.

Decode is one row per step, and rocBLAS SGEMM is tuned for a wide `n`: at
`n == 1` it reads the weights at roughly a third of the bandwidth SGEMV
reaches. `SuryaGpuRunner._gemm` therefore routes `rows == 1` to
`Rocblas.sgemv_rowmajor_nt`. Both accumulate in fp32, so only the summation
order changes; the greedy oracle gates are unaffected. End-to-end this took a
full OCR request from 2.051 s to 1.276 s on gfx1151 (decode 32.9 to 55.5
tok/s). The same fp32 `rows == 1` pattern exists in `runtime/evie.py`,
`runtime/timesfm_decode.py` and `runtime/timesfm3_decode.py`, which have not
been converted.

### Serving path and lifecycle (2026-09-12)

There is one implementation per backend, and both public entry points reach it
through the four-axis registry rather than through a parallel copy:

| entry point | path |
| --- | --- |
| `LLM("datalab-to/surya-ocr-2", backend=...)` + `generate_multimodal_detailed` | `LLM` → `resolve_text_generator(model="surya_ocr2", backend, quant="fp32")` → `SuryaOCRGenerator` / `SuryaOCRGeneratorGPU` |
| `run_surya_ocr(model_dir, image, ...)` | same registry resolution, CPU backend |

`run_surya_ocr` used to carry its own copy of preprocess → vision → prefill →
greedy loop and default to the ad-hoc `"Transcribe this page."` prompt. It now
defaults to `FULL_PAGE_HTML_PROMPT` and delegates to the registered
`(surya_ocr2, cpu_reference, fp32)` generator, so the public entry point and
`LLM(...)` share one capacity check, one stop-token rule, and one
cancellation/deadline path. It registers the built-in generators itself before
resolving, so it works from a fresh process that never constructed an `LLM`.
`prompt=None` means the checkpoint's full-page prompt; a caller driving a
different protocol passes it explicitly.

**Cancellation and deadlines are checked at stage boundaries, not only per
token.** A request that is already cancelled or past its `deadline_at` must not
pay for work it will throw away, so both generators check before preprocessing,
again immediately before the vision tower, and again before the text prefill;
`greedy_decode_tokens` keeps checking before every decode step. Previously the
only check was per generated token, so an abandoned request still ran the whole
vision tower and prefill first.

**Vision admission is memory *and* time (2026-09-13).** `check_vision_capacity`
runs three checks before the patch embed, because they fail for different
reasons: the declared score-tile bytes, the declared vision seconds
(`max_vision_seconds`), and free device memory. The byte budget alone admitted
a page nothing downstream could serve — `SURYA_MAX_PIXELS` (256x256, 65536
patches) fits the 512 MiB tile at 502 MB, but one forward there is 1.7e14
FLOPs, 93% of it bidirectional attention, and about 2.7 minutes on gfx1151 — so
the time budget is what actually bounds a page-scale grid.
`vision_forward_seconds` estimates a forward from the plan the runner will
execute: the linear GEMMs, the quadratic attention, and the per-tile key/value
re-read, which is why the estimate follows `max_vision_scratch_bytes` (the same
A4 page estimates at 43.4 s under the default budget and 195.7 s under an 8 MiB
one). It is calibrated against the retained tiling sweeps to -6%/+14% at
production shapes. The default budget is 120 s, which admits the 34320-patch A4
page at 43.4 s with 2.8x headroom and rejects the ceiling at 161.4 s, naming
the largest patch count that would fit. The same estimate rejects a page that
cannot finish inside the request's remaining `deadline_at` as
`GenerationDeadlineExceeded`; without it the deadline was only noticed at the
next stage boundary, after the client's timeout had already elapsed.

**Failure recovery is exact.** An abandoned request leaves the runner holding a
partially written KV cache and advanced conv/GDN state — and, because the
abandoned request is typically the longer one, more written KV slots than the
next request will attend over. `prefill` zeroes the conv and GDN state and
rewrites KV from slot 0, and decode attends only over `_seq_len + 1` slots, so
the follow-up request is bit-identical to the same request on a runner that
never saw the abandoned one. `tests/test_surya_gpu.py`
`test_gpu_generator_recovers_after_an_abandoned_request` gates the
long-then-short ordering, and also that an over-capacity rejection leaves the
runner usable.

**KV ABI closed (2026-09-13).** The KV write and decode kernels read the same
`(base_offsets, live_counts, token_positions, evict_mask)` `KVLiveSpans` ABI as
the rest of the tree. `surya_scatter_kv_f32_spans` maps each logical token
index through the page table and skips it when it is outside `live_counts`, has
a negative `token_positions` entry, or is marked in `evict_mask`;
`surya_full_attn_decode_f32_spans` is a fused fp32 GQA-4 split-K producer plus
its reduce, replacing the batched-SGEMM scores + scale + row-softmax + AV chain
and never materializing an `(nq, max_seq)` score row. The dense policy fills
every span field uniformly (identity page table, `arange` positions, empty
eviction mask), so the default path exercises the metadata rather than merely
declaring it, and the request's `live_counts`/`row_positions` move once per
decode step. The pre-spans `surya_scatter_kv_f32` and
`attention_decode_rocblas_f32` remain registered as the bisection oracle and the
strict fallback for a shape the fused kernel does not compile for. Measured on
gfx1151, the fused decode attention is 2.3-9.5x the rocBLAS parent over context
lengths 128-16384 with the same-fill output agreeing to 3.3e-07 absolute; the
ratio is monotone from 512 tokens up, while below that both routes are tens of
microseconds and the ratio moves by ~0.5x between runs
(`tests/test_surya_kv_spans.py`, `scripts/surya_kv_spans_bench.py`).

### HTTP serving (2026-09-12)

Surya is reachable through `hipserver`'s `/v1/chat/completions` multimodal
branch. That branch was written for Qwen4Exp and hardcoded three things that
made it unusable for a page-scale OCR model; each is now an engine declaration
resolved from the attached generator, so a new vision model declares its
bounds instead of editing the server.

| declaration | Surya value | why |
| --- | --- | --- |
| `vision_max_pixels` | `SURYA_MAX_PIXELS` (16.78 MP) | the checkpoint's own preprocessor ceiling; a 300-DPI A4 page is 8.7 MP and the old 1024 px per-side cap rejected every real page |
| `vision_media_input` | `"image_array"` | the Surya generator takes one RGB array, not the Qwen4Exp `{"items": [...]}` mapping |
| `vision_prompt_marker` | `""` | `render_chat_prompt` derives the image pad span from the patch grid, so the prompt is the bare text; Qwen4Exp needs an inline `<\|vision_start\|>...` marker |
| `vision_default_prompt` | `FULL_PAGE_HTML_PROMPT` | the text *is* the task, so an image-only request means the default task, not an empty prompt |

The compressed-payload bound scales with the admitted pixel area (16 MiB at
16.78 MP), because a 300-DPI document does not compress like a thumbnail. An
engine that declares nothing keeps the previous 1 MP / 1024 px / 8 MiB scope
exactly, which `tests/test_surya_server_multimodal.py` pins in both directions
using the committed A4 fixture. `--vision-max-pixels` and
`--vision-max-image-bytes` override either bound.

**The declarations live on the generator, not on the engine the server holds.**
The serving front end holds an `LLM`, which wraps the generator, so `LLM`
forwards `vision_max_pixels`, `vision_media_input`, `vision_prompt_marker`,
`vision_video_prompt_marker` and `vision_default_prompt` from it. It forwards
them *by raising `AttributeError`* when the generator declares nothing, so
`getattr(engine, name, default)` in a caller still sees the absence and applies
its own default — Qwen4Exp declares no bounds and must keep the caller's,
including its inline prompt marker. An earlier version of this path read the
declarations off `LLM`, got `None`, silently kept the 1 MP default, and rejected
every real A4 page; `supports_vision` was true throughout, which is why the
failure looked like a bound problem rather than a missing forward.

`usage.prompt_tokens` comes from the generator when it reports one, because a
vision prompt is text *plus* image tokens: the A4 request is 8701 tokens, of
which 121 are text and 8580 are image. `GenerationOutput.prompt_tokens` carries
it and the Surya generators set it from the prompt they built; the server falls
back to a tokenizer count for generators that report nothing.

**Verified end to end against a loaded checkpoint.**
`scripts/surya_http_e2e.py` serves one page through `TestClient`, scores it with
the same ground-truth scorer the direct path is gated by, and asserts equality
with the committed direct-call row rather than similarity: 2108 completion
tokens, 13 blocks, 12/12 paragraph units, CER 0.0000, 32/32 table cells, 0
reading-order violations, `stop`, 119.9 s wall against the direct call's 114.3 s
(artifact `benchmarks/results/2026-09-12-gfx1151-surya-http-serving-e2e.json`).

### Measured inventory (2026-09-11, revision `3b3d4cdf`)

Checkpoint downloaded and inventoried; the geometry table above matches the
actual config and tensors on every value checked.

- **Safetensors (canonical):** single BF16 `model.safetensors`, 1.37 GB,
  488 tensors, 686.2M parameters (language 565.1M + visual 100.6M + MTP
  20.5M; advertised "650M" excludes MTP). Tied LM head confirmed: no
  `lm_head` tensor; output uses `model.language_model.embed_tokens.weight`.
- **Weight namespaces:** `model.language_model.layers.N.*` (GDN layers use
  `linear_attn.in_proj_qkv/a/b/z`, `conv1d (6144,1,4)`, `A_log (16,)`,
  `dt_bias (16,)`, `norm (128,)`, `out_proj (1024,2048)`); full-attention
  layers use fused `q/k/v_proj` with per-head `q_norm`/`k_norm (256,)`;
  vision uses `model.visual.*` with fused `attn.qkv (2304,768)` **with
  biases** (EVIE has no vision biases), `patch_embed.proj (768,3,2,16,16)`,
  learned `pos_embed.weight (2304,768)`, `merger.norm (768,)` +
  `linear_fc1 (3072,3072)` / `linear_fc2 (1024,3072)` with biases.
- **MTP tensors present:** 15 tensors (`mtp.fc`, one full-attention layer,
  `pre_fc_norm_{embedding,hidden}`, `mtp.norm`); shared embeddings
  (`mtp_use_dedicated_embeddings: false`). Optional follow-up, as above.
- **Token IDs (trap 1 resolved):** tokenizer + generation_config agree —
  EOS **2** (`<|im_end|>`), pad 0, vision_start 9 / end 10, image_pad 11,
  video_pad 12, unk 65424. `text_config.eos_token_id: 248044` is stale
  metadata outside the 65,425 vocab; never used. Chat template renders
  `<|im_start|>assistant\n` with no thinking prefix, as documented.
- **GGUF pair (same repo family, `6a3a4c30`):** `surya-2.gguf` 1.27 GB +
  `surya-2-mmproj.gguf` 0.20 GB. **Unquantized** (file_type F16/F32 only),
  so a same-precision cross-oracle rather than a quant question. Metadata
  matches the safetensors config on all geometry keys
  (`qwen35.*`, `ssm.*`, `clip.*`, projector `qwen3vl_merger`).
  Anomalies: `tokenizer.ggml.tokens` holds **130,854** entries against
  65,425 `token_type` values (doubled vocab array — do not trust GGUF
  tokenizer contents; use the HF tokenizer files), `merges` is a stub
  `[1 x 8]`, and `general.finetune = '_patched_ckpt'` records a
  llama.cpp-conversion patch. EOS=2 / pad=0 confirmed in GGUF metadata.
- The safetensors path is the implementation target; GGUF is a cross-check
  oracle only.

Surya needs a vision-to-generation path, but most of its computation already
has close relatives here: EVIE supplies a Qwen3.5 vision encoder and GDN
prefill, the Qwen3.5 GGUF generator supplies autoregressive decoding, and
Qwen4Exp supplies multimodal serving and position-control examples. The work
is to adapt and connect these pieces with Surya's weights, tokenizer, image
processing, and OCR protocol. Existing family support does not establish that
this checkpoint already loads or generates correctly.

## Model and scope

**Checkpoint:** [datalab-to/surya-ocr-2][hf], advertised as 650M parameters,
BF16 safetensors. It is `Qwen3_5ForConditionalGeneration`: an image encoder
feeding a causal hybrid language decoder. Eighteen Gated DeltaNet layers are
normal for this Qwen3.5 configuration; they are not a new kernel class for
hipEngine. [Configuration][config]

**First target:** one page image → generated HTML with layout labels and
bounding boxes. Current upstream `RecognitionPredictor` chooses full-page
mode when no layout results are supplied. Layout JSON, block OCR, and table
structure are additional prompts to the same VLM. Failed full-page output
can fall back to VLM layout followed by block OCR. [Recognition source][recognition]

The separate text-line detector and RF-DETR `FastLayoutPredictor` are optional
adjacent capabilities, not prerequisites for this path. Porting the complete
Surya/Marker toolkit would be a larger task. The older TODO claim that two
auxiliary torch models block Surya OCR support does not describe the current
full-page path. [Fast-layout source][fastlayout]

The upstream code is Apache-2.0; the weights have the **modified AI Pubs
OpenRAIL-M** license. Keep the weight license distinct from the runtime's
license and preserve its notices for any distributed conversion. [Weight license][license]

### Geometry

Values below are from the pinned checkpoint config, rather than inferred
from the Qwen family name. Validate actual tensor names/shapes before loading.

| Component | Surya OCR 2 |
| --- | --- |
| Text width / layers / FFN | 1024 / 24 / 3584, SiLU gated MLP |
| Layer schedule | Three GDN then one full attention; 18 GDN + 6 full |
| Full attention | 8 query heads, 2 KV heads, head dim 256; sigmoid output gate |
| GDN | 16 key heads × 128, 16 value heads × 128; causal conv width 4 |
| Text normalization | RMSNorm, epsilon 1e-6 |
| Rotary | 64 rotated dimensions per 256-d head; theta 10,000,000; interleaved mRoPE sections `[11,11,10]` |
| Vocabulary / head | 65,425; tied input/output embeddings |
| Configured context ceiling | 262,144; not a validated OCR serving budget |
| Vision | 12 blocks, width 768, 12 heads × 64, FFN width 3072 |
| Patches | RGB, spatial 16×16, temporal 2, spatial merge 2×2 |
| Learned vision positions | 2304 entries = 48×48 table, interpolated to the patch grid |
| Vision activations | Block MLP: tanh-approximate GELU; merger: erf GELU |
| Merger | Per-patch LayerNorm → concatenate four patches → 3072→3072→1024 MLP |
| DeepStack | Empty visual-index list; no intermediate feature injection |
| MTP metadata | One next-token prediction layer declared; optional follow-up |

Sources: [checkpoint config][config] and [Transformers Qwen3.5 implementation][transformers].
MTP tensor availability and runner compatibility still need an inventory;
metadata alone does not establish usable speculative decoding.

### Forward path and positional semantics

```text
RGB page → resize/normalize → packed temporal/spatial patches
         → vision patch projection + interpolated learned positions
         → 12 bidirectional vision blocks with 2D rotary positions
         → 2×2 merger → 1024-d image features

Surya chat template → token IDs + expanded image placeholders
                   → token embeddings with image-feature substitution
                   → 24 causal hybrid decoder layers → tied LM head
                   → cached autoregressive decode → HTML/JSON → page results
```

At final processor dimensions H×W (multiples of 32), a still image produces
`P = (H/16)*(W/16)` vision rows and `P/4 = H*W/1024` decoder image tokens.
The temporal pair duplicates the still image; it does not double the decoder
token count. A **processor-output** size of 1024×1024 therefore gives 4096
vision rows and 1024 image tokens. This is geometry, not a measured workload.

Vision attention is bidirectional within each image; text attention is causal.
Decoder mRoPE coordinates and physical KV/token offsets are different objects.
Image coordinates compress spatial progress, so the next generated token must
use the processor's continuation offset while cache writes advance through
physical sequence slots. Do not replace mRoPE with scalar positions without a
separate production-quality evaluation. [Transformers implementation][transformers]

## Existing implementations and reuse

| Reference | What to use it for | Limitation |
| --- | --- | --- |
| Upstream Surya + vLLM | Exact task prompts, client preprocessing, parsing, retries, and full-page/block orchestration | External oracle/application; do not import its torch dependency tree into generation |
| Official GGUF + llama.cpp multimodal path | Same-artifact quantized decoder comparison and independent end-to-end image oracle | Requires both `surya-2.gguf` and `surya-2-mmproj.gguf`; inspect tensor types and metadata rather than guessing quant from filename |
| Transformers Qwen3.5 | FP32 teacher, BF16 reference, intermediate tensors, processor and cached-position semantics | Keep torch in fixture/oracle tools only; pin a working revision |
| MLX-VLM Qwen3.5 + community Surya conversion | Independent shape/weight adaptation cross-check; its vision module reuses Qwen3-VL | Apple-specific, not a ROCm performance baseline; conversion-specific quantization is not a correctness contract |

Reviewed [GGUF inventory][gguf], [Surya OpenAI client][client],
[llama.cpp backend wrapper][llamacpp], and [MLX conversion card][mlxcard].
The [MLX vision adapter][mlxvision] is a secondary implementation reference;
its moving `main` source must be commit-pinned before copying any code.
No external implementation was executed during this review.

### In-tree candidates

| Piece | Existing path | Adaptation needed |
| --- | --- | --- |
| Vision math and orchestration | `hipengine/runtime/evie.py`, `hipengine/kernels/cpu_reference/evie.py`, `hipengine/kernels/hip_gfx1100/evie/evie_ops.{hip,py}` | Extract shared configuration-driven encoder; Surya shapes and independent weight/spec contract |
| Model/weight validation pattern | `hipengine/models/evie.py`, `hipengine/loading/evie.py` | No EVIE retrieval projection or MaxSim; map Surya checkpoint namespaces and tied LM head |
| Cached text generation | `hipengine/models/qwen35.py`, `hipengine/loading/qwen35_gguf.py`, `hipengine/runtime/qwen35_gguf_runner.py`, `hipengine/generation/qwen35_gguf.py` | Validate small decoder geometry and quant coverage; add image embedding injection and three-axis rotary/continuation control |
| GGUF vision loading example | `hipengine/loading/qwen4_exp_vision_gguf.py`, `hipengine/loading/qwen4_exp_vision_materialize.py` | Surya mmproj tensor inventory; parameterize validated geometry |
| Multimodal control/serving | `hipengine/generation/qwen4_exp_multimodal.py`, `hipengine/server/multimodal.py` | Surya template/token IDs; capability-based integration and realistic page limits |
| Alternative encoder reference | `hipengine/runtime/qwen4_exp_vision.py` | Currently fixes width 1152 and FFN 4304; cannot directly load Surya |

EVIE already distinguishes tanh GELU in blocks from erf GELU in the merger.
Its recent 8B repair also derives merger width from `merge² * vision_width`,
not the block FFN width. Preserve that distinction even though both happen to
be 3072 for Surya. See the [8B repair handoff](../worklog/entries/20260910T113713.093689Z-lhl-evie-cbf1b8.md).

Prefer the existing GGUF decoder as the initial generation route if inventory
confirms coverage. Safetensors vision + GGUF text is a reasonable temporary
bridge if mmproj loading is slower to implement; record both source hashes
and remove duplicate loaders through `REFACTOR.md` once the final route works.
Do not turn EVIE's prefill-only retrieval runtime into a second independent
language-generation engine.

## Correctness traps to address first

1. **Special tokens and EOS disagree across files.** The tokenizer and
   `generation_config.json` identify EOS as ID **2** (`<|im_end|>`), padding as
   **0**, and image/start/end as **11/9/10**. `text_config.eos_token_id` is
   **248044**, outside this 65,425-token vocabulary. Resolve effective stopping
   from validated tokenizer/generation metadata (and GGUF equivalents), with a
   regression test; never carry stock Qwen token IDs into Surya. Use the supplied
   chat template, which ends at the assistant prefix without a thinking prefix.
   [Tokenizer][tokenizer], [generation config][generation], [chat template][template]
2. **There are two resizing stages in upstream serving.** Surya's client
   `scale_to_fit` uses Lanczos and a 28-pixel grid with default area bounds
   `1792*28` to `3072*2048`; the model processor subsequently uses patch 16 /
   merge 2, bicubic resampling, RGB rescale 1/255, mean/std 0.5. Processor size
   fields are pixel-area bounds (65,536–16,777,216), not literal side lengths.
   Capture both stages, patch packing order, temporal duplication, interpolation,
   and final `grid_thw`. A cleaner one-resize implementation would change inputs.
   [Client resizing][util], [processor configuration][processor]
3. **Embedding injection alone is insufficient.** Validate placeholder counts,
   feature order, mRoPE axes, decode offset, and per-request GDN convolution and
   recurrent state. Preserve state across prefill chunks; reset between requests.
   EVIE prefill parity is useful evidence, not a cached-decoding gate.
4. **Existing server limits are too small to assume page compatibility.** The
   bounded PNG helper defaults to 1024-pixel sides; the Qwen4Exp vision runner
   defaults to only 256 patch rows. Audit actual callers and resource budgets.
   Avoid silent image reduction or truncation. Use configured limits with clear
   errors and account for image tokens, prompt tokens, output, and concurrency.
5. **Upstream defaults contain a context-budget mismatch.** Pinned settings use
   a 12,288-token full-page output cap and 12,288 context tokens per llama.cpp
   slot; the comment still budgets an 8192-token output. Copy neither budget
   blindly: image and prompt tokens also occupy context. [Settings][settings]
6. **Task behavior is more than raw generation.** Preserve upstream prompt
   strings, normalized 0–1000 box coordinates, original-page coordinate mapping,
   reading order, label mapping, HTML/table structure, and errors. The client
   requests token logprobs and computes mean token probability. Guided layout
   defaults on; guided table decoding defaults off. Check hipEngine's schema
   subset against the actual layout schema, including the bbox string pattern.
   [Prompts/schema][prompts], [parsers][parsers], [client][client]

## Suggested implementation sequence

These were proposed milestones and paths. Steps 1–4 are implemented (see the
status table above); step 5 is partially implemented (public OCR API and
parsing, no OpenAI-compatible service path); step 6 is pending.

1. **Freeze contracts and generate oracle fixtures.** Add
   `hipengine/models/surya.py` and targeted contract tests; inventory safetensors
   and both GGUF files, including tied weights, token IDs, and optional MTP.
   Add `scripts/surya_oracle_torch.py` plus small fixtures under
   `tests/fixtures/surya/`. Capture identical processed inputs, vision boundary
   tensors, merged features, first logits, and teacher-forced cached steps.
2. **Prove the text checkpoint independently.** Exercise the existing Qwen3.5
   decoder with Surya tokenization and known text prefixes, including EOS and
   chunked prefill. This isolates weight/quant/head/state failures before images.
3. **Adapt the shared vision encoder.** Implement a Surya loader and a reusable
   encoder interface; expose merged features and grid metadata without EVIE's
   retrieval head. Validate tiny, rectangular, odd-input-size, and full-page
   grids against common processor outputs. Add CPU-reference coverage before
   any new kernel math. Re-run affected EVIE gates if shared code changes.
4. **Connect image prefill to generation.** Register the Surya model/generator
   through existing plugin interfaces; add image embedding overrides and mRoPE
   control to the reused runner. First acceptance: one image generates valid
   full-page OCR HTML through a torch-free public path, with correct cached state.
5. **Expose the OCR protocol.** Add parsing and test the OpenAI-compatible image
   request path. Upstream `SURYA_INFERENCE_URL` is a useful eventual integration
   point after model naming, images, schema support, and logprobs are verified.
   Add layout/table tasks and full-page→block fallback as explicit subsequent
   coverage. PDF rendering can stay at the application boundary.
6. **Qualify production and optimize.** Measure full-page costs first; only then
   choose batching, tiled vision attention, GEMM precision, or decode tuning.
   Auxiliary detector/fast-layout ports and MTP are independent follow-ups.

Keep kernels behind `(backend, layer, quant, variant)` registrations and retain
registered strict fallbacks. `KVLiveSpans` remains the attention ABI; mRoPE
coordinates do not replace cache ownership metadata. The Surya KV write and
decode kernels now read that ABI. The remaining deviation is the direct
`hip_gfx1100` JIT-library import, which the existing EVIE runtime also uses;
it is recorded in `docs/REFACTOR.md` with concrete removal conditions, and
migrating it is a repo-wide refactor, not a Surya-only change. Read
`KERNELS.md` and run
`python3 scripts/check_lineage.py --kind kernel --diff stat` before any kernel
port.

## Validation and performance plan

Use three independent layers of evidence:

- **Exact control:** token/template/patch counts, feature placement, box scaling,
  EOS, positions, cache ownership, GDN reset, cancellation, chunk boundaries,
  same-schedule repeats, and neighboring-request isolation. Include text-only,
  image-plus-text, mixed-size batches, empty/blank pages, and truncated outputs.
- **Numerics:** common-input CPU/FP32 oracle at patch, block, merger, logits,
  and cached-step boundaries. For new kernels the outer floor is KL ≤ 0.05
  and top-1 ≥ 90% where output distributions apply; continuous vision features
  also need calibrated feature-error gates and downstream-logit evaluation.
  Production promotion requires the complete mean/p95/p99/max KL, category
  top-1, determinism/isolation, BF16-relative and task gates in
  [EXECUTION-PROFILES.md](EXECUTION-PROFILES.md). Calibrate the OCR task envelope
  before candidate tuning. Non-bit-identical candidates get production review.
- **OCR quality:** a frozen multi-page corpus plus held-outs covering English,
  Japanese/mixed scripts, dense small text, scans, multi-column reading order,
  tables with spans, math, and blank pages. Report CER/WER where appropriate,
  layout/box and order accuracy, table structure, malformed output, omissions,
  loops, and truncation; add olmOCR-bench for broader task validation. Compare
  raw one-pass inference separately from retries/fallback-assisted results.

Benchmark the same image bytes, processor settings, prompts, output budgets,
quantized artifacts, concurrency, and retry policy on the **same physical host**.
Report cold load, preprocessing, vision, prefill/TTFT, decode, parsing, complete
page latency, pages/s, output tokens, peak memory, and quality.

Measured memory envelope (gfx1151, fp32, `scripts/surya_attention_memory.py`):
the runner holds 2663 MB of fp32 weights, `2 * nk * max_seq * hd * 4` bytes per
full-attention layer of KV planes (403 MB at `max_seq` 16384), 18.9 MB of GDN
recurrence state, 1.8 MB of conv windows, and 0.68 MB of resident spans state:
one split-K partial set of `nq * num_splits * (hd + 2) * 4` bytes (528 KB at
the 16384-token plan) plus the page table, `arange` position table, and empty
eviction mask (147 KB at `max_seq` 16384). Both attention score matrices are
tiled by query rows under a 512 MiB score-tile budget. The vision tower's is
`12 * n^2 * 4` bytes dense, so a 300-DPI A4 page (220x156 grid, 34320 patches)
needs 527 MB instead of the 56.5 GB a dense score matrix would, and the
`SURYA_MAX_PIXELS` ceiling (256x256, 65536 patches) needs 1.01 GB instead of
206 GB. The text prefill's causal scores are `8 * tokens^2 * 4` bytes dense, so
the page's 8580 image tokens need 70 MB instead of 2.36 GB and a full
16384-token prompt 268 MB instead of 8.59 GB; the measured prefill peak at
`max_seq` 16384 falls from 7.03 GB to 4.75 GB for the page and from 14.71 GB to
6.39 GB at a full prompt, with the remaining scratch linear at 180.8 KiB/token
across every shape. What the tiling costs is the shape envelope's, not the
tiling's: swept across query blocks at both lengths
(`benchmarks/results/2026-09-13-gfx1151-surya-text-prefill-shape-sweep.json`),
the widest tile the 512 MiB budget admits is the text optimum — 16.927 s at
16384 tokens against 17.436 s for the envelope's 512 rows (3.0%), and 7.535 s
at 8580 tokens against 7.692 s for its 256 rows (2.1%) — and both budget widths
are at or below the dense time (16.984 s and 7.619 s). The text curve is
monotone in tile count, because every tile still computes the full key range,
so fewer tiles is less per-tile work at unchanged arithmetic.
The generator admits
`max_seq` 16384 by default (upstream Surya budgets 12,288 context tokens per OCR
slot and 18,000 for vLLM), configurable through `LLM(max_sequence_length=...)`;
a 300-DPI A4 page's 8580 image tokens fit that context with room for a
full-page output, whereas the previous 2048 default rejected every real
document. That default is measured, not assumed
(`scripts/surya_attention_memory.py --max-seq 8192 12288 16384 20480 32768
--decode-steps 16`, artifact
`benchmarks/results/2026-09-13-gfx1151-surya-context-default.json`). The cost is
exactly 24.58 KiB per context token (KV planes 24.0 KiB/token over the six
full-attention layers plus ~9 B/token of span metadata), measured linear across
the five candidates, so 16384 is 403 MB of the 3087 MB resident and 32768 would
be 805 MB of 3490 MB. The reach a candidate buys is `max_seq` minus the
121-token prompt minus the output budget, in megapixels of vision grid:

| `max_seq` | resident | reach at a 4000-token output | reach at the A4's measured 2108 | vision time at the measured output |
| --- | --- | --- | --- | --- |
| 8192 | 2886 MB | 4.2 MP | 6.1 MP | 25 s |
| 12288 | 2986 MB | 8.4 MP | 10.3 MP | 57 s |
| **16384** | **3087 MB** | **12.6 MP** | **14.5 MP** | **119 s** |
| 20480 | 3188 MB | 16.8 MP | 18.7 MP | 161 s |
| 32768 | 3490 MB | 29.3 MP | 31.3 MP | 437 s |

Three measured facts fix the value. The lower bound is a gate that is actually
run: the transcription acceptance suite's A4 row declares a 4000-token output
budget and needs 121 + 8580 + 4000 = 12701 tokens, so 12288 — upstream's own
llama.cpp slot budget — cannot serve it, and 13312 is the smallest
1024-multiple that can, with 5% headroom. The upper bound is the vision time
budget: with the A4's measured output held fixed, 16384 reaches 14.49 MP
against the 120 s budget's 14.56 MP ceiling, so the two policy defaults agree to
0.5% and a larger context buys only pages costing more than 119 s of vision —
pages past the vision ceiling need `max_vision_seconds` raised first. The
measured workload has room: the A4 request uses 10809 of 16384 (66%) and its
output 2108 of the 7683 the default allows for that page, and every other
measured page needs at most 4321 tokens. Prefill is `max_seq`-independent (at
16384 tokens, 17.17/17.19/17.18 s across 16384/20480/32768 with the same score
tile and scratch), while decode is weakly not: the split chunk grows with
`max_seq` while the split count stays 64, so at a fixed live context 1024 tokens
decode in 18.31/18.36/18.88/18.96/19.55 ms — a further reason not to
over-provision the default. The two score tiles are configurable through
`LLM(vision_max_scratch_bytes=...)` and `LLM(prefill_max_scratch_bytes=...)`,
and the vision forward's wall-clock budget through
`LLM(vision_max_seconds=...)` (120 s by default, `math.inf` for unbounded); a
budget that cannot hold even one query row is a configuration-time rejection,
not a silent fallback to the quadratic matrix.

The budget is an upper bound on the tile, not the shape rule itself. `rows` is
the patch count for vision and the token count for text, and `plan_score_tiles`
caps the block at `max(128, ceil(rows / 32))` rows and rounds it down to a whole
number of 32-lane wavefronts, then takes the widest value that fits the budget.
The cap is 128 rows for every grid up to 4096 patches and grows with the grid
above that, so the budget stays the binding constraint at page scale: at 34320
patches the cap is 1073 rows and the 512 MiB budget still chooses the shape. Two
effects motivate that shape. Wide tiles waste the tile GEMMs where the dense
matrix is small — at 1024 patches the budget admits the whole dense score matrix
(193.54 ms) where a 128-row tile runs 135.52 ms, and at 4096 patches it admits
2730 rows (1123.42 ms) where 128 rows runs 874.38 ms — while narrow tiles make
every tile re-read the whole key range, which is what page-scale grids cannot
afford (on the A4 page 48 rows across 715 tiles takes 65983 ms against 44022 ms
for 512 rows across 68). The wavefront rounding is what the page-scale curve
actually separates on: every measured A4 width that is a multiple of 32 runs
43862-44668 ms and every width that is not runs 45050-48496 ms, and the 512 MiB
budget derives 325 rows, on the slow side of that line. The envelope is not
optimal at every grid and the misses are recorded rather than hidden: at 6400
patches a 96-row tile is 10% faster than the 192 rows the cap picks, the A4
page's own best measured width is 2048 rows (41622 ms, 5.1% below the default),
and the text prefill is a third miss in the opposite direction. Its curve is
monotone in tile count (fewer, wider tiles are always faster), so the cap costs
3.0% at 16384 tokens and 2.1% at 8580 while the widest tile the 512 MiB budget
admits is at or below the dense time — the shared planner wants a text-aware
rule, not the vision envelope, and `docs/REFACTOR.md` records it as open.

No Surya rate or memory target is established here. EVIE retrieval throughput
and upstream NVIDIA/Apple results are not hipEngine OCR baselines. Begin on
the declared ROCm host (default W7900/gfx1100); gfx1151 is an independent lane.
New kernels require guarded GPU tests and a prebuilt-cache `rocprofv3` trace.
Any accepted benchmark must update the result artifact, scoreboard, and
changelog under the repository's evidence policy.

## Source pins

Reviewed against hipEngine `d64eb6d10d0d31d762d6df7943776fe46804f652`.
External references were read on 2026-09-11; weights were not downloaded.

- HF safetensors/config/tokenizer: `3b3d4cdf88d6928b0acdc75181b13206ea67c4a3`.
- Official GGUF repository: `6a3a4c30e5e74446d4f8b6afd05b2f2da970f470`.
- Surya source: `c0377481c10527ec8c87bec5a10a6567fdf613fd`.
- Transformers source: tag `v5.2.0`, matching checkpoint metadata; oracle execution
  still needs a tested environment pin.
- MLX conversion card: `dd513812189b8d9bfbe76f32e84f7a496c31fac1`.

[hf]: https://huggingface.co/datalab-to/surya-ocr-2/tree/3b3d4cdf88d6928b0acdc75181b13206ea67c4a3
[config]: https://huggingface.co/datalab-to/surya-ocr-2/blob/3b3d4cdf88d6928b0acdc75181b13206ea67c4a3/config.json
[processor]: https://huggingface.co/datalab-to/surya-ocr-2/blob/3b3d4cdf88d6928b0acdc75181b13206ea67c4a3/processor_config.json
[tokenizer]: https://huggingface.co/datalab-to/surya-ocr-2/blob/3b3d4cdf88d6928b0acdc75181b13206ea67c4a3/tokenizer.json
[generation]: https://huggingface.co/datalab-to/surya-ocr-2/blob/3b3d4cdf88d6928b0acdc75181b13206ea67c4a3/generation_config.json
[template]: https://huggingface.co/datalab-to/surya-ocr-2/blob/3b3d4cdf88d6928b0acdc75181b13206ea67c4a3/chat_template.jinja
[license]: https://huggingface.co/datalab-to/surya-ocr-2/blob/3b3d4cdf88d6928b0acdc75181b13206ea67c4a3/LICENSE
[gguf]: https://huggingface.co/datalab-to/surya-ocr-2-gguf/tree/6a3a4c30e5e74446d4f8b6afd05b2f2da970f470
[transformers]: https://github.com/huggingface/transformers/blob/v5.2.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py
[recognition]: https://github.com/datalab-to/surya/blob/c0377481c10527ec8c87bec5a10a6567fdf613fd/surya/recognition/__init__.py
[fastlayout]: https://github.com/datalab-to/surya/blob/c0377481c10527ec8c87bec5a10a6567fdf613fd/surya/fast_layout/__init__.py
[client]: https://github.com/datalab-to/surya/blob/c0377481c10527ec8c87bec5a10a6567fdf613fd/surya/inference/backends/openai_client.py
[llamacpp]: https://github.com/datalab-to/surya/blob/c0377481c10527ec8c87bec5a10a6567fdf613fd/surya/inference/backends/llamacpp.py
[util]: https://github.com/datalab-to/surya/blob/c0377481c10527ec8c87bec5a10a6567fdf613fd/surya/inference/util.py
[settings]: https://github.com/datalab-to/surya/blob/c0377481c10527ec8c87bec5a10a6567fdf613fd/surya/settings.py
[prompts]: https://github.com/datalab-to/surya/blob/c0377481c10527ec8c87bec5a10a6567fdf613fd/surya/inference/prompts.py
[parsers]: https://github.com/datalab-to/surya/blob/c0377481c10527ec8c87bec5a10a6567fdf613fd/surya/inference/parsers.py
[mlxcard]: https://huggingface.co/aglaia-models/surya-ocr-2-mlx/blob/dd513812189b8d9bfbe76f32e84f7a496c31fac1/README.md
[mlxvision]: https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/qwen3_5/vision.py
