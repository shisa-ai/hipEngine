# Long-context NextN proposal graphs

The greedy GGUF NextN proposal graph uses a per-depth attention schedule up to
resident cache capacity. Each depth matches eager split count and kernel choice;
page edges and policy transitions change the graph binding. Cached-only lookup
rejects a changed binding, and the ordinary host-result route captures its
replacement. Target verification can remain eager. Sampled proposals are unchanged.

## Protocol and result

Physical host `gfx1151`, Radeon 8060S / Ryzen AI MAX+ 395, Qwen3.8-27B
`UD-Q4_K_M`, production profile, BF16 KV, MTP budget 3, prefix cache off,
capacity 4608, one request and 48 greedy outputs with ignored EOS. Compiler,
model fingerprint, source hashes and profile manifest are in `summary.json`.
Baseline source is `92d785a4b`; the comparison disables proposal graph launches
only, leaving target verification and arithmetic unchanged.

Public `LLM.generate_detailed()` uses one resident owner in
eager/graph/graph/eager order with a discarded warmup before each arm. The ten
canonical category prompts and eight held-outs retain their task messages and
receive neutral context padding. Actual prompt lengths range from 970 to 4082;
the artifact lists every length. This includes requests crossing the split
threshold and page edges, as well as unrelated lengths.

| Proposal execution | Mean whole-request suite wall | Mean proposal time per suite |
| --- | ---: | ---: |
| Eager | 210.789 s | 5472.313 ms |
| Graph | 207.329 s | 4929.836 ms |

Whole-request wall is 1.64% lower; proposal time is 9.91% lower. Every category
improves. Output IDs, acceptance counts, selected depths and fallback controls
match across arms and repeats. Graph call records confirm execution, with no
capture errors. Closing the owner returns tracked allocations to zero.
This is a same-owner diagnostic, not a fresh-session speedup or MTP-versus-AR
claim. It does not resolve the separate fresh-owner acceptance variation.

## Validation and reproduction

The live oracle checks 81 proposal chains at positions 993–4116 through public
generation: graph and eager hidden-state bytes, top-1 IDs and top-1 values all
match. It restores the provider checkpoint between executions and reruns the
graph before returning to the caller. Both validation drivers return tracked
allocations to zero after close.

CPU schedule tests cover budgets 1/3/7, both sides of transitions, unrelated
lengths, a non-page-aligned policy threshold and changed kernel identity.
Cache tests cover reuse, retirement, cached-only rejection and cleanup failures.
Serving tests confirm proposal eligibility is independent of target graphs.
The CPU deterministic bundle, documentation gates and audit budget pass.
The independent review attempt failed at provider authentication; review was
performed directly instead. That review corrected schedule selection to use the
runner's adapted one-layer draft configuration. All published measurements and
the byte oracle were rerun after that correction; preliminary timings are not
included in these results.

The first smoke exposed a serving restriction that bypassed the graph launcher;
it is not graph-path evidence. A second public smoke confirmed long graph
launches with matching output/control results. The category comparison below
uses the reachable route. Drivers are preserved as executed; output paths are
under `/tmp/`. Use the existing environment wrapper from the repository root:

```bash
D=benchmarks/results/2026-11-12-gfx1151-nextn-long-graphs
P=benchmarks/results/2026-11-12-gfx1151-qwen38-mixed-pair-batch
bash "$P/run.sh" "$D/public_ab.py"
bash "$P/run.sh" "$D/live_oracle.py"
uv run pytest tests/test_unit_qwen35_gguf_nextn_attention_schedule.py tests/test_unit_qwen35_gguf_nextn_proposal_graph_cache.py tests/test_unit_qwen35_gguf_mtp2_seam.py tests/test_live_qwen35_gguf_nextn_provider.py -q
uv run pytest -q
python3 scripts/docs/check_all.py
python3 audit/audit.py check
```
