# Archived external-forks survey material

Raw evidence from the external-engine comparison survey (August-September
2026), archived from `~/.local/state/hipengine-external-survey/` before that
directory's deletion (the 106 GB of downloaded engine binaries under
`models/` and the `repos/` source/build trees were NOT archived — engine
commits are recorded in the artifacts; binaries are re-downloadable).

Contents:

- `results/` — raw survey captures: llama.cpp mainline Vulkan, q38rocm,
  laurent, nathan, mike, yandaq, kyanite, piebru, rocmfpx, and hipEngine
  head/q4-standard comparison logs and JSONs. These are the data sources
  cited by committed artifacts, e.g.
  `benchmarks/results/2026-08-29-parity-p1-protocol-attribution.json`
  (references `results/q4-standard/*.json` frozen rows and server_command
  fields) and the `2026-09-01-gfx1151-qwen38-z1/z2-laurent-*` artifacts
  (reference the survey paths and recorded server commands).
- `llama_q4_c1c8.py` — the standardized cross-engine survey harness named
  in those artifacts' `harness_command` fields.
- `campaign-plan.json` — the survey campaign plan.

Note: the original absolute paths under
`/home/lhl/.local/state/hipengine-external-survey/` recorded inside
committed artifacts now resolve to this archive (`results/` here is that
`results/` verbatim).
