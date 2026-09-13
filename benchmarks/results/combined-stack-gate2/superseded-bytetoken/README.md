Superseded 2026-09-10 (review round 3): these category-gate runs fed raw
UTF-8 bytes as token IDs instead of tokenizing the fixture prompts, so they
never exercised the fixtures' tasks nor the >=129-row IQ4_XS fused prefill
path. The repaired gate (real tokenizer + chat template, incumbent-extended
512-token prompts, pooled binding predicate) lives in
scripts/gguf_ud_combined_stack_gate.py; its results are category-km-seed7.json
(FAIL) and category-ks-seed7.json (PASS).
