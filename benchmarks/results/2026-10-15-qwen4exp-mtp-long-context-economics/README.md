# Qwen4Exp MTP long-context economics

A seven-prompt disjoint long-context suite was measured in one resident Qwen4Exp target process. Prompts were expanded to approximately 7.7K tokens, with `max_sequence_length=8192`, production profile, Q8_0 MTP sidecar, candidate budget 3, and 16 generated tokens. Each request has a true autoregressive arm and an MTP arm.

- AR: 2.495 tok/s over the suite.
- MTP: 0.856 tok/s.
- MTP/AR: 0.343x.
- Draft acceptance: 38.6%.
- 5/7 complete MTP output sequences matched their same-process AR controls; the two non-identical rows had six-token common prefixes.
- All seven requests used the `batched_state_rows` verifier path.
- 732 contemporaneous `rocm-smi` samples were recorded without errors; temperature ranged from 44 C to 84 C.

This establishes capacity and transaction execution beyond the prior 1K cap, but it is a negative economics result. Longer-context MTP is not promoted: the verifier/proposal overhead dominates at this context and gives 0.343x true AR. The artifact is diagnostic and does not turn the short-context 1.279x result into a longer-context claim.

Artifacts: `mtp-long-4k.json`, `telemetry.jsonl`.
