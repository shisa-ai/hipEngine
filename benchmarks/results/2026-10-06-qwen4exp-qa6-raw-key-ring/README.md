# QA-6: QSA raw-index key ring audit

QA-6 proposes replacing the full per-token FP32 QSA raw-index key history with
an open-block-plus-verifier-width ring, while retaining the pooled block bank as
authoritative history. This packet audits the current in-tree owner and its
consumers. It does not implement or promote a ring.

## Current allocation

`Qwen4ExpQSAIndexDeviceState` allocates one FP32 `raw_keys` buffer per QSA
layer, with one 128-element key per configured token position. With 12 QSA
layers, the raw owner is 6,144 B/token: 1.50 GiB at native 262,144 context.
The complete index owner is larger because it also holds member indices, block
starts, pooled FP32 keys, scores, selection outputs and host mirrors. The exact
current formula in `hipengine/loading/qwen4_exp_materialize.py` gives
2,035,532,208 bytes (1.896 GiB) per request at 262,144 context before the
separate KV, recurrent-state and scratch owners.

The compact audit artifact is
[`audit.json`](audit.json). It records the arithmetic and source-level consumer
map without claiming a performance or capacity improvement.

## Consumer map

- `append`, `append_device_position` and `append_rows` write raw keys into the
  request's paged physical positions.
- `prepare_complete_blocks` reads raw keys only when newly completed compression
  blocks are closed, normalizes/RoPEs them and appends one pooled key per block.
  It short-circuits when `complete <= pooled_count`; the pooled bank is already
  incremental.
- Device scoring reads `pooled_keys`, not the raw-key history. Raw keys are not
  needed for ordinary scoring after their block has been pooled.
- `restore_count(position)` rewinds the logical cursor and sets
  `pooled_count = position // compression_ratio`; it does not copy or read raw
  key history. Target transactions and the MTP draft use this cursor rollback
  contract.
- The standalone `Qwen4ExpHostQSAIndexState.snapshot()` copies
  `raw_keys[:count]` to host memory. The production device transaction instead
  snapshots recurrent device state and restores attention/index cursors by
  position; the append-only KV/index payload is bounded by that cursor.

This is the same lifetime pattern that makes a ring plausible: once a complete
block has entered the pooled bank, its raw members are not required for normal
selection. It is not yet a proof that a ring is safe, because the current state
has no ring ownership, partial-block retention, or pooled-bank lifetime metadata.

## Missing qualification

A ring would change the state and snapshot ABI rather than merely remove
redundant work. Before implementation can be considered, the test matrix must
cover raw keys spanning a partial block, ring wrap, a block close that evicts
old raw members, rollback before and after block close, dirty restore
destinations, the live 2051/2052 dense-to-sparse boundary, c2 ownership and
full logits/KV/recurrent/index-state equality. The current tests cover
incremental pooled-block call counts and cursor rollback, plus host snapshot
round trips, but do not cover those ring-specific cases.

The current context-capacity evidence shows that native 262K operation is
served and that the raw-key saving would be material if memory becomes the
binding resource. It does not show that the resource currently binds, and it
does not license changing the representation.

## Decision

No code, default, allocator, snapshot format or benchmark claim changes. QA-6
remains gated after the memory/consumer audit. Reopen it only if the native
capacity ledger identifies the raw-key owner as a binding constraint or if a
separate capacity target requires the saving. Then build the ring as a declared
state/snapshot variant and run the full exactness, rollback, lifecycle and
capacity gate; do not infer safety from the existing pooled-key cache.

Source and test audit only; no GPU run was performed.
