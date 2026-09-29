# QA-9: pooled-key scorer audit - closes as already implemented

QA-9 asks for an audit, not an implementation: "Inspect the pinned scorer formula
and compare actual operands/selections with our existing pooled bank before
proposing code. If only redundant work disappears, test under `QA-4`; if selection
policy/representation changes, declare a separate configuration rather than
silently promoting it."

**Audit result: the cache is already implemented, incrementally and
rollback-safely, and the scorer formula is pinned and oracle-tested. "Add a cache"
is not new work.** The external variant is non-bit-exact and therefore falls under
the row's second branch: a separate declared configuration.

## The pooled bank already exists and caches incrementally

`Qwen4ExpQSAIndexDeviceState.append` carries the docstring *"Append only newly
completed normalized/RoPE index-key blocks"* and short-circuits when nothing new
completed:

```python
first = self.pooled_count
if complete <= first:
    return
count = complete - first
qwen4_exp_qsa_pool_norm_rope_f32(... self.pooled_keys.ptr + first * self.index_dim * 4 ...)
self.pooled_count = complete
```

Only `count = complete - first` blocks are pooled, written at `first * index_dim`
- an append, not a recompute.

## And it is tested, including the incrementality

`tests/test_gpu_qwen4_exp_runner_qsa.py` monkeypatches the pool kernel and asserts
the **call counts**, which is what makes this a proof rather than a reading:

| action | asserted |
|---|---|
| first select after 6 appends | `pooled_count == 3`, pooled counts `[3]` |
| append 1 more, select | counts still `[3]` - **nothing pooled** |
| append 1 more, select | `pooled_count == 4`, counts `[3, 1]` - **only the new block** |
| offsets of the second call | `member_indices.ptr + 3*2*4`, `pooled_keys.ptr + 3*4*4` |
| `restore_count(6)` | `pooled_count == 3` - **rollback works** |

So the bank is incremental, offset-correct, and rollback-safe, and the test fails
if any of those stop holding.

## The scorer formula is pinned and oracle-tested

`kernels/cpu_reference/qwen4_exp.py::qsa_index_scores` documents the contract -
*"Rectify each QSA index-head dot product before summing heads"*:

```python
dots = np.einsum("qhd,bd->qhb", query, keys, dtype=np.float32)
scores = np.sum(np.maximum(dots, 0.0), axis=1, dtype=np.float32)
return (scores / np.sqrt(index_dim)).astype(np.float32)
```

The GPU kernel `qsa_score_f32_kernel` matches it: per head, `dot` over `index_dim`,
then `value += fmaxf(dot, 0.0f)`, reduced across threads.

`tests/test_gpu_qwen4_exp_qsa_hip.py::test_qwen4_exp_qsa_native_pool_score_select_match_cpu`
compares the GPU path against `qsa_index_scores`, so the rectification order is
covered by an oracle rather than asserted by comment.

Operands: `pooled_keys` is `[blocks, index_dim]` - one pooled key per **block**,
not per token - which is the same granularity the external cache operates on.

## Disposition

- **"Add a cache"**: not new work. The bank exists, is incremental, is
  rollback-safe, and is tested.
- **The external variant**: the row records it as **non-bit-exact**. A
  non-bit-exact pooled-key path changes representation or arithmetic, so per the
  row's own instruction it must be a **separately declared configuration** - not
  silently promoted into the default path, and not gated by `QA-4` (which is the
  branch for work that only removes redundancy).

## What this audit did not do

- **The external report is not in-tree**, so its formula was **not** compared
  against ours. This audit establishes the state of *our* implementation and the
  disposition rule for the external one; it does not verify any claim about the
  external implementation's behaviour.
- **No rate was transferred**, per the row's explicit instruction, and none is
  reported here.
- No code was changed. This is an audit packet only.

## Limits

- Source and test reading, no new measurement. The call-count assertions are the
  strongest evidence available without instrumenting a live run.
- One host; the audit is arithmetic and control-flow, which is host-independent.
