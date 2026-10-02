// Gemma 4 BF16 full-attention (head_dim 512) decode on gfx11, on the matrix cores.
//
// The same source as the prefill kernel, built with the query-block shape a
// decode wants. A decode block is one query tile against the whole context, so
// it is short of blocks and every one of them walks the entire band: at
// `kGqaHeads` 2 a KV band is read four times over, once per GQA tile, and there
// is no prefill work competing for the machine to make that worth anything.
// Eight GQA heads in one block reads each band once.
//
// That shape only fits because a decode needs one query row. `kColumns` is
// `kQueryRows * kGqaHeads` and the Q tile is staged in shared memory, so at
// sixteen rows and eight heads it is 130 KB against a 64 KB budget; at two rows
// it is 16.2 KB and the whole kernel is 32.5 KB. The second row is padding that
// brings the WMMA M tile to sixteen columns -- the kernel is bound by the K/V
// walk, not by the arithmetic, so a half-empty M tile costs nothing.
//
// Selection is in `gemma4_attention_prefill_wmma_full.py`, which builds this
// shape when the query block is small enough to be a decode.
//
// See `gemma4_attention_prefill_wmma_full.hip` for the arithmetic lineage, the
// mask contract and the split-walk workspace layout; all of it is shared.

#define GEMMA4_WMMA_FULL_QUERY_ROWS 2
#define GEMMA4_WMMA_FULL_GQA_HEADS 8
// Two column groups against four dim groups, so the same 128 threads the prefill
// shape runs. At the shared 2 this build is 64 threads, and the staging loop's
// own comment -- "128 threads cover two rows per iteration" -- is written for
// the 128 case: `kRowStep` is 2 there and 1 here.
#define GEMMA4_WMMA_FULL_DIM_GROUPS 4
#define GEMMA4_WMMA_FULL_SYMBOL hipengine_gemma4_attention_decode_wmma_full_bf16
#define GEMMA4_WMMA_FULL_KERNEL_NAME gemma4_attention_decode_wmma_full_kernel

#include "gemma4_attention_prefill_wmma_full.hip"
