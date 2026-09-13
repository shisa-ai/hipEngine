"""Torch-free Surya OCR 2 HIP runtime (fp32 strict path, gfx1100/gfx1151).

Runs the Surya text decoder on the HIP device, mirroring the CPU reference
(``hipengine.kernels.cpu_reference.surya``) stage for stage:

- embedding lookup with sequential visual-feature injection at image-pad
  positions (``hipengine_evie_embed_lookup_f32``),
- 18 gated-DeltaNet layers: rocBLAS SGEMM projections, segment-aware fp32
  causal-conv prefill (SiLU inside the kernel, final window written to a
  persistent per-layer state slot), q/K l2-normalization, sigmoid/decay
  gate prep, the normalized cluster8 delta-rule recurrence (persistent
  per-layer state), RMSNormGated, out projection,
- 6 causal full-attention layers: per-head RMSNorm, per-head fused q|gate
  deinterleave, interleaved partial mRoPE (tables built on the host with
  the shared family helper), batched SGEMM scores + causal mask + softmax
  over query-row tiles, GQA-mapped AV product, sigmoid gate, o projection,
  persistent per-layer KV caches that decode continues from,
- SiLU MLPs, final RMSNorm, tied LM head over the embedding table.

The single-token decode step reuses the prefill recurrence kernel with
``tokens=1`` (algebraically identical to the CPU reference decode loop:
state decay, keyed memory read, delta write-back, query read-out) plus the
fp32 conv decode kernel. The correctness gate is greedy-token parity and
logit agreement against the CPU reference, not a loose statistical gate.

The vision tower runs on the same device: patch embed, half-split 2-axis
rotary, bidirectional full-image attention tiled by query rows, the tanh-GELU
MLP blocks, and the merger. Preprocessing (resize, normalize, patchify, the
mRoPE and position tables) stays on the host — it is data preparation, not
model math, and keeps the runtime torch-free.
"""

from __future__ import annotations

import ctypes
import math

import numpy as np

from hipengine.core.device import Device
from hipengine.core.hip import HipRuntime, get_hip_runtime
from hipengine.core.memory import (
    DeviceBuffer,
    copy_device_to_host,
    copy_host_to_device,
    host_array_ptr,
    malloc,
)
from hipengine.core.memory import free as hip_free
from hipengine.core.rocblas import Rocblas
from hipengine.core.tensor import Tensor
from hipengine.kernels.cpu_reference.evie import text_rope_tables
from hipengine.kernels.cpu_reference.surya import SuryaSpec, SuryaWeights
from hipengine.kernels.hip_gfx1100.evie.evie_ops import build_evie_ops
from hipengine.kernels.hip_gfx1100.linear_attn.conv import (
    build_qwen35_linear_attn_conv,
    qwen35_linear_attn_conv_decode_f32,
    qwen35_linear_attn_conv_prefill_segments_f32,
)
from hipengine.kernels.hip_gfx1100.linear_attn.gdn import (
    build_qwen35_linear_attn_gdn,
    qwen35_gdn_prefill_recurrent_f32,
    qwen35_gdn_prefill_recurrent_normalized_cluster8_f32,
)
from hipengine.kernels.hip_gfx1100.surya.surya_ops import (
    SURYA_DECODE_HEAD_DIM,
    SURYA_DECODE_Q_PER_KV,
    build_surya_ops,
    plan_surya_dense_spans,
    surya_causal_mask_scale_f32,
    surya_full_attn_decode_f32_spans,
    surya_gdn_l2norm_f32,
    surya_scatter_kv_f32_spans,
    surya_split_qgate_f32,
)
from hipengine.kvcache import KVLiveSpans

_P = ctypes.c_void_p
_F = ctypes.c_float
_I = ctypes.c_int64
_S = ctypes.c_void_p

_GEMM_PAD_BYTES = 512

# Vision attention is bidirectional over the whole page: every patch attends to
# every other patch, so the score matrix for `n` patches is `heads * n^2 * 4`
# bytes. Materializing it needs 56.5 GB for a 300-DPI A4 page (220x156 grid,
# 34320 patches) and 206 GB at the checkpoint's `max_pixels` ceiling (256x256,
# 65536 patches). The tower instead walks the score matrix in query-row tiles:
# each tile holds the full key range for `block` queries, so its softmax rows
# are exactly the rows the dense path computes, and only
# `heads * n * block * 4` bytes are live at once.
#
# This value is the per-request ceiling on that live tile. The block size is
# derived from it (`plan_vision_attention`), so the budget is honored rather
# than merely checked, and any grid the checkpoint admits fits: the tile never
# has to exceed one query row. Pass ``max_vision_scratch_bytes=None`` to run a
# single dense tile.
DEFAULT_MAX_VISION_SCRATCH_BYTES = 512 * 1024**2

# Text prefill has the same shape of problem: the causal score matrix for `t`
# tokens is `num_attention_heads * t^2 * 4` bytes, so a 300-DPI A4 page's 8580
# image tokens need 2.36 GB and a full 16384-token prompt 8.59 GB. The prefill
# walks it in query-row tiles the same way, so the two paths share one planner.
# The default matches the vision budget so there is a single number to reason
# about; pass ``max_prefill_scratch_bytes=None`` to run a single dense tile.
DEFAULT_MAX_PREFILL_SCRATCH_BYTES = 512 * 1024**2

# The tile's query block is a shape choice, not simply the widest tile the byte
# budget admits. When the budget alone would leave the grid in one or two query
# tiles, the block is capped by the measured envelope
# ``max(SHAPE_TILE_ROWS, ceil(rows / SHAPE_TILE_DIVISOR))`` and then rounded
# down to a whole number of wavefronts; the constants are calibrated with
# ``scripts/surya_vision_tiling_cost.py`` on gfx1151, as the median of 3-8
# timed forwards per shape, and the text prefill shares both the planner and
# that cap (``scripts/surya_attention_memory.py``) because the two paths tile
# the same matrix. The cap is a small-grid correction and not a global ceiling:
# see "When the cap applies" below, which is where the two sweeps disagree.
#
# Why cap at all: the budget alone takes the widest tile that fits, and where
# the dense matrix is small that is a bad shape. At 1024 patches the budget
# admits the whole dense matrix (193.54 ms) where a 128-row tile runs 135.52 ms,
# and at 4096 patches it picks a 2730-row tile (1123.42 ms) where 128 rows runs
# 874.38 ms. The tile GEMMs' `n` dimension is the block, so a tile far wider
# than the grid needs buys nothing back.
#
# Why the cap grows with the grid: every tile re-reads the whole key range, so
# the key/value re-read traffic and the launch count grow with the tile count.
# On the 34320-patch A4 page 715 tiles (48 rows) take 65983 ms, 269 tiles (128
# rows) 45639 ms, 68 tiles (512 rows) 43766 ms and 17 tiles (2048 rows)
# 41374 ms, so a fixed 128-row cap would cost 5.4% against what the 512 MiB
# budget can buy (108 tiles, 320 rows, 43318 ms) wherever it bound. The
# conditional rule below already keeps the cap off page-scale grids under the
# default budget -- at 34320 patches the budget's 325 rows leaves 108 tiles, so
# the cap is never consulted -- but a raised budget on a mid-size grid does
# enter the capped regime, and there the cap must not force a tile so narrow
# that the re-read traffic costs more than the shape saves. Capping at a 32nd
# of the grid admits 1073 rows at 34320 patches, which leaves the byte budget
# the binding constraint there: the 512 MiB default still chooses the shape, so
# raising the budget still buys a wider tile. Under the default budget the
# capped regime is rows <= 4729 (where the budget's block first exceeds half the
# grid), and rows/32 is 148 there, rounding down to the same 128 the floor
# gives, so the divisor only matters for a raised budget. The page's curve
# flattens above the budget rather than stopping: same-run 320 rows 43288 ms,
# 1024 41711, 2048 40618, 3072 40362, and 4096 40076 (steps of 1.43/0.68/0.16
# ms per MiB, then a 0.93% turn-up at 8192 rows), so the default sits 7.4% above
# the page's optimum for 1/12.8th of its scratch. SHAPE_TILE_ROWS is where the
# two bounds meet: the envelope is 128 rows for every grid up to 4096 patches,
# and rows/32 above it.
#
# Why the rounding: on the A4 page, in the two retained sweeps, every measured
# width that is a multiple of 32 lands in 43318-44831 ms (8 measurements of 256,
# 288, 320, 352, 512) and every width that is not lands in 46168-48496 ms (5
# measurements of 300, 325, 336), so every non-multiple is slower than every
# multiple in its own sweep and the means differ by 7.2%. The tile's `n`
# dimension is the block, so a width that is not a whole number of 32-lane
# wavefronts leaves the last wavefront partly idle. The 512 MiB budget derives
# 325 rows there, which is the slow side of that line; rounding down to 320
# takes it 47836.19 -> 43318.30 ms (1.104x) in the second sweep and
# 48496.26 -> 43862.87 ms (1.106x) in the first, and shrinks the tile
# 510.6 -> 502.7 MiB. Rounding down can only shrink the tile, so it cannot break
# the budget. The rounding is a tie-break, not a guarantee: an earlier
# cliff-mapping sweep recorded 400 rows at 45050 ms, within 1.1% of the slowest
# multiple that sweep measured (352 rows, 44568 ms), and at 4096 patches a
# 512-row tile is a multiple of 32 and is the slowest shape measured on that
# grid (1667 ms against 881 ms at 128 rows), with 192/256/341/1024 rows 24-50%
# slower than 128 there too. Where those spikes fall is not explained by width
# or by divisibility; see docs/REFACTOR.md.
#
# Grid divisibility was the other candidate shape rule and it
# was tested too: the sweep records `even` (full last tile) per row, and even
# division wins at 256 and 6400 patches and loses at 1024 and 4096, so it does
# not predict the curve.
#
# The envelope is not optimal at every grid, and the misses are recorded in
# ``docs/REFACTOR.md`` rather than hidden: at 6400 patches a 96-row tile (67
# tiles, 1940.93 ms) beats the envelope's 192 rows (2130.11 ms) by 10%, and the
# text prefill's curve is monotone in the other direction. Measured across
# query blocks at 8580 and 16384 tokens on gfx1151 (the sweep artifact
# ``benchmarks/results/2026-09-13-gfx1151-surya-text-prefill-shape-sweep.json``),
# every halving of the tile count is faster: at 16384 tokens 128/64/32/16/9/8
# tiles cost 17.799/17.611/17.397/16.927/16.853/16.808 s, so the envelope's 512
# rows cost 3.0% against the 1024 the 512 MiB budget admits, and at 8580 tokens
# its 256 rows cost 2.1% against 1955. Both budget widths are at or below the
# dense time, so on the text path the cap is pure loss: it buys 268 MB of a
# 6.4 GB peak for that 3.0%, and a user who needs the memory sets a smaller
# budget, which derives a smaller tile anyway. The batched score GEMMs are
# shape-invariant in arithmetic (each tile still computes the full key range,
# ``sum(bq) == tokens``), so the cost is in the GEMM shapes and the per-tile
# key/value re-read, not in extra FLOPs.
# It recovers 28.5-42.8% at the grids where the budget alone picks a bad shape.
#
# When the cap applies: only while the byte budget alone would leave the grid in
# ``SHAPE_TILE_CAP_MAX_TILES`` tiles or fewer. That is where a tile wider than
# the grid needs starves the batched GEMM of output tiles, and it is exactly
# what the sweeps separate on. At 1024 patches the budget admits the whole dense
# matrix (193.54 ms) against 135.52 ms at 128 rows, and at 4096 patches it
# admits 2730 rows (1123.42 ms) against 874.38 ms at 128 rows -- both one or two
# tiles. Once the budget leaves four or more tiles its own choice is at least as
# good as the cap: at 6400 patches it takes 1747 rows (4 tiles) for 2004.85 ms
# against the cap's 192 rows at 2125.18 ms, 5.7%; at the 34320-patch A4 page it
# takes 320 rows (43318 ms) where the cap's 1073 rows never binds; and on the
# text prefill, whose curve is monotone in the tile count, at 16384 tokens it
# takes 1024 rows (16.927 s) against the cap's 512 at 17.436 s, 3.0%, and at 8580
# tokens 1955 rows (7.535 s) against the cap's 256 at 7.692 s, 2.1%. Applying
# the cap below four tiles costs that 2.1-3.0% on the text path for nothing,
# which is the regression
# ``benchmarks/results/2026-09-13-gfx1151-surya-text-prefill-shape-sweep.json``
# was run to catch. Three tiles is not measured on either path; docs/REFACTOR.md
# records it.
SHAPE_TILE_ROWS = 128
SHAPE_TILE_DIVISOR = 32
SHAPE_TILE_MULTIPLE = 32
SHAPE_TILE_CAP_MAX_TILES = 2

# Text-decoder context the generator admits by default. Upstream Surya budgets
# 12,288 context tokens per OCR slot (image prefill + full-page output +
# chat-template overhead) and sizes its vLLM lane at 18,000. A 300-DPI A4 page
# alone is 8580 image tokens, so the previous 2048 default rejected every real
# document before the prompt or any output token.
#
# 16384 is measured rather than assumed: ``scripts/surya_attention_memory.py
# --max-seq 8192 12288 16384 20480 32768 --decode-steps 16`` on gfx1151
# (``benchmarks/results/2026-09-13-gfx1151-surya-context-default.json``), one
# fresh runner per row, three timed prefills and sixteen timed decode steps per
# row:
#
#   max_seq  KV planes  resident  chunk  reach@4000-tok  reach@2108-tok  vision
#      8192    201 MB   2886 MB    128         4.2 MP          6.1 MP    25 s
#     12288    302 MB   2986 MB    192         8.4 MP         10.3 MP    57 s
#     16384    403 MB   3087 MB    256        12.6 MP         14.5 MP   119 s
#     20480    503 MB   3188 MB    320        16.8 MP         18.7 MP   161 s
#     32768    805 MB   3490 MB    512        29.3 MP         31.3 MP   437 s
#
# The cost is 24.58 KiB per context token, measured exactly linear across those
# five candidates (KV planes 24.0 KiB/token over the six full-attention layers
# plus ~9 B/token of ``KVLiveSpans`` metadata), so 16384 is 403 MB of the
# 3087 MB resident. Reach is ``max_seq`` minus the 121-token prompt minus the
# output budget, and the two reach columns are the two defensible output models:
# a fixed 4000-token budget (the largest any retained gate row declares) and the
# largest measured page-scale output (2108 tokens, the A4 page's). A page's text
# does not grow with its resolution, so the fixed-output column is the physical
# one; the third column is where that page's vision forward lands in time.
#
# Three measured facts fix the default. First, the lower bound is a gate we run:
# the transcription acceptance A4 row declares a 4000-token output budget and
# needs 121 + 8580 + 4000 = 12701 tokens, so 12288 -- upstream's own llama.cpp
# slot budget -- cannot serve it, and 13312 is the smallest 1024-multiple that
# can, with 5% headroom. Second, the upper bound is the vision time budget: with
# the A4's measured output held fixed, 16384 reaches 14155 image tokens =
# 14.49 MP against the 120 s vision budget's 14.56 MP ceiling, so the two
# policy defaults agree to 0.5% and a larger context only buys pages that cost
# more than 119 s of vision -- pages past the vision ceiling need
# ``max_vision_seconds`` raised first, and ``max_sequence_length`` with it.
# Third, the measured workload has headroom: the A4 page's end-to-end request
# uses 10809 of 16384 (66%) and its output 2108 of the 7683 the default allows
# for that page, while every other measured page needs at most 4321 tokens
# (``page_long``: 1600 image tokens and a 2600-token budget).
#
# Prefill does not see ``max_seq`` at all (at 16384 tokens: 17.17/17.19/17.18 s
# across 16384/20480/32768, with the same 268.4 MB score tile and 3032.9 MB of
# non-attention scratch), but decode does, weakly: at a *fixed* live context the
# split chunk grows with ``max_seq`` (128/192/256/320/512) while the split count
# stays 64, so fewer blocks have work and 1024 live tokens decode in
# 18.31/18.36/18.88/18.96/19.55 ms. That is 3% from 8192 to 16384 and 4% more to
# 32768, i.e. a reason not to over-provision the default; the chunk is derived
# from ``max_seq`` rather than the live count, which ``docs/REFACTOR.md``
# records as a recoverable inefficiency.
DEFAULT_MAX_SEQ = 16384


def plan_score_tiles(
    rows: int, num_heads: int, budget_bytes: int | None
) -> tuple[int, int]:
    """Plan one query-row tile of an attention score matrix.

    Shared by the vision tower and the text prefill: both materialize
    ``num_heads * rows * rows`` fp32 scores unless the queries are tiled, and
    both attend over the full key range from every query row, so the tiles
    partition the queries and not the keys. ``rows`` is the patch count for
    vision and the token count for text.

    The tile is ``num_heads * rows * block`` fp32 elements — the full key range
    for ``block`` query rows. ``block`` is the largest value whose tile fits
    ``budget_bytes``, unless that would leave the grid in
    :data:`SHAPE_TILE_CAP_MAX_TILES` tiles or fewer, in which case it is bounded
    by the measured shape envelope (:data:`SHAPE_TILE_ROWS` /
    :data:`SHAPE_TILE_DIVISOR`); either way it is rounded down to a whole number
    of wavefronts (:data:`SHAPE_TILE_MULTIPLE`). ``None`` disables the budget and
    the envelope and returns one tile covering every query. A budget too small
    for even one query row still returns ``block == 1`` so admission can report
    the shortfall instead of silently producing an unusable plan.
    """

    n = int(rows)
    heads = int(num_heads)
    if n <= 0 or heads <= 0:
        raise ValueError("attention tiling needs a positive row count and head count")
    per_row = heads * n * 4
    if budget_bytes is None:
        return n, per_row * n
    block = int(budget_bytes) // per_row
    # The cap is a small-grid correction, not a global ceiling: it applies only
    # while the budget's own plan would leave the grid in
    # ``SHAPE_TILE_CAP_MAX_TILES`` tiles or fewer, which is where a tile wider
    # than the grid needs starves the batched GEMM of output tiles. Where the
    # budget already splits the grid further, its own choice is measured to be
    # at least as good; see the block comment above the constants.
    if block * SHAPE_TILE_CAP_MAX_TILES >= n:
        envelope = max(SHAPE_TILE_ROWS, -(-n // SHAPE_TILE_DIVISOR))
        if block > envelope:
            block = envelope
    if block > SHAPE_TILE_MULTIPLE:
        block -= block % SHAPE_TILE_MULTIPLE
    if block > n:
        block = n
    if block < 1:
        block = 1
    return block, per_row * block


def plan_vision_attention(
    n_patches: int, num_heads: int, budget_bytes: int | None
) -> tuple[int, int]:
    """Vision-shaped name for :func:`plan_score_tiles`.

    Kept because the vision path, its tests, and its recorded evidence refer to
    the planner by this name; the arithmetic is shared with the text prefill.
    """

    return plan_score_tiles(n_patches, num_heads, budget_bytes)


# The byte budget bounds the score tile, not the time. At the checkpoint's
# ``max_pixels`` ceiling (a 256x256 patch grid, 65536 patches) one vision
# forward is 1.7e14 FLOPs, 93% of it the bidirectional attention, and about
# 2.7 minutes on gfx1151 — long enough that a memory-only admission let a page
# through that no caller could wait for. ``vision_forward_seconds`` estimates
# that cost from the plan the runner will execute, and ``max_vision_seconds``
# is the declared budget it is admitted against.
#
# Calibrated from ``scripts/surya_vision_tiling_cost.py`` on gfx1151 (Radeon
# 8060S, fp32), one discarded warm-up, median wall clock around
# ``vision_forward`` only, at the production shape envelope plus the A4 page's
# query-block sweep (``benchmarks/results/2026-09-13-gfx1151-surya-vision-
# tiling-cost.json`` and ``...-vision-tiling-a4-v2.json``):
#
#     patches  tiles   measured   estimate
#         256      2   28.07 ms   26.21 ms
#        1024      8  136.00 ms  134.07 ms
#        4096     32  881.59 ms 1004.23 ms
#        6400      4 2004.85 ms 1924.05 ms
#       34320    108 43318.30 ms 43378.91 ms  (300-DPI A4, 512 MiB budget)
#       34320    358 49602.21 ms 49015.97 ms  (same page, 96-row tiles)
#
# The 6400 row is the plan the conditional cap now picks (1747 rows, 4 tiles);
# the same sweep measured the old cap's 192 rows at 34 tiles as 2125.18 ms, so
# the rule change is worth 5.7% on that grid, and 1924.05 ms estimates it to
# -4.0%. The page's 1-tile dense row measures 1917.35 ms there, 4.6% faster
# than the new plan and 4x the scratch.
#
# i.e. -6%/+14% over the shapes the planner picks at the production grids, and
# -11%/+3% across the 33 A4 query-block rows the per-tile term is fitted to
# (the worst case is the raw 325-row budget, which is not a multiple of 32 and
# so is a shape the planner never picks). Because it is a plan model it follows
# the tile count: at 34320 patches the same page estimates at 43.4 s under the
# default 512 MiB budget and 195.7 s under an 8 MiB one, since every extra
# query tile re-reads the whole key range. Measured shapes the planner never
# picks fall outside that band, because a tile far wider than a small grid
# needs wastes the tile GEMMs (see the shape envelope above): the unbudgeted
# dense tile is 194.01 ms against a 129.36 ms estimate at 1024 patches, 1039.95
# against 920.80 ms at 4096, and a 512-row tile at 4096 patches is 1667.07 ms
# against 939.64 ms. All of those are under a second at these sizes, and
# admission only ever sees planner-chosen shapes.
#
# ``DEFAULT_MAX_VISION_SECONDS`` is a policy budget, not a hardware limit: it
# admits every page the benchmark suite measures (the largest, a 300-DPI A4
# page at 34320 patches, estimates at 43.4 s) with 2.8x headroom, and rejects
# the 65536-patch ``max_pixels`` ceiling at 161.4 s. Raise it for a bigger page
# or pass ``None`` to admit anything the byte budget allows.
VISION_LINEAR_FLOPS_PER_S = 1.91e12
VISION_ATTENTION_FLOPS_PER_S = 1.15e12
VISION_TILE_PATCH_SECONDS = 6.57e-7
DEFAULT_MAX_VISION_SECONDS = 120.0


def vision_flops(spec: SuryaSpec, n_patches: int) -> tuple[float, float]:
    """``(linear, attention)`` FLOPs for one vision forward at ``n_patches``.

    Pure geometry from the checkpoint's vision contract: the patch embed, the
    per-layer QKV/projection/MLP GEMMs, and the merger are linear in the patch
    count, and the bidirectional attention is ``4 * vh * depth`` per patch
    pair (QK^T plus AV, two FLOPs per multiply-add). At the 65536-patch
    ``max_pixels`` ceiling the attention term is 1.58e14 of 1.70e14 total.
    """

    n = int(n_patches)
    if n <= 0:
        raise ValueError("a vision FLOP count needs a positive patch count")
    vh = int(spec.vision_hidden_size)
    vi = int(spec.vision_intermediate_size)
    depth = int(spec.vision_depth)
    merged = n // (int(spec.vision_spatial_merge_size) ** 2)
    linear = (
        depth * (8 * vh * vh + 4 * vh * vi) * n  # 3*vh qkv, vh proj, 2*vh*vi mlp
        + 2 * vh * vh * n  # patch embed
        + 2 * (4 * vh) * (4 * vh) * merged  # merger fc1
        + 2 * (4 * vh) * int(spec.vision_out_hidden_size) * merged  # merger fc2
    )
    attention = 4 * vh * depth * n * n
    return float(linear), float(attention)


def vision_forward_seconds(spec: SuryaSpec, n_patches: int, tiles: int) -> float:
    """Estimated wall clock for one vision forward, from measured gfx1151 rates.

    Three measured terms: the linear GEMMs, the bidirectional attention (88-93%
    of the FLOPs at page scale), and the per-tile key/value re-read, which is
    why the estimate takes the tile count as well as the patch count. Monotone
    in both arguments, so a budget can be inverted for the largest admitted
    page. See the constants above for the calibration table and its error band.
    """

    n = int(n_patches)
    tile_count = int(tiles)
    if n <= 0 or tile_count <= 0:
        raise ValueError("a vision time estimate needs a positive patch and tile count")
    linear, attention = vision_flops(spec, n)
    return (
        linear / VISION_LINEAR_FLOPS_PER_S
        + attention / VISION_ATTENTION_FLOPS_PER_S
        + tile_count * n * VISION_TILE_PATCH_SECONDS
    )


class SuryaGpuRuntimeError(RuntimeError):
    pass


def _raw_buffer(ptr: int, nbytes: int) -> DeviceBuffer:
    """A non-owning view over an existing device pointer (for copies)."""

    return DeviceBuffer(ptr=ptr, nbytes=nbytes)


def attention_decode_rocblas_f32(
    rocblas,
    evie_library,
    runtime: HipRuntime,
    ptr_array,
    *,
    q_ptr: int,
    k_cache_ptr: int,
    v_cache_ptr: int,
    out_ptr: int,
    scores_ptr: int,
    total: int,
    num_q_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    max_seq: int,
) -> None:
    """Strict unfused single-query attention: batched SGEMM, scale, softmax, AV.

    The pre-``KVLiveSpans`` Surya decode path, kept as the registered fallback
    for :func:`surya_full_attn_decode_f32_spans` and as its parent-parity
    oracle.  ``scores`` is the ``(num_q_heads, max_seq)`` fp32 scratch row the
    four dispatches round-trip through.  ``ptr_array`` uploads (and may cache)
    an int64 device pointer array for a batched call.
    """

    repeat = num_q_heads // num_key_value_heads
    head_stride = max_seq
    plane = max_seq * head_dim
    k_planes = [k_cache_ptr + (h // repeat) * plane * 4 for h in range(num_q_heads)]
    q_rows = [q_ptr + h * head_dim * 4 for h in range(num_q_heads)]
    score_rows = [scores_ptr + h * head_stride * 4 for h in range(num_q_heads)]
    v_planes = [v_cache_ptr + (h // repeat) * plane * 4 for h in range(num_q_heads)]
    out_rows = [out_ptr + h * head_dim * 4 for h in range(num_q_heads)]

    rocblas.sgemm_batched(
        ptr_array(k_planes).ptr,
        ptr_array(q_rows).ptr,
        ptr_array(score_rows).ptr,
        batch=num_q_heads, m=total, n=1, k=head_dim,
        lda=head_dim, ldb=head_dim, ldc=head_stride,
        trans_a=True, trans_b=False,
    )
    scale = _fn_evie(evie_library, "hipengine_evie_scale_f32", [_P, _P, _F, _I, _S])
    _check_err(
        scale(_P(scores_ptr), _P(scores_ptr), _F(head_dim ** -0.5),
              _I(num_q_heads * head_stride), _S(0)),
        runtime,
        "surya decode scale",
    )
    softmax = _fn_evie(
        evie_library, "hipengine_evie_softmax_rows_f32", [_P, _I, _I, _I, _I, _S]
    )
    _check_err(
        softmax(_P(scores_ptr), _I(num_q_heads), _I(total), _I(1), _I(head_stride), _S(0)),
        runtime,
        "surya decode softmax",
    )
    rocblas.sgemm_batched(
        ptr_array(v_planes).ptr,
        ptr_array(score_rows).ptr,
        ptr_array(out_rows).ptr,
        batch=num_q_heads, m=head_dim, n=1, k=total,
        lda=head_dim, ldb=head_stride, ldc=num_q_heads * head_dim,
        trans_a=False, trans_b=False,
    )


def _fn_evie(library, symbol: str, argtypes: list):
    fn = getattr(library, symbol, None)
    if fn is None:
        raise SuryaGpuRuntimeError(f"missing symbol {symbol}")
    fn.argtypes = argtypes
    fn.restype = ctypes.c_int
    return fn


def _check_err(err: int, runtime: HipRuntime, what: str) -> None:
    if err != 0:
        raise SuryaGpuRuntimeError(f"{what} failed: {err}")


class SuryaGpuRunner:
    """Surya OCR 2 fp32 text decoder on the HIP device."""

    def __init__(
        self,
        weights: SuryaWeights | str,
        spec: SuryaSpec | None = None,
        *,
        max_seq: int = 2048,
        max_vision_scratch_bytes: int | None = DEFAULT_MAX_VISION_SCRATCH_BYTES,
        max_vision_seconds: float | None = DEFAULT_MAX_VISION_SECONDS,
        max_prefill_scratch_bytes: int | None = DEFAULT_MAX_PREFILL_SCRATCH_BYTES,
        rocblas: Rocblas | None = None,
        runtime: HipRuntime | None = None,
        library: ctypes.CDLL | None = None,
        conv_library: ctypes.CDLL | None = None,
        gdn_library: ctypes.CDLL | None = None,
    ):
        if isinstance(weights, str):
            weights = SuryaWeights.load(weights)
        if int(max_seq) <= 0:
            raise ValueError("max_seq must be positive")
        if max_vision_scratch_bytes is not None and int(max_vision_scratch_bytes) <= 0:
            raise ValueError("max_vision_scratch_bytes must be positive when set")
        if max_vision_seconds is not None and not float(max_vision_seconds) > 0:
            raise ValueError("max_vision_seconds must be positive when set")
        if max_prefill_scratch_bytes is not None and int(max_prefill_scratch_bytes) <= 0:
            raise ValueError("max_prefill_scratch_bytes must be positive when set")
        self.spec = spec or SuryaSpec()
        self.max_seq = int(max_seq)
        self.max_vision_scratch_bytes = (
            None
            if max_vision_scratch_bytes is None
            else int(max_vision_scratch_bytes)
        )
        self.max_vision_seconds = (
            None if max_vision_seconds is None else float(max_vision_seconds)
        )
        self.max_prefill_scratch_bytes = (
            None
            if max_prefill_scratch_bytes is None
            else int(max_prefill_scratch_bytes)
        )
        self.runtime = runtime or get_hip_runtime()
        self.rocblas = rocblas or Rocblas.load()
        self.rocblas.set_workspace(0, 0)
        self.library = library or build_evie_ops(load=True)
        self.conv_library = conv_library or build_qwen35_linear_attn_conv(load=True)
        self.gdn_library = gdn_library or build_qwen35_linear_attn_gdn(load=True)
        self.surya_library = build_surya_ops(load=True)
        self._w: dict[str, DeviceBuffer] = {}
        # every long-lived allocation, freed by close(); named for what it holds
        self._permanent_bufs: list[DeviceBuffer] = []

        s = self.spec
        self.n_gdn_layers = sum(1 for l in range(s.num_layers) if not s.is_full_attention(l))
        self.n_attn_layers = s.num_layers - self.n_gdn_layers
        # persistent per-layer device state
        self._conv_state: dict[int, DeviceBuffer] = {}
        self._gdn_state: dict[int, DeviceBuffer] = {}
        self._kv_cache: dict[int, tuple[DeviceBuffer, DeviceBuffer]] = {}
        self._upload_weights(weights)
        self._alloc_state()
        # scratch cache (per-key device buffers grown on demand)
        self._scratch: dict[str, DeviceBuffer] = {}
        # cached device pointer arrays (keyed by pointer tuple)
        self._ptr_array_bufs: dict[tuple, DeviceBuffer] = {}
        # persistent tiny staging buffers for per-step scalar/rope uploads
        self._ids_buf = self._permanent(8)
        self._pos_buf = self._permanent(8)
        self._cu_buf = self._permanent(16)
        self._state_idx_buf = self._permanent(8)
        self._rope_cos_buf = self._permanent(s.num_attention_heads * 0 + 256)
        self._rope_sin_buf = self._permanent(256)
        self._visual_buf: DeviceBuffer | None = None
        self._seq_len = 0

    # -- setup ------------------------------------------------------------------

    def _upload_weights(self, weights: SuryaWeights) -> None:
        s = self.spec
        needed: list[str] = ["model.language_model.embed_tokens.weight",
                             "model.language_model.norm.weight"]
        for layer in range(s.num_layers):
            lp = f"model.language_model.layers.{layer}."
            needed += [lp + "input_layernorm.weight", lp + "post_attention_layernorm.weight",
                       lp + "mlp.gate_proj.weight", lp + "mlp.up_proj.weight",
                       lp + "mlp.down_proj.weight"]
            if s.is_full_attention(layer):
                p = lp + "self_attn."
                needed += [p + n for n in ("q_proj.weight", "k_proj.weight", "v_proj.weight",
                                           "q_norm.weight", "k_norm.weight", "o_proj.weight")]
            else:
                p = lp + "linear_attn."
                needed += [p + n for n in ("in_proj_qkv.weight", "in_proj_z.weight",
                                           "in_proj_b.weight", "in_proj_a.weight",
                                           "conv1d.weight", "A_log", "dt_bias",
                                           "norm.weight", "out_proj.weight")]
        # vision tower + merger
        needed += ["model.visual.patch_embed.proj.weight", "model.visual.patch_embed.proj.bias",
                   "model.visual.pos_embed.weight",
                   "model.visual.merger.norm.weight", "model.visual.merger.norm.bias",
                   "model.visual.merger.linear_fc1.weight", "model.visual.merger.linear_fc1.bias",
                   "model.visual.merger.linear_fc2.weight", "model.visual.merger.linear_fc2.bias"]
        for i in range(s.vision_depth):
            vp = f"model.visual.blocks.{i}."
            needed += [vp + n for n in ("norm1.weight", "norm1.bias",
                                        "attn.qkv.weight", "attn.qkv.bias",
                                        "attn.proj.weight", "attn.proj.bias",
                                        "norm2.weight", "norm2.bias",
                                        "mlp.linear_fc1.weight", "mlp.linear_fc1.bias",
                                        "mlp.linear_fc2.weight", "mlp.linear_fc2.bias")]
        for name in needed:
            arr = np.ascontiguousarray(weights[name].astype(np.float32))
            if name.endswith("conv1d.weight"):
                arr = arr.reshape(-1, arr.shape[-1])  # (channels, 1, k) -> (channels, k)
            buf = malloc(arr.nbytes + _GEMM_PAD_BYTES)
            self._upload(buf, arr)
            self._w[name] = buf

    def _alloc_state(self) -> None:
        s = self.spec
        n_conv = 3 * s.gdn_num_value_heads * s.gdn_value_head_dim  # qkv channels
        for layer in range(s.num_layers):
            if not s.is_full_attention(layer):
                self._conv_state[layer] = self._permanent(
                    n_conv * s.gdn_conv_kernel * 4)
                self._gdn_state[layer] = self._permanent(
                    s.gdn_num_value_heads * s.gdn_key_head_dim * s.gdn_value_head_dim * 4)
            else:
                nk, hd = s.num_key_value_heads, s.head_dim
                self._kv_cache[layer] = (
                    self._permanent(nk * self.max_seq * hd * 4),
                    self._permanent(nk * self.max_seq * hd * 4),
                )
        self._alloc_span_state()

    def _alloc_span_state(self) -> None:
        """Dense uniform ``KVLiveSpans`` backing store for the KV path.

        The page table, absolute token positions, and eviction mask are static
        for a dense fill, so they are uploaded once; only ``live_counts`` and
        ``row_positions`` move per decode step.  The scalar host mirrors stay
        alive for the process lifetime so a per-step H2D does not need the
        synchronizing ``_upload`` path (which exists for transient temporaries).
        """

        s = self.spec
        self._span_plan = plan_surya_dense_spans(self.max_seq)
        plan = self._span_plan
        self._span_page_table = self._permanent(plan.page_table.nbytes)
        self._span_token_positions = self._permanent(plan.token_positions.nbytes)
        self._span_evict_mask = self._permanent(plan.evict_mask.nbytes)
        self._span_live_counts = self._permanent(8)
        self._span_row_positions = self._permanent(8)
        self._span_live_host = np.zeros(1, dtype=np.int64)
        self._span_row_host = np.zeros(1, dtype=np.int64)
        self._upload(self._span_page_table, plan.page_table, dtype=np.int32)
        self._upload(self._span_token_positions, plan.token_positions, dtype=np.int64)
        self._upload(self._span_evict_mask, plan.evict_mask, dtype=np.bool_)
        nq, hd = s.num_attention_heads, s.head_dim
        splits = plan.num_splits
        self._span_partial_out = self._permanent(nq * splits * hd * 4)
        self._span_partial_m = self._permanent(nq * splits * 4)
        self._span_partial_l = self._permanent(nq * splits * 4)
        # One immutable view per request: only the live-count/row-position device
        # values move, so rebuilding the dataclass per layer would be pure host
        # overhead on the decode path.
        self._span_view = KVLiveSpans.paged_dense(
            block_table=Tensor.from_handle(
                self._span_page_table.ptr, (plan.block_table_len,), "int32",
                Device("hip", 0)),
            live_counts=Tensor.from_handle(
                self._span_live_counts.ptr, (1,), "int64", Device("hip", 0)),
            token_positions=Tensor.from_handle(
                self._span_token_positions.ptr, (self.max_seq,), "int64",
                Device("hip", 0)),
            evict_mask=Tensor.from_handle(
                self._span_evict_mask.ptr, (self.max_seq,), "bool",
                Device("hip", 0)),
            row_positions=Tensor.from_handle(
                self._span_row_positions.ptr, (1,), "int64", Device("hip", 0)),
            capacity=self.max_seq,
            block_size=plan.block_size,
            storage_dtype="fp32",
        )

    def _set_span_extent(self, live_count: int, row_position: int) -> None:
        """Publish the current live-token count and query row for this step."""

        self._span_live_host[0] = int(live_count)
        self._span_row_host[0] = int(row_position)
        copy_host_to_device(
            self._span_live_counts, host_array_ptr(self._span_live_host), 8)
        copy_host_to_device(
            self._span_row_positions, host_array_ptr(self._span_row_host), 8)

    def _spans(self) -> KVLiveSpans:
        """The request's dense uniform spans over the persistent KV planes."""

        return self._span_view

    def _permanent(self, nbytes: int) -> DeviceBuffer:
        buf = malloc(nbytes + _GEMM_PAD_BYTES)
        self._permanent_bufs.append(buf)
        return buf

    def _buf(self, key: str, nbytes: int) -> DeviceBuffer:
        buf = self._scratch.get(key)
        if buf is None or buf.nbytes < nbytes + _GEMM_PAD_BYTES:
            if buf is not None:
                hip_free(buf)
            buf = malloc(nbytes + _GEMM_PAD_BYTES)
            self._scratch[key] = buf
        return buf

    # -- low-level helpers --------------------------------------------------------

    def _upload(self, buf: DeviceBuffer, arr: np.ndarray, dtype=np.float32) -> None:
        """H2D copy that pins the host source's lifetime across the transfer.

        On this stack the DMA of an unpinned-source hipMemcpy reads the host
        buffer after the call returns, so a same-statement temporary can be
        freed and recycled before the copy lands (observed as stale-heap
        garbage at the destination). Keep the source alive and synchronize
        before releasing it.
        """
        src = np.ascontiguousarray(arr, dtype=dtype)
        copy_host_to_device(buf, host_array_ptr(src), src.nbytes)
        self.runtime.device_synchronize()

    def _k(self, symbol: str, argtypes: list) -> ctypes._FuncPtr:
        fn = getattr(self.library, symbol, None)
        if fn is None:
            raise SuryaGpuRuntimeError(f"missing symbol {symbol}")
        fn.argtypes = argtypes
        fn.restype = ctypes.c_int
        return fn

    def _check(self, err: int, what: str) -> None:
        if err != 0:
            raise SuryaGpuRuntimeError(f"{what} failed: hip error {err}")

    def _gemm(self, x_ptr: int, w_ptr: int, out_ptr: int, rows: int, fin: int, fout: int) -> None:
        if rows == 1:
            # Decode is one row: rocBLAS SGEMM runs at ~128 GB/s there (it is
            # tuned for a wide n), while SGEMV reads the same weights at
            # ~316 GB/s. Measured 2.48x on a 3185x1152 fp32 weight on gfx1151.
            # Both accumulate in fp32; the only difference is summation order.
            self.rocblas.sgemv_rowmajor_nt(
                x_ptr, w_ptr, out_ptr, in_features=fin, out_features=fout
            )
            return
        self.rocblas.sgemm_rowmajor_nt(
            x_ptr, w_ptr, out_ptr, rows=rows, in_features=fin, out_features=fout
        )

    def _rmsnorm(self, x_ptr: int, w_ptr: int, out_ptr: int, rows: int, dim: int) -> None:
        # family convention (shared with the Surya CPU reference oracle):
        # out = x * rsqrt(mean(x^2) + eps) * (1 + w)
        err = self._k("hipengine_evie_rmsnorm_f32", [_P, _P, _P, _I, _I, _F, _S])(
            _P(x_ptr), _P(w_ptr), _P(out_ptr), _I(rows), _I(dim), _F(1e-6), _S(0)
        )
        self._check(err, "rmsnorm")

    def _add(self, x_ptr: int, y_ptr: int, out_ptr: int, n: int) -> None:
        err = self._k("hipengine_evie_add_f32", [_P, _P, _P, _I, _S])(
            _P(x_ptr), _P(y_ptr), _P(out_ptr), _I(n), _S(0)
        )
        self._check(err, "add")

    def _dev_ptr_array(self, ptrs: list[int]) -> DeviceBuffer:
        """Upload a host pointer list for rocBLAS batched GEMMs.

        Cached per unique pointer list so the A/B/C arrays of one call
        never alias (the EVIE runner made the same fix).
        """

        key = tuple(ptrs)
        cached = self._ptr_array_bufs.get(key)
        if cached is not None:
            return cached
        arr = np.array(ptrs, dtype=np.uint64)
        buf = malloc(arr.nbytes + _GEMM_PAD_BYTES)
        self._upload(buf, arr, dtype=np.uint64)
        self._ptr_array_bufs[key] = buf
        return buf

    def _h2d_i64(self, values: list[int], buf: DeviceBuffer) -> None:
        self._upload(buf, np.array(values, dtype=np.int64), dtype=np.int64)

    def _h2d_i32(self, values: list[int], buf: DeviceBuffer) -> None:
        self._upload(buf, np.array(values, dtype=np.int32), dtype=np.int32)

    def _rope_tables_device(self, positions: np.ndarray) -> tuple[DeviceBuffer, DeviceBuffer]:
        """Host-built interleaved partial mRoPE tables (shared family math)."""
        cos, sin = text_rope_tables(self.spec, positions)  # (seq, 64) each
        cos_buf = self._buf("rope_cos", cos.nbytes)
        sin_buf = self._buf("rope_sin", sin.nbytes)
        self._upload(cos_buf, cos)
        self._upload(sin_buf, sin)
        return cos_buf, sin_buf

    # -- attention ---------------------------------------------------------------

    def _attention_packed(self, q_ptr, k_ptr, v_ptr, out_ptr, tokens) -> None:
        """Causal prompt attention against the planar KV cache planes.

        q: (tokens, nq*hd); k/v caches: (nk, max_seq, hd) per-head planes.

        Tiled by query rows under ``max_prefill_scratch_bytes``: each tile holds
        the full key range for ``bq`` queries, so a softmax row is the same
        ``tokens``-long row the dense path computed and the causal mask stays
        absolute through ``query_offset``. The tile's scores are the col-major
        ``(tokens x bq)`` C of a batched SGEMM with ``ldc=tokens``, so tile h
        starts at ``h * tokens * bq``. ``block >= tokens`` reproduces the dense
        call exactly (one tile, ``bq == tokens``).
        """
        s = self.spec
        nq, nk, hd = s.num_attention_heads, s.num_key_value_heads, s.head_dim
        repeat = nq // nk
        q_row = nq * hd
        plane = self.max_seq * hd
        head_stride = tokens
        block = self.prefill_block(tokens)
        scores = self._text_scores(tokens, block)
        k_planes = [k_ptr + (h // repeat) * plane * 4 for h in range(nq)]
        v_planes = [v_ptr + (h // repeat) * plane * 4 for h in range(nq)]
        for start in range(0, tokens, block):
            bq = min(block, tokens - start)
            tile = tokens * bq
            # scores tile h is the col-major (tokens x bq) C of batch h with
            # ldc=head_stride, so tile h starts at h*tokens*bq (the mask and
            # softmax kernels index the same (heads, bq, head_stride) row-major
            # layout).
            self.rocblas.sgemm_batched(
                self._dev_ptr_array(k_planes).ptr,
                self._dev_ptr_array([q_ptr + h * hd * 4 + start * q_row * 4 for h in range(nq)]).ptr,
                self._dev_ptr_array([scores.ptr + h * tile * 4 for h in range(nq)]).ptr,
                batch=nq, m=tokens, n=bq, k=hd,
                lda=hd, ldb=q_row, ldc=head_stride,
                trans_a=True, trans_b=False,
            )
            err = self._fn_surya("hipengine_surya_causal_mask_scale_f32",
                                 [_P, _F, _I, _I, _I, _I, _S])(
                _P(scores.ptr), _F(hd ** -0.5), _I(nq), _I(bq), _I(head_stride),
                _I(start), _S(0))
            self._check(err, "causal mask")
            err = self._k("hipengine_evie_softmax_rows_f32", [_P, _I, _I, _I, _I, _S])(
                _P(scores.ptr), _I(nq * bq), _I(tokens), _I(bq), _I(tile), _S(0))
            self._check(err, "softmax")
            self.rocblas.sgemm_batched(
                self._dev_ptr_array(v_planes).ptr,
                self._dev_ptr_array([scores.ptr + h * tile * 4 for h in range(nq)]).ptr,
                self._dev_ptr_array([out_ptr + h * hd * 4 + start * q_row * 4 for h in range(nq)]).ptr,
                batch=nq, m=hd, n=bq, k=tokens,
                lda=hd, ldb=head_stride, ldc=q_row,
                trans_a=False, trans_b=False,
            )

    def _attention_decode(self, q_ptr, k_cache_ptr, v_cache_ptr, out_ptr, total) -> None:
        """Strict unfused decode fallback (batched SGEMM + row softmax).

        The default route is :func:`surya_full_attn_decode_f32_spans`; this
        parent path stays registered for shapes the fused kernel does not
        compile for (head_dim != 256, GQA repeat != 4) and as the bisection
        oracle.  It reads no span metadata, which is exactly why it is the
        fallback rather than the contract.
        """
        s = self.spec
        nq, nk, hd = s.num_attention_heads, s.num_key_value_heads, s.head_dim
        scores = self._buf("scores", nq * self.max_seq * 4)
        attention_decode_rocblas_f32(
            self.rocblas, self.library, self.runtime, self._dev_ptr_array,
            q_ptr=q_ptr, k_cache_ptr=k_cache_ptr, v_cache_ptr=v_cache_ptr,
            out_ptr=out_ptr, scores_ptr=scores.ptr, total=total,
            num_q_heads=nq, num_key_value_heads=nk, head_dim=hd,
            max_seq=self.max_seq,
        )

    def _attention_decode_spans(self, q_ptr, k_cache_ptr, v_cache_ptr, out_ptr) -> None:
        """Fused fp32 GQA decode over the request's ``KVLiveSpans``."""
        s = self.spec
        nq, nk, hd = s.num_attention_heads, s.num_key_value_heads, s.head_dim
        if hd != SURYA_DECODE_HEAD_DIM or nq != SURYA_DECODE_Q_PER_KV * nk:
            self._attention_decode(q_ptr, k_cache_ptr, v_cache_ptr, out_ptr,
                                   self._seq_len + 1)
            return
        surya_full_attn_decode_f32_spans(
            q_ptr, k_cache_ptr, v_cache_ptr, out_ptr,
            self._span_partial_out.ptr, self._span_partial_m.ptr,
            self._span_partial_l.ptr, self._spans(),
            self._span_plan.block_size, nq, nk, hd, hd ** -0.5,
            library=self.surya_library, runtime=self.runtime,
        )

    def _fn_surya(self, symbol: str, argtypes: list) -> ctypes._FuncPtr:
        fn = getattr(self.surya_library, symbol, None)
        if fn is None:
            raise SuryaGpuRuntimeError(f"missing symbol {symbol}")
        fn.argtypes = argtypes
        fn.restype = ctypes.c_int
        return fn

    # -- layers -------------------------------------------------------------------

    def _gdn_layer_prefill(self, layer: int, norm_ptr: int, out_ptr: int, tokens: int) -> None:
        s = self.spec
        h = s.hidden_size
        nv, hk, hv = s.gdn_num_value_heads, s.gdn_key_head_dim, s.gdn_value_head_dim
        hkv = s.gdn_num_key_heads
        channels = 3 * nv * hv
        p = f"model.language_model.layers.{layer}.linear_attn."
        qkv = self._buf("gdn_qkv", tokens * channels * 4)
        z = self._buf("gdn_z", tokens * nv * hv * 4)
        b_in = self._buf("gdn_b", tokens * nv * 4)
        a_in = self._buf("gdn_a", tokens * nv * 4)
        conv_out = self._buf("gdn_conv", tokens * channels * 4)
        q = self._buf("gdn_q", tokens * nv * hk * 4)
        k = self._buf("gdn_k", tokens * nv * hk * 4)
        v = self._buf("gdn_v", tokens * nv * hv * 4)
        beta = self._buf("gdn_beta", tokens * nv * 4)
        decay = self._buf("gdn_decay", tokens * nv * 4)
        gdn_out = self._buf("gdn_out", tokens * nv * hv * 4)
        gdn_normed = self._buf("gdn_normed", tokens * nv * hv * 4)

        self._gemm(norm_ptr, self._w[p + "in_proj_qkv.weight"].ptr, qkv.ptr, tokens, h, channels)
        self._gemm(norm_ptr, self._w[p + "in_proj_z.weight"].ptr, z.ptr, tokens, h, nv * hv)
        self._gemm(norm_ptr, self._w[p + "in_proj_b.weight"].ptr, b_in.ptr, tokens, h, nv)
        self._gemm(norm_ptr, self._w[p + "in_proj_a.weight"].ptr, a_in.ptr, tokens, h, nv)

        conv_state = self._conv_state[layer]
        # Zero the state slot, then the segment-aware prefill writes the
        # final (channels, k) window into slot 0 for the decode step. This has
        # to be a real device memset: a scale-by-zero kernel (``x * 0.0``) does
        # not clear a NaN or an Inf left in recycled device memory
        # (``NaN * 0 == NaN``), and the recurrent state then propagates it into
        # every logit. A fresh process only worked because hipMalloc hands back
        # zeroed pages for allocations this size.
        self.runtime.memset(conv_state.ptr, 0, conv_state.nbytes)
        # cu_seqlens is int32 in the conv kernel ABI; state_indices is int64
        self._h2d_i32([0, tokens], self._cu_buf)
        self._h2d_i64([0], self._state_idx_buf)
        qwen35_linear_attn_conv_prefill_segments_f32(
            qkv.ptr, conv_state.ptr, self._w[p + "conv1d.weight"].ptr, conv_out.ptr,
            self._cu_buf.ptr, self._state_idx_buf.ptr, tokens, 1, channels,
            s.gdn_conv_kernel, stream=0, library=self.conv_library, runtime=self.runtime,
        )
        # q/K l2-normalization with the 1/sqrt(d_k) query scale applied on
        # the input (plain kernel recurrence consumes q as-is); plain
        # (tokens, heads, dim) output rows
        surya_gdn_l2norm_f32(conv_out.ptr, q.ptr, k.ptr,
                             1.0 / math.sqrt(hk), tokens, hkv, hk, channels,
                             nv * hv, library=self.surya_library,
                             runtime=self.runtime)
        err = self._k("hipengine_evie_gdn_gates_f32", [_P, _P, _P, _P, _P, _P, _I, _I, _S])(
            _P(b_in.ptr), _P(a_in.ptr), _P(self._w[p + "A_log"].ptr),
            _P(self._w[p + "dt_bias"].ptr), _P(beta.ptr), _P(decay.ptr),
            _I(tokens), _I(nv), _S(0))
        self._check(err, "gdn gates")
        # v plane sits at offset 2*key_dim in each conv row (q|k|v)
        err = self._k("hipengine_evie_expand_heads_f32",
                      [_P, _P, _I, _I, _I, _I, _I, _I, _S])(
            _P(conv_out.ptr), _P(v.ptr), _I(tokens), _I(2 * hkv * hk), _I(channels),
            _I(nv), _I(hv), _I(1), _S(0))
        self._check(err, "gdn v expand")
        gdn_state = self._gdn_state[layer]
        # Same reason as the conv state above: ``x * 0.0`` cannot clear a NaN.
        self.runtime.memset(gdn_state.ptr, 0, gdn_state.nbytes)
        qwen35_gdn_prefill_recurrent_f32(
            q.ptr, k.ptr, v.ptr, beta.ptr, decay.ptr, gdn_state.ptr, gdn_out.ptr,
            tokens, nv, hk, hv, stream=0, library=self.gdn_library, runtime=self.runtime,
        )
        err = self._k("hipengine_evie_gdn_rmsnorm_gate_f32",
                      [_P, _P, _P, _P, _I, _I, _F, _S])(
            _P(gdn_out.ptr), _P(z.ptr), _P(self._w[p + "norm.weight"].ptr),
            _P(gdn_normed.ptr), _I(tokens * nv), _I(hv), _F(1e-6), _S(0))
        self._check(err, "gdn rmsnorm gate")
        self._gemm(gdn_normed.ptr, self._w[p + "out_proj.weight"].ptr, out_ptr, tokens, nv * hv, h)

    def _gdn_layer_decode(self, layer: int, norm_ptr: int, out_ptr: int) -> None:
        s = self.spec
        h = s.hidden_size
        nv, hk, hv = s.gdn_num_value_heads, s.gdn_key_head_dim, s.gdn_value_head_dim
        hkv = s.gdn_num_key_heads
        channels = 3 * nv * hv
        p = f"model.language_model.layers.{layer}.linear_attn."
        qkv = self._buf("dec_qkv", channels * 4)
        z = self._buf("dec_z", nv * hv * 4)
        b_in = self._buf("dec_b", nv * 4)
        a_in = self._buf("dec_a", nv * 4)
        conv_out = self._buf("dec_conv", channels * 4)
        q = self._buf("dec_q", nv * hk * 4)
        k = self._buf("dec_k", nv * hk * 4)
        v = self._buf("dec_v", nv * hv * 4)
        beta = self._buf("dec_beta", nv * 4)
        decay = self._buf("dec_decay", nv * 4)
        gdn_out = self._buf("dec_gdn_out", nv * hv * 4)
        gdn_normed = self._buf("dec_gdn_normed", nv * hv * 4)

        self._gemm(norm_ptr, self._w[p + "in_proj_qkv.weight"].ptr, qkv.ptr, 1, h, channels)
        self._gemm(norm_ptr, self._w[p + "in_proj_z.weight"].ptr, z.ptr, 1, h, nv * hv)
        self._gemm(norm_ptr, self._w[p + "in_proj_b.weight"].ptr, b_in.ptr, 1, h, nv)
        self._gemm(norm_ptr, self._w[p + "in_proj_a.weight"].ptr, a_in.ptr, 1, h, nv)
        qwen35_linear_attn_conv_decode_f32(
            qkv.ptr, self._conv_state[layer].ptr, self._w[p + "conv1d.weight"].ptr,
            conv_out.ptr, channels, s.gdn_conv_kernel,
            stream=0, library=self.conv_library, runtime=self.runtime,
        )
        err = self._k("hipengine_evie_gdn_l2norm_scale_f32",
                      [_P, _P, _P, _P, _F, _I, _I, _S])(
            _P(conv_out.ptr), _P(conv_out.ptr + nv * hv * 4), _P(q.ptr), _P(k.ptr),
            _F(1.0 / math.sqrt(hk)), _I(nv), _I(hk), _S(0))
        self._check(err, "gdn l2norm decode")
        err = self._k("hipengine_evie_gdn_gates_f32", [_P, _P, _P, _P, _P, _P, _I, _I, _S])(
            _P(b_in.ptr), _P(a_in.ptr), _P(self._w[p + "A_log"].ptr),
            _P(self._w[p + "dt_bias"].ptr), _P(beta.ptr), _P(decay.ptr),
            _I(1), _I(nv), _S(0))
        self._check(err, "gdn gates decode")
        err = self._k("hipengine_evie_expand_heads_f32",
                      [_P, _P, _I, _I, _I, _I, _I, _I, _S])(
            _P(conv_out.ptr), _P(v.ptr), _I(1), _I(2 * hkv * hk), _I(channels),
            _I(nv), _I(hv), _I(1), _S(0))
        self._check(err, "gdn v expand decode")
        qwen35_gdn_prefill_recurrent_f32(
            q.ptr, k.ptr, v.ptr, beta.ptr, decay.ptr, self._gdn_state[layer].ptr,
            gdn_out.ptr, 1, nv, hk, hv, stream=0, library=self.gdn_library,
            runtime=self.runtime,
        )
        err = self._k("hipengine_evie_gdn_rmsnorm_gate_f32",
                      [_P, _P, _P, _P, _I, _I, _F, _S])(
            _P(gdn_out.ptr), _P(z.ptr), _P(self._w[p + "norm.weight"].ptr),
            _P(gdn_normed.ptr), _I(nv), _I(hv), _F(1e-6), _S(0))
        self._check(err, "gdn rmsnorm gate decode")
        self._gemm(gdn_normed.ptr, self._w[p + "out_proj.weight"].ptr, out_ptr, 1, nv * hv, h)

    def _attn_layer_prefill(self, layer: int, norm_ptr: int, out_ptr: int, tokens: int,
                            cos_buf: DeviceBuffer, sin_buf: DeviceBuffer) -> None:
        s = self.spec
        h = s.hidden_size
        nq, nk, hd = s.num_attention_heads, s.num_key_value_heads, s.head_dim
        p = f"model.language_model.layers.{layer}.self_attn."
        qp = self._buf("attn_qp", tokens * nq * hd * 2 * 4)
        q = self._buf("attn_q", tokens * nq * hd * 4)
        gate = self._buf("attn_gate", tokens * nq * hd * 4)
        k = self._buf("attn_k", tokens * nk * hd * 4)
        v = self._buf("attn_v", tokens * nk * hd * 4)
        heads_out = self._buf("attn_heads_out", tokens * nq * hd * 4)

        self._gemm(norm_ptr, self._w[p + "q_proj.weight"].ptr, qp.ptr, tokens, h, nq * hd * 2)
        self._gemm(norm_ptr, self._w[p + "k_proj.weight"].ptr, k.ptr, tokens, h, nk * hd)
        self._gemm(norm_ptr, self._w[p + "v_proj.weight"].ptr, v.ptr, tokens, h, nk * hd)
        err = self._fn_surya("hipengine_surya_split_qgate_f32", [_P, _P, _P, _I, _I, _I, _S])(
            _P(qp.ptr), _P(q.ptr), _P(gate.ptr), _I(tokens), _I(nq), _I(hd), _S(0))
        self._check(err, "split qgate")
        self._rmsnorm(q.ptr, self._w[p + "q_norm.weight"].ptr, q.ptr, tokens * nq, hd)
        self._rmsnorm(k.ptr, self._w[p + "k_norm.weight"].ptr, k.ptr, tokens * nk, hd)
        err = self._k("hipengine_evie_rope_f32", [_P, _P, _P, _I, _I, _I, _I, _I, _S])(
            _P(q.ptr), _P(cos_buf.ptr), _P(sin_buf.ptr), _I(tokens), _I(nq),
            _I(hd), _I(int(hd * s.partial_rotary_factor)), _I(nq * hd), _S(0))
        self._check(err, "rope q")
        err = self._k("hipengine_evie_rope_f32", [_P, _P, _P, _I, _I, _I, _I, _I, _S])(
            _P(k.ptr), _P(cos_buf.ptr), _P(sin_buf.ptr), _I(tokens), _I(nk),
            _I(hd), _I(int(hd * s.partial_rotary_factor)), _I(nk * hd), _S(0))
        self._check(err, "rope k")
        # write k/v into the persistent cache planes through the span ABI, then
        # attend causally (prefill attention is not a decode/paged-write kernel,
        # so it keeps the dense plane view)
        k_cache, v_cache = self._kv_cache[layer]
        spans = self._spans()
        block_size = self._span_plan.block_size
        surya_scatter_kv_f32_spans(k.ptr, k_cache.ptr, spans, tokens, 0,
                                   block_size, nk, hd,
                                   library=self.surya_library, runtime=self.runtime)
        surya_scatter_kv_f32_spans(v.ptr, v_cache.ptr, spans, tokens, 0,
                                   block_size, nk, hd,
                                   library=self.surya_library, runtime=self.runtime)
        self._attention_packed(q.ptr, k_cache.ptr, v_cache.ptr, heads_out.ptr, tokens)
        err = self._k("hipengine_evie_sigmoid_mul_f32", [_P, _P, _I, _S])(
            _P(gate.ptr), _P(heads_out.ptr), _I(tokens * nq * hd), _S(0))
        self._check(err, "attn gate")
        self._gemm(heads_out.ptr, self._w[p + "o_proj.weight"].ptr, out_ptr, tokens, nq * hd, h)

    def _attn_layer_decode(self, layer: int, norm_ptr: int, out_ptr: int,
                           cos_buf: DeviceBuffer, sin_buf: DeviceBuffer) -> None:
        s = self.spec
        h = s.hidden_size
        nq, nk, hd = s.num_attention_heads, s.num_key_value_heads, s.head_dim
        p = f"model.language_model.layers.{layer}.self_attn."
        qp = self._buf("dec_attn_qp", nq * hd * 2 * 4)
        q = self._buf("dec_attn_q", nq * hd * 4)
        gate = self._buf("dec_attn_gate", nq * hd * 4)
        k = self._buf("dec_attn_k", nk * hd * 4)
        v = self._buf("dec_attn_v", nk * hd * 4)
        heads_out = self._buf("dec_attn_out", nq * hd * 4)

        self._gemm(norm_ptr, self._w[p + "q_proj.weight"].ptr, qp.ptr, 1, h, nq * hd * 2)
        self._gemm(norm_ptr, self._w[p + "k_proj.weight"].ptr, k.ptr, 1, h, nk * hd)
        self._gemm(norm_ptr, self._w[p + "v_proj.weight"].ptr, v.ptr, 1, h, nk * hd)
        err = self._fn_surya("hipengine_surya_split_qgate_f32", [_P, _P, _P, _I, _I, _I, _S])(
            _P(qp.ptr), _P(q.ptr), _P(gate.ptr), _I(1), _I(nq), _I(hd), _S(0))
        self._check(err, "split qgate decode")
        self._rmsnorm(q.ptr, self._w[p + "q_norm.weight"].ptr, q.ptr, nq, hd)
        self._rmsnorm(k.ptr, self._w[p + "k_norm.weight"].ptr, k.ptr, nk, hd)
        err = self._k("hipengine_evie_rope_f32", [_P, _P, _P, _I, _I, _I, _I, _I, _S])(
            _P(q.ptr), _P(cos_buf.ptr), _P(sin_buf.ptr), _I(1), _I(nq),
            _I(hd), _I(int(hd * s.partial_rotary_factor)), _I(nq * hd), _S(0))
        self._check(err, "rope q decode")
        err = self._k("hipengine_evie_rope_f32", [_P, _P, _P, _I, _I, _I, _I, _I, _S])(
            _P(k.ptr), _P(cos_buf.ptr), _P(sin_buf.ptr), _I(1), _I(nk),
            _I(hd), _I(int(hd * s.partial_rotary_factor)), _I(nk * hd), _S(0))
        self._check(err, "rope k decode")
        k_cache, v_cache = self._kv_cache[layer]
        pos = self._seq_len
        spans = self._spans()
        block_size = self._span_plan.block_size
        surya_scatter_kv_f32_spans(k.ptr, k_cache.ptr, spans, 1, pos,
                                   block_size, nk, hd,
                                   library=self.surya_library, runtime=self.runtime)
        surya_scatter_kv_f32_spans(v.ptr, v_cache.ptr, spans, 1, pos,
                                   block_size, nk, hd,
                                   library=self.surya_library, runtime=self.runtime)
        self._attention_decode_spans(q.ptr, k_cache.ptr, v_cache.ptr, heads_out.ptr)
        err = self._k("hipengine_evie_sigmoid_mul_f32", [_P, _P, _I, _S])(
            _P(gate.ptr), _P(heads_out.ptr), _I(nq * hd), _S(0))
        self._check(err, "attn gate decode")
        self._gemm(heads_out.ptr, self._w[p + "o_proj.weight"].ptr, out_ptr, 1, nq * hd, h)

    def _mlp(self, prefix: str, norm_ptr: int, out_ptr: int, tokens: int) -> None:
        s = self.spec
        inter = s.intermediate_size
        gate_p = self._buf("mlp_gate", tokens * inter * 4)
        up_p = self._buf("mlp_up", tokens * inter * 4)
        self._gemm(norm_ptr, self._w[prefix + "mlp.gate_proj.weight"].ptr, gate_p.ptr, tokens, s.hidden_size, inter)
        self._gemm(norm_ptr, self._w[prefix + "mlp.up_proj.weight"].ptr, up_p.ptr, tokens, s.hidden_size, inter)
        err = self._k("hipengine_evie_silu_mul_f32", [_P, _P, _I, _S])(
            _P(gate_p.ptr), _P(up_p.ptr), _I(tokens * inter), _S(0))
        self._check(err, "swiglu")
        self._gemm(gate_p.ptr, self._w[prefix + "mlp.down_proj.weight"].ptr, out_ptr, tokens, inter, s.hidden_size)

    # -- public API -----------------------------------------------------------------

    def _embed(self, input_ids: np.ndarray, visual_features: np.ndarray | None) -> DeviceBuffer:
        s = self.spec
        tokens = len(input_ids)
        ids = np.ascontiguousarray(input_ids, dtype=np.int64)
        ids_buf = self._buf("ids", ids.nbytes)
        self._upload(ids_buf, ids, dtype=np.int64)
        x = self._buf("x", tokens * s.hidden_size * 4)
        visual_ptr = 0
        image_token_id = -1
        if visual_features is not None:
            vf = np.ascontiguousarray(visual_features, dtype=np.float32)
            n_img = int((np.asarray(input_ids) == s.image_token_id).sum())
            if vf.ndim != 2 or vf.shape[0] != n_img or vf.shape[1] != s.hidden_size:
                raise SuryaGpuRuntimeError(
                    f"visual_features must be (n_image_tokens, hidden) = "
                    f"({n_img}, {s.hidden_size}); got {vf.shape}")
            if self._visual_buf is None or self._visual_buf.nbytes < vf.nbytes + _GEMM_PAD_BYTES:
                if self._visual_buf is not None:
                    hip_free(self._visual_buf)
                self._visual_buf = malloc(vf.nbytes + _GEMM_PAD_BYTES)
            self._upload(self._visual_buf, vf)
            visual_ptr = self._visual_buf.ptr
            image_token_id = s.image_token_id
        err = self._k("hipengine_evie_embed_lookup_f32", [_P, _P, _P, _P, _I, _I, _I, _S])(
            _P(ids_buf.ptr), _P(self._w["model.language_model.embed_tokens.weight"].ptr),
            _P(visual_ptr), _P(x.ptr), _I(tokens), _I(s.hidden_size), _I(image_token_id), _S(0))
        self._check(err, "embed lookup")
        return x

    def _decode_stack(self, x_ptr: int, tokens: int, cos_buf: DeviceBuffer,
                      sin_buf: DeviceBuffer, *, decode: bool) -> None:
        s = self.spec
        h = s.hidden_size
        norm = self._buf("norm", tokens * h * 4)
        attn_out = self._buf("attn_out", tokens * h * 4)
        for layer in range(s.num_layers):
            lp = f"model.language_model.layers.{layer}."
            self._rmsnorm(x_ptr, self._w[lp + "input_layernorm.weight"].ptr, norm.ptr, tokens, h)
            if s.is_full_attention(layer):
                if decode:
                    self._attn_layer_decode(layer, norm.ptr, attn_out.ptr, cos_buf, sin_buf)
                else:
                    self._attn_layer_prefill(layer, norm.ptr, attn_out.ptr, tokens, cos_buf, sin_buf)
            else:
                if decode:
                    self._gdn_layer_decode(layer, norm.ptr, attn_out.ptr)
                else:
                    self._gdn_layer_prefill(layer, norm.ptr, attn_out.ptr, tokens)
            self._add(x_ptr, attn_out.ptr, x_ptr, tokens * h)
            self._rmsnorm(x_ptr, self._w[lp + "post_attention_layernorm.weight"].ptr, norm.ptr, tokens, h)
            self._mlp(lp, norm.ptr, attn_out.ptr, tokens)
            self._add(x_ptr, attn_out.ptr, x_ptr, tokens * h)
        self._rmsnorm(x_ptr, self._w["model.language_model.norm.weight"].ptr, norm.ptr, tokens, h)
        self._final_norm_ptr = norm.ptr
        # row of the final-norm buffer the lm_head must read (last token)
        self._final_norm_row = norm.ptr + (tokens - 1) * h * 4

    def debug_read(self, ptr: int, n: int) -> np.ndarray:
        # sync first: on this stack a D2H hipMemcpy can complete its call
        # before the producing kernel does, returning stale destination bytes
        self.runtime.device_synchronize()
        out = np.empty(n, dtype=np.float32)
        copy_device_to_host(host_array_ptr(out), _raw_buffer(ptr, n * 4))
        return out

    # -- vision tower -------------------------------------------------------------

    def vision_forward(self, pixel_rows: np.ndarray, grid_thw) -> np.ndarray:
        """Surya vision tower on the HIP device; returns merged features.

        Mirrors ``kernels.cpu_reference.surya.vision_forward``: patch embed
        (conv-as-matmul) + bias, host bilinear position embed, full-dim
        half-split 2-axis rotary, bidirectional packed attention, tanh-GELU
        MLP blocks, then the merger (LayerNorm -> square fc1 -> erf GELU ->
        fc2). Preprocessing and the small pos-embed/rotary tables stay on
        the host; every tensor op runs on the device. Returns the merged
        features (n / merge^2, vision_out_hidden_size) as host fp32.

        Attention is the dense full-image bidirectional attention, evaluated in
        query-row tiles so the score matrix is never fully materialized; see
        ``_vision_attention_packed``.
        """
        from hipengine.kernels.cpu_reference.surya import (
            _merge_block_major_coords,
            _pixel_rows_to_patches,
            vision_pos_embed,
            vision_rotary,
        )

        # Admit before any device work: an over-budget page must be rejected
        # here, not after patch embed and a failed multi-GB allocation.
        self.check_vision_capacity(grid_thw)

        s = self.spec
        vh = s.vision_hidden_size
        nh, hd = s.vision_num_heads, s.vision_head_dim()
        merge = s.vision_spatial_merge_size
        inter = s.vision_intermediate_size

        patches = _pixel_rows_to_patches(pixel_rows, s)
        n = patches.shape[0]
        n_merged = n // (merge * merge)
        vis_inter = merge * merge * vh
        block = self.vision_block(grid_thw)

        x = self._buf("vis_x", n * vh * 4)
        norm = self._buf("vis_norm", n * vh * 4)
        qkv = self._buf("vis_qkv", n * 3 * vh * 4)
        attn = self._buf("vis_attn", n * vh * 4)
        out = self._buf("vis_out", n * vh * 4)
        mlp = self._buf("vis_mlp", n * inter * 4)
        merged_in = self._buf("vis_merged_in", n_merged * vis_inter * 4)
        merged = self._buf("vis_merged", n_merged * s.vision_out_hidden_size * 4)
        scores = self._vis_scores(n, nh, block)

        # patch embed: (n, ch*t*p*p) @ (ch*t*p*p, vh) + bias (separate C —
        # never alias the GEMM input)
        patches_flat = patches.reshape(n, -1)
        patches_buf = self._buf("vis_patches", patches_flat.nbytes)
        self._upload(patches_buf, patches_flat)
        self._gemm(patches_buf.ptr, self._w["model.visual.patch_embed.proj.weight"].ptr,
                   x.ptr, n, patches_flat.shape[1], vh)
        self._add_bias(x.ptr, self._w["model.visual.patch_embed.proj.bias"].ptr,
                       n * vh, vh)

        # position embed: host bilinear resample of the learned table
        coords = _merge_block_major_coords(grid_thw, merge)[:2]
        pos = vision_pos_embed(
            {"model.visual.pos_embed.weight": self._pos_embed_table()},
            s, list(grid_thw), coords,
        )
        self._upload(out, pos)
        self._add(x.ptr, out.ptr, x.ptr, n * vh)

        cos, sin = vision_rotary(s, *coords)
        cos_buf = self._buf("vis_cos", cos.nbytes)
        sin_buf = self._buf("vis_sin", sin.nbytes)
        self._upload(cos_buf, cos)
        self._upload(sin_buf, sin)

        scale = hd ** -0.5
        for i in range(s.vision_depth):
            p = f"model.visual.blocks.{i}."
            self._layernorm(x.ptr, self._w[p + "norm1.weight"].ptr,
                            self._w[p + "norm1.bias"].ptr, norm.ptr, n, vh)
            self._gemm(norm.ptr, self._w[p + "attn.qkv.weight"].ptr, qkv.ptr,
                       n, vh, 3 * vh)
            self._add_bias(qkv.ptr, self._w[p + "attn.qkv.bias"].ptr,
                           n * 3 * vh, 3 * vh)
            # full-dim half-split rotary on the q and k planes of packed qkv
            self._rope(qkv.ptr, cos_buf.ptr, sin_buf.ptr, n, nh, hd, hd, 3 * vh)
            self._rope(qkv.ptr + vh * 4, cos_buf.ptr, sin_buf.ptr, n, nh, hd, hd, 3 * vh)
            self._vision_attention_packed(
                qkv.ptr, qkv.ptr + vh * 4, qkv.ptr + 2 * vh * 4,
                attn.ptr, n, nh, hd, 3 * vh, scores.ptr, scale, block)
            self._gemm(attn.ptr, self._w[p + "attn.proj.weight"].ptr, out.ptr,
                       n, vh, vh)
            self._add_bias(out.ptr, self._w[p + "attn.proj.bias"].ptr, n * vh, vh)
            self._add(x.ptr, out.ptr, x.ptr, n * vh)

            self._layernorm(x.ptr, self._w[p + "norm2.weight"].ptr,
                            self._w[p + "norm2.bias"].ptr, norm.ptr, n, vh)
            self._gemm(norm.ptr, self._w[p + "mlp.linear_fc1.weight"].ptr,
                       mlp.ptr, n, vh, inter)
            self._add_bias(mlp.ptr, self._w[p + "mlp.linear_fc1.bias"].ptr,
                           n * inter, inter)
            self._gelu_tanh(mlp.ptr, mlp.ptr, n * inter)
            self._gemm(mlp.ptr, self._w[p + "mlp.linear_fc2.weight"].ptr,
                       out.ptr, n, inter, vh)
            self._add_bias(out.ptr, self._w[p + "mlp.linear_fc2.bias"].ptr,
                           n * vh, vh)
            self._add(x.ptr, out.ptr, x.ptr, n * vh)

        # merger: LayerNorm -> (n/4, merge^2*vh) -> fc1 -> erf GELU -> fc2
        self._layernorm(x.ptr, self._w["model.visual.merger.norm.weight"].ptr,
                        self._w["model.visual.merger.norm.bias"].ptr,
                        norm.ptr, n, vh)
        self._gemm(norm.ptr, self._w["model.visual.merger.linear_fc1.weight"].ptr,
                   merged_in.ptr, n_merged, vis_inter, vis_inter)
        self._add_bias(merged_in.ptr,
                       self._w["model.visual.merger.linear_fc1.bias"].ptr,
                       n_merged * vis_inter, vis_inter)
        self._gelu_erf(merged_in.ptr, merged_in.ptr, n_merged * vis_inter)
        self._gemm(merged_in.ptr,
                   self._w["model.visual.merger.linear_fc2.weight"].ptr,
                   merged.ptr, n_merged, vis_inter, s.vision_out_hidden_size)
        self._add_bias(merged.ptr,
                       self._w["model.visual.merger.linear_fc2.bias"].ptr,
                       n_merged * s.vision_out_hidden_size,
                       s.vision_out_hidden_size)

        result = np.empty(n_merged * s.vision_out_hidden_size, dtype=np.float32)
        self.runtime.device_synchronize()
        copy_device_to_host(host_array_ptr(result),
                            _raw_buffer(merged.ptr, result.nbytes))
        return result.reshape(n_merged, s.vision_out_hidden_size)

    def _pos_embed_table(self) -> np.ndarray:
        if getattr(self, "_pos_table_host", None) is None:
            pe = self.spec.vision_num_position_embeddings
            vh = self.spec.vision_hidden_size
            self._pos_table_host = self.debug_read(
                self._w["model.visual.pos_embed.weight"].ptr, pe * vh
            ).reshape(pe, vh)
        return self._pos_table_host

    @staticmethod
    def _vis_tile_elements(n: int, block: int) -> int:
        """Elements per head in one query-row tile of the score matrix."""

        return n * block

    def _vis_scores(self, n: int, heads: int, block: int) -> DeviceBuffer:
        """Score scratch for one query tile of a ``n``-patch grid."""

        tile = self._vis_tile_elements(n, block)
        return self._buf("vis_scores", heads * tile * 4)

    @staticmethod
    def _text_tile_elements(tokens: int, block: int) -> int:
        """Elements per head in one query-row tile of the causal scores."""

        return tokens * block

    def _text_scores(self, tokens: int, block: int) -> DeviceBuffer:
        """Score scratch for one query tile of a ``tokens``-long prefill."""

        tile = self._text_tile_elements(tokens, block)
        return self._buf("scores", self.spec.num_attention_heads * tile * 4)

    # -- text-prefill admission --------------------------------------------

    def prefill_block(self, tokens: int) -> int:
        """Query rows per causal score tile for a ``tokens``-long prefill."""

        return plan_score_tiles(
            tokens,
            self.spec.num_attention_heads,
            self.max_prefill_scratch_bytes,
        )[0]

    def prefill_scratch_bytes(self, tokens: int) -> int:
        """Peak causal score-tile bytes a ``tokens``-long prefill needs.

        Linear in the token count once the query block is bounded, and derived
        from the same plan the allocation uses so admission can never disagree
        with it. The dense path this replaced was quadratic
        (``nq * tokens^2 * 4``).
        """

        _, need = plan_score_tiles(
            tokens,
            self.spec.num_attention_heads,
            self.max_prefill_scratch_bytes,
        )
        return need

    def check_prefill_capacity(self, tokens: int) -> None:
        """Admit a prompt length before any device work or allocation runs.

        The same two independent checks as the vision tower: the configured
        budget bounds one request regardless of free memory, and free device
        memory catches weights plus other live runners having consumed it.
        """

        need = self.prefill_scratch_bytes(tokens)
        cap = self.max_prefill_scratch_bytes
        if cap is not None and need > cap:
            raise SuryaGpuRuntimeError(
                f"text prefill score tile for {tokens} tokens is "
                f"{need / 1e9:.2f} GB, above the {cap / 1e9:.2f} GB budget; "
                f"one query row needs "
                f"{self.spec.num_attention_heads * tokens * 4 / 1e6:.1f} MB, "
                f"so raise SuryaGpuRunner(max_prefill_scratch_bytes=...) to at "
                f"least that"
            )
        # the scratch buffer is cached per key and reused when it is already big
        # enough, so only the growth is charged against free memory
        existing = self._scratch.get("scores")
        growth = max(0, need + _GEMM_PAD_BYTES - (existing.nbytes if existing else 0))
        if growth == 0:
            return
        try:
            free_bytes, _total = self.runtime.mem_get_info()
        except Exception:  # runtime without mem_get_info: budget is the only bound
            return
        if growth > int(free_bytes):
            raise SuryaGpuRuntimeError(
                f"text prefill score tile for {tokens} tokens needs "
                f"{growth / 1e9:.2f} GB more device memory but only "
                f"{int(free_bytes) / 1e9:.2f} GB is free"
            )

    # -- vision admission --------------------------------------------------

    def vision_block(self, grid_thw) -> int:
        """Query rows per score tile for ``grid_thw``, from the budget."""

        return plan_vision_attention(
            self._grid_patches(grid_thw),
            self.spec.vision_num_heads,
            self.max_vision_scratch_bytes,
        )[0]

    @staticmethod
    def _grid_patches(grid_thw) -> int:
        return int(grid_thw[0][1]) * int(grid_thw[0][2])

    def vision_scratch_bytes(self, grid_thw) -> int:
        """Peak score-tile bytes the vision attention needs for ``grid_thw``.

        Linear in patch count once the query block is bounded, and derived from
        the same plan the allocation uses so admission can never disagree with
        it. The dense path this replaced was quadratic
        (``heads * n^2 * 4``).
        """

        n = self._grid_patches(grid_thw)
        _, need = plan_vision_attention(
            n, self.spec.vision_num_heads, self.max_vision_scratch_bytes
        )
        return need

    def _vision_plan_seconds(self, n: int) -> tuple[float, int, int]:
        """``(seconds, tiles, block)`` estimated for a raw patch count."""

        block = plan_vision_attention(
            n, self.spec.vision_num_heads, self.max_vision_scratch_bytes
        )[0]
        tiles = -(-int(n) // block)
        return vision_forward_seconds(self.spec, n, tiles), tiles, block

    def vision_time_seconds(self, grid_thw) -> float:
        """Estimated wall clock for the vision forward of ``grid_thw``.

        The plan the runner will execute, so the estimate follows
        ``max_vision_scratch_bytes``: a narrower tile means more query tiles and
        more key/value re-reads. See :func:`vision_forward_seconds`.
        """

        return self._vision_plan_seconds(self._grid_patches(grid_thw))[0]

    def vision_patch_ceiling(self, seconds: float | None = None) -> int | None:
        """Largest patch count whose estimated vision forward fits ``seconds``.

        ``seconds=None`` uses ``max_vision_seconds``, and an unbounded budget
        has no ceiling (``None``). Bisection over the estimate, re-planning the
        tile at every candidate so the answer accounts for the tile count the
        byte budget would pick. The returned count is always sound: its own
        estimate fits, because the search only raises ``low`` on a candidate
        that fits. The estimate is not monotone in the patch count, though: at
        the default budget the shape cap stops applying at 4730 patches (the
        budget's own block first exceeds half the grid there), the plan drops
        from 37 tiles of 128 rows to 3 of 2336, and the estimate falls 8.3%
        (1270.91 -> 1165.67 ms) even as the grid grows. So for a time budget
        inside that dip the search can stop short of the largest count that
        fits; it matches a brute-force scan at the 120 s default (56856
        patches) and at every budget tried from 0.5 s to 200 s except the dip
        window itself (1.20 s: 4565 against the true 4815). Admission does not
        use this: ``check_vision_capacity`` compares the estimate of the grid
        it was given, which the dip cannot make unsound.
        """

        limit = self.max_vision_seconds if seconds is None else float(seconds)
        if limit is None:
            return None
        if not limit > 0:
            return 0
        high = 1
        while self._vision_plan_seconds(high)[0] <= limit:
            if high >= 1 << 24:  # unreachable at any real budget; keeps it total
                return high
            high *= 2
        low = high // 2
        while low < high:
            mid = (low + high + 1) // 2
            if self._vision_plan_seconds(mid)[0] <= limit:
                low = mid
            else:
                high = mid - 1
        return low

    def check_vision_capacity(self, grid_thw) -> None:
        """Admit a vision grid before any device work or allocation runs.

        Three independent checks, because they fail for different reasons:

        - the configured byte budget bounds a single request regardless of how
          much memory the host happens to have;
        - the configured time budget bounds how long a single request may run,
          which is what actually limits a page-scale grid (65536 patches is
          1.7e14 FLOPs and about 2.7 minutes);
        - free device memory catches the case where the weights plus other
          live runners have already consumed the budget.

        Called before patch embed so an over-budget page costs nothing.
        """

        n = self._grid_patches(grid_thw)
        need = self.vision_scratch_bytes(grid_thw)
        grid = (int(grid_thw[0][0]), int(grid_thw[0][1]), int(grid_thw[0][2]))
        cap = self.max_vision_scratch_bytes
        if cap is not None and need > cap:
            raise SuryaGpuRuntimeError(
                f"vision attention score tile for grid {grid} is "
                f"{need / 1e9:.2f} GB ({n} patches), above the "
                f"{cap / 1e9:.2f} GB budget; one query row needs "
                f"{self.spec.vision_num_heads * n * 4 / 1e6:.1f} MB, so raise "
                f"SuryaGpuRunner(max_vision_scratch_bytes=...) to at least that"
            )
        seconds, tiles, _ = self._vision_plan_seconds(n)
        limit = self.max_vision_seconds
        if limit is not None and seconds > limit:
            _, attention = vision_flops(self.spec, n)
            fits = self.vision_patch_ceiling()
            fits_note = (
                ""
                if fits is None
                else f"; the {limit:.0f} s budget admits up to {fits} patches"
            )
            raise SuryaGpuRuntimeError(
                f"vision forward for grid {grid} is estimated at "
                f"{seconds:.1f} s ({n} patches, {tiles} query tiles, "
                f"{attention / 1e12:.1f} TFLOP of bidirectional attention at "
                f"the measured {VISION_ATTENTION_FLOPS_PER_S / 1e12:.2f} TFLOP/s), "
                f"above the {limit:.0f} s vision time budget{fits_note}; "
                f"downscale the page or raise "
                f"SuryaGpuRunner(max_vision_seconds=...)"
            )
        # the scratch buffer is cached per key and reused when it is already big
        # enough, so only the growth is charged against free memory
        existing = self._scratch.get("vis_scores")
        growth = max(0, need + _GEMM_PAD_BYTES - (existing.nbytes if existing else 0))
        if growth == 0:
            return
        try:
            free_bytes, _total = self.runtime.mem_get_info()
        except Exception:  # runtime without mem_get_info: budget is the only bound
            return
        if growth > int(free_bytes):
            raise SuryaGpuRuntimeError(
                f"vision attention score tile for grid {grid} needs "
                f"{growth / 1e9:.2f} GB more device memory but only "
                f"{int(free_bytes) / 1e9:.2f} GB is free"
            )

    def _add_bias(self, x_ptr: int, bias_ptr: int, n: int, row: int) -> None:
        err = self._k("hipengine_evie_add_bias_f32", [_P, _P, _I, _I, _S])(
            _P(x_ptr), _P(bias_ptr), _I(n), _I(row), _S(0))
        self._check(err, "add bias")

    def _layernorm(self, x_ptr: int, w_ptr: int, b_ptr: int, out_ptr: int,
                   rows: int, dim: int) -> None:
        err = self._k("hipengine_evie_layernorm_f32",
                      [_P, _P, _P, _P, _I, _I, _F, _S])(
            _P(x_ptr), _P(w_ptr), _P(b_ptr), _P(out_ptr), _I(rows), _I(dim),
            _F(1e-6), _S(0))
        self._check(err, "layernorm")

    def _gelu_tanh(self, x_ptr: int, out_ptr: int, n: int) -> None:
        err = self._k("hipengine_evie_gelu_tanh_f32", [_P, _P, _I, _S])(
            _P(x_ptr), _P(out_ptr), _I(n), _S(0))
        self._check(err, "gelu tanh")

    def _gelu_erf(self, x_ptr: int, out_ptr: int, n: int) -> None:
        err = self._k("hipengine_evie_gelu_erf_f32", [_P, _P, _I, _S])(
            _P(x_ptr), _P(out_ptr), _I(n), _S(0))
        self._check(err, "gelu erf")

    def _rope(self, x_ptr: int, cos_ptr: int, sin_ptr: int, tokens: int,
              heads: int, head_dim: int, rotary_dim: int, row_stride: int) -> None:
        err = self._k("hipengine_evie_rope_f32",
                      [_P, _P, _P, _I, _I, _I, _I, _I, _S])(
            _P(x_ptr), _P(cos_ptr), _P(sin_ptr), _I(tokens), _I(heads),
            _I(head_dim), _I(rotary_dim), _I(row_stride), _S(0))
        self._check(err, "rope")

    def _vision_attention_packed(self, q_ptr: int, k_ptr: int, v_ptr: int,
                                 out_ptr: int, tokens: int, heads: int,
                                 head_dim: int, row_stride: int,
                                 scores_ptr: int, scale: float,
                                 block: int) -> None:
        """Bidirectional full-image attention, tiled by query rows.

        Every query attends to every key — the tiles partition the queries, not
        the image, so the result is the dense full-image attention result. Each
        tile's scores are the col-major ``(tokens x bq)`` C of a strided-batched
        SGEMM with ``ldc=tokens``, so tile h starts at ``h * tokens * bq`` and a
        softmax row is the full ``tokens``-long key range for one query.

        ``block`` is the number of query rows per tile. The final partial tile
        packs tighter (``tokens * bq``) than the largest one, so no element of
        the scratch buffer is read without having been written by this tile.
        """

        for start in range(0, tokens, block):
            bq = min(block, tokens - start)
            tile = tokens * bq
            q_blk = q_ptr + start * row_stride * 4
            self.rocblas.sgemm_strided_batched(
                k_ptr, q_blk, scores_ptr,
                m=tokens, n=bq, k=head_dim,
                lda=row_stride, ldb=row_stride, ldc=tokens,
                stride_a=head_dim, stride_b=head_dim, stride_c=tile,
                batch=heads, trans_a=True, trans_b=False,
            )
            err = self._k("hipengine_evie_scale_f32", [_P, _P, _F, _I, _S])(
                _P(scores_ptr), _P(scores_ptr), _F(scale),
                _I(heads * tile), _S(0))
            self._check(err, "vision score scale")
            err = self._k("hipengine_evie_softmax_rows_f32", [_P, _I, _I, _I, _I, _S])(
                _P(scores_ptr), _I(heads * bq), _I(tokens), _I(bq),
                _I(tile), _S(0))
            self._check(err, "vision softmax")
            self.rocblas.sgemm_strided_batched(
                v_ptr, scores_ptr, out_ptr + start * heads * head_dim * 4,
                m=head_dim, n=bq, k=tokens,
                lda=row_stride, ldb=tokens, ldc=heads * head_dim,
                stride_a=head_dim, stride_b=tile, stride_c=head_dim,
                batch=heads, trans_a=False, trans_b=False,
            )

    def _logits_last(self) -> np.ndarray:
        s = self.spec
        logits = self._buf("logits", s.vocab_size * 4)
        self._gemm(self._final_norm_row, self._w["model.language_model.embed_tokens.weight"].ptr,
                   logits.ptr, 1, s.hidden_size, s.vocab_size)
        self.runtime.device_synchronize()
        out = np.empty(s.vocab_size, dtype=np.float32)
        copy_device_to_host(host_array_ptr(out), _raw_buffer(logits.ptr, s.vocab_size * 4))
        return out

    def prefill(self, input_ids, positions, visual_features=None) -> np.ndarray:
        """Run the prompt; returns last-token logits and leaves device state
        (conv windows, GDN recurrence states, KV caches) ready for decode."""
        ids = np.asarray(input_ids).reshape(-1)
        pos = np.asarray(positions)
        if pos.shape[0] == 3 and pos.ndim == 3:  # (b, 3, s) fixture layout
            pos = pos[0]
        if pos.ndim != 2 or pos.shape[0] != 3:
            raise ValueError(f"positions must be (3, s); got {pos.shape}")
        # Admit before any device work: an over-budget prompt must be rejected
        # here, not after embed, rope, and a failed multi-GB allocation.
        self.check_prefill_capacity(len(ids))
        self._seq_len = len(ids)
        self._set_span_extent(len(ids), len(ids) - 1)
        x = self._embed(ids, visual_features)
        cos_buf, sin_buf = self._rope_tables_device(np.ascontiguousarray(pos, dtype=np.int64))
        self._decode_stack(x.ptr, len(ids), cos_buf, sin_buf, decode=False)
        return self._logits_last()

    def decode_step(self, token_id: int, position: int) -> np.ndarray:
        """Advance one token; position is the absolute rope position."""
        x = self._embed(np.array([token_id]), None)
        cos_buf, sin_buf = self._rope_tables_device(
            np.array([[position], [position], [position]], dtype=np.int64))
        # ``_seq_len`` is the next free KV slot: ``_attn_layer_decode`` scatters
        # into that slot and attends over ``_seq_len + 1`` positions. Advance it
        # only after the step, otherwise the scatter lands one slot too high and
        # attention reads a slot this request never wrote -- stale KV from any
        # earlier, longer request on the same runner. ``_set_span_extent``
        # publishes the same extent through ``KVLiveSpans`` for this step.
        self._set_span_extent(self._seq_len + 1, self._seq_len)
        self._decode_stack(x.ptr, 1, cos_buf, sin_buf, decode=True)
        self._seq_len += 1
        return self._logits_last()

    def generate(self, input_ids, positions, visual_features=None,
                 max_new_tokens: int = 64, eos_token_id: int = 2) -> list[int]:
        """Greedy generation; mirrors the CPU reference generate loop."""
        logits = self.prefill(input_ids, positions, visual_features)
        generated: list[int] = []
        pos = int(np.asarray(positions).reshape(3, -1)[:, -1].max()) if np.asarray(positions).ndim >= 2 else 0
        for step in range(max_new_tokens):
            nxt = int(np.argmax(logits))
            if nxt == eos_token_id:
                break
            generated.append(nxt)
            logits = self.decode_step(nxt, pos + 1 + step)
        return generated

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        for buf in self._permanent_bufs:
            hip_free(buf)
        self._permanent_bufs.clear()
        for buf in self._ptr_array_bufs.values():
            hip_free(buf)
        self._ptr_array_bufs.clear()
        for buf in self._scratch.values():
            hip_free(buf)
        self._scratch.clear()
        if self._visual_buf is not None:
            hip_free(self._visual_buf)
            self._visual_buf = None
        for buf in self._w.values():
            hip_free(buf)
        self._w.clear()
        self._conv_state.clear()
        self._gdn_state.clear()
        self._kv_cache.clear()
