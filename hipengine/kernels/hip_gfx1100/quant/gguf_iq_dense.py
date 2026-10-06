"""Raw dense IQ projections with a fixed strict F32 reduction."""
import ctypes
import hashlib
import os
from pathlib import Path

from hipengine.core.build import build_hip
from hipengine.core.hip import get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

SOURCE = Path(__file__).with_suffix('.hip')
TABLE_HASH = hashlib.sha256(b''.join(SOURCE.with_name(name).read_bytes() for name in
    ('gguf_iq_dense_tables.h', 'gguf_iq2_xs_dense_table.h',
     'gguf_iq3_s_grid.h'))).hexdigest()
QUANTS = {'gguf_iq4_xs': 0, 'gguf_iq4_nl': 1, 'gguf_iq3_s': 2, 'gguf_q3_k': 3,
          'gguf_iq3_xxs': 4, 'gguf_iq2_s': 5, 'gguf_iq2_xs': 6}
# Quants the local32 decode owner serves; the split/scale family routes to
# its own export (one kernel, per-quant decode blocks).
_LOCAL32_SPLIT_QUANTS = ('gguf_iq3_s', 'gguf_iq3_xxs', 'gguf_iq2_s',
                         'gguf_iq2_xs')
OUTPUTS = {'f32': 0, 'bf16': 1}
# Rows per block. R only changes which prompt rows share a block, so every
# value is bit-identical to R=1; it trades registers for an R-fold cut in
# weight traffic. Measured on gfx1151: R=1 43 VGPRs, R=2 53, R=4 74 (all still
# 16 waves/SIMD), R=8 132 VGPRs at 10 waves/SIMD, none with scratch.
ROW_BATCHES = (1, 2, 4, 8)
# Rollback and bisection switch for the row-slab rule. Unset/empty keeps the
# smallest slab at or above the row count, which is the measured default;
# "1"/"true"/"on" restores the largest slab at or below it.
_ROW_BATCH_DOWN_ENV = "HIPENGINE_GGUF_IQ_DENSE_ROW_BATCH_DOWN"
_HANDLES = {}
_LIBRARY = None
_LOCAL32_HANDLES = {}
_LOCAL32_DUAL_HANDLES = {}
_LOCAL32_ROWS_HANDLES = {}
_LOCAL32_RESIDUAL_HANDLES = {}
_DENSE_RESIDUAL_HANDLES = {}


def _local32_waves(in_features, out_features):
    """Split-K wave count shared by the rows==1 owner and its rows 2-4 sibling.

    Narrow-N shapes leave too few single-wave blocks on a 96-CU part
    (ffn_down at N=5120 -> 640), so they take 4 waves; wide-N takes 2
    (measured best-or-tied on every real shape, 2026-09-09, W7900). Each wave
    needs at least a few 256-element blocks to stay efficient. Both owners
    call this, so the cross-wave reduction order - and with it the sibling's
    bit-exactness claim - is shared code rather than a copied constant.
    """
    waves = 4 if out_features < 8192 else 2
    while waves > 1 and in_features // 256 < 4 * waves:
        waves //= 2
    return waves


def launch_local32(x_ptr, qweight_ptr, out_ptr, rows, in_features, out_features, *,
                   quant='gguf_iq4_xs', output='bf16', stream=0, library=None,
                   runtime=None):
    """Launch the local32 IQ4_XS decode GEMV (rows=1, bf16 out).

    One wave per 8 output columns with each lane owning 8 contiguous K
    values; narrow-N shapes split K across the block's waves. The
    accumulation order differs from the strict per-row GEMV (lane-grouped
    contiguous k, wave32 shuffle tree, fixed-order cross-wave sum), so this
    owner is approximate and routes admitting it owe the production-referenced
    accuracy gate. Measured max relative error against the strict owner on
    real tensors is <= 2.3e-4 (2026-09-09, W7900).
    """
    if quant not in ('gguf_iq4_xs', 'gguf_iq4_nl') + _LOCAL32_SPLIT_QUANTS:
        raise ValueError('local32 dense IQ decode supports the IQ4/IQ3/IQ2 '
                         'family only')
    if output != 'bf16':
        raise ValueError('local32 dense IQ decode writes bf16 only')
    if (rows != 1 or in_features <= 0 or in_features % 256
            or out_features <= 0 or out_features % 8):
        raise ValueError('local32 dense IQ decode requires rows=1, K divisible '
                         'by 256 and N divisible by 8')
    if not all((x_ptr, qweight_ptr, out_ptr)):
        raise ValueError('local32 dense IQ decode pointers must be nonzero')
    library = library or _default_library()
    key = id(library)
    fn = _LOCAL32_HANDLES.get((key, quant))
    if fn is None:
        if quant in _LOCAL32_SPLIT_QUANTS:
            fn = library.hipengine_gguf_iq_split_local32_gemv
            fn.argtypes = ([ctypes.c_void_p] * 3 + [ctypes.c_int64] * 3
                           + [ctypes.c_int32] * 2 + [ctypes.c_void_p])
        else:
            symbol = ('hipengine_gguf_iq4_xs_local32_gemv'
                      if quant == 'gguf_iq4_xs'
                      else 'hipengine_gguf_iq4_nl_local32_gemv')
            fn = getattr(library, symbol)
            fn.argtypes = ([ctypes.c_void_p] * 3 + [ctypes.c_int64] * 4
                           + [ctypes.c_void_p])
        fn.restype = ctypes.c_int
        _LOCAL32_HANDLES[(key, quant)] = fn
    # Split-K wave count: see _local32_waves. Shared with the rows 2-4
    # sibling so the two owners' cross-wave reduction order cannot drift.
    waves = _local32_waves(in_features, out_features)
    if quant in _LOCAL32_SPLIT_QUANTS:
        err = fn(ctypes.c_void_p(x_ptr), ctypes.c_void_p(qweight_ptr),
                 ctypes.c_void_p(out_ptr), rows, in_features, out_features,
                 QUANTS[quant], waves, ctypes.c_void_p(stream))
    else:
        err = fn(ctypes.c_void_p(x_ptr), ctypes.c_void_p(qweight_ptr),
                 ctypes.c_void_p(out_ptr), rows, in_features, out_features,
                 waves, ctypes.c_void_p(stream))
    if err:
        rt = runtime or get_hip_runtime()
        raise RuntimeError(f'local32 dense IQ decode failed: {rt.error_string(err)}')


def launch_local32_rows(x_ptr, qweight_ptr, out_ptr, rows, in_features,
                        out_features, *, quant='gguf_iq4_xs', output='bf16',
                        stream=0, library=None, runtime=None):
    """Launch the rows 2-4 verifier sibling of the local32 IQ decode owner.

    One block owns ``rows`` prompt rows and shares the weight decode across
    them. Every row keeps the rows == 1 owner's per-(row, column) k ownership,
    FMA order, shuffle tree and cross-wave sum, so each row's output is
    bit-identical to that owner's output for the same row - the sibling adds
    no arithmetic of its own.

    The owner is still approximate relative to the strict per-row GEMV (the
    local32 accumulation order), so routes admitting it owe the
    production-referenced accuracy gate exactly as the rows == 1 owner does.
    """
    if quant not in ('gguf_iq4_xs', 'gguf_iq4_nl') + _LOCAL32_SPLIT_QUANTS:
        raise ValueError('local32 dense IQ rows decode supports the IQ4/IQ3/'
                         'IQ2 family only')
    if output != 'bf16':
        raise ValueError('local32 dense IQ rows decode writes bf16 only')
    if (rows not in (2, 3, 4) or in_features <= 0 or in_features % 256
            or out_features <= 0 or out_features % 8):
        raise ValueError('local32 dense IQ rows decode requires rows in 2..4, '
                         'K divisible by 256 and N divisible by 8')
    if not all((x_ptr, qweight_ptr, out_ptr)):
        raise ValueError('local32 dense IQ rows decode pointers must be nonzero')
    library = library or _default_library()
    key = (id(library), quant)
    fn = _LOCAL32_ROWS_HANDLES.get(key)
    if fn is None:
        if quant in _LOCAL32_SPLIT_QUANTS:
            fn = library.hipengine_gguf_iq_split_local32_rows_gemv
            fn.argtypes = ([ctypes.c_void_p] * 3 + [ctypes.c_int64] * 3
                           + [ctypes.c_int32] * 2 + [ctypes.c_void_p])
        else:
            symbol = ('hipengine_gguf_iq4_xs_local32_rows_gemv'
                      if quant == 'gguf_iq4_xs'
                      else 'hipengine_gguf_iq4_nl_local32_rows_gemv')
            fn = getattr(library, symbol)
            fn.argtypes = ([ctypes.c_void_p] * 3 + [ctypes.c_int64] * 4
                           + [ctypes.c_void_p])
        fn.restype = ctypes.c_int
        _LOCAL32_ROWS_HANDLES[key] = fn
    # Same split-K rule as the rows == 1 owner so the per-row arithmetic is
    # identical, not merely equivalent.
    waves = _local32_waves(in_features, out_features)
    if quant in _LOCAL32_SPLIT_QUANTS:
        err = fn(ctypes.c_void_p(x_ptr), ctypes.c_void_p(qweight_ptr),
                 ctypes.c_void_p(out_ptr), rows, in_features, out_features,
                 QUANTS[quant], waves, ctypes.c_void_p(stream))
    else:
        err = fn(ctypes.c_void_p(x_ptr), ctypes.c_void_p(qweight_ptr),
                 ctypes.c_void_p(out_ptr), rows, in_features, out_features,
                 waves, ctypes.c_void_p(stream))
    if err:
        rt = runtime or get_hip_runtime()
        raise RuntimeError(
            f'local32 dense IQ rows decode failed: {rt.error_string(err)}')


def launch_local32_dual_silu(x_ptr, qweight_a_ptr, qweight_b_ptr, out_ptr,
                             rows, in_features, out_features, *,
                             quant='gguf_iq4_xs', output='bf16', stream=0,
                             library=None, runtime=None):
    """Launch the fused IQ4_XS gate/up dual local32 + SiLU decode GEMV.

    Bit-exact with the unfused single/single/silu_mul path: the per-column
    accumulation is the single owner's, and both accumulators are
    bf16-rounded before the SiLU exactly where the separate elementwise
    kernel would read the two bf16 buffers.
    """
    if quant != 'gguf_iq4_xs':
        raise ValueError('local32 dense IQ decode supports gguf_iq4_xs only')
    if output != 'bf16':
        raise ValueError('local32 dense IQ decode writes bf16 only')
    if (rows != 1 or in_features <= 0 or in_features % 256
            or out_features <= 0 or out_features % 8):
        raise ValueError('local32 dense IQ decode requires rows=1, K divisible '
                         'by 256 and N divisible by 8')
    if not all((x_ptr, qweight_a_ptr, qweight_b_ptr, out_ptr)):
        raise ValueError('local32 dense IQ decode pointers must be nonzero')
    library = library or _default_library()
    key = id(library)
    fn = _LOCAL32_DUAL_HANDLES.get(key)
    if fn is None:
        fn = library.hipengine_gguf_iq4_xs_local32_dual_silu
        fn.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int64] * 4 + [ctypes.c_void_p]
        fn.restype = ctypes.c_int
        _LOCAL32_DUAL_HANDLES[key] = fn
    waves = 4 if out_features < 8192 else 2
    while waves > 1 and in_features // 256 < 4 * waves:
        waves //= 2
    err = fn(ctypes.c_void_p(x_ptr), ctypes.c_void_p(qweight_a_ptr),
             ctypes.c_void_p(qweight_b_ptr), ctypes.c_void_p(out_ptr),
             rows, in_features, out_features, waves,
             ctypes.c_void_p(stream))
    if err:
        rt = runtime or get_hip_runtime()
        raise RuntimeError(f'local32 dense IQ dual decode failed: {rt.error_string(err)}')


def _row_batch_round_down():
    """True when the pre-2026-09-13 largest-slab-at-or-below rule is requested."""

    raw = os.environ.get(_ROW_BATCH_DOWN_ENV, "").strip().lower()
    if not raw:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{_ROW_BATCH_DOWN_ENV} must be a boolean value")


def _row_batch(rows):
    """Row slab that keeps this prompt's weight traffic to a single read.

    A block reads its columns' whole weight slice once and applies it to every
    row it owns, so a prompt of ``rows`` rows is read ``ceil(rows/R)`` times
    across ``grid.y``. Picking the largest supported slab at or *below* the row
    count - the original rule - is right only when the kernel is compute-bound:
    it left rows=3 at R=2 and so read every weight twice. Padding rows load
    ``0.0f`` and are never stored (``gguf_iq_dense.hip``, phase B), and the
    declared contract is that every R is bit-identical, so the smallest slab at
    or *above* the row count is available and removes the re-read. It is never
    worse on compute either: ``ceil(rows/R) * R`` is the row-lane work, and for
    every ``rows <= 8`` the new rule returns the next power of two at or above
    ``rows``, which makes ``grid.y`` one and leaves that product at or below the
    old rule's. Measured on the gfx1100 UD-Q4_K_M verifier at rows=3, R 2 -> 4
    takes the ``gguf_iq_dense_strict_kernel`` family from 4.278 to 3.190
    ms/step (-25.5%), and its largest shape from 380.3 to 252.4 us/call
    (-33.6%).

    ``HIPENGINE_GGUF_IQ_DENSE_ROW_BATCH_DOWN=1`` restores the largest slab at or
    below the row count for rollback and bisection.
    """

    if rows > ROW_BATCHES[-1] or _row_batch_round_down():
        for candidate in reversed(ROW_BATCHES):
            if rows >= candidate:
                return candidate
        return 1
    for candidate in ROW_BATCHES:
        if rows <= candidate:
            return candidate
    return ROW_BATCHES[-1]


def build_gguf_iq_dense(**kwargs):
    return build_hip(sources=[SOURCE], family='gguf_iq_dense', profile='decode',
                     extra_flags=['-ffp-contract=off', f'-DGGUF_IQ_TABLE_HASH={TABLE_HASH}'], output_name='gguf_iq_dense.so', **kwargs)


def _default_library():
    """Cache the loaded CDLL. ``build_gguf_iq_dense`` re-reads and hashes the
    kernel sources to recompute its cache key on every call; on the hot path
    that is once per projection per layer."""
    global _LIBRARY
    if _LIBRARY is None:
        _LIBRARY = build_gguf_iq_dense()
    return _LIBRARY


def launch(x_ptr, qweight_ptr, out_ptr, rows, in_features, out_features, *,
           quant, output, threads=128, stream=0, library=None, runtime=None,
           row_batch=None):
    if quant not in QUANTS or output not in OUTPUTS:
        raise ValueError('unsupported dense IQ quant or output dtype')
    if row_batch is not None and row_batch not in ROW_BATCHES:
        raise ValueError(f'dense IQ row batch must be one of {ROW_BATCHES}')
    block = 32 if quant == 'gguf_iq4_nl' else 256
    if rows <= 0 or rows > 65535 or in_features <= 0 or in_features % block or out_features <= 0:
        raise ValueError('dense IQ requires positive dimensions, rows <= 65535 and block-aligned K')
    if threads != 128:
        raise ValueError('strict dense IQ reduction requires 128 threads')
    if not all((x_ptr, qweight_ptr, out_ptr)):
        raise ValueError('dense IQ pointers must be nonzero')
    library = library or _default_library()
    key = id(library)
    fn = _HANDLES.get(key)
    if fn is None:
        fn = library.hipengine_gguf_iq_dense
        fn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int64] * 3 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
        fn.restype = ctypes.c_int
        _HANDLES[key] = fn
    err = fn(x_ptr, qweight_ptr, out_ptr, rows, in_features, out_features,
             QUANTS[quant], OUTPUTS[output],
             _row_batch(rows) if row_batch is None else row_batch, stream)
    if err:
        rt = runtime or get_hip_runtime()
        raise RuntimeError(f'dense IQ launch failed: {rt.error_string(err)}')


def launch_local32_residual(x_ptr, qweight_ptr, residual_ptr, out_ptr, rows,
                            in_features, out_features, *, quant='gguf_iq4_xs',
                            output='bf16', stream=0, library=None,
                            runtime=None):
    """Rows-1 local32 decode GEMV plus rounded-BF16 residual (E6c-2).

    Bit-exact with ``launch_local32`` followed by ``gguf_bf16_add``: the
    kernel replays the parent's arithmetic and store rounding, then the two
    conversions the standalone add performs. Only the IQ4_XS / IQ4_NL
    owners register this sibling; the split family has no residual variant
    and fails closed to the unfused chain.
    """

    if quant not in ('gguf_iq4_xs', 'gguf_iq4_nl'):
        raise ValueError('local32 residual decode supports IQ4_XS/IQ4_NL only')
    if output != 'bf16':
        raise ValueError('local32 residual decode writes bf16 only')
    if (rows != 1 or in_features <= 0 or in_features % 256
            or out_features <= 0 or out_features % 8):
        raise ValueError('local32 residual decode requires rows=1, K divisible '
                         'by 256 and N divisible by 8')
    if not all((x_ptr, qweight_ptr, residual_ptr, out_ptr)):
        raise ValueError('local32 residual decode pointers must be nonzero')
    library = library or _default_library()
    key = id(library)
    fn = _LOCAL32_RESIDUAL_HANDLES.get((key, quant))
    if fn is None:
        symbol = ('hipengine_gguf_iq4_xs_local32_gemv_residual'
                  if quant == 'gguf_iq4_xs'
                  else 'hipengine_gguf_iq4_nl_local32_gemv_residual')
        fn = getattr(library, symbol)
        fn.argtypes = ([ctypes.c_void_p] * 4 + [ctypes.c_int64] * 3
                       + [ctypes.c_int32] + [ctypes.c_void_p])
        fn.restype = ctypes.c_int
        _LOCAL32_RESIDUAL_HANDLES[(key, quant)] = fn
    waves = _local32_waves(in_features, out_features)
    err = fn(ctypes.c_void_p(x_ptr), ctypes.c_void_p(qweight_ptr),
             ctypes.c_void_p(residual_ptr), ctypes.c_void_p(out_ptr), rows,
             in_features, out_features, waves, ctypes.c_void_p(stream))
    if err:
        rt = runtime or get_hip_runtime()
        raise RuntimeError(f'local32 residual decode failed: {rt.error_string(err)}')


def launch_dense_residual(x_ptr, qweight_ptr, residual_ptr, out_ptr, rows,
                          in_features, out_features, *, quant, output='bf16',
                          stream=0, library=None, runtime=None, row_batch=None):
    """Rows-1 strict dense-IQ GEMV plus rounded-BF16 residual (E6c-2).

    Bit-exact with ``launch`` (bf16 out) followed by ``gguf_bf16_add``.
    The strict owner is the production parent for pinned slots and
    policy-less quants, and for every raw slot while the dense-IQ session is
    unbound, so this sibling must exist for all seven dense-IQ quants.
    """

    if quant not in QUANTS or output != 'bf16':
        raise ValueError('dense IQ residual decode supports the IQ4/IQ3/IQ2 '
                         'family with bf16 out only')
    if rows != 1 or in_features <= 0 or in_features % (32 if quant == 'gguf_iq4_nl' else 256) \
            or out_features <= 0:
        raise ValueError('dense IQ residual decode requires rows=1 and '
                         'block-aligned K')
    if row_batch not in (None, 1):
        raise ValueError('dense IQ residual decode is rows-1 only')
    if not all((x_ptr, qweight_ptr, residual_ptr, out_ptr)):
        raise ValueError('dense IQ residual decode pointers must be nonzero')
    library = library or _default_library()
    key = id(library)
    fn = _DENSE_RESIDUAL_HANDLES.get(key)
    if fn is None:
        fn = library.hipengine_gguf_iq_dense_residual
        fn.argtypes = ([ctypes.c_void_p] * 4 + [ctypes.c_int64] * 3
                       + [ctypes.c_int32] + [ctypes.c_void_p])
        fn.restype = ctypes.c_int
        _DENSE_RESIDUAL_HANDLES[key] = fn
    err = fn(ctypes.c_void_p(x_ptr), ctypes.c_void_p(qweight_ptr),
             ctypes.c_void_p(residual_ptr), ctypes.c_void_p(out_ptr), rows,
             in_features, out_features, QUANTS[quant], ctypes.c_void_p(stream))
    if err:
        rt = runtime or get_hip_runtime()
        raise RuntimeError(f'dense IQ residual launch failed: {rt.error_string(err)}')


def embedding(token_ids_ptr, qweight_ptr, out_ptr, rows, hidden_size, vocab_size, *,
              threads=256, stream=0, library=None, runtime=None):
    if not 0 < rows <= 65535 or hidden_size <= 0 or hidden_size % 256 or vocab_size <= 0:
        raise ValueError('Q3_K embedding requires positive dimensions and block-aligned hidden size')
    if threads != 256 or not all((token_ids_ptr, qweight_ptr, out_ptr)):
        raise ValueError('Q3_K embedding requires 256 threads and nonzero pointers')
    library = library or _default_library()
    fn = library.hipengine_gguf_q3_k_embedding
    fn.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int64]*3 + [ctypes.c_void_p]
    fn.restype = ctypes.c_int
    err = fn(token_ids_ptr, qweight_ptr, out_ptr, rows, hidden_size, vocab_size, stream)
    if err:
        raise RuntimeError(f'Q3_K embedding launch failed: {(runtime or get_hip_runtime()).error_string(err)}')


def register_gguf_iq_dense_kernels(*, backend='hip_gfx1100', replace=True):
    from functools import partial
    register(KernelKey(backend, 'embedding', 'gguf_q3_k', 'lookup_bf16_out'),
             embedding, replace=replace)
    for quant in QUANTS:
        for output in OUTPUTS:
            fn = partial(launch, quant=quant, output=output)
            for prefix in ('gemv', 'prefill'):
                register(KernelKey(backend, 'linear', quant, f'{prefix}_bf16_{output}_out'),
                         fn, replace=replace)
    # The raw-ABI launcher does not pass the quant (the strict kernels bind
    # it the same way): each routed quant registers a partial so the decode
    # owner reads its own block format.
    register(KernelKey(backend, 'linear', 'gguf_iq4_xs',
                       'local32_gemv_bf16_bf16_out'),
             partial(launch_local32, quant='gguf_iq4_xs'), replace=replace)
    register(KernelKey(backend, 'linear', 'gguf_iq4_nl',
                       'local32_gemv_bf16_bf16_out'),
             partial(launch_local32, quant='gguf_iq4_nl'), replace=replace)
    for quant in _LOCAL32_SPLIT_QUANTS:
        register(KernelKey(backend, 'linear', quant,
                           'local32_gemv_bf16_bf16_out'),
                 partial(launch_local32, quant=quant), replace=replace)
    # Verifier sibling (2026-09-12): the same local32 owner geometry at rows
    # 2-4. Registered under its own variant so the strict per-row GEMV stays
    # the default owner everywhere the backend policy does not admit it.
    register(KernelKey(backend, 'linear', 'gguf_iq4_xs',
                       'local32_rows_gemv_bf16_bf16_out'),
             partial(launch_local32_rows, quant='gguf_iq4_xs'), replace=replace)
    register(KernelKey(backend, 'linear', 'gguf_iq4_nl',
                       'local32_rows_gemv_bf16_bf16_out'),
             partial(launch_local32_rows, quant='gguf_iq4_nl'), replace=replace)
    for quant in _LOCAL32_SPLIT_QUANTS:
        register(KernelKey(backend, 'linear', quant,
                           'local32_rows_gemv_bf16_bf16_out'),
                 partial(launch_local32_rows, quant=quant), replace=replace)
    register(KernelKey(backend, 'linear_pair_silu', 'gguf_iq4_xs',
                       'local32_pair_silu_bf16_bf16_out'),
             launch_local32_dual_silu, replace=replace)
    # E6c-2 residual siblings (rows-1 FFN-down fold). The strict composite
    # covers all seven quants (pinned slots, policy-less quants and every
    # session-unbound launch run the strict owner); the local32 composite
    # covers the two redirected IQ4 owners this artifact's down slots take.
    for quant in QUANTS:
        register(KernelKey(backend, 'linear+residual', quant,
                           'gemv_bf16_residual_bf16_out'),
                 partial(launch_dense_residual, quant=quant), replace=replace)
    for quant in ('gguf_iq4_xs', 'gguf_iq4_nl'):
        register(KernelKey(backend, 'linear+residual', quant,
                           'local32_gemv_bf16_residual_bf16_out'),
                 partial(launch_local32_residual, quant=quant), replace=replace)


register_gguf_iq_dense_kernels()
