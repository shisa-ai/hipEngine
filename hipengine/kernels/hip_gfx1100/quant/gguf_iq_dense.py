"""Raw dense IQ projections with a fixed strict F32 reduction."""
import ctypes
import hashlib
from pathlib import Path

from hipengine.core.build import build_hip
from hipengine.core.hip import get_hip_runtime
from hipengine.kernels.registry import KernelKey, register

SOURCE = Path(__file__).with_suffix('.hip')
TABLE_HASH = hashlib.sha256(b''.join(SOURCE.with_name(name).read_bytes() for name in
    ('gguf_iq_dense_tables.h', 'gguf_iq2_xs_dense_table.h'))).hexdigest()
QUANTS = {'gguf_iq4_xs': 0, 'gguf_iq4_nl': 1, 'gguf_iq3_s': 2, 'gguf_q3_k': 3,
          'gguf_iq3_xxs': 4, 'gguf_iq2_s': 5, 'gguf_iq2_xs': 6}
OUTPUTS = {'f32': 0, 'bf16': 1}
# Rows per block. R only changes which prompt rows share a block, so every
# value is bit-identical to R=1; it trades registers for an R-fold cut in
# weight traffic. Measured on gfx1151: R=1 43 VGPRs, R=2 53, R=4 74 (all still
# 16 waves/SIMD), R=8 132 VGPRs at 10 waves/SIMD, none with scratch.
ROW_BATCHES = (1, 2, 4, 8)
_HANDLES = {}
_LIBRARY = None


def _row_batch(rows):
    """Largest supported slab that a prompt of ``rows`` rows actually fills."""
    for candidate in reversed(ROW_BATCHES):
        if rows >= candidate:
            return candidate
    return 1


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


register_gguf_iq_dense_kernels()
