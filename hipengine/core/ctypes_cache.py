"""ctypes function-pointer signature caching.

Every kernel-launcher wrapper in ``hipengine/kernels/`` follows the pattern::

    fn = getattr(library, _SYMBOL)
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ..., ctypes.c_int64, ...]
    fn.restype = ctypes.c_int
    err = fn(ctypes.c_void_p(ptr_a), ctypes.c_void_p(ptr_b), ...)
    _check_launch(runtime, err)

The argtypes/restype assignment runs on every kernel launch even though the
signature is identical across calls.  Microbenchmark (libc.strlen, single arg):

  - per-call ``fn.argtypes = [...]; fn.restype = ...`` : ~383 ns/call
  - cached argtypes/restype, just ``fn(...)``         : ~170 ns/call

Plus, the explicit ``ctypes.c_void_p(ptr)`` and ``ctypes.c_int64(x)`` calls at
the call site add ~172 ns/call vs passing raw ints (which auto-coerce when
argtypes is set).

The MTP verifier hot path makes ~1000 kernel launches per verifier pass and
~25 verifier passes per decoded token.  Per the M13.C cProfile attribution,
``run_moe_c1_fp16`` cumulative time across 40 layers is ~7 ms/pass of pure
Python + ctypes work.  Caching argtypes/restype + dropping call-site ctypes
constructors reclaims a measurable slice of that without touching kernel code.

``signed_kernel_fn`` resolves the function pointer once per library handle,
sets argtypes/restype once, and tags the fn object with a sentinel attribute
so subsequent lookups short-circuit.  ``ctypes.CDLL.__getattr__`` returns the
same ``_FuncPtr`` object on every access for a given library, so the sentinel
is stable across calls.

Callers should:

1. Define ``_ARGTYPES_X`` as a module-level tuple of ctypes types.
2. Replace ``fn.argtypes = [...]; fn.restype = ...; fn = getattr(library, S)``
   with ``fn = signed_kernel_fn(library, S, _ARGTYPES_X, ctypes.c_int)``.
3. Replace ``fn(ctypes.c_void_p(ptr), ctypes.c_int64(x))`` with ``fn(ptr, x)``.

This module is hot-path code; keep it lean.  Do not log, do not validate beyond
the assertion sentinel.
"""

from __future__ import annotations

import ctypes
from typing import Any, Sequence

_SIGNATURE_ATTR = "_he_signed"


def signed_kernel_fn(
    library: ctypes.CDLL,
    symbol: str,
    argtypes: Sequence[Any],
    restype: Any,
) -> Any:
    """Return ``library.<symbol>`` with argtypes/restype set, cached on the fn.

    First call per ``(library, symbol)`` pair assigns argtypes/restype and tags
    the function pointer.  Subsequent calls short-circuit on the tag.  The
    sentinel value is ``argtypes`` itself, so any inconsistency (different
    argtypes tuple identity for the same symbol) refreshes the signature — a
    hard requirement when the same symbol is reused with different signatures,
    which the existing wrappers never do but the helper is safe under.
    """
    fn = getattr(library, symbol)
    if getattr(fn, _SIGNATURE_ATTR, None) is argtypes:
        return fn
    fn.argtypes = argtypes
    fn.restype = restype
    setattr(fn, _SIGNATURE_ATTR, argtypes)
    return fn


__all__ = ["signed_kernel_fn"]
