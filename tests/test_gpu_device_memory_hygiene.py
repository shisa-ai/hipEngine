"""Mechanical guards for the device-memory hygiene rules.

``docs/KERNELS.md`` "Device-memory hygiene" states three rules. Two of them are
enforced at runtime by ``tests/_poison_probe.py`` (poison a device buffer with
``0xFF`` and require a bit-identical result); the third is a property of the
*source text* of an H2D upload and cannot be observed by running the code, so it
is enforced here.

Why the source rule needs a guard: ``copy_host_to_device`` takes a bare integer
address, so the array has to outlive the call by itself. In
``copy(buf, host_array_ptr(np.zeros_like(x)))`` CPython drops the temporary's
last reference when ``host_array_ptr`` returns, so the array is already freed
*before the copy is entered* -- measured on gfx1151, the freed block is handed
straight back to the next same-size allocation at the same address, and the copy
then reads whatever that allocation wrote. The failure is heap-layout dependent,
so it shows up as "this test passes alone and fails after another test" --
``tests/test_gpu_surya_kv_spans.py`` did exactly that with 4 denormal values in
slots the scatter never writes. Nothing about the call site looks wrong, which is
why it needs a lint rather than a review convention.

The transfer itself does not need the source after it returns: overwriting the
source in place immediately after ``copy_host_to_device`` returns leaves the
destination untouched at 1, 16, 64, and 128 MiB on gfx1151. So a *named* local is
sufficient, and the guard is about the missing reference, not about DMA timing.

**This is a partial lint.** It flags an always-allocating call in the argument
position, which is the form that can never be correct. It does not flag:

- an allocating expression hidden under a view, e.g.
  ``host_array_ptr(np.zeros(n).reshape(2, -1))`` (the ``reshape`` is the outer
  call, and the allocation happens inside it);
- ``host_array_ptr(np.ascontiguousarray(x))``, which copies only when ``x`` is
  non-contiguous. Flagging it would mean editing ~100 call sites that are safe
  as written, so the conditional case is recorded in ``docs/REFACTOR.md``
  instead;
- ``np.asarray`` on an array that is already the right dtype, which returns its
  argument rather than allocating. It stays in the flagged set because the
  cheap way to satisfy the lint (bind it to a local) is also correct for the
  allocating case, and because a guard that silently misses a real allocation is
  worse than one that asks for a local.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCANNED_ROOTS = ("hipengine", "tests", "scripts", "benchmarks")

# Calls that always return a freshly allocated array, so the only reference is
# the one held by the enclosing expression.
_ALWAYS_ALLOCATES = frozenset(
    {
        "np.zeros",
        "np.zeros_like",
        "np.ones",
        "np.ones_like",
        "np.full",
        "np.full_like",
        "np.array",
        "np.asarray",
        "np.empty",
        "np.empty_like",
        "np.tile",
        "np.repeat",
        "np.concatenate",
        "np.stack",
        "np.pad",
        "np.arange",
        "np.linspace",
        "np.fromiter",
    }
)
# Methods that always return a fresh array (``astype`` copies unless asked not to,
# ``flatten`` copies by definition).  ``view``/``reshape``/``ravel`` are excluded:
# they return a view whose buffer is owned by the original, which the caller
# normally holds, and a temporary *original* is already caught on its own.
_ALWAYS_ALLOCATES_METHODS = frozenset({"astype", "copy", "flatten"})


def _dotted(node: ast.AST) -> str | None:
    """``np.zeros`` for an attribute chain, or ``None`` for anything else."""

    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _is_always_allocating(node: ast.AST) -> bool:
    """True when evaluating ``node`` always yields a newly allocated array."""

    if not isinstance(node, ast.Call):
        return False
    dotted = _dotted(node.func)
    if dotted in _ALWAYS_ALLOCATES:
        return True
    if isinstance(node.func, ast.Attribute):
        if node.func.attr in _ALWAYS_ALLOCATES_METHODS:
            return True
        # ``np.ascontiguousarray(x.astype(...))``: a no-op only when its input is
        # already contiguous, so it is a temporary exactly when the inner call is.
        if node.func.attr == "ascontiguousarray":
            return any(_is_always_allocating(arg) for arg in node.args)
    return False


def _inline_temporary_uploads() -> list[str]:
    offenders: list[str] = []
    for root in SCANNED_ROOTS:
        for path in sorted((ROOT / root).rglob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if _dotted(node.func) != "host_array_ptr" or not node.args:
                    continue
                if _is_always_allocating(node.args[0]):
                    source = ast.unparse(node.args[0])
                    offenders.append(
                        f"{path.relative_to(ROOT)}:{node.lineno}: host_array_ptr({source})"
                    )
    return offenders


def test_unpinned_h2d_source_is_never_an_inline_temporary() -> None:
    offenders = _inline_temporary_uploads()
    assert not offenders, (
        "these H2D copies use a same-statement temporary as the host source; the array "
        "is freed when host_array_ptr returns, before the copy is entered "
        "(docs/KERNELS.md \"Device-memory hygiene\"). Bind it to a local:\n  "
        + "\n  ".join(offenders)
    )


def test_the_lint_flags_the_pattern_it_is_meant_to_catch() -> None:
    """A guard that cannot fail is not a guard."""

    samples = [
        "np.zeros_like(x)",
        "np.ascontiguousarray(x.astype(np.int32))",
        "np.asarray([0], dtype=np.int32)",
        "a.b.copy()",
        "np.tile(np.arange(4, dtype=np.float32), (2, 1))",
        "x_f16.astype(np.float32)",
    ]
    safe = [
        "src",
        "np.ascontiguousarray(src)",
        "host_view[offset:offset + n]",
        "np.ascontiguousarray(src, dtype=np.float32)",
        "x_f16.view(np.uint16)",
        "query.reshape(1, 4, 128)",
    ]
    for sample in samples:
        node = ast.parse(sample).body[0].value
        assert _is_always_allocating(node), sample
    for sample in safe:
        node = ast.parse(sample).body[0].value
        assert not _is_always_allocating(node), sample


def test_the_lint_is_documented_as_partial_and_its_gaps_are_pinned() -> None:
    """Pin the known misses, so "partial lint" is a measured statement.

    These are the forms the guard does *not* catch.  If one of them is ever
    closed, this test fails and the docstring's list has to shrink with it --
    the point is that the module never claims more coverage than it has.
    """

    known_misses = [
        "np.zeros(8).reshape(2, -1)",   # allocation hidden under the outer call
        "np.ascontiguousarray(x.T)",    # copies only when x.T is non-contiguous
    ]
    for sample in known_misses:
        node = ast.parse(sample).body[0].value
        assert not _is_always_allocating(node), (
            f"{sample} is now flagged; remove it from the documented gaps in the "
            "module docstring and from this list"
        )
    # A deliberate over-flag: np.asarray returns its argument when the dtype
    # already matches, so this can be a false positive.  It stays flagged
    # because binding it to a local is correct either way and a guard that
    # misses a real allocation is worse than one that asks for a local.
    assert _is_always_allocating(ast.parse("np.asarray(x)").body[0].value)
    docstring = __doc__ or ""
    for phrase in ("partial lint", "does not flag", "docs/REFACTOR.md"):
        assert phrase in docstring, phrase


@pytest.mark.parametrize("name", ["tests/_poison_probe.py"])
def test_unit_poison_probe_helper_is_present(name: str) -> None:
    """The runtime half of the hygiene rules must stay available to runners."""

    text = (ROOT / name).read_text()
    for symbol in ("assert_poison_invariant", "collect_device_buffers", "group_by_prefix"):
        assert f"def {symbol}" in text, f"{name} must keep providing {symbol}"


def _hip_available() -> bool:
    import ctypes

    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


@pytest.mark.skipif(not _hip_available(), reason="ROCm/HIP runtime not available")
def test_cached_dev_ptr_array_uploads_once_and_keeps_its_source_alive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``EvieRunner._dev_ptr_array`` must upload on cache miss only.

    The cache key is the whole pointer list, so a hit is already correct.  The
    pre-fix version re-uploaded on every call from a host array that went out of
    scope when the method returned, so the array could be freed and its block
    recycled before the copy read it (``docs/KERNELS.md`` "Device-memory
    hygiene").  Surya's twin of this method was already fixed; this pins Evie's.

    The upload count is the RED assertion: the fix is that a hit transfers
    nothing, which is also what removes the window.
    """

    import hipengine.runtime.evie as evie_module
    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import copy_device_to_host, free, host_array_ptr
    from hipengine.runtime.evie import EvieRunner

    uploads: list[int] = []
    real_copy = evie_module.copy_host_to_device

    def counting_copy(buffer, source, nbytes=None, **kwargs):
        uploads.append(int(nbytes if nbytes is not None else buffer.nbytes))
        return real_copy(buffer, source, nbytes, **kwargs)

    monkeypatch.setattr(evie_module, "copy_host_to_device", counting_copy)

    runner = object.__new__(EvieRunner)
    runner.runtime = get_hip_runtime()
    runner._ptr_array_bufs = {}
    pointers = [0x1000, 0x2000, 0x3000, 0x4000]
    try:
        first = runner._dev_ptr_array(pointers)
        uploads_after_first = len(uploads)
        second = runner._dev_ptr_array(pointers)
        assert second == first, "a cache hit must return the same device array"
        assert len(uploads) == uploads_after_first, (
            "a cache hit re-uploaded the pointer array; the upload is the only "
            "place the host source can be recycled mid-transfer"
        )

        buffer = runner._ptr_array_bufs[tuple(pointers)]
        host = np.zeros(len(pointers), dtype=np.uint64)
        copy_device_to_host(host_array_ptr(host), buffer, host.nbytes)
        np.testing.assert_array_equal(host, np.asarray(pointers, dtype=np.uint64))
    finally:
        for buffer in runner._ptr_array_bufs.values():
            free(buffer)
        runner._ptr_array_bufs.clear()
