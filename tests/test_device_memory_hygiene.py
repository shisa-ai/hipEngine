"""Mechanical guards for the device-memory hygiene rules.

``docs/KERNELS.md`` "Device-memory hygiene" states three rules. Two of them are
enforced at runtime by ``tests/_poison_probe.py`` (poison a device buffer with
``0xFF`` and require a bit-identical result); the third is a property of the
*source text* of an H2D upload and cannot be observed by running the code, so it
is enforced here.

Why the source rule needs a guard at all: on this stack the DMA of an *unpinned*
host source reads the host buffer after ``hipMemcpy`` returns. A temporary
inlined into the call can therefore be freed and recycled before the copy lands,
and the destination silently receives stale heap bytes. The failure is
heap-layout dependent, so it shows up as "this test passes alone and fails after
another test" -- ``tests/test_surya_kv_spans.py`` did exactly that with 4
denormal values in slots the scatter never writes. Nothing about the call site
looks wrong, which is why it needs a lint rather than a review convention.

Only *always*-allocating sources are flagged: ``np.zeros``/``np.ones``/
``np.full``/``np.array``/``np.asarray``/``np.tile``/``.astype(...)``/``.copy()``
and friends. ``host_array_ptr(np.ascontiguousarray(x))`` is deliberately *not*
flagged: on a contiguous ``x`` it returns the same object and the caller keeps it
alive, which is the common case. It is still a latent violation for a
non-contiguous ``x``, and that residue is recorded in ``docs/REFACTOR.md``.
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
        "these unpinned H2D copies use a same-statement temporary as the host source; "
        "the DMA can read the freed buffer (docs/KERNELS.md \"Device-memory hygiene\"). "
        "Hoist the array into a local and device_synchronize() before releasing it:\n  "
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


@pytest.mark.parametrize("name", ["tests/_poison_probe.py"])
def test_poison_probe_helper_is_present(name: str) -> None:
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
    scope when the method returned; on this stack the DMA of an unpinned source
    reads the host buffer *after* ``hipMemcpy`` returns, so the device array
    could receive stale heap bytes (``docs/KERNELS.md`` "Device-memory
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
