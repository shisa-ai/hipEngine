"""The Gemma 4 decode route ABI, on CPU.

The decode entry points take an explicit route selection plus the flash
key-slice count. Before that they took a signed slice count whose sign was the
flash request, so the same integer meant "single kernel", "dimension partition"
or "flash at N slices" depending on its value. These tests pin the replacement
at the host boundary:

* **The argument tail.** The decode tail is
  ``(workspace, route, flash_slices, live_extent)``; the prefill tail is still
  ``(window, row_offset)``. A tail that drifts is a silent misread of a count as
  a pointer.
* **The route.** The single-kernel oracle below the partition threshold, the
  dimension partition from it, the flash request with its own count, and the
  owned-scratch strict route where the resident logit row does not fit.
* **The count.** The flash count travels in its own argument and is zero on
  every non-flash route, so a route and a count cannot disagree.
* **The workspace.** The partition reserves its weights buffer, flash reserves
  the larger of that and its partials, and the strict route reserves nothing
  unless the resident row needs owned scratch.

The kernel side is pinned from the source: the exported parameter lists are
compared against the host argtypes, because a dropped or reordered parameter
only shows up as a misread on device. The same pass checks that no tracked raw
caller still passes the removed arguments.

No build, no device, no kernel launch.
"""

from __future__ import annotations

import ast
import ctypes
import pathlib
from types import SimpleNamespace

import pytest

from hipengine.kernels.hip_gfx1100.gemma4 import gemma4_attention as attention

# Position of each decode-only argument in the launched argument tuple, counted
# from the end so a change to the shared prefix does not silently re-index it.
_WORKSPACE = -4
_ROUTE = -3
_FLASH_SLICES = -2
_LIVE_EXTENT = -1

# Gemma 4's global geometry: 16 query heads over 2 KV heads, head_dim 512. This
# is one of the two packings the flash kernels are instantiated for.
_GLOBAL = dict(tokens=1, num_heads=16, num_kv_heads=2, head_dim=512, scale=1.0)

# The same head width with GQA ratio 4: outside the flash kernels' ratio-2 and
# ratio-8 packings, so it keeps the dimension partition at the same key count.
_PARTITIONED = dict(tokens=1, num_heads=16, num_kv_heads=4, head_dim=512, scale=1.0)


class _Scratch:
    """Records every workspace reservation the launcher makes."""

    def __init__(self):
        self.sizes: list[int] = []

    def buffer(self, nbytes, *, stream, runtime):
        self.sizes.append(int(nbytes))
        return SimpleNamespace(ptr=0xB000 + len(self.sizes))


def _capture(monkeypatch, *, split_bytes=100, flash_bytes=200, request_slices=9):
    """Stub the launcher's sizing and kernel lookups; return the capture state."""

    calls: list[tuple] = []
    sized: dict[str, list[tuple]] = {"split": [], "flash": []}

    def launch(*args):
        calls.append(args)
        return 0

    def split_workspace_bytes(tokens, num_heads, keys, *, library=None):
        sized["split"].append((int(tokens), int(num_heads), int(keys)))
        return split_bytes

    def flash_workspace_bytes(tokens, num_heads, head_dim, slices, *, library=None):
        sized["flash"].append(
            (int(tokens), int(num_heads), int(head_dim), int(slices))
        )
        return flash_bytes

    monkeypatch.setattr(attention, "signed_kernel_fn", lambda *a: launch)
    monkeypatch.setattr(attention, "split_workspace_bytes", split_workspace_bytes)
    monkeypatch.setattr(attention, "flash_workspace_bytes", flash_workspace_bytes)
    monkeypatch.setattr(attention, "flash_request_slices", lambda **kw: request_slices)

    scratch = _Scratch()
    return calls, scratch, sized


def _launch(monkeypatch, *, keys, prefer_flash=False, **shape):
    calls, scratch, sized = _capture(monkeypatch)
    geometry = dict(_GLOBAL)
    geometry.update(shape)
    attention._launch_prefill(
        attention._SYMBOL_DECODE_F32, 1, 2, 3, 4, 5,
        keys=keys,
        stream=3,
        library=object(),
        runtime=object(),
        scratch=scratch,
        prefer_flash=prefer_flash,
        **geometry,
    )
    assert len(calls) == 1, "the decode ABI is one launch"
    return calls[0], scratch, sized


def test_decode_argument_tail_is_workspace_route_count_slot():
    """The tail is positional ctypes; a drift misreads an int as a pointer."""

    shared = len(attention._ARGTYPES_COMMON)
    assert attention._ARGTYPES_DECODE[shared:] == (
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
    )
    assert attention._ARGTYPES_PREFILL[shared:] == (
        ctypes.c_int64, ctypes.c_int64,
    )


def test_route_values_name_the_kernel_families():
    """The host constants mirror the kernel's own route enum."""

    assert attention.DECODE_ROUTE_STRICT == 0
    assert attention.DECODE_ROUTE_SPLIT == 1
    assert attention.DECODE_ROUTE_FLASH == 2


def test_short_row_keeps_the_single_kernel_oracle(monkeypatch):
    """Below the partition threshold and without a preference, nothing splits."""

    args, scratch, sized = _launch(monkeypatch, keys=512)
    assert args[_ROUTE].value == attention.DECODE_ROUTE_STRICT
    assert args[_FLASH_SLICES].value == 0
    assert not args[_WORKSPACE].value
    assert args[_LIVE_EXTENT].value is None
    assert scratch.sizes == []
    assert sized["split"] == [] and sized["flash"] == []


def test_partition_route_reserves_weights_and_carries_no_count(monkeypatch):
    """A shape flash cannot serve takes the dimension partition at 1024 keys."""

    args, scratch, sized = _launch(monkeypatch, keys=1024, **_PARTITIONED)
    assert args[_ROUTE].value == attention.DECODE_ROUTE_SPLIT
    assert args[_FLASH_SLICES].value == 0
    assert args[_WORKSPACE].value
    assert sized["split"] == [(1, 16, 1024)]
    assert sized["flash"] == []
    assert scratch.sizes == [100]


def test_flash_route_carries_its_key_slice_count(monkeypatch):
    """The count is a separate argument, and the workspace covers both routes."""

    args, scratch, sized = _launch(monkeypatch, keys=1024)
    assert args[_ROUTE].value == attention.DECODE_ROUTE_FLASH
    assert args[_FLASH_SLICES].value == 9
    assert args[_WORKSPACE].value
    # The reservation is max(split weights, flash partials): a rejected flash
    # request falls through to the partition on the same allocation.
    assert sized["split"] == [(1, 16, 1024)]
    assert sized["flash"] == [(1, 16, 512, 9)]
    assert scratch.sizes == [200]


def test_flash_preference_takes_a_row_below_the_partition_threshold(monkeypatch):
    """A short row only the preference moves: flash engages from 256 keys."""

    args, _, sized = _launch(monkeypatch, keys=256, prefer_flash=True)
    assert args[_ROUTE].value == attention.DECODE_ROUTE_FLASH
    assert args[_FLASH_SLICES].value == 9
    assert sized["flash"] == [(1, 16, 512, 9)]


def test_unservable_flash_request_keeps_the_partition_route(monkeypatch):
    """A request with no slice count that fits is not a flash launch."""

    calls, scratch, sized = _capture(monkeypatch, request_slices=0)
    attention._launch_prefill(
        attention._SYMBOL_DECODE_F32, 1, 2, 3, 4, 5,
        keys=1024, stream=3, library=object(), runtime=object(), scratch=scratch,
        prefer_flash=True, **_GLOBAL,
    )
    args = calls[0]
    assert args[_ROUTE].value == attention.DECODE_ROUTE_SPLIT
    assert args[_FLASH_SLICES].value == 0
    assert sized["flash"] == []
    assert scratch.sizes == [100]


def test_owned_scratch_keeps_the_strict_route(monkeypatch):
    """Past the resident logit bound the strict class kernel owns global scratch."""

    args, scratch, sized = _launch(monkeypatch, keys=16640, **_PARTITIONED)
    assert args[_ROUTE].value == attention.DECODE_ROUTE_STRICT
    assert args[_FLASH_SLICES].value == 0
    assert args[_WORKSPACE].value
    assert sized["split"] == [(1, 16, 16640)]
    assert scratch.sizes == [100]


def test_capability_miss_never_requests_flash(monkeypatch):
    """Admission is the kernel's own shape test, not a count or an identity."""

    assert not attention.flash_admits(tokens=1, head_dim=512, num_heads=16, num_kv_heads=4)
    assert not attention.flash_admits(tokens=1, head_dim=256, num_heads=16, num_kv_heads=2)

    args, _, sized = _launch(monkeypatch, keys=4096, prefer_flash=True, **_PARTITIONED)
    assert args[_ROUTE].value == attention.DECODE_ROUTE_SPLIT
    assert sized["flash"] == []


def test_split_workspace_asks_for_three_shape_fields(monkeypatch):
    """The sizing export lost its unused head width and partition count."""

    seen: list[tuple] = []

    def signed(library, symbol, argtypes, restype):
        seen.append((symbol, argtypes, restype))

        def fn(tokens, num_heads, keys):
            seen.append((int(tokens), int(num_heads), int(keys)))
            return 7

        return fn

    monkeypatch.setattr(attention, "signed_kernel_fn", signed)
    assert attention.split_workspace_bytes(2, 3, 5, library=object()) == 7
    symbol, argtypes, restype = seen[0]
    assert symbol == attention._SYMBOL_SPLIT_WORKSPACE_BYTES
    assert argtypes == (ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
    assert restype is ctypes.c_size_t
    assert seen[1] == (2, 3, 5)


@pytest.mark.parametrize(
    ("keys", "prefer_flash", "engages", "short_row"),
    [
        (1, True, False, False),
        (255, True, False, False),
        (256, True, True, True),
        (700, True, True, True),
        (1023, True, True, True),
        (1024, True, True, False),
        (256, False, False, True),
        (1023, False, False, True),
        (1024, False, True, False),
        (5000, False, True, False),
    ],
)
def test_route_entry_points_are_the_partition_threshold_and_preference(
    keys, prefer_flash, engages, short_row
):
    """The two entry expressions the layer selector and the launcher share."""

    assert (
        attention.flash_decode_engages(keys=keys, prefer_flash=prefer_flash)
        is engages
    )
    assert attention.flash_decode_short_row(keys=keys) is short_row


# --- The kernel side of the ABI, from the source --------------------------

_REPO = pathlib.Path(__file__).resolve().parents[1]
_HIP_SOURCE = pathlib.Path(attention.__file__).with_suffix(".hip").read_text()

# The C types the decode and workspace exports use, and their ctypes mirror.
# A pointer is a pointer on the host side, whatever it points at.
_CTYPES_BY_C_TYPE = {
    "int64_t": ctypes.c_int64,
    "float": ctypes.c_float,
    "int": ctypes.c_int,
    "hipStream_t": ctypes.c_void_p,
}


def _export_parameters(name: str) -> list[str]:
    """One exported function's parameter declarations, split on top-level commas."""

    # The definition, not a comment that names it: every export is written as
    # ``name(`` at the end of its declaration line.
    start = _HIP_SOURCE.index(f"{name}(\n")
    end = _HIP_SOURCE.index(") {", start)
    return [part.strip() for part in _HIP_SOURCE[start:end].split("(", 1)[1].split(",")]


def _host_type(declaration: str) -> type:
    """The ctypes type for one ``<type> <name>`` parameter declaration."""

    if "*" in declaration:
        return ctypes.c_void_p
    return _CTYPES_BY_C_TYPE[declaration.rsplit(None, 1)[0]]


def _names(parameters: list[str]) -> list[str]:
    return [parameter.split()[-1].lstrip("*") for parameter in parameters]


_DECODE_NAMES = [
    "query", "key", "value", "keep_mask", "out", "tokens", "num_heads",
    "num_kv_heads", "head_dim", "scale", "stream", "keys", "workspace",
    "route", "flash_slices", "live_extent",
]


@pytest.mark.parametrize(
    "symbol",
    ["hipengine_gemma4_attention_decode_bf16", "hipengine_gemma4_attention_decode_f32"],
)
def test_kernel_decode_export_is_the_host_argtypes(symbol):
    """The callee's parameter list and the host's argtypes are one list.

    A dropped, added, or reordered parameter changes no Python and no import, so
    without this the first evidence of drift would be a device launch reading a
    count as a pointer.
    """

    parameters = _export_parameters(symbol)
    assert len(parameters) == len(attention._ARGTYPES_DECODE) == 16
    assert tuple(_host_type(parameter) for parameter in parameters) == (
        attention._ARGTYPES_DECODE
    )
    assert _names(parameters) == _DECODE_NAMES


@pytest.mark.parametrize(
    ("symbol", "names", "types"),
    [
        (
            "hipengine_gemma4_decode_split_workspace_bytes",
            ["tokens", "num_heads", "keys"],
            (ctypes.c_int64,) * 3,
        ),
        (
            "hipengine_gemma4_decode_flash_workspace_bytes",
            ["tokens", "num_heads", "head_dim", "slices"],
            (ctypes.c_int64,) * 3 + (ctypes.c_int,),
        ),
        (
            "hipengine_gemma4_decode_flash_admits",
            ["tokens", "num_heads", "num_kv_heads", "head_dim", "keys", "slices"],
            (ctypes.c_int64,) * 5 + (ctypes.c_int,),
        ),
    ],
)
def test_kernel_sizing_and_admission_exports_keep_their_arity(symbol, names, types):
    """The three exports a caller sizes or admits with, at their own arities."""

    parameters = _export_parameters(symbol)
    assert _names(parameters) == names
    assert tuple(_host_type(parameter) for parameter in parameters) == types


# A raw caller builds the argument list itself, so a removed argument is not a
# TypeError: ctypes happily passes one extra value and the kernel reads the next
# slot. That is how the signed count survived a rewrite in the first place.
_RAW_CALLER_ROOTS = ("hipengine", "scripts", "scratch")
# The literal export names, plus the module constants a caller imports instead.
_DECODE_SYMBOL_TEXT = (
    "gemma4_attention_decode_bf16",
    "gemma4_attention_decode_f32",
    "_SYMBOL_DECODE_BF16",
    "_SYMBOL_DECODE_F32",
)
_REMOVED_ARGUMENTS = ("split_slices", "split_workspace=")


def test_no_raw_decode_caller_keeps_the_removed_arguments():
    """Every tracked caller of the raw decode symbol speaks the new tail."""

    scanned, offenders = [], []
    for root in _RAW_CALLER_ROOTS:
        for path in sorted((_REPO / root).rglob("*.py")):
            text = path.read_text(encoding="utf-8", errors="replace")
            if not any(symbol in text for symbol in _DECODE_SYMBOL_TEXT):
                continue
            scanned.append(path.relative_to(_REPO).as_posix())
            offenders.extend(
                f"{path.relative_to(_REPO)}: {argument}"
                for argument in _REMOVED_ARGUMENTS
                if argument in text
            )
    assert offenders == []
    # The two diagnostic probes are the callers this check exists for; if they
    # stop matching, the scan is vacuous rather than clean.
    assert {
        "scratch/flash_scale_probe.py",
        "scratch/gemma4_flash_slice_sweep.py",
    } <= set(scanned)


def test_the_raw_launch_helper_speaks_the_new_tail():
    """The shared raw helper's keywords are what the probes above pass.

    Both probes build their argument tail through this helper, so its keyword
    names are part of the ABI surface they depend on.
    """

    path = _REPO / "tests/test_gpu_gemma4_attention_decode_parity.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    helper = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_raw_launch"
    )
    keywords = {argument.arg for argument in helper.args.args + helper.args.kwonlyargs}
    assert {"workspace", "route", "flash_slices", "live_extent"} <= keywords
    assert not {"split_workspace", "split_slices"} & keywords
