"""Owner lifetime and bounds at the synchronous array upload boundary."""
import ctypes
import ast
from pathlib import Path
import weakref
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.core import memory


QWEN4_EXP_AUXILIARY_UPLOAD_MODULES = (
    "runtime/qwen4_exp_batch_gdn",
    "runtime/qwen4_exp_batch_ple",
    "runtime/qwen4_exp_batch_qsa",
    "runtime/qwen4_exp_mixed_target",
    "runtime/qwen4_exp_mtp",
    "runtime/qwen4_exp_mtp_prefill",
    "runtime/qwen4_exp_runner",
)

QWEN35_GGUF_MTP_NEXTN_UPLOAD_MODULES = (
    "runtime/qwen35_gguf_nextn",
    "runtime/qwen35_gguf_mtp",
)

MAPLE_UPLOAD_MODULES = (
    "runtime/maple",
    "runtime/maple_graph",
    "runtime/maple_cuda",
)

QWEN35_GGUF_RUNNER_UPLOAD_MODULES = (
    "runtime/qwen35_gguf_runner",
)

QWEN35_PARO_RUNNER_UPLOAD_MODULES = (
    "runtime/qwen35_paro_runner",
)

MOONSHINE_RUNTIME_UPLOAD_MODULES = (
    "runtime/moonshine",
    "runtime/moonshine_cuda",
    "runtime/moonshine_cuda_batch",
    "runtime/moonshine_cuda_continuous",
    "runtime/moonshine_encoder_cuda",
    "runtime/moonshine_encoder_cuda_batch",
)

EVIE_TIMESFM_NAMED_LOCAL_UPLOAD_MODULES = (
    "runtime/evie",
    "runtime/timesfm_decode",
    "runtime/timesfm3_decode",
)

DMS_SYNCHRONOUS_ARRAY_UPLOAD_MODULES = (
    "kvcache/dms_device",
    "runtime/dms_prefix",
    "runtime/dms_prefill_cache",
)

GGUF_RUNTIME_SYNCHRONOUS_ARRAY_UPLOAD_MODULES = (
    "runtime/gguf_decode_graph",
    "runtime/gguf_packed_decode_graph",
    "runtime/gguf_native_spec_cycle",
    "runtime/native_sampler",
)

SPECULATIVE_RUNTIME_SYNCHRONOUS_ARRAY_UPLOAD_MODULES = (
    "speculative/mtp_native",
    "speculative/mtp_resident_draft",
    "speculative/mtp_cached_draft",
    "speculative/laguna_dflash",
)


@pytest.mark.parametrize("module", [
    "runtime/evie", "runtime/timesfm_decode", "runtime/timesfm3_decode",
    "runtime/qwen35_gguf_runner", "runtime/moonshine", "runtime/qwen35_paro_runner",
    "speculative/mtp_cached_draft", "speculative/mtp_resident_draft",
    "kernels/hip_gfx1100/speculative/mtp_nextn",
    *QWEN4_EXP_AUXILIARY_UPLOAD_MODULES,
    *QWEN35_GGUF_MTP_NEXTN_UPLOAD_MODULES,
])
def test_reviewed_runners_do_not_erase_temporary_array_owners(module):
    """Bounded syntax guard, not a proof of arbitrary pointer lifetimes."""
    path = Path(__file__).parents[1] / "hipengine" / f"{module}.py"
    tree = ast.parse(path.read_text())
    hazards = [node.lineno for node in ast.walk(tree)
               if isinstance(node, ast.Call)
               and isinstance(node.func, ast.Name)
               and node.func.id == "host_array_ptr"
               and node.args and isinstance(node.args[0], ast.Call)]
    assert not hazards, f"{path.name}: retain array owners at lines {hazards}"


def _is_named_host_array_pointer(node):
    """Return whether ``node`` is ``host_array_ptr(named_local)``."""

    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "host_array_ptr"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Name)
    )


def _is_array_pointer_expression(node):
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == "host_array_ptr":
            return True
        if isinstance(node.func, ast.Name) and node.func.id == "int":
            return bool(node.args) and _is_array_pointer_expression(node.args[0])
        if isinstance(node.func, ast.Attribute) and node.func.attr == "addressof":
            return bool(node.args) and _is_array_pointer_expression(node.args[0])
        return (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in {"data_as", "ctypes"}
        )
    if isinstance(node, ast.Attribute) and node.attr == "data":
        owner = node.value
        return isinstance(owner, ast.Attribute) and owner.attr == "ctypes"
    return False


@pytest.mark.parametrize("module", QWEN4_EXP_AUXILIARY_UPLOAD_MODULES)
def test_qwen4exp_auxiliary_uploads_use_the_owning_helper(module):
    """No synchronous array upload may erase its source owner.

    This is a bounded syntax guard over the reviewed Qwen4Exp auxiliary
    modules, not a proof of arbitrary pointer lifetimes. Direct runtime
    ``memcpy`` calls and asynchronous uploads are outside this syntax guard;
    that exclusion does not establish their ownership safety.
    """
    path = Path(__file__).parents[1] / "hipengine" / f"{module}.py"
    tree = ast.parse(path.read_text())
    hazards = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "copy_host_to_device"
        ):
            continue
        source = (
            node.args[1]
            if len(node.args) > 1
            else next((kw.value for kw in node.keywords if kw.arg == "host_ptr"), None)
        )
        if source is not None and _is_array_pointer_expression(source):
            hazards.append(node.lineno)
    assert not hazards, f"{path.name}: use copy_host_array_to_device at lines {hazards}"


@pytest.mark.parametrize("module", MAPLE_UPLOAD_MODULES)
def test_maple_synchronous_array_uploads_use_the_owning_helper(module):
    """No synchronous array upload may erase its source owner.

    This is a bounded syntax guard over the Maple runtime modules, not a proof
    of arbitrary pointer lifetimes. Asynchronous uploads, raw pointer uploads,
    and D2H/D2D copies are outside this guard.
    """
    path = Path(__file__).parents[1] / "hipengine" / f"{module}.py"
    tree = ast.parse(path.read_text())
    hazards = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "copy_host_to_device":
            source = (
                node.args[1]
                if len(node.args) > 1
                else next((kw.value for kw in node.keywords if kw.arg == "host_ptr"), None)
            )
            if source is not None and _is_array_pointer_expression(source):
                hazards.append((node.lineno, "copy_host_to_device"))
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "memcpy"):
            continue
        kind = (
            node.args[3]
            if len(node.args) > 3
            else next((kw.value for kw in node.keywords if kw.arg == "kind"), None)
        )
        if not (isinstance(kind, ast.Attribute) and kind.attr == "HOST_TO_DEVICE"):
            continue
        source = (
            node.args[1]
            if len(node.args) > 1
            else next((kw.value for kw in node.keywords if kw.arg == "src"), None)
        )
        if source is not None and _is_array_pointer_expression(source):
            hazards.append((node.lineno, "runtime.memcpy"))
    assert not hazards, f"{path.name}: use copy_host_array_to_device at {hazards}"


@pytest.mark.parametrize("module", QWEN35_GGUF_RUNNER_UPLOAD_MODULES)
def test_qwen35_gguf_runner_synchronous_array_uploads_use_the_owning_helper(module):
    """No synchronous array upload may erase its source owner.

    This is a bounded syntax guard over ``qwen35_gguf_runner``, not a proof of
    arbitrary pointer lifetimes. Asynchronous uploads, raw pointer uploads, and
    D2H/D2D copies are outside this guard.
    """
    path = Path(__file__).parents[1] / "hipengine" / f"{module}.py"
    tree = ast.parse(path.read_text())
    hazards = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "copy_host_to_device":
            source = (
                node.args[1]
                if len(node.args) > 1
                else next((kw.value for kw in node.keywords if kw.arg == "host_ptr"), None)
            )
            if source is not None and _is_array_pointer_expression(source):
                hazards.append((node.lineno, "copy_host_to_device"))
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "memcpy"):
            continue
        kind = (
            node.args[3]
            if len(node.args) > 3
            else next((kw.value for kw in node.keywords if kw.arg == "kind"), None)
        )
        if not (isinstance(kind, ast.Attribute) and kind.attr == "HOST_TO_DEVICE"):
            continue
        source = (
            node.args[1]
            if len(node.args) > 1
            else next((kw.value for kw in node.keywords if kw.arg == "src"), None)
        )
        if source is not None and _is_array_pointer_expression(source):
            hazards.append((node.lineno, "runtime.memcpy"))
    assert not hazards, f"{path.name}: use copy_host_array_to_device at {hazards}"


@pytest.mark.parametrize("module", QWEN35_PARO_RUNNER_UPLOAD_MODULES)
def test_qwen35_paro_runner_synchronous_array_uploads_use_the_owning_helper(module):
    """No synchronous array upload may erase its source owner.

    This is a bounded syntax guard over ``qwen35_paro_runner``, not a proof of
    arbitrary pointer lifetimes. Asynchronous uploads, raw pointer uploads, and
    D2H/D2D copies are outside this guard.
    """
    path = Path(__file__).parents[1] / "hipengine" / f"{module}.py"
    tree = ast.parse(path.read_text())
    hazards = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "copy_host_to_device":
            source = (
                node.args[1]
                if len(node.args) > 1
                else next((kw.value for kw in node.keywords if kw.arg == "host_ptr"), None)
            )
            if source is not None and _is_array_pointer_expression(source):
                hazards.append((node.lineno, "copy_host_to_device"))
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "memcpy"):
            continue
        kind = (
            node.args[3]
            if len(node.args) > 3
            else next((kw.value for kw in node.keywords if kw.arg == "kind"), None)
        )
        if not (isinstance(kind, ast.Attribute) and kind.attr == "HOST_TO_DEVICE"):
            continue
        source = (
            node.args[1]
            if len(node.args) > 1
            else next((kw.value for kw in node.keywords if kw.arg == "src"), None)
        )
        if source is not None and _is_array_pointer_expression(source):
            hazards.append((node.lineno, "runtime.memcpy"))
    assert not hazards, f"{path.name}: use copy_host_array_to_device at {hazards}"


@pytest.mark.parametrize("module", MOONSHINE_RUNTIME_UPLOAD_MODULES)
def test_moonshine_runtime_synchronous_array_uploads_use_the_owning_helper(module):
    """No synchronous array upload may erase its source owner.

    This is a bounded syntax guard over the Moonshine runtime family, not a
    proof of arbitrary pointer lifetimes. Asynchronous uploads, raw pointer
    uploads, and D2H/D2D copies are outside this guard.
    """
    path = Path(__file__).parents[1] / "hipengine" / f"{module}.py"
    tree = ast.parse(path.read_text())
    hazards = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "copy_host_to_device":
            source = (
                node.args[1]
                if len(node.args) > 1
                else next((kw.value for kw in node.keywords if kw.arg == "host_ptr"), None)
            )
            if source is not None and _is_array_pointer_expression(source):
                hazards.append((node.lineno, "copy_host_to_device"))
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "memcpy"):
            continue
        kind = (
            node.args[3]
            if len(node.args) > 3
            else next((kw.value for kw in node.keywords if kw.arg == "kind"), None)
        )
        if not (isinstance(kind, ast.Attribute) and kind.attr == "HOST_TO_DEVICE"):
            continue
        source = (
            node.args[1]
            if len(node.args) > 1
            else next((kw.value for kw in node.keywords if kw.arg == "src"), None)
        )
        if source is not None and _is_array_pointer_expression(source):
            hazards.append((node.lineno, "runtime.memcpy"))
    assert not hazards, f"{path.name}: use copy_host_array_to_device at {hazards}"


@pytest.mark.parametrize("module", EVIE_TIMESFM_NAMED_LOCAL_UPLOAD_MODULES)
def test_evie_timesfm_named_local_uploads_use_the_owning_helper(module):
    """Named-local H2D arrays must also retain their owner through the copy.

    The general temporary-owner guard covers allocating expressions in the
    pointer argument. These modules bind the array to a local first, which is
    sufficient for lifetime but still uses the pointer-only API. This guard
    pins the owning-helper migration for that named-local form, including
    direct ``runtime.memcpy(..., HOST_TO_DEVICE)`` calls.
    """
    path = Path(__file__).parents[1] / "hipengine" / f"{module}.py"
    tree = ast.parse(path.read_text())
    hazards = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "copy_host_to_device":
            source = (
                node.args[1]
                if len(node.args) > 1
                else next((kw.value for kw in node.keywords if kw.arg == "host_ptr"), None)
            )
            if source is not None and _is_named_host_array_pointer(source):
                hazards.append((node.lineno, "copy_host_to_device"))
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "memcpy"):
            continue
        kind = (
            node.args[3]
            if len(node.args) > 3
            else next((kw.value for kw in node.keywords if kw.arg == "kind"), None)
        )
        if not (isinstance(kind, ast.Attribute) and kind.attr == "HOST_TO_DEVICE"):
            continue
        source = (
            node.args[1]
            if len(node.args) > 1
            else next((kw.value for kw in node.keywords if kw.arg == "src"), None)
        )
        if source is not None and _is_named_host_array_pointer(source):
            hazards.append((node.lineno, "runtime.memcpy"))
    assert not hazards, f"{path.name}: use copy_host_array_to_device at {hazards}"


@pytest.mark.parametrize("module", DMS_SYNCHRONOUS_ARRAY_UPLOAD_MODULES)
def test_dms_synchronous_array_uploads_use_the_owning_helper(module):
    """Named-local DMS H2D arrays must retain their owner through the copy.

    This is a bounded syntax guard over the DMS device, prefix, and prefill
    cache modules, not a proof of arbitrary pointer lifetimes. Asynchronous
    uploads, raw pointer uploads, and D2H/D2D copies are outside this guard.
    """
    path = Path(__file__).parents[1] / "hipengine" / f"{module}.py"
    tree = ast.parse(path.read_text())
    hazards = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "copy_host_to_device":
            source = (
                node.args[1]
                if len(node.args) > 1
                else next((kw.value for kw in node.keywords if kw.arg == "host_ptr"), None)
            )
            if source is not None and _is_named_host_array_pointer(source):
                hazards.append((node.lineno, "copy_host_to_device"))
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "memcpy"):
            continue
        kind = (
            node.args[3]
            if len(node.args) > 3
            else next((kw.value for kw in node.keywords if kw.arg == "kind"), None)
        )
        if not (isinstance(kind, ast.Attribute) and kind.attr == "HOST_TO_DEVICE"):
            continue
        source = (
            node.args[1]
            if len(node.args) > 1
            else next((kw.value for kw in node.keywords if kw.arg == "src"), None)
        )
        if source is not None and _is_named_host_array_pointer(source):
            hazards.append((node.lineno, "runtime.memcpy"))
    assert not hazards, f"{path.name}: use copy_host_array_to_device at {hazards}"


@pytest.mark.parametrize("module", GGUF_RUNTIME_SYNCHRONOUS_ARRAY_UPLOAD_MODULES)
def test_gguf_runtime_synchronous_array_uploads_use_the_owning_helper(module):
    """GGUF graph and sampler H2D arrays must retain their owner through the copy.

    This is a bounded syntax guard over the GGUF decode graph, packed decode
    graph, native speculative cycle, and native sampler modules, not a proof of
    arbitrary pointer lifetimes. Asynchronous uploads, raw pointer uploads, and
    D2H/D2D copies are outside this guard.
    """
    path = Path(__file__).parents[1] / "hipengine" / f"{module}.py"
    tree = ast.parse(path.read_text())
    hazards = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "copy_host_to_device":
            source = (
                node.args[1]
                if len(node.args) > 1
                else next((kw.value for kw in node.keywords if kw.arg == "host_ptr"), None)
            )
            if source is not None and _is_array_pointer_expression(source):
                hazards.append((node.lineno, "copy_host_to_device"))
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "memcpy"):
            continue
        kind = (
            node.args[3]
            if len(node.args) > 3
            else next((kw.value for kw in node.keywords if kw.arg == "kind"), None)
        )
        if not (isinstance(kind, ast.Attribute) and kind.attr == "HOST_TO_DEVICE"):
            continue
        source = (
            node.args[1]
            if len(node.args) > 1
            else next((kw.value for kw in node.keywords if kw.arg == "src"), None)
        )
        if source is not None and _is_array_pointer_expression(source):
            hazards.append((node.lineno, "runtime.memcpy"))
    assert not hazards, f"{path.name}: use copy_host_array_to_device at {hazards}"


@pytest.mark.parametrize("module", QWEN35_GGUF_MTP_NEXTN_UPLOAD_MODULES)
def test_qwen35_gguf_mtp_nextn_synchronous_array_uploads_use_the_owning_helper(module):
    """No synchronous array upload may erase its source owner.

    This is a bounded syntax guard over the reviewed Qwen3.5 GGUF NextN/MTP
    runtime modules, not a proof of arbitrary pointer lifetimes. Direct
    runtime ``memcpy`` calls and asynchronous uploads are outside this syntax
    guard; that exclusion does not establish their ownership safety.
    """
    path = Path(__file__).parents[1] / "hipengine" / f"{module}.py"
    tree = ast.parse(path.read_text())
    hazards = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "copy_host_to_device"
        ):
            continue
        source = (
            node.args[1]
            if len(node.args) > 1
            else next((kw.value for kw in node.keywords if kw.arg == "host_ptr"), None)
        )
        if source is not None and _is_array_pointer_expression(source):
            hazards.append(node.lineno)
    assert not hazards, f"{path.name}: use copy_host_array_to_device at lines {hazards}"


def test_qwen35_gguf_mtp_copy_array_uses_the_owning_upload(monkeypatch):
    from hipengine.runtime import qwen35_gguf_mtp as mtp

    calls = []

    def upload(buffer, array, nbytes=None, *, runtime=None):
        calls.append((buffer, array, nbytes, runtime))

    monkeypatch.setattr(mtp, "copy_host_array_to_device", upload)
    tensor = SimpleNamespace(ptr=0x4000, numel=4, dtype=SimpleNamespace(itemsize=4))
    values = np.arange(8, dtype=np.int32)[::2]
    runtime = object()

    mtp._copy_array(tensor, values, runtime)

    assert len(calls) == 1
    buffer, array, nbytes, call_runtime = calls[0]
    assert (buffer.ptr, buffer.nbytes, nbytes, call_runtime) == (
        0x4000,
        16,
        16,
        runtime,
    )
    assert array.flags.c_contiguous
    assert array.tolist() == [0, 2, 4, 6]


def test_qwen35_gguf_nextn_restore_checkpoint_uploads_owning_cursor_arrays(
    monkeypatch,
):
    from hipengine.runtime import qwen35_gguf_nextn as nextn

    calls = []

    def upload(buffer, array, nbytes=None, *, runtime=None):
        calls.append((buffer, array, nbytes, runtime))

    monkeypatch.setattr(nextn, "copy_host_array_to_device", upload)
    position_host = np.zeros(1, dtype=np.int64)
    context_host = np.zeros(1, dtype=np.int64)
    slot = SimpleNamespace(
        position_host=position_host,
        context_host=context_host,
        position_buf=memory.DeviceBuffer(0x3000, 8),
        context_buf=memory.DeviceBuffer(0x4000, 8),
    )
    executor = object.__new__(nextn.Qwen35GGUFNextNExecutor)
    executor._request_slots = {7: 0}
    executor._batch_sessions = None
    executor.scratch = SimpleNamespace(
        for_slot=lambda index, span_role="decode": slot
    )
    executor.runtime = object()
    checkpoint = SimpleNamespace(
        released=False,
        request_id=7,
        slot=0,
        state_pairs=(),
        position=17,
        context_length=18,
    )

    executor.restore_request_checkpoint(checkpoint)

    assert [call[0].ptr for call in calls] == [0x3000, 0x4000]
    assert [call[1] for call in calls] == [position_host, context_host]
    assert [call[2] for call in calls] == [8, 8]
    assert all(call[3] is executor.runtime for call in calls)
    assert position_host[0] == 17
    assert context_host[0] == 18


def test_qwen4exp_qsa_row_upload_uses_the_owning_helper():
    """A direct row-array H2D memcpy must not erase its host row view."""
    path = (
        Path(__file__).parents[1]
        / "hipengine/runtime/qwen4_exp_batch_qsa.py"
    )
    tree = ast.parse(path.read_text())
    hazards = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "memcpy"
        ):
            continue
        kind = (
            node.args[3]
            if len(node.args) > 3
            else next((kw.value for kw in node.keywords if kw.arg == "kind"), None)
        )
        if isinstance(kind, ast.Attribute) and kind.attr == "HOST_TO_DEVICE":
            hazards.append(node.lineno)
    assert not hazards, f"{path.name}: use copy_host_array_to_device at lines {hazards}"


def test_upload_retains_temporary_through_copy():
    refs = []

    def source():
        array = np.arange(8, dtype=np.int32)
        refs.append(weakref.ref(array))
        return array

    def memcpy(dst, src, nbytes, kind):
        assert refs[0]() is not None
        assert nbytes == 32
        assert list((ctypes.c_int32 * 8).from_address(src)) == list(range(8))

    memory.copy_host_array_to_device(
        memory.DeviceBuffer(1234, 64), source(),
        runtime=SimpleNamespace(memcpy=memcpy),
    )
    assert refs[0]() is None


@pytest.mark.parametrize("case", ["source_bounds", "destination_bounds", "strided"])
def test_upload_rejects_invalid_range_before_runtime(case):
    array = np.arange(8, dtype=np.int32)
    buffer = memory.DeviceBuffer(1234, 32)
    nbytes = 36 if case == "source_bounds" else None
    if case == "destination_bounds":
        buffer = memory.DeviceBuffer(1234, 16)
    if case == "strided":
        array = array[::2]
    with pytest.raises(ValueError):
        memory.copy_host_array_to_device(buffer, array, nbytes, runtime=object())


def test_upload_accepts_owned_contiguous_view_and_explicit_prefix():
    array = np.arange(8, dtype=np.int32)[2:]
    seen = []
    memory.copy_host_array_to_device(
        memory.DeviceBuffer(1234, 8), array, 8,
        runtime=SimpleNamespace(memcpy=lambda dst, src, size, kind: seen.append(
            list((ctypes.c_int32 * 2).from_address(src)))),
    )
    assert seen == [[2, 3]]


@pytest.mark.parametrize("strided", [False, True])
def test_moonshine_encoder_upload_retains_contiguous_source(monkeypatch, strided):
    from hipengine.runtime import moonshine

    hidden = np.arange(16, dtype=np.float16).reshape(1, 4, 4)
    mask = np.ones((1, 4), dtype=np.int32)
    if strided:
        hidden = hidden[:, ::2, :]
        mask = mask[:, ::2]
    refs = []
    pointer = memory.host_array_ptr

    def watched_pointer(array):
        refs.append(weakref.ref(array))
        return pointer(array)

    # Observe both the raw-pointer and owning APIs without retaining the source.
    monkeypatch.setattr(memory, "host_array_ptr", watched_pointer)
    monkeypatch.setattr(moonshine, "host_array_ptr", watched_pointer)
    copies = []

    def memcpy(dst, src, size, kind):
        assert refs[-1]() is not None, "upload source was freed before memcpy"
        copies.append(ctypes.string_at(src, size))

    buffers = {
        "encoder_hidden": memory.DeviceBuffer(4096, hidden.nbytes),
        "encoder_attention_mask": memory.DeviceBuffer(8192, mask.nbytes),
    }
    runner = object.__new__(moonshine.MoonshineResidentRuntime)
    runner.closed = False
    runner.spec = SimpleNamespace(hidden_size=4)
    runner.encoder_frames = hidden.shape[1]
    runner.self_cache_length = 0
    runner.decode_position = None
    runner.runtime = SimpleNamespace(memcpy=memcpy)
    runner.workspace = SimpleNamespace(
        allocation=lambda name: SimpleNamespace(buffer=buffers[name]))
    runner.set_encoder_state(hidden, mask)
    assert copies == [hidden.tobytes(), mask.tobytes()]
    assert runner.encoder_state_valid
    assert not runner.cross_cache_valid


def test_evie_pointer_cache_uploads_only_new_content(monkeypatch):
    from hipengine.runtime import evie

    uploads = []
    allocations = []

    def allocate(nbytes):
        buffer = memory.DeviceBuffer(4096 + len(allocations) * 4096, nbytes)
        allocations.append(buffer)
        return buffer

    def upload(buffer, array, nbytes=None, *, runtime=None):
        uploads.append(np.asarray(array).tolist())

    monkeypatch.setattr(evie, "_malloc_committed", allocate)
    monkeypatch.setattr(evie, "copy_host_array_to_device", upload, raising=False)
    runner = object.__new__(evie.EvieRunner)
    a = runner._dev_ptr_array([123, 456])
    assert runner._dev_ptr_array([123, 456]) == a
    assert runner._dev_ptr_array([456, 123]) != a
    assert uploads == [[123, 456], [456, 123]]


def test_evie_to_dev_uploads_owning_named_array(monkeypatch):
    from hipengine.runtime import evie

    calls = []

    def upload(buffer, array, nbytes=None, *, runtime=None):
        calls.append((buffer, array, nbytes, runtime))

    monkeypatch.setattr(
        evie,
        "_malloc_committed",
        lambda nbytes: memory.DeviceBuffer(0x9000, nbytes),
    )
    monkeypatch.setattr(evie, "copy_host_array_to_device", upload, raising=False)
    runner = object.__new__(evie.EvieRunner)
    runner._misc_buffers = []
    source = np.arange(4, dtype=np.float32)

    buffer = runner._to_dev(source)

    assert buffer.ptr == 0x9000
    assert len(calls) == 1
    call_buffer, array, nbytes, call_runtime = calls[0]
    assert call_buffer is buffer
    assert nbytes == source.nbytes
    assert call_runtime is None
    assert array.flags.c_contiguous
    assert array.tolist() == source.tolist()
    assert runner._misc_buffers == [buffer]


def _fake_timesfm_loaded(names, *, head_dim, layers=0):
    tensors = {}
    for index, name in enumerate(names):
        tensors[name] = SimpleNamespace(
            source=SimpleNamespace(shape=(1, 1)),
            buffer=memory.DeviceBuffer(0x8000 + index * 4, 4),
        )
    return SimpleNamespace(
        spec=SimpleNamespace(
            head_dim=head_dim,
            num_hidden_layers=layers,
            num_layers=layers,
            model_dims=4,
        ),
        weights=SimpleNamespace(tensors=tensors),
    )


def test_timesfm_decode_constructor_uploads_owning_named_arrays(monkeypatch):
    from hipengine.runtime import timesfm_decode as mod

    uploads = []

    def upload(buffer, array, nbytes=None, *, runtime=None):
        uploads.append((buffer, array, nbytes, runtime))

    allocations = []

    def allocate(nbytes):
        buffer = memory.DeviceBuffer(0x1000 + len(allocations) * 0x100, nbytes)
        allocations.append(buffer)
        return buffer

    monkeypatch.setattr(mod, "malloc", allocate)
    monkeypatch.setattr(mod, "copy_device_to_host", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod, "copy_host_array_to_device", upload, raising=False)
    loaded = _fake_timesfm_loaded(
        (
            *mod._GEMM_WEIGHT_NAMES,
            "tokenizer.hidden_layer.bias",
            "tokenizer.output_layer.bias",
            "tokenizer.residual_layer.bias",
        ),
        head_dim=4,
    )

    decoder = mod.TimesFMGPUDecoder(loaded, rocblas=object(), precision="fp16")

    assert len(uploads) == 1 + len(mod._GEMM_WEIGHT_NAMES)
    timescale_buffer, timescale, timescale_nbytes, timescale_runtime = uploads[0]
    assert timescale_buffer is decoder._timescale
    assert timescale.dtype == np.float32
    assert timescale.nbytes == 8
    assert timescale_nbytes is None
    assert timescale_runtime is None
    for buffer, array, nbytes, call_runtime in uploads[1:]:
        assert nbytes is None
        assert call_runtime is None
        assert array.dtype == np.float16
        assert array.flags.c_contiguous
        assert array.nbytes == 2
        assert buffer.nbytes == 2


def test_timesfm3_decode_constructor_uploads_owning_named_arrays(monkeypatch):
    from hipengine.runtime import timesfm3_decode as mod

    uploads = []

    def upload(buffer, array, nbytes=None, *, runtime=None):
        uploads.append((buffer, array, nbytes, runtime))

    allocations = []

    def allocate(nbytes):
        buffer = memory.DeviceBuffer(0x2000 + len(allocations) * 0x100, nbytes)
        allocations.append(buffer)
        return buffer

    names = (
        "pre_transformer_resblock.hidden_layer.weight",
        "pre_transformer_resblock.output_layer.weight",
        "pre_transformer_resblock.residual_layer.weight",
        "output_head.weight",
        "output_head.bias",
    )
    monkeypatch.setattr(mod, "malloc", allocate)
    monkeypatch.setattr(mod, "copy_device_to_host", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod, "copy_host_array_to_device", upload, raising=False)
    loaded = _fake_timesfm_loaded(names, head_dim=4)

    decoder = mod.TimesFM3GPUDecoder(loaded, rocblas=object(), precision="fp16")

    assert len(uploads) == 1 + 4
    timescale_buffer, timescale, timescale_nbytes, timescale_runtime = uploads[0]
    assert timescale_buffer is decoder._timescale
    assert timescale.dtype == np.float32
    assert timescale.nbytes == 8
    assert timescale_nbytes is None
    assert timescale_runtime is None
    for buffer, array, nbytes, call_runtime in uploads[1:]:
        assert nbytes is None
        assert call_runtime is None
        assert array.dtype == np.float16
        assert array.flags.c_contiguous
        assert array.nbytes == 2
        assert buffer.nbytes == 2


def test_qwen35_gguf_runner_packed_verify_token_upload_uses_owning_array(monkeypatch):
    from hipengine.runtime import qwen35_gguf_runner as runner_mod

    calls = []

    def upload(buffer, array, nbytes=None, *, runtime):
        calls.append((buffer, array, nbytes, runtime))

    monkeypatch.setattr(runner_mod, "copy_host_array_to_device", upload)
    layout = SimpleNamespace(
        input_token_ids=np.asarray([11, 12, 13], dtype=np.int64),
        slot_count=1,
    )
    destination = memory.DeviceBuffer(0x5000, 64)
    runtime = object()

    runner_mod._stage_gguf_packed_verify_token_ids(
        layout,
        [{}],
        destination,
        runtime=runtime,
        stream=0,
        runtime_state_library=object(),
    )

    assert len(calls) == 1
    buffer, array, nbytes, call_runtime = calls[0]
    assert (buffer.ptr, buffer.nbytes, nbytes, call_runtime) == (
        0x5000,
        24,
        24,
        runtime,
    )
    assert array.flags.c_contiguous
    assert array.tolist() == [11, 12, 13]


def test_qwen35_gguf_runner_full_attention_position_uploads_owning_arrays(monkeypatch):
    from hipengine.runtime import qwen35_gguf_runner as runner_mod

    calls = []

    def upload(buffer, array, nbytes=None, *, runtime):
        calls.append((buffer, array, nbytes, runtime))

    monkeypatch.setattr(runner_mod, "copy_host_array_to_device", upload)
    session = object.__new__(runner_mod._FullStackScratch)
    object.__setattr__(session, "max_positions", 16)
    object.__setattr__(session, "position_host", np.zeros(1, dtype=np.int64))
    object.__setattr__(session, "context_host", np.zeros(1, dtype=np.int64))
    object.__setattr__(session, "position_buf", memory.DeviceBuffer(0x6000, 8))
    object.__setattr__(session, "context_buf", memory.DeviceBuffer(0x7000, 8))
    runtime = object()

    session.set_full_attention_position(7, runtime)

    assert [call[0].ptr for call in calls] == [0x6000, 0x7000]
    assert [call[1] for call in calls] == [session.position_host, session.context_host]
    assert [call[2] for call in calls] == [None, None]
    assert all(call[3] is runtime for call in calls)
    assert session.position_host[0] == 7
    assert session.context_host[0] == 8


def test_qwen35_gguf_runner_sidecar_upload_owns_contiguous_copy(monkeypatch):
    from hipengine.runtime import qwen35_gguf_runner as runner_mod

    calls = []

    def upload(buffer, array, nbytes=None, *, runtime):
        calls.append((buffer, array, nbytes, runtime))

    monkeypatch.setattr(
        runner_mod,
        "malloc",
        lambda nbytes, *, runtime: memory.DeviceBuffer(0x8000, int(nbytes)),
    )
    monkeypatch.setattr(runner_mod, "copy_host_array_to_device", upload)
    runtime = object()
    source = np.arange(8, dtype=np.int32)[::2]

    buffer = runner_mod._copy_sidecar_array_to_device(source, runtime=runtime)

    assert (buffer.ptr, buffer.nbytes) == (0x8000, 16)
    assert len(calls) == 1
    call_buffer, array, nbytes, call_runtime = calls[0]
    assert call_buffer is buffer
    assert nbytes is None
    assert call_runtime is runtime
    assert array.flags.c_contiguous
    assert array.tolist() == [0, 2, 4, 6]


def test_qwen35_paro_runner_dev_uploads_owning_array(monkeypatch):
    from hipengine.runtime import qwen35_paro_runner as runner_mod

    calls = []

    def upload(buffer, array, nbytes=None, *, runtime):
        calls.append((buffer, array, nbytes, runtime))

    monkeypatch.setattr(runner_mod, "copy_host_array_to_device", upload)
    monkeypatch.setattr(
        runner_mod,
        "malloc",
        lambda nbytes, *, runtime: memory.DeviceBuffer(0xA000, int(nbytes)),
    )
    session = object.__new__(runner_mod.Qwen35ParoResidentSession)
    session.runtime = object()
    session.buffers = []
    source = np.arange(8, dtype=np.int32)[::2]

    buffer = session._dev(source)

    assert (buffer.ptr, buffer.nbytes) == (0xA000, source.nbytes)
    assert len(calls) == 1
    call_buffer, array, nbytes, call_runtime = calls[0]
    assert call_buffer is buffer
    assert nbytes is None
    assert call_runtime is session.runtime
    assert array.flags.c_contiguous
    assert array.tolist() == [0, 2, 4, 6]
    assert session.buffers == [buffer]


def test_qwen35_paro_runner_copy_zero_uploads_owning_array(monkeypatch):
    from hipengine.runtime import qwen35_paro_runner as runner_mod

    calls = []

    def upload(buffer, array, nbytes=None, *, runtime):
        calls.append((buffer, array, nbytes, runtime))

    monkeypatch.setattr(runner_mod, "copy_host_array_to_device", upload)
    runtime = object()
    buffer = memory.DeviceBuffer(0x9000, 16)
    zeros = np.zeros(4, dtype=np.int32)

    runner_mod._copy_zero(runtime, buffer, zeros)

    assert len(calls) == 1
    call_buffer, array, nbytes, call_runtime = calls[0]
    assert call_buffer is buffer
    assert array is zeros
    assert nbytes == 16
    assert call_runtime is runtime


def test_qwen35_paro_runner_copy_zero_keeps_destination_count_with_unequal_template(monkeypatch):
    from hipengine.runtime import qwen35_paro_runner as runner_mod

    calls = []

    def upload(buffer, array, nbytes=None, *, runtime):
        calls.append((buffer, array, nbytes, runtime))

    monkeypatch.setattr(runner_mod, "copy_host_array_to_device", upload)
    runtime = object()
    buffer = memory.DeviceBuffer(0x9000, 16)
    zeros = np.zeros(8, dtype=np.int32)

    runner_mod._copy_zero(runtime, buffer, zeros)

    assert len(calls) == 1
    call_buffer, array, nbytes, call_runtime = calls[0]
    assert call_buffer is buffer
    assert array is zeros
    assert nbytes == 16
    assert call_runtime is runtime


def test_moonshine_batch_cross_cache_row_uploads_owning_offsets(monkeypatch):
    from hipengine.runtime import moonshine_cuda_batch as runner_mod

    calls = []

    def upload(buffer, array, nbytes=None, *, runtime):
        calls.append((buffer, array, nbytes, runtime))

    monkeypatch.setattr(runner_mod, "copy_host_array_to_device", upload)
    mask_buffer = memory.DeviceBuffer(0x3000, 3 * 4)
    workspace = SimpleNamespace(
        allocation=lambda name: SimpleNamespace(buffer=mask_buffer)
    )
    runtime = SimpleNamespace(
        stream_synchronize=lambda stream: None,
        memcpy=lambda *args: None,
    )
    session = object.__new__(runner_mod.MoonshineCudaBatchRuntime)
    session.closed = False
    session.spec = SimpleNamespace(decoder_layers=1, decoder_kv_heads=2, head_dim=4)
    session.max_batch = 3
    session.encoder_frames = 3
    session.runtime = runtime
    session.stream = 0
    session.workspace = workspace
    session.cross_cache = lambda layer: SimpleNamespace(
        key=SimpleNamespace(ptr=0x1000),
        value=SimpleNamespace(ptr=0x2000),
    )
    session.cross_cache_valid = False
    session.encoder_state_valid = False
    key = np.arange(2 * 3 * 4, dtype=np.float16).reshape(2, 3, 4)
    value = (key + 1).astype(np.float16)
    mask = np.array([1, 0, 1], dtype=np.int32)

    session.load_cross_cache_row(1, [key], [value], mask=mask)

    assert [(call[0].ptr, call[2]) for call in calls] == [
        (0x1030, 48),
        (0x2030, 48),
        (0x300C, 12),
    ]
    assert calls[0][1].tolist() == key.tolist()
    assert calls[1][1].tolist() == value.tolist()
    assert calls[2][1].tolist() == mask.tolist()
    assert session.cross_cache_valid is True
    assert session.encoder_state_valid is True


def test_moonshine_encoder_audio_upload_keeps_exact_bucket_bytes(monkeypatch):
    from hipengine.runtime import moonshine_encoder_cuda as runner_mod

    calls = []

    def upload(buffer, array, nbytes=None, *, runtime):
        calls.append((buffer, array, nbytes, runtime))

    monkeypatch.setattr(runner_mod, "copy_host_array_to_device", upload)
    audio_buffer = memory.DeviceBuffer(0x4000, 16000 * 2)
    workspace = SimpleNamespace(
        allocation=lambda name: SimpleNamespace(buffer=audio_buffer)
    )
    runtime = object()
    session = object.__new__(runner_mod.MoonshineCudaEncoderRuntime)
    session.closed = False
    session.audio_samples = 16000
    session.encoder_frames = 40
    session.workspace = workspace
    session.runtime = runtime
    session._upload_mask = lambda *args, **kwargs: None
    audio = np.ones((1, 9727), dtype=np.float32)

    session.upload_input(audio)

    assert len(calls) == 1
    buffer, array, nbytes, call_runtime = calls[0]
    assert buffer is audio_buffer
    assert array.dtype == np.float16
    assert array.flags.c_contiguous
    assert nbytes == 9727 * np.dtype(np.float16).itemsize
    assert call_runtime is runtime
    assert session._last_real_samples == 9727
    assert session._last_real_frames == 24
    assert session._input_uploaded is True


def test_qwen35_gguf_runner_zero_upload_keeps_destination_count(monkeypatch):
    from hipengine.runtime import qwen35_gguf_runner as runner_mod

    calls = []

    def upload(buffer, array, nbytes=None, *, runtime):
        calls.append((buffer, array, nbytes, runtime))

    monkeypatch.setattr(runner_mod, "copy_host_array_to_device", upload)
    runtime = SimpleNamespace(
        memset=lambda *args, **kwargs: pytest.fail("mismatched zero template used memset")
    )
    buffer = memory.DeviceBuffer(0x9000, 16)
    zeros = np.zeros(8, dtype=np.int32)

    runner_mod._zero(runtime, buffer, zeros, stream=0)

    assert len(calls) == 1
    call_buffer, array, nbytes, call_runtime = calls[0]
    assert call_buffer is buffer
    assert array is zeros
    assert nbytes == 16
    assert call_runtime is runtime


def test_dms_device_upload_uses_owning_contiguous_array(monkeypatch):
    from hipengine.kvcache import dms_device as mod

    uploads = []
    legacy = []

    def upload(buffer, array, nbytes=None, *, runtime=None):
        uploads.append((buffer, array, nbytes, runtime))

    def old_upload(*args, **kwargs):
        legacy.append((args, kwargs))

    monkeypatch.setattr(mod, "copy_host_array_to_device", upload, raising=False)
    monkeypatch.setattr(mod, "copy_host_to_device", old_upload, raising=False)
    buffer = memory.DeviceBuffer(0x1000, 16)
    store = object.__new__(mod.DMSDevicePayloadStore)
    store._stg = {"pack_k": buffer}
    source = np.arange(8, dtype=np.int32)[::2]

    store._upload("pack_k", source)

    assert legacy == []
    assert len(uploads) == 1
    call_buffer, array, nbytes, call_runtime = uploads[0]
    assert call_buffer is buffer
    assert array.flags.c_contiguous
    assert array.tolist() == [0, 2, 4, 6]
    assert nbytes == 16
    assert call_runtime is None


def test_dms_device_configure_layer_uploads_owning_metadata(monkeypatch):
    from hipengine.kvcache import dms_device as mod

    uploads = []
    legacy = []

    def upload(buffer, array, nbytes=None, *, runtime=None):
        uploads.append((buffer, array, nbytes, runtime))

    def old_upload(*args, **kwargs):
        legacy.append((args, kwargs))

    monkeypatch.setattr(mod, "copy_host_array_to_device", upload, raising=False)
    monkeypatch.setattr(mod, "copy_host_to_device", old_upload, raising=False)
    store = object.__new__(mod.DMSDevicePayloadStore)
    store._closed = False
    store._layers = 1
    store._heads = 2
    store._base_meta = [memory.DeviceBuffer(0x2000, 8)]
    store._capacity_meta = [memory.DeviceBuffer(0x3000, 8)]
    store._live_meta = [memory.DeviceBuffer(0x4000, 8)]
    store._ensure_layer = lambda layer: None

    store.configure_layer(
        0,
        base=np.array([1, 2], dtype=np.int32),
        capacity=np.array([3, 4], dtype=np.int32),
        live=np.array([5, 6], dtype=np.int32),
    )

    assert legacy == []
    assert [(call[0].ptr, call[2], call[3]) for call in uploads] == [
        (0x2000, 8, None),
        (0x3000, 8, None),
        (0x4000, 8, None),
    ]
    assert [call[1].tolist() for call in uploads] == [[1, 2], [3, 4], [5, 6]]


def test_dms_device_bind_row_spans_uploads_owning_rows(monkeypatch):
    from hipengine.kvcache import dms_device as mod

    uploads = []
    legacy = []

    def upload(buffer, array, nbytes=None, *, runtime=None):
        uploads.append((buffer, array, nbytes, runtime))

    def old_upload(*args, **kwargs):
        legacy.append((args, kwargs))

    monkeypatch.setattr(mod, "copy_host_array_to_device", upload, raising=False)
    monkeypatch.setattr(mod, "copy_host_to_device", old_upload, raising=False)
    store = object.__new__(mod.DMSDevicePayloadStore)
    store._closed = False
    store._layers = 1
    store._heads = 2
    store._max_rows = 2
    store._rows_capacity = 2
    store._rows_base = [memory.DeviceBuffer(0x5000, 16)]
    store._rows_live = [memory.DeviceBuffer(0x6000, 16)]
    store._ensure_layer = lambda layer: None

    pointers = store.bind_row_spans(
        0,
        base=np.array([[1, 2], [3, 4]], dtype=np.int32),
        live=np.array([[5, 6], [7, 8]], dtype=np.int32),
    )

    assert pointers == (0x5000, 0x6000)
    assert legacy == []
    assert [(call[0].ptr, call[2], call[3]) for call in uploads] == [
        (0x5000, 16, None),
        (0x6000, 16, None),
    ]
    assert [call[1].tolist() for call in uploads] == [
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]],
    ]


def test_dms_device_restore_uploads_owning_planes_with_offsets(monkeypatch):
    from hipengine.kvcache import dms_device as mod

    uploads = []
    legacy = []

    def upload(buffer, array, nbytes=None, *, runtime=None):
        uploads.append((buffer, array, nbytes, runtime))

    def old_upload(*args, **kwargs):
        legacy.append((args, kwargs))

    monkeypatch.setattr(mod, "copy_host_array_to_device", upload, raising=False)
    monkeypatch.setattr(mod, "copy_host_to_device", old_upload, raising=False)
    store = object.__new__(mod.DMSDevicePayloadStore)
    store._closed = False
    store._layers = 1
    store._heads = 1
    store._slots = 4
    store._dim = 2
    store._payload_dtype = np.dtype(np.uint16)
    store._k_scales = []
    store._v_scales = []
    store._k_slot = [memory.DeviceBuffer(0x7000, 16)]
    store._v_slot = [memory.DeviceBuffer(0x8000, 16)]
    store._positions = [memory.DeviceBuffer(0x9000, 16)]
    store._slot_evict = [memory.DeviceBuffer(0xA000, 4)]
    store._live_meta = [memory.DeviceBuffer(0xB000, 4)]
    store._ensure_layer = lambda layer: None
    snapshot = mod.DMSDevicePayloadSnapshot(
        extents=(
            mod.DMSDeviceExtentSnapshot(
                layer=0,
                head=0,
                start=2,
                length=1,
                k_bits=np.array([[1, 2]], dtype=np.uint16),
                v_bits=np.array([[3, 4]], dtype=np.uint16),
                positions=np.array([5], dtype=np.int32),
                evict=np.array([1], dtype=np.uint8),
            ),
        ),
        live_counts=np.array([[4]], dtype=np.int32),
    )

    store.restore(snapshot)

    assert legacy == []
    assert [
        (call[0].ptr, call[0].nbytes, call[2], call[3]) for call in uploads
    ] == [
        (0x7008, 4, 4, None),
        (0x8008, 4, 4, None),
        (0x9008, 4, 4, None),
        (0xA002, 1, 1, None),
        (0xB000, 4, 4, None),
    ]
    assert [call[1].reshape(-1).tolist() for call in uploads] == [
        [1, 2],
        [3, 4],
        [5],
        [1],
        [4],
    ]


def test_dms_prefix_restore_recurrent_prefix_uploads_owning_arrays(monkeypatch):
    from hipengine.runtime import dms_prefix as mod

    uploads = []
    legacy = []

    def upload(buffer, array, nbytes=None, *, runtime=None):
        uploads.append((buffer, array, nbytes, runtime))

    def old_upload(*args, **kwargs):
        legacy.append((args, kwargs))

    monkeypatch.setattr(mod, "copy_host_array_to_device", upload, raising=False)
    monkeypatch.setattr(mod, "copy_host_to_device", old_upload, raising=False)
    conv = memory.DeviceBuffer(0x1000, 8)
    recurrent = memory.DeviceBuffer(0x2000, 8)
    session = SimpleNamespace(
        scratch=SimpleNamespace(
            layer_conv_states=[conv],
            layer_recurrent_states=[recurrent],
        )
    )
    first = np.arange(2, dtype=np.int32)
    second = np.arange(2, dtype=np.int32) + 2
    snapshot = SimpleNamespace(recurrent_and_logits=(first, second))

    mod.restore_recurrent_prefix(session, snapshot)

    assert legacy == []
    assert [(call[0].ptr, call[2], call[3]) for call in uploads] == [
        (0x1000, 8, None),
        (0x2000, 8, None),
    ]
    assert [call[1].tolist() for call in uploads] == [[0, 1], [2, 3]]


def test_dms_prefix_restore_prompt_uploads_owning_arrays_after_restore(monkeypatch):
    from hipengine.runtime import dms_prefix as mod

    calls = []
    legacy = []

    def upload(buffer, array, nbytes=None, *, runtime=None):
        calls.append(("upload", buffer, array, nbytes, runtime))

    def old_upload(*args, **kwargs):
        legacy.append((args, kwargs))

    def restore_prefix(backend, layer, compact):
        calls.append(("restore_prefix", backend, layer, compact))

    monkeypatch.setattr(mod, "copy_host_array_to_device", upload, raising=False)
    monkeypatch.setattr(mod, "copy_host_to_device", old_upload, raising=False)
    monkeypatch.setattr(mod, "restore_prefix", restore_prefix)
    conv = memory.DeviceBuffer(0x3000, 8)
    recurrent = memory.DeviceBuffer(0x4000, 8)
    first = np.arange(2, dtype=np.int32)
    second = np.arange(2, dtype=np.int32) + 2

    class Backend:
        def estimate(self, request, none, detail):
            return 1

        def reserve(self, nbytes):
            calls.append(("reserve", nbytes))

        def close(self):
            pass

    backend = Backend()
    compact = SimpleNamespace(live_counts=np.array([1], dtype=np.int32))
    snapshot = SimpleNamespace(
        matches=lambda session, tokens, sampled: True,
        tokens=(1, 2),
        sampled=False,
        compact=compact,
        recurrent_and_logits=(first, second),
        result=SimpleNamespace(value=7),
    )
    session = SimpleNamespace(
        scratch=SimpleNamespace(
            max_positions=8,
            layer_conv_states=[conv],
            layer_recurrent_states=[recurrent],
        ),
        position=0,
        _dms_backend=None,
        _dms_source=SimpleNamespace(
            config=SimpleNamespace(num_kv_heads=1, num_layers=1)
        ),
        dms_max_new_tokens=2,
        dms_backend_factory=lambda **kwargs: backend,
        _dms_dense_prefill_pool=None,
        runner=SimpleNamespace(),
        backend=object(),
    )

    result = mod.restore_prompt(session, snapshot)

    assert result.value == 7
    assert legacy == []
    assert [call[0] for call in calls] == [
        "reserve",
        "restore_prefix",
        "upload",
        "upload",
    ]
    assert [(call[1].ptr, call[3], call[4]) for call in calls[2:]] == [
        (0x3000, 8, None),
        (0x4000, 8, None),
    ]
    assert [call[2].tolist() for call in calls[2:]] == [[0, 1], [2, 3]]
    assert session._position == 2
    assert session._dms_backend is backend
    assert session.runner.__dict__["_dms_decode_owner"] is session


def test_dms_prefill_cache_write_prefix_uploads_owning_array(monkeypatch):
    from hipengine.runtime import dms_prefill_cache as mod

    uploads = []
    legacy = []

    def upload(buffer, array, nbytes=None, *, runtime=None):
        uploads.append((buffer, array, nbytes, runtime))

    def old_upload(*args, **kwargs):
        legacy.append((args, kwargs))

    monkeypatch.setattr(mod, "copy_host_array_to_device", upload, raising=False)
    monkeypatch.setattr(mod, "copy_host_to_device", old_upload, raising=False)
    buffer = memory.DeviceBuffer(0x5000, 32)
    data = np.arange(8, dtype=np.uint8)

    mod.write_prefix(buffer, data)

    assert legacy == []
    assert len(uploads) == 1
    call_buffer, array, nbytes, call_runtime = uploads[0]
    assert (call_buffer.ptr, call_buffer.nbytes) == (0x5000, 8)
    assert array is data
    assert nbytes == 8
    assert call_runtime is None

    with pytest.raises(ValueError):
        mod.write_prefix(buffer, np.zeros(33, dtype=np.uint8))
    assert len(uploads) == 1
    assert legacy == []


def _capture_uploads(monkeypatch, module):
    """Install owning/legacy upload recorders on ``module``."""

    uploads = []
    legacy = []

    def upload(buffer, array, nbytes=None, *, runtime=None):
        uploads.append((buffer, array, nbytes, runtime))

    def old_upload(*args, **kwargs):
        legacy.append((args, kwargs))

    monkeypatch.setattr(module, "copy_host_array_to_device", upload, raising=False)
    monkeypatch.setattr(module, "copy_host_to_device", old_upload, raising=False)
    return uploads, legacy


def _fake_malloc(monkeypatch, module, base_ptr=0x7000):
    mallocs = []

    def fake_malloc(nbytes, *, runtime=None):
        buffer = memory.DeviceBuffer(base_ptr + 0x10 * len(mallocs), int(nbytes))
        mallocs.append(buffer)
        return buffer

    monkeypatch.setattr(module, "malloc", fake_malloc)
    monkeypatch.setattr(module, "free", lambda *args, **kwargs: None)
    return mallocs


def test_native_sampler_named_upload_uses_owning_contiguous_array(monkeypatch):
    from hipengine.runtime import native_sampler as mod

    uploads, legacy = _capture_uploads(monkeypatch, mod)
    mallocs = _fake_malloc(monkeypatch, mod)
    workspace = object.__new__(mod.NativeSamplerWorkspace)
    workspace.runtime = object()
    workspace._buffers = []
    workspace._named_buffers = {}
    workspace._cached_uploads = {}
    source = np.arange(8, dtype=np.int32)[::2]

    buffer = workspace._upload("min_tokens", source)

    assert legacy == []
    assert len(uploads) == 1
    call_buffer, array, nbytes, call_runtime = uploads[0]
    assert call_buffer is buffer is mallocs[0]
    assert array.flags.c_contiguous
    assert array.tolist() == [0, 2, 4, 6]
    assert nbytes == 16
    assert call_runtime is workspace.runtime


def test_native_sampler_cached_upload_uses_owning_contiguous_array(monkeypatch):
    from hipengine.runtime import native_sampler as mod

    uploads, legacy = _capture_uploads(monkeypatch, mod)
    mallocs = _fake_malloc(monkeypatch, mod)
    workspace = object.__new__(mod.NativeSamplerWorkspace)
    workspace.runtime = object()
    workspace._buffers = []
    workspace._named_buffers = {}
    workspace._cached_uploads = {}
    source = np.arange(8, dtype=np.int32)[::2]

    first = workspace._cached_upload(("bias",), source)
    second = workspace._cached_upload(("bias",), source)

    assert second is first
    assert legacy == []
    assert len(uploads) == 1
    call_buffer, array, nbytes, call_runtime = uploads[0]
    assert call_buffer is first is mallocs[0]
    assert array.flags.c_contiguous
    assert array.tolist() == [0, 2, 4, 6]
    assert nbytes == 16
    assert call_runtime is workspace.runtime


def test_gguf_native_spec_cycle_copy_array_uploads_owning_contiguous_array(
    monkeypatch,
):
    from hipengine.runtime import gguf_native_spec_cycle as mod

    uploads, legacy = _capture_uploads(monkeypatch, mod)
    runtime = object()
    tensor = SimpleNamespace(ptr=0x4000, numel=4, dtype=SimpleNamespace(itemsize=4))
    values = np.arange(8, dtype=np.int32)[::2]

    mod._copy_array_to_tensor(tensor, values, runtime=runtime)

    assert legacy == []
    assert len(uploads) == 1
    call_buffer, array, nbytes, call_runtime = uploads[0]
    assert (call_buffer.ptr, call_buffer.nbytes) == (0x4000, 16)
    assert array.flags.c_contiguous
    assert array.tolist() == [0, 2, 4, 6]
    assert nbytes == 16
    assert call_runtime is runtime

    with pytest.raises(ValueError, match="does not match tensor capacity"):
        mod._copy_array_to_tensor(tensor, np.zeros(3, dtype=np.int32), runtime=runtime)
    assert len(uploads) == 1
    assert legacy == []


def _fake_decode_graph_session(runtime):
    def buffer(ptr):
        return memory.DeviceBuffer(ptr, 16)

    scratch = SimpleNamespace(
        block_size=256,
        max_positions=768,
        position_buf=buffer(1),
        context_buf=buffer(2),
        hidden_seed_fp32=buffer(3),
        norm=buffer(4),
        attn_out=buffer(5),
        layer_conv_states=(buffer(6), None),
        layer_recurrent_states=(buffer(7), None),
        full_key_caches=(None, buffer(8)),
        full_value_caches=(None, buffer(9)),
        full_k_scale_caches=(None, buffer(10)),
        full_v_scale_caches=(None, buffer(11)),
    )
    weight = SimpleNamespace(
        spec=SimpleNamespace(
            slot_path="layers.0.attn_q",
            quant_key="gguf_q8_0_t16_v1",
            source=SimpleNamespace(shape=(2048, 2048)),
        ),
        allocations={"t16": SimpleNamespace(tensor=buffer(200))},
    )
    from hipengine.core.dtype import DType
    import contextlib

    return SimpleNamespace(
        backend="hip_gfx1151",
        model_path="/models/example.gguf",
        position=4,
        runtime=runtime,
        runner=SimpleNamespace(
            target_arch="gfx1151",
            hidden_size=2048,
            vocab_size=248320,
            weights=SimpleNamespace(
                weights=[weight],
                config=SimpleNamespace(
                    layer_types=("linear_attention", "full_attention"),
                    context_length=4096,
                ),
            ),
        ),
        scratch=scratch,
        kv_storage_dtype=DType.BF16,
        kv_storage_layout="uniform",
        kv_scale_dtype=DType.FP16,
        kv_scale_granularity="per_token_head",
        use_wmma_prefill=True,
        use_gemv_decode=True,
        host_token_embedding_enabled=False,
        _hidden_a=buffer(100),
        _hidden_b=buffer(101),
        _token_buf=buffer(102),
        _logits_buf=buffer(103),
        _lm_block_values=buffer(104),
        _lm_block_indices=buffer(105),
        _lm_out_index=buffer(106),
        _lm_out_value=buffer(107),
        _lm_head_threads=128,
        _lm_head_stage1_blocks=970,
        _decode_graphs=[],
        _set_full_attention_position_device=lambda position, *, stream: None,
        _iq_dense_mmq_context=contextlib.nullcontext,
    )


def test_gguf_decode_graph_record_index_zero_upload_uses_owning_array(monkeypatch):
    import contextlib

    from hipengine.runtime import gguf_decode_graph as mod

    uploads, legacy = _capture_uploads(monkeypatch, mod)
    mallocs = _fake_malloc(monkeypatch, mod)
    monkeypatch.setattr(mod, "_enqueue_decode_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        mod,
        "gemv_decode_session",
        lambda *args, **kwargs: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        mod,
        "create_graph_submission",
        lambda **kwargs: SimpleNamespace(graph_exec=0x55, close=lambda: None),
    )
    runtime = SimpleNamespace(
        stream_create=lambda: 0x31,
        stream_synchronize=lambda stream: None,
        stream_begin_capture=lambda stream: None,
        stream_end_capture=lambda stream: 0x77,
        stream_destroy=lambda stream: None,
        graph_destroy=lambda graph: None,
        memset=lambda *args, **kwargs: None,
    )
    session = _fake_decode_graph_session(runtime)

    graph = mod.capture_qwen35_gguf_decode_graph(
        session,
        position=4,
        steps_per_replay=1,
        max_replay_steps=4,
        record_steps=1,
    )

    assert graph.generated_index is mallocs[1]
    assert legacy == []
    assert len(uploads) == 1
    call_buffer, array, nbytes, call_runtime = uploads[0]
    assert call_buffer is mallocs[1]
    assert (call_buffer.ptr, call_buffer.nbytes) == (mallocs[1].ptr, 8)
    # The pointer-only default copied the destination size; keep it explicit.
    assert array.dtype == np.int64
    assert array.tolist() == [0]
    assert nbytes == 8
    assert call_runtime is runtime


def test_gguf_packed_decode_graph_capture_uploads_owning_inputs_in_order(
    monkeypatch,
):
    from hipengine.core.dtype import DType
    from hipengine.runtime import gguf_packed_decode_graph as mod

    uploads, legacy = _capture_uploads(monkeypatch, mod)
    mallocs = _fake_malloc(monkeypatch, mod)
    monkeypatch.setattr(mod, "replace", lambda obj, **kwargs: obj)
    monkeypatch.setattr(
        mod,
        "_singleton_layout",
        lambda *args, **kwargs: SimpleNamespace(blocks_per_slot=1),
    )
    monkeypatch.setattr(
        mod,
        "_packed_graph_buffer_ptrs",
        lambda *args, **kwargs: (0x3000,),
    )
    monkeypatch.setattr(
        mod,
        "_resolve_packed_graph_kernels",
        lambda *args, **kwargs: (
            lambda *a, **k: None,
            lambda *a, **k: None,
            lambda *a, **k: None,
        ),
    )
    monkeypatch.setattr(
        mod,
        "build_qwen35_gguf_packed_decode_graph_key",
        lambda *args, **kwargs: SimpleNamespace(
            as_dict=lambda: {"stub": True},
            key_sha256="stub",
        ),
    )
    monkeypatch.setattr(
        mod,
        "build_packed_decode_execution_manifest",
        lambda **kwargs: {"layer_families": {"sampler": {}}},
    )
    monkeypatch.setattr(
        mod,
        "create_graph_submission",
        lambda **kwargs: SimpleNamespace(
            graph_exec=0x55,
            close=lambda: None,
            provenance=lambda: {"stub": True},
        ),
    )
    import hipengine.runtime.qwen35_gguf_runner as runner_mod

    monkeypatch.setattr(
        runner_mod,
        "_rebind_packed_verify_layout_pages",
        lambda layout, packed_state: layout,
        raising=False,
    )
    runtime = SimpleNamespace(
        stream_create=lambda: 0x31,
        stream_synchronize=lambda stream: None,
        stream_begin_capture=lambda stream: None,
        stream_end_capture=lambda stream: 0x77,
        stream_destroy=lambda stream: None,
        graph_destroy=lambda graph: None,
        memset=lambda *args, **kwargs: None,
    )

    def buffer(ptr):
        return memory.DeviceBuffer(ptr, 64)

    packed_scratch = SimpleNamespace(
        block_size=256,
        max_positions=1024,
        block_table=buffer(0x401),
        positions=buffer(0x402),
        context_counts=buffer(0x403),
        cu_q=buffer(0x404),
        cu_k=buffer(0x405),
        atomic=buffer(0x406),
        gdn_cu_seqlens=buffer(0x407),
        gdn_state_indices=buffer(0x408),
        append_spans=SimpleNamespace(max_live_count=0),
        prefill_spans=SimpleNamespace(max_live_count=0),
        buffers=(),
    )
    owner = SimpleNamespace(
        runner=SimpleNamespace(
            backend="hip_gfx1100",
            target_arch="gfx1100",
            hidden_size=64,
            vocab_size=128,
            _packed_decode_metadata_kernel=lambda: (lambda *a, **k: None),
            weights=SimpleNamespace(
                weights=(),
                config=SimpleNamespace(
                    layer_types=("full_attention",),
                    is_moe=False,
                    expert_used_count=0,
                ),
            ),
        ),
        scratch=SimpleNamespace(),
        runtime=runtime,
        host_token_embedding_enabled=False,
        kv_storage_dtype=DType.BF16,
        use_expert_sidecar=False,
        _packed_decode_state_dirty=False,
        _prefill_token_buf=memory.DeviceBuffer(0x1000, 16),
        _bulk_prefill_scratch=object(),
        _prefill_hidden_a=None,
        _prefill_hidden_b=None,
        _verify_logits_buf=None,
        _verify_lm_block_values=None,
        _verify_lm_block_indices_i32=None,
        _verify_lm_out_indices_i32=memory.DeviceBuffer(0x2000, 8),
        _verify_lm_out_values=None,
        _runtime_state_library=object(),
        _last_packed_lm_head_decode_path="stub",
        _last_packed_sampler_decode_path="stub",
        _decode_graphs=[],
        _normalize_layer_output_capture=lambda value: tuple(value),
        _sync_packed_decode_initial_state=lambda *args, **kwargs: (),
        _ensure_verify_lm_head_buffers=lambda *args, **kwargs: None,
        _enqueue_packed_decode_model_step=lambda **kwargs: ({"strict"}, {"strict"}),
    )
    owner._ensure_packed_verify_workspace = lambda **kwargs: (
        SimpleNamespace(buffers=(), layer_conv_states=(), layer_recurrent_states=()),
        SimpleNamespace(for_packed_verify_layout=lambda *a, **k: packed_scratch),
    )
    sessions = (
        SimpleNamespace(position=4, runner=owner.runner, scratch=owner.scratch),
        SimpleNamespace(position=8, runner=owner.runner, scratch=owner.scratch),
    )

    graph = mod.capture_qwen35_gguf_packed_decode_graph(
        owner,
        token_ids=(11, 22),
        sessions=sessions,
        steps_per_replay=4,
        max_replay_steps=4,
        record_steps=4,
    )

    assert graph.generated_tokens is mallocs[1]
    assert graph.record_index is mallocs[2]
    assert graph.active_mask_device is mallocs[0]
    assert legacy == []
    assert [(call[0].ptr, call[2], call[3]) for call in uploads] == [
        (0x1000, 16, runtime),
        (mallocs[0].ptr, 2, runtime),
        (mallocs[2].ptr, 8, runtime),
    ]
    token_array, mask_array, zero = (call[1] for call in uploads)
    assert token_array.dtype == np.int64
    assert token_array.tolist() == [11, 22]
    assert mask_array.dtype == np.uint8
    assert mask_array.tolist() == [1, 1]
    assert zero.tolist() == [0]


VISION_RUNTIME_SYNCHRONOUS_ARRAY_UPLOAD_MODULES = (
    "runtime/gemma4_vision",
    "runtime/qwen4_exp_vision",
)


@pytest.mark.parametrize("module", VISION_RUNTIME_SYNCHRONOUS_ARRAY_UPLOAD_MODULES)
def test_vision_runtime_synchronous_array_uploads_use_the_owning_helper(module):
    """Vision H2D arrays must retain their owner through the copy.

    This is a bounded syntax guard over the Gemma 4 and Qwen4Exp vision
    runtime modules, not a proof of arbitrary pointer lifetimes. Asynchronous
    uploads, raw pointer uploads, and D2H/D2D copies are outside this guard.
    """
    path = Path(__file__).parents[1] / "hipengine" / f"{module}.py"
    tree = ast.parse(path.read_text())
    hazards = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "copy_host_to_device":
            source = (
                node.args[1]
                if len(node.args) > 1
                else next((kw.value for kw in node.keywords if kw.arg == "host_ptr"), None)
            )
            if source is not None and _is_array_pointer_expression(source):
                hazards.append((node.lineno, "copy_host_to_device"))
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "memcpy"):
            continue
        kind = (
            node.args[3]
            if len(node.args) > 3
            else next((kw.value for kw in node.keywords if kw.arg == "kind"), None)
        )
        if not (isinstance(kind, ast.Attribute) and kind.attr == "HOST_TO_DEVICE"):
            continue
        source = (
            node.args[1]
            if len(node.args) > 1
            else next((kw.value for kw in node.keywords if kw.arg == "src"), None)
        )
        if source is not None and _is_array_pointer_expression(source):
            hazards.append((node.lineno, "runtime.memcpy"))
    assert not hazards, f"{path.name}: use copy_host_array_to_device at {hazards}"


def test_gemma4_vision_upload_uses_owning_contiguous_array(monkeypatch):
    from hipengine.runtime import gemma4_vision as mod

    uploads, legacy = _capture_uploads(monkeypatch, mod)
    mallocs = _fake_malloc(monkeypatch, mod)
    runner = object.__new__(mod.Gemma4VisionRunner)
    runner._owned = []
    source = np.arange(8, dtype=np.float32)[::2]

    buffer = runner._upload(source)

    assert runner._owned == [buffer]
    assert legacy == []
    assert len(uploads) == 1
    call_buffer, array, nbytes, call_runtime = uploads[0]
    assert call_buffer is buffer is mallocs[0]
    assert array.flags.c_contiguous
    assert array.tolist() == [0.0, 2.0, 4.0, 6.0]
    assert nbytes == 16
    assert call_runtime is None


class _StopVisionForward(Exception):
    """Sentinel raised by the stubbed BLAS to bound a vision forward."""


def _fake_gemma4_vision_runner():
    from hipengine.runtime import gemma4_vision as mod

    runner = object.__new__(mod.Gemma4VisionRunner)
    runner.closed = False
    runner.config = SimpleNamespace(
        hidden_size=4,
        intermediate_size=8,
        heads=2,
        head_dim=2,
        spatial_merge_size=1,
        patch_size=2,
        block_count=1,
        output_size=4,
        norm_epsilon=1e-6,
    )
    # Real checkpoints store one lookup table per spatial axis:
    # v.position_embd.weight is [2, position_table_size, hidden].
    runner.positions = np.arange(2 * 8 * 4, dtype=np.float32).reshape(2, 8, 4)
    runner._weights = {
        "v.patch_embd.weight": memory.DeviceBuffer(0x9000, 16),
    }
    runner.ops = {}
    runner.blas = SimpleNamespace(
        sgemm_rowmajor_nt=lambda *args, **kwargs: (_ for _ in ()).throw(
            _StopVisionForward()
        )
    )
    return runner


def test_gemma4_vision_forward_alloc_uploads_owning_arrays(monkeypatch):
    from hipengine.runtime import gemma4_vision as mod

    uploads, legacy = _capture_uploads(monkeypatch, mod)
    mallocs = _fake_malloc(monkeypatch, mod)
    monkeypatch.setattr(mod, "free", lambda *args, **kwargs: None)
    runner = _fake_gemma4_vision_runner()
    patches = np.arange(48, dtype=np.float32).reshape(4, 12) / 48.0
    positions = np.asarray(
        [[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.int32
    )

    with pytest.raises(_StopVisionForward):
        runner.forward(patches, positions, (2, 2))

    assert legacy == []
    assert [(call[0].ptr, call[2], call[3]) for call in uploads] == [
        (mallocs[0].ptr, 192, None),
        (mallocs[1].ptr, 32, None),
        # Learned position rows are [rows, hidden], so the copy covers the whole
        # destination allocation for valid checkpoint geometry.
        (mallocs[2].ptr, 64, None),
    ]
    assert mallocs[2].nbytes == 64
    patch_array, position_array, learned_array = (call[1] for call in uploads)
    np.testing.assert_array_equal(patch_array, np.ascontiguousarray(2.0 * patches - 1.0))
    np.testing.assert_array_equal(position_array, positions)
    expected_learned = runner.positions[0, positions[:, 0]] + runner.positions[1, positions[:, 1]]
    np.testing.assert_array_equal(learned_array, expected_learned)


def test_qwen4_exp_vision_encode_pair_uploads_owning_arrays_in_order(monkeypatch):
    from hipengine.runtime import qwen4_exp_vision as mod

    uploads, legacy = _capture_uploads(monkeypatch, mod)
    for name in (
        "qwen4_exp_vision_layernorm_f32",
        "qwen4_exp_vision_attention_f32",
        "qwen4_exp_vision_add_bias_residual_f32",
        "qwen4_exp_vision_bias_gelu_tanh_f32",
        "launch_gguf_linear",
    ):
        monkeypatch.setattr(mod, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(mod, "copy_device_to_host", lambda *args, **kwargs: None)
    runtime = SimpleNamespace(
        memcpy=lambda *args, **kwargs: None,
        device_synchronize=lambda: None,
    )

    def buffer(ptr, nbytes):
        return memory.DeviceBuffer(ptr, nbytes)

    runner = object.__new__(mod.Qwen4ExpVisionRunner)
    runner.closed = False
    runner.runtime = runtime
    runner.config = SimpleNamespace(norm_epsilon=1e-6)
    runner._buffers = {
        "initial": buffer(0x1000, 8 * 1152 * 4),
        "norm": buffer(0x1100, 8 * 1152 * 4),
        "qkv": buffer(0x1200, 8 * 3 * 1152 * 4),
        "attention": buffer(0x1300, 8 * 1152 * 4),
        "projected": buffer(0x1400, 8 * 1152 * 4),
        "ff1": buffer(0x1500, 8 * 4304 * 4),
        "gelu": buffer(0x1600, 8 * 4304 * 4),
        "ff2": buffer(0x1700, 8 * 1152 * 4),
        "pos_h": buffer(0x1800, 8 * 4),
        "pos_w": buffer(0x1900, 8 * 4),
        "merge_input": buffer(0x1A00, 8 * 1152 * 4),
        "merge_hidden": buffer(0x1B00, 2 * 4 * 1152 * 4),
        "merge_output": buffer(0x1C00, 2 * 2560 * 4),
    }
    runner._p = lambda name: 0x8000
    runner._w = lambda name: SimpleNamespace(name=name)
    initial = np.arange(8 * 1152, dtype=np.float32).reshape(8, 1152)
    pos_h = np.arange(8, dtype=np.int32)
    pos_w = np.arange(8, dtype=np.int32)[::-1].copy()
    runner.preprocess_temporal_pair = lambda *args, **kwargs: (
        initial,
        pos_h,
        pos_w,
        (2, 4),
    )

    features = runner._encode_pair(object(), None, modality="image")

    assert legacy == []
    assert [(call[0].ptr, call[2], call[3]) for call in uploads] == [
        (0x1000, initial.nbytes, runtime),
        (0x1800, pos_h.nbytes, runtime),
        (0x1900, pos_w.nbytes, runtime),
    ]
    assert [call[1] for call in uploads] == [initial, pos_h, pos_w]
    assert features.grid_thw == (1, 2, 4)
    assert features.embeddings.shape == (2, 2560)


LOADING_RUNTIME_SYNCHRONOUS_ARRAY_UPLOAD_MODULES = (
    "loading/materialize",
    "runtime/qwen35_gguf_probe",
)


@pytest.mark.parametrize("module", LOADING_RUNTIME_SYNCHRONOUS_ARRAY_UPLOAD_MODULES)
def test_loading_runtime_synchronous_array_uploads_use_the_owning_helper(module):
    """Materializer and probe H2D arrays must retain their owner through the copy.

    This is a bounded syntax guard over the safetensors materializer and the
    Qwen3.5 GGUF one-layer probe, not a proof of arbitrary pointer lifetimes.
    Asynchronous uploads, raw pointer uploads, and D2H/D2D copies are outside
    this guard.
    """
    path = Path(__file__).parents[1] / "hipengine" / f"{module}.py"
    tree = ast.parse(path.read_text())
    hazards = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "copy_host_to_device":
            source = (
                node.args[1]
                if len(node.args) > 1
                else next((kw.value for kw in node.keywords if kw.arg == "host_ptr"), None)
            )
            if source is not None and _is_array_pointer_expression(source):
                hazards.append((node.lineno, "copy_host_to_device"))
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "memcpy"):
            continue
        kind = (
            node.args[3]
            if len(node.args) > 3
            else next((kw.value for kw in node.keywords if kw.arg == "kind"), None)
        )
        if not (isinstance(kind, ast.Attribute) and kind.attr == "HOST_TO_DEVICE"):
            continue
        source = (
            node.args[1]
            if len(node.args) > 1
            else next((kw.value for kw in node.keywords if kw.arg == "src"), None)
        )
        if source is not None and _is_array_pointer_expression(source):
            hazards.append((node.lineno, "runtime.memcpy"))
    assert not hazards, f"{path.name}: use copy_host_array_to_device at {hazards}"


def test_materialize_tensor_info_upload_uses_owning_contiguous_array(monkeypatch):
    from hipengine.loading import materialize as mod
    from hipengine.loading.safetensors import TensorInfo

    uploads, legacy = _capture_uploads(monkeypatch, mod)
    mallocs = _fake_malloc(monkeypatch, mod)
    freed = []
    monkeypatch.setattr(mod, "free", lambda buffer, *, runtime=None: freed.append(buffer))
    source = np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2]
    assert not source.flags.c_contiguous
    monkeypatch.setattr(mod, "_read_numpy_tensor", lambda info: source)
    info = TensorInfo(
        name="model.weight",
        shard_path=Path("/models/fake/model.safetensors"),
        dtype="F32",
        shape=(3, 2),
    )
    runtime = object()

    allocation = mod.load_tensor_info_to_device(info, runtime=runtime)

    assert freed == []
    assert legacy == []
    assert len(uploads) == 1
    call_buffer, array, nbytes, call_runtime = uploads[0]
    assert call_buffer is allocation.buffer is mallocs[0]
    assert call_buffer.nbytes == 24
    assert array.flags.c_contiguous
    assert array.tolist() == [[0.0, 2.0], [4.0, 6.0], [8.0, 10.0]]
    assert nbytes == 24
    assert call_runtime is runtime


def test_materialize_tensor_info_upload_failure_frees_the_buffer(monkeypatch):
    from hipengine.loading import materialize as mod
    from hipengine.loading.safetensors import TensorInfo

    uploads, legacy = _capture_uploads(monkeypatch, mod)
    mallocs = _fake_malloc(monkeypatch, mod)
    freed = []
    monkeypatch.setattr(mod, "free", lambda buffer, *, runtime=None: freed.append(buffer))
    monkeypatch.setattr(
        mod,
        "_read_numpy_tensor",
        lambda info: np.zeros((2,), dtype=np.float32),
    )
    info = TensorInfo(
        name="model.weight",
        shard_path=Path("/models/fake/model.safetensors"),
        dtype="F32",
        shape=(2,),
    )

    def fail_upload(*args, **kwargs):
        raise RuntimeError("injected upload failure")

    monkeypatch.setattr(mod, "copy_host_array_to_device", fail_upload, raising=False)
    monkeypatch.setattr(mod, "copy_host_to_device", fail_upload, raising=False)

    with pytest.raises(RuntimeError, match="injected upload failure"):
        mod.load_tensor_info_to_device(info, runtime=object())

    assert freed == [mallocs[0]]
    assert uploads == []
    assert legacy == []


def test_materialize_as_dtype_upload_uses_owning_array_and_borrowed_cleanup(
    monkeypatch,
):
    from hipengine.core.dtype import DType
    from hipengine.loading import materialize as mod

    uploads, legacy = _capture_uploads(monkeypatch, mod)
    buffer = memory.DeviceBuffer(0x3100, 16)
    released = []
    allocator = SimpleNamespace(
        allocate=lambda nbytes: buffer,
        owns=lambda candidate: True,
        release=lambda candidate: released.append(candidate),
    )
    runtime = object()
    source = np.arange(8, dtype=np.int32)[::2]

    allocation = mod.load_host_array_to_device_as_dtype(
        "arena_prepared",
        source,
        DType.INT32,
        runtime=runtime,
        allocator=allocator,
    )

    assert released == []
    assert allocation.owns_buffer is False
    assert legacy == []
    assert len(uploads) == 1
    call_buffer, array, nbytes, call_runtime = uploads[0]
    assert call_buffer is buffer
    assert array.flags.c_contiguous
    assert array.tolist() == [0, 2, 4, 6]
    assert nbytes == 16
    assert call_runtime is runtime

    def fail_upload(*args, **kwargs):
        raise RuntimeError("injected upload failure")

    monkeypatch.setattr(mod, "copy_host_array_to_device", fail_upload, raising=False)
    monkeypatch.setattr(mod, "copy_host_to_device", fail_upload, raising=False)

    with pytest.raises(RuntimeError, match="injected upload failure"):
        mod.load_host_array_to_device_as_dtype(
            "arena_prepared",
            source,
            DType.INT32,
            runtime=runtime,
            allocator=allocator,
        )
    assert released == [buffer]


def _fake_qwen35_gguf_probe(runtime):
    from hipengine.runtime import qwen35_gguf_probe as mod

    weight = SimpleNamespace(
        allocation=lambda: SimpleNamespace(tensor=SimpleNamespace(ptr=0xA000))
    )
    weights = SimpleNamespace(
        config=SimpleNamespace(
            hidden_size=8,
            ssm_inner_size=4,
            vocab_size=16,
            rms_norm_eps=1e-6,
        ),
        root=lambda slot: weight,
        layer=lambda layer_id: SimpleNamespace(weight=lambda name: weight),
    )
    probe = object.__new__(mod.Qwen35GGUFOneLayerProbe)
    probe.runtime = runtime
    probe.layer_id = 0
    probe.weights = weights
    return probe


def test_qwen35_gguf_probe_uploads_owning_arrays_with_exact_counts(monkeypatch):
    from hipengine.runtime import qwen35_gguf_probe as mod

    uploads, legacy = _capture_uploads(monkeypatch, mod)
    mallocs = _fake_malloc(monkeypatch, mod)
    for name in (
        "launch_gguf_embedding",
        "gguf_rmsnorm_bf16_f32_weight",
        "launch_gguf_linear",
    ):
        monkeypatch.setattr(mod, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(mod, "copy_device_to_host", lambda *args, **kwargs: None)
    runtime = SimpleNamespace(device_synchronize=lambda: None)
    probe = _fake_qwen35_gguf_probe(runtime)

    probe.run_token(760)
    probe.logits_from_hidden_bits(np.zeros((1, 8), dtype=np.uint16))

    assert legacy == []
    assert [(call[0].ptr, call[2], call[3]) for call in uploads] == [
        (mallocs[0].ptr, 8, runtime),
        (mallocs[5].ptr, 16, runtime),
    ]
    token_array, hidden_array = (call[1] for call in uploads)
    assert token_array.dtype == np.int64
    assert token_array.tolist() == [760]
    assert hidden_array.dtype == np.uint16
    assert hidden_array.shape == (1, 8)


AUDIO_KV_RUNTIME_SYNCHRONOUS_ARRAY_UPLOAD_MODULES = (
    "runtime/gemma4_int8_kv",
    "runtime/surya",
    "runtime/yue2_ar",
    "runtime/vibevoice_qwen2",
)


@pytest.mark.parametrize("module", AUDIO_KV_RUNTIME_SYNCHRONOUS_ARRAY_UPLOAD_MODULES)
def test_audio_kv_runtime_synchronous_array_uploads_use_the_owning_helper(module):
    """Audio/KV H2D arrays must retain their owner through the copy.

    This is a bounded syntax guard over the Gemma 4 INT8 KV owner, Surya, YuE2
    AR and VibeVoice Qwen2 runtimes, not a proof of arbitrary pointer lifetimes.
    Asynchronous uploads, raw pointer uploads, and D2H/D2D copies are outside
    this guard.
    """
    path = Path(__file__).parents[1] / "hipengine" / f"{module}.py"
    tree = ast.parse(path.read_text())
    hazards = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "copy_host_to_device":
            source = (
                node.args[1]
                if len(node.args) > 1
                else next((kw.value for kw in node.keywords if kw.arg == "host_ptr"), None)
            )
            if source is not None and _is_array_pointer_expression(source):
                hazards.append((node.lineno, "copy_host_to_device"))
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "memcpy"):
            continue
        kind = (
            node.args[3]
            if len(node.args) > 3
            else next((kw.value for kw in node.keywords if kw.arg == "kind"), None)
        )
        if not (isinstance(kind, ast.Attribute) and kind.attr == "HOST_TO_DEVICE"):
            continue
        source = (
            node.args[1]
            if len(node.args) > 1
            else next((kw.value for kw in node.keywords if kw.arg == "src"), None)
        )
        if source is not None and _is_array_pointer_expression(source):
            hazards.append((node.lineno, "runtime.memcpy"))
    assert not hazards, f"{path.name}: use copy_host_array_to_device at {hazards}"


def test_gemma4_int8_kv_page_table_upload_uses_owning_array(monkeypatch):
    from hipengine.core.dtype import DType
    from hipengine.runtime import gemma4_int8_kv as mod

    uploads, legacy = _capture_uploads(monkeypatch, mod)
    _fake_malloc(monkeypatch, mod)
    monkeypatch.setattr(mod, "free", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod, "enqueue_host_to_device", lambda *args, **kwargs: None)

    cache = mod.Gemma4Int8KVCache(
        capacity=512,
        max_block=8,
        attentions=((16, 8, 256),),
        scale_dtype=DType.FP16,
        block_size=64,
    )

    identity = np.tile(
        np.arange(cache.blocks, dtype=np.int32), (cache.max_block, 1)
    ).reshape(-1)
    assert legacy == []
    assert len(uploads) == 1
    call_buffer, array, nbytes, call_runtime = uploads[0]
    assert call_buffer is cache._page_table
    assert call_buffer.nbytes == identity.nbytes
    assert array.tolist() == identity.tolist()
    assert nbytes == identity.nbytes
    assert call_runtime is None
    cache.close()


def _fake_surya_runner():
    from hipengine.runtime import surya as mod

    runner = object.__new__(mod.SuryaGpuRunner)
    runner._span_live_counts = memory.DeviceBuffer(0x5000, 8)
    runner._span_row_positions = memory.DeviceBuffer(0x5100, 8)
    runner._span_live_host = np.zeros(1, dtype=np.int64)
    runner._span_row_host = np.zeros(1, dtype=np.int64)
    return runner


def test_surya_span_extent_uploads_owning_scalars(monkeypatch):
    from hipengine.runtime import surya as mod

    uploads, legacy = _capture_uploads(monkeypatch, mod)
    runner = _fake_surya_runner()

    runner._set_span_extent(7, 3)

    assert legacy == []
    assert [(call[0].ptr, call[2], call[3]) for call in uploads] == [
        (0x5000, 8, None),
        (0x5100, 8, None),
    ]
    live_array, row_array = (call[1] for call in uploads)
    assert live_array.tolist() == [7]
    assert row_array.tolist() == [3]
    assert runner._span_live_host.tolist() == [7]
    assert runner._span_row_host.tolist() == [3]


def test_surya_upload_converts_owning_contiguous_source(monkeypatch):
    from hipengine.runtime import surya as mod

    uploads, legacy = _capture_uploads(monkeypatch, mod)
    synchronizations = []
    runner = SimpleNamespace(
        runtime=SimpleNamespace(
            device_synchronize=lambda: synchronizations.append(True)
        )
    )
    buffer = memory.DeviceBuffer(0x5200, 16)

    mod.SuryaGpuRunner._upload(runner, buffer, np.arange(8)[::2])

    assert legacy == []
    assert synchronizations == []
    assert len(uploads) == 1
    call_buffer, array, nbytes, call_runtime = uploads[0]
    assert call_buffer is buffer
    assert array.flags.c_contiguous
    assert array.dtype == np.float32
    assert array.tolist() == [0.0, 2.0, 4.0, 6.0]
    assert nbytes == 16
    assert call_runtime is None


def test_yue2_ar_push_token_uploads_owning_context_length(monkeypatch):
    from hipengine.runtime import yue2_ar as mod

    uploads, legacy = _capture_uploads(monkeypatch, mod)
    runtime = object.__new__(mod.Yue2ArRuntime)
    runtime.branches = 1
    runtime.max_context = 16
    runtime.spec = SimpleNamespace(hidden_size=4)
    runtime._hidden = [memory.DeviceBuffer(0x6000, 8)]
    runtime._ctx_len = [memory.DeviceBuffer(0x6100, 8)]
    runtime._ctx_len_host = [np.zeros(1, dtype=np.int64)]
    runtime._logits_pair_cache = object()
    runtime._logits_window_filled = True

    runtime.push_token(np.zeros(4, dtype=np.float32), 0)

    assert legacy == []
    # The hidden row keeps the helper's source-sized default; the context length
    # passes the former destination-sized default explicitly.
    assert [
        (call[0].ptr, call[1].nbytes if call[2] is None else call[2], call[3])
        for call in uploads
    ] == [(0x6000, 8, None), (0x6100, 8, None)]
    hidden_array, context_array = (call[1] for call in uploads)
    assert hidden_array.dtype == np.uint16
    assert context_array.tolist() == [1]
    assert runtime._ctx_len_host[0].tolist() == [1]
    assert runtime._logits_pair_cache is None
    assert runtime._logits_window_filled is False


def test_vibevoice_qwen2_push_token_uploads_owning_hidden_and_context(monkeypatch):
    from hipengine.runtime import vibevoice_qwen2 as mod

    uploads, legacy = _capture_uploads(monkeypatch, mod)
    runtime = object.__new__(mod.VibevoiceQwen2Runtime)
    runtime.max_context = 16
    runtime.spec = SimpleNamespace(hidden_size=4)
    runtime._hidden = memory.DeviceBuffer(0x7000, 8)
    runtime._ctx_len = memory.DeviceBuffer(0x7100, 8)
    runtime._ctx_len_host = np.zeros(1, dtype=np.int64)

    runtime.push_token(np.zeros(4, dtype=np.float32), 0)

    assert legacy == []
    assert [(call[0].ptr, call[2], call[3]) for call in uploads] == [
        (0x7000, 8, None),
        (0x7100, 8, None),
    ]
    hidden_array, context_array = (call[1] for call in uploads)
    assert hidden_array.dtype == np.uint16
    assert context_array.tolist() == [1]
    assert runtime._ctx_len_host[0] == 1


def test_vibevoice_qwen2_batched_prefill_publishes_owning_context_length(
    monkeypatch,
):
    """The batched prefill tail publishes the new context length from the owner.

    ``layers`` is empty so the test exercises the prefill entry validation, the
    arena scratch request, the position/count staging and the final context
    publication without any kernel, model, or hipBLASLt problem.
    """
    from hipengine.runtime import vibevoice_qwen2 as mod

    uploads, legacy = _capture_uploads(monkeypatch, mod)
    runtime = object.__new__(mod.VibevoiceQwen2Runtime)
    runtime.max_context = 16
    runtime.spec = SimpleNamespace(
        hidden_size=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        intermediate_size=8,
        rms_norm_eps=1e-6,
    )
    scratch = []

    def take(sizes):
        views = [
            memory.DeviceBuffer(0x8000 + 0x100 * index, int(size))
            for index, size in enumerate(sizes)
        ]
        scratch.extend(views)
        return views

    runtime._prefill_scratch = SimpleNamespace(take=take)
    runtime._spans = lambda positions, counts, rows: object()
    runtime.layers = ()
    runtime._ctx_len = memory.DeviceBuffer(0x8100, 8)
    runtime._ctx_len_host = np.zeros(1, dtype=np.int64)
    hidden_rows = memory.DeviceBuffer(0x8200, 4 * 4 * 2)

    # The rejected window must not stage or publish anything.
    with pytest.raises(ValueError, match="prefill exceeds max_context"):
        runtime._prefill_batched(hidden_rows, 14, 3)
    assert uploads == []
    assert legacy == []
    assert runtime._ctx_len_host.tolist() == [0]

    runtime._prefill_batched(hidden_rows, 4, 3)

    assert legacy == []
    positions, counts = scratch[0], scratch[1]
    assert [
        (call[0].ptr, call[1].nbytes if call[2] is None else call[2], call[3])
        for call in uploads
    ] == [
        (positions.ptr, 4 * 8, None),
        (counts.ptr, 4 * 8, None),
        (0x8100, 8, None),
    ]
    position_array, count_array, context_array = (call[1] for call in uploads)
    assert position_array.tolist() == [3, 4, 5, 6]
    assert count_array.tolist() == [4, 5, 6, 7]
    # The publication retains the runtime-owned host scalar and passes the
    # former destination-sized default explicitly.
    assert context_array is runtime._ctx_len_host
    assert uploads[2][2] == runtime._ctx_len.nbytes == 8
    assert runtime._ctx_len_host.tolist() == [7]


@pytest.mark.parametrize("module", SPECULATIVE_RUNTIME_SYNCHRONOUS_ARRAY_UPLOAD_MODULES)
def test_speculative_runtime_synchronous_array_uploads_use_the_owning_helper(
    module,
):
    """MTP native/resident/cached drafts and the Laguna DFlash drafter.

    A bounded syntax guard over synchronous H2D uploads, not a proof of
    arbitrary pointer lifetimes. Asynchronous ``memcpy_async`` staging, raw
    pointer staging, and D2H/D2D copies are outside this guard.
    """
    path = Path(__file__).parents[1] / "hipengine" / f"{module}.py"
    tree = ast.parse(path.read_text())
    hazards = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "copy_host_to_device":
            source = (
                node.args[1]
                if len(node.args) > 1
                else next((kw.value for kw in node.keywords if kw.arg == "host_ptr"), None)
            )
            if source is not None and _is_array_pointer_expression(source):
                hazards.append((node.lineno, "copy_host_to_device"))
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "memcpy"):
            continue
        kind = (
            node.args[3]
            if len(node.args) > 3
            else next((kw.value for kw in node.keywords if kw.arg == "kind"), None)
        )
        if not (isinstance(kind, ast.Attribute) and kind.attr == "HOST_TO_DEVICE"):
            continue
        source = (
            node.args[1]
            if len(node.args) > 1
            else next((kw.value for kw in node.keywords if kw.arg == "src"), None)
        )
        if source is not None and _is_array_pointer_expression(source):
            hazards.append((node.lineno, "runtime.memcpy"))
    assert not hazards, f"{path.name}: use copy_host_array_to_device at {hazards}"


def test_mtp_native_device_array_uploads_owning_contiguous_source(monkeypatch):
    """``_device_array`` retains its own contiguous copy through the upload."""

    from hipengine.speculative import mtp_native as mod

    uploads, legacy = _capture_uploads(monkeypatch, mod)
    _fake_malloc(monkeypatch, mod)
    runtime = object()
    source = np.arange(8, dtype=np.float32)[::2]
    buffers: list[object] = []

    buffer = mod._device_array(source, buffers, runtime=runtime)

    assert legacy == []
    assert buffers == [buffer]
    assert buffer.nbytes == 16
    (call,) = uploads
    assert call[0] is buffer
    assert call[1].flags["C_CONTIGUOUS"]
    assert call[1].tolist() == [0.0, 2.0, 4.0, 6.0]
    assert call[2] == 16
    assert call[3] is runtime


def test_mtp_resident_draft_seed_upload_owns_contiguous_seed(monkeypatch):
    """``propose_chain`` uploads its own float32 copy of the seed rows."""

    from hipengine.speculative import mtp_resident_draft as mod

    uploads, legacy = _capture_uploads(monkeypatch, mod)
    runner = object.__new__(mod.Qwen35GGUFResidentMTPDraftRunner)
    runner.runtime = object()
    runner.hidden_size = 4
    runner.seed_a = memory.DeviceBuffer(0x9000, 16)
    sentinel = object()
    runner._propose_chain_from_seed_buffer = lambda **_kwargs: sentinel

    seed = np.arange(8, dtype=np.float32)[::2].reshape(1, 4)
    result = runner.propose_chain(
        seed,
        start_token=1,
        start_position=2,
        draft_n_max=1,
        top_k=2,
        rope_cos=np.zeros((4, 4), dtype=np.float32),
        rope_sin=np.zeros((4, 4), dtype=np.float32),
    )

    assert result is sentinel
    assert legacy == []
    (call,) = uploads
    assert call[0] is runner.seed_a
    assert call[1].dtype == np.float32
    assert call[1].shape == (1, 4)
    assert call[1].tolist() == [[0.0, 2.0, 4.0, 6.0]]
    assert call[2] == 16
    assert call[3] is runner.runtime


class _StopDraftForward(Exception):
    """Sentinel raised by the stubbed layer forward."""


def test_mtp_resident_draft_prepare_uploads_owning_arrays_in_order(monkeypatch):
    """Per-depth prepare staging uploads five owning arrays in kernel order."""

    from hipengine.speculative import mtp_resident_draft as mod

    uploads, legacy = _capture_uploads(monkeypatch, mod)
    runner = object.__new__(mod.Qwen35GGUFResidentMTPDraftRunner)
    runtime = object()
    runner.runtime = runtime
    runner.qk_head_dim = 4
    runner._device_chain_enabled = False
    runner.seed_a = memory.DeviceBuffer(0x9500, 16)
    runner.seed_b = memory.DeviceBuffer(0x9600, 16)
    runner.token_embd_f32 = np.arange(32, dtype=np.float32).reshape(8, 4)
    runner.token_embed = memory.DeviceBuffer(0x9000, 16)
    runner.cos = memory.DeviceBuffer(0x9100, 16)
    runner.sin = memory.DeviceBuffer(0x9200, 16)
    runner.position_i64 = memory.DeviceBuffer(0x9300, 8)
    runner.context_i64 = memory.DeviceBuffer(0x9400, 8)

    def stop_forward(*_args, **_kwargs):
        raise _StopDraftForward

    runner._run_one = stop_forward
    rope_cos = np.arange(16, dtype=np.float32).reshape(4, 4)
    rope_sin = rope_cos + 100.0

    with pytest.raises(_StopDraftForward):
        runner._propose_chain_from_seed_buffer(
            start_token=2,
            start_position=3,
            draft_n_max=1,
            top_k=2,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            dense_key_cache=None,
            dense_value_cache=None,
            dense_cache_len=5,
            draft_p_min=0.0,
        )

    assert legacy == []
    assert [
        (call[0].ptr, call[2], call[3] is runtime) for call in uploads
    ] == [
        (0x9000, 16, True),
        (0x9100, 16, True),
        (0x9200, 16, True),
        (0x9300, 8, True),
        (0x9400, 8, True),
    ]
    embed, cos, sin, pos, ctx = (call[1] for call in uploads)
    assert embed.tolist() == runner.token_embd_f32[2:3].tolist()
    assert cos.tolist() == [[12.0, 13.0, 14.0, 15.0]]
    assert sin.tolist() == [[112.0, 113.0, 114.0, 115.0]]
    assert pos.tolist() == [3]
    assert ctx.tolist() == [6]
    assert [call[1].dtype for call in uploads] == [
        np.float32,
        np.float32,
        np.float32,
        np.int64,
        np.int64,
    ]


def test_mtp_cached_draft_shared_head_upload_owns_dequantized_array(monkeypatch):
    """The Q6_K shared-head dequant upload publishes its own float32 copy."""

    from hipengine.speculative import mtp_cached_draft as mod

    uploads, legacy = _capture_uploads(monkeypatch, mod)
    _fake_malloc(monkeypatch, mod)
    tensors = [
        SimpleNamespace(name="blk.40.attn_q.weight", ggml_type=0, shape=(4,)),
        SimpleNamespace(
            name="output.weight",
            ggml_type=mod.GGMLQuantizationType.Q6_K,
            shape=(4,),
        ),
    ]
    reader = SimpleNamespace(
        info=SimpleNamespace(tensors=tensors),
        tensor_data=lambda _name: np.zeros(4, dtype=np.uint8),
    )
    monkeypatch.setattr(mod, "GGUFReader", lambda _path: reader)
    monkeypatch.setattr(
        mod,
        "dequantize_gguf_data",
        lambda _data, _quant: np.arange(4, dtype=np.float32),
    )
    runtime = SimpleNamespace(device_synchronize=lambda: None)

    runner = mod.MtpCachedDraftRunner("/unused.gguf", runtime=runtime)

    assert legacy == []
    # Both cached tensors upload first, then the dequantized shared head.
    weight_uploads, shared_head_upload = uploads[:2], uploads[2]
    assert [call[0].ptr for call in weight_uploads] == [
        runner._dev_weights["blk.40.attn_q.weight"].ptr,
        runner._dev_weights["output.weight"].ptr,
    ]
    assert [call[1].dtype for call in weight_uploads] == [np.uint8, np.uint8]
    assert [call[2] for call in weight_uploads] == [None, None]
    assert shared_head_upload[0] is runner._dev_shared_head_f32
    assert shared_head_upload[1].dtype == np.float32
    assert shared_head_upload[1].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert shared_head_upload[2] == runner._dev_shared_head_f32.nbytes == 16
    assert shared_head_upload[3] is runtime
    assert runner._buffers == [
        runner._dev_weights["blk.40.attn_q.weight"],
        runner._dev_weights["output.weight"],
        runner._dev_shared_head_f32,
    ]


def test_laguna_dflash_expanded_norm_upload_owns_contiguous_f32(monkeypatch):
    """The BF16 norm expansion uploads the FP32 array it just materialized."""

    from hipengine.core.dtype import DType
    from hipengine.speculative import laguna_dflash as mod

    uploads, legacy = _capture_uploads(monkeypatch, mod)

    def fake_copy_device_to_host(dst, _src, nbytes, *, runtime=None):
        out = (ctypes.c_uint16 * (int(nbytes) // 2)).from_address(int(dst))
        for index in range(len(out)):
            out[index] = 0x3F80

    monkeypatch.setattr(mod, "copy_device_to_host", fake_copy_device_to_host)
    drafter = object.__new__(mod.LagunaDFlashResidentDrafter)
    drafter.runtime = object()
    drafter.config = SimpleNamespace(num_hidden_layers=1)
    drafter.weights = SimpleNamespace(
        tensor=lambda _name: SimpleNamespace(
            ptr=0xA000,
            dtype=DType.BF16,
            ndim=1,
            shape=(4,),
        )
    )
    expanded = []

    def empty(shape, dtype):
        tensor = SimpleNamespace(ptr=0xB000 + 0x100 * len(expanded), shape=shape, dtype=dtype)
        expanded.append(tensor)
        return tensor

    drafter._empty = empty
    drafter._buffer_for = lambda tensor: memory.DeviceBuffer(
        tensor.ptr,
        int(np.prod(tensor.shape)) * tensor.dtype.itemsize,
    )
    drafter._f32_norm_weights = {}

    drafter._materialize_f32_norm_weights()

    assert legacy == []
    assert len(expanded) == 3
    assert len(uploads) == 3
    for upload, tensor in zip(uploads, expanded):
        assert upload[0].ptr == tensor.ptr
        assert upload[0].nbytes == tensor.shape[0] * 4
        assert upload[1].dtype == np.float32
        assert upload[1].tolist() == [1.0, 1.0, 1.0, 1.0]
        assert upload[2] == tensor.shape[0] * 4 == 16
        assert upload[3] is drafter.runtime


def test_laguna_dflash_scratch_zero_upload_owns_contiguous_zeros(monkeypatch):
    """The query zero buffer is published from the owning float32 array."""

    from hipengine.core.dtype import DType
    from hipengine.speculative import laguna_dflash as mod

    uploads, legacy = _capture_uploads(monkeypatch, mod)
    drafter = object.__new__(mod.LagunaDFlashResidentDrafter)
    runtime = object()
    drafter.runtime = runtime
    drafter.query_rows = 2
    drafter.max_append_rows = 3
    drafter.candidate_budget = 2
    drafter.top_k = 2
    drafter.config = SimpleNamespace(
        hidden_size=4,
        q_features=4,
        kv_features=4,
        intermediate_size=8,
        vocab_size=8,
        num_attention_heads=2,
        target_hidden_concat_size=8,
    )
    allocated = []

    def empty(shape, dtype):
        tensor = SimpleNamespace(ptr=0xB000 + 0x100 * len(allocated), shape=shape, dtype=dtype)
        allocated.append(tensor)
        return tensor

    drafter._empty = empty
    drafter._buffer_for = lambda tensor: memory.DeviceBuffer(
        tensor.ptr,
        int(np.prod(tensor.shape)) * tensor.dtype.itemsize,
    )

    drafter._allocate_scratch()

    assert legacy == []
    (call,) = uploads
    assert call[0].ptr == drafter.query_zero_f32.ptr
    assert call[0].nbytes == 2 * 4 * 4
    assert call[1].dtype == np.float32
    assert call[1].shape == (2, 4)
    assert call[1].tolist() == [[0.0] * 4, [0.0] * 4]
    assert call[2] == 2 * 4 * 4
    assert call[3] is runtime


GRAPH_REPLAY_RESTORE_UPLOAD_MODULES = (
    "runtime/moe_graph",
    "runtime/mtp_draft_head",
)


@pytest.mark.parametrize("module", GRAPH_REPLAY_RESTORE_UPLOAD_MODULES)
def test_graph_replay_restore_synchronous_array_uploads_use_the_owning_helper(
    module,
):
    """Graph replay restoration and draft-head row staging.

    A bounded syntax guard over synchronous H2D uploads, not a proof of
    arbitrary pointer lifetimes. Device-to-device row copies, D2H readbacks and
    graph launch ordering are outside this guard.
    """
    path = Path(__file__).parents[1] / "hipengine" / f"{module}.py"
    tree = ast.parse(path.read_text())
    hazards = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "copy_host_to_device":
            source = (
                node.args[1]
                if len(node.args) > 1
                else next((kw.value for kw in node.keywords if kw.arg == "host_ptr"), None)
            )
            if source is not None and _is_array_pointer_expression(source):
                hazards.append((node.lineno, "copy_host_to_device"))
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "memcpy"):
            continue
        kind = (
            node.args[3]
            if len(node.args) > 3
            else next((kw.value for kw in node.keywords if kw.arg == "kind"), None)
        )
        if not (isinstance(kind, ast.Attribute) and kind.attr == "HOST_TO_DEVICE"):
            continue
        source = (
            node.args[1]
            if len(node.args) > 1
            else next((kw.value for kw in node.keywords if kw.arg == "src"), None)
        )
        if source is not None and _is_array_pointer_expression(source):
            hazards.append((node.lineno, "runtime.memcpy"))
    assert not hazards, f"{path.name}: use copy_host_array_to_device at {hazards}"


class _GraphRuntime:
    """CPU stand-in for the HIP stream/graph surface ``MoeGraphCache`` uses."""

    def __init__(self, unit) -> None:
        self.unit = unit
        self.events: list[tuple] = []

    def stream_create(self, *, nonblocking=False):
        self.events.append(("stream_create", bool(nonblocking)))
        return 0x51

    def stream_destroy(self, stream):
        self.events.append(("stream_destroy", int(stream)))

    def stream_begin_capture(self, stream, mode):
        self.events.append(("begin_capture", int(stream), int(mode)))

    def stream_end_capture(self, stream):
        self.events.append(("end_capture", int(stream)))
        return 0x6001

    def graph_instantiate(self, graph):
        self.events.append(("instantiate", int(graph)))
        return 0x7001

    def graph_launch(self, graph_exec, stream):
        self.events.append(("launch", int(graph_exec), int(stream)))
        self.unit(stream)

    def graph_exec_destroy(self, graph_exec):
        self.events.append(("exec_destroy", int(graph_exec)))

    def graph_destroy(self, graph):
        self.events.append(("graph_destroy", int(graph)))

    def device_synchronize(self):
        self.events.append(("sync",))

    def stream_synchronize(self, stream):
        self.events.append(("stream_sync", int(stream)))

    def memcpy(self, dst, src, nbytes, kind):
        self.events.append(("memcpy", int(dst), int(nbytes), kind))
        ctypes.memmove(int(dst), int(src), int(nbytes))


def test_moe_graph_capture_restore_uploads_owning_snapshot(monkeypatch):
    """The replay restore publishes the capture-time snapshot it still owns.

    The unit reuses its input scratch for the next layer's output, so the
    restore is what makes the validating replay bit-exact: a replay reading the
    mutated scratch would fail parity and demote the key to eager.
    """

    from hipengine.runtime import moe_graph as mod

    uploads = []
    legacy = []
    hidden = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    input_buffer = ctypes.create_string_buffer(hidden.nbytes)
    out_buffer = ctypes.create_string_buffer(hidden.nbytes)
    ctypes.memmove(input_buffer, hidden.ctypes.data, hidden.nbytes)
    input_ptr = ctypes.addressof(input_buffer)
    out_ptr = ctypes.addressof(out_buffer)

    def unit(_stream):
        current = np.frombuffer(
            ctypes.string_at(input_ptr, hidden.nbytes), dtype=np.float32
        ).copy()
        scaled = np.ascontiguousarray(current * 3.0, dtype=np.float32)
        ctypes.memmove(input_ptr, scaled.ctypes.data, hidden.nbytes)
        ctypes.memmove(out_ptr, scaled.ctypes.data, hidden.nbytes)

    runtime = _GraphRuntime(unit)
    events = runtime.events
    real_upload = memory.copy_host_array_to_device

    def upload(buffer, array, nbytes=None, *, runtime=None):
        # Record, then delegate: the restore must still reach device memory for
        # the validating replay to reproduce the eager reference.
        uploads.append((buffer, array, nbytes, runtime))
        events.append(("upload", int(buffer.ptr), int(buffer.nbytes)))
        real_upload(buffer, array, nbytes, runtime=runtime)

    def old_upload(*args, **kwargs):
        legacy.append((args, kwargs))

    monkeypatch.setattr(mod, "copy_host_array_to_device", upload, raising=False)
    monkeypatch.setattr(mod, "copy_host_to_device", old_upload, raising=False)

    cache = mod.MoeGraphCache(runtime, enabled=True)
    status = cache.run(
        ("ffn", 0),
        eager=unit,
        out_ptr=out_ptr,
        out_nbytes=hidden.nbytes,
        mutable_inputs=((input_ptr, hidden.nbytes),),
        stream=0,
    )

    # Parity only holds if the restore re-published the capture-time input.
    assert status == "capture"
    assert legacy == []
    assert [call[0].ptr for call in uploads] == [input_ptr]
    buffer, array, nbytes, call_runtime = uploads[0]
    assert buffer.nbytes == hidden.nbytes
    assert array.dtype == np.uint8
    assert array.tobytes() == hidden.tobytes()
    assert nbytes == hidden.nbytes
    assert call_runtime is runtime
    kinds = [event[0] for event in events]
    assert kinds.index("instantiate") < kinds.index("upload") < kinds.index("launch")
    replayed = np.frombuffer(ctypes.string_at(out_ptr, hidden.nbytes), dtype=np.float32)
    np.testing.assert_array_equal(replayed, hidden * 3.0)
    cache.close()
