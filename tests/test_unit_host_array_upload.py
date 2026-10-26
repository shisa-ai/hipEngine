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

    def upload(buffer, ptr, nbytes=None, **kwargs):
        uploads.append(list((ctypes.c_uint64 * (nbytes // 8)).from_address(ptr)))

    monkeypatch.setattr(evie, "_malloc_committed", allocate)
    monkeypatch.setattr(evie, "copy_host_to_device", upload)
    runner = object.__new__(evie.EvieRunner)
    a = runner._dev_ptr_array([123, 456])
    assert runner._dev_ptr_array([123, 456]) == a
    assert runner._dev_ptr_array([456, 123]) != a
    assert uploads == [[123, 456], [456, 123]]
