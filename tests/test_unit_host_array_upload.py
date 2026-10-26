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
