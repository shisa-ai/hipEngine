"""Host-only ABI and selected-down call-order checks for the C dispatcher."""
from __future__ import annotations

import ctypes
from pathlib import Path
import shutil
import subprocess

import pytest

from hipengine.kernels.hip_gfx1100.dispatch.moe_c1 import MoeC1Args, MoeC1Fns


@pytest.fixture(scope="module")
def host_dispatcher(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler; no HIP runtime is used")
    source = Path(__file__).resolve().parents[1] / "hipengine/kernels/hip_gfx1100/dispatch/moe_c1_dispatch.hip"
    root = tmp_path_factory.mktemp("moe-host-dispatch")
    wrapper = root / "wrapper.cpp"
    wrapper.write_text(
        '#include <cstddef>\n'
        f'#include "{source}"\n'
        'extern "C" int selected_down_test(const MoeC1Fns* f, const MoeC1Args* a, int64_t r) { return dispatch_selected_down(f, a, r); }\n'
        'extern "C" int selected_gate_up_test(const MoeC1Fns* f, const MoeC1Args* a, int64_t r) { return dispatch_selected_gate_up(f, a, r); }\n'
        'extern "C" uint64_t args_size_test() { return sizeof(MoeC1Args); }\n'
        'extern "C" uint64_t fns_size_test() { return sizeof(MoeC1Fns); }\n'
        'extern "C" uint64_t args_offset_test(uint64_t i) { const uint64_t offsets[] = {'
        + ','.join(f'offsetof(MoeC1Args, {name})' for name, _ctype in MoeC1Args._fields_)
        + '}; return offsets[i]; }\n'
        'extern "C" uint64_t fns_offset_test(uint64_t i) { const uint64_t offsets[] = {'
        + ','.join(f'offsetof(MoeC1Fns, {name})' for name, _ctype in MoeC1Fns._fields_)
        + '}; return offsets[i]; }\n',
        encoding="utf-8",
    )
    library = root / "dispatcher.so"
    subprocess.run([compiler, "-shared", "-fPIC", "-std=c++17", str(wrapper), "-o", str(library)], check=True, capture_output=True, text=True)
    dll = ctypes.CDLL(str(library))
    for name in ("selected_down_test", "selected_gate_up_test"):
        getattr(dll, name).argtypes = [ctypes.POINTER(MoeC1Fns), ctypes.POINTER(MoeC1Args), ctypes.c_int64]
        getattr(dll, name).restype = ctypes.c_int32
    for name in ("args_size_test", "fns_size_test"):
        getattr(dll, name).restype = ctypes.c_uint64
    for name in ("args_offset_test", "fns_offset_test"):
        getattr(dll, name).argtypes = [ctypes.c_uint64]
        getattr(dll, name).restype = ctypes.c_uint64
    return dll


@pytest.mark.parametrize("tokens", [1, 3, 7])
@pytest.mark.parametrize("rotate_status,projection_status", [(0, 0), (7, 0), (0, 11)])
def test_selected_gate_up_dispatch_uses_unfused_chain(host_dispatcher, tokens, rotate_status, projection_status):
    assert host_dispatcher.args_size_test() == ctypes.sizeof(MoeC1Args)
    assert host_dispatcher.fns_size_test() == ctypes.sizeof(MoeC1Fns)
    for structure, offset_fn in ((MoeC1Args, host_dispatcher.args_offset_test),
                                 (MoeC1Fns, host_dispatcher.fns_offset_test)):
        for index, (name, _ctype) in enumerate(structure._fields_):
            assert offset_fn(index) == getattr(structure, name).offset
    fields = dict(MoeC1Args._fields_)
    assert not {"selected_rotate_barrier_target", "selected_rotate_barrier_epoch", "selected_rotate_fuse_barrier"} & fields.keys()
    assert "gemv_awq_selected_dual_pack8_transposed_rotate_staged_keyed_fp16" not in dict(MoeC1Fns._fields_)
    ptr, num = ctypes.c_void_p, ctypes.c_int64
    rotate_type = ctypes.CFUNCTYPE(ctypes.c_int32, *([ptr] * 5 + [num] * 4 + [ptr]))
    project_type = ctypes.CFUNCTYPE(ctypes.c_int32, *([ptr] * 9 + [num] * 8 + [ptr]))
    calls = []

    def rotate(*args):
        calls.append(("rotate", args))
        return rotate_status

    def project(*args):
        calls.append(("project", args))
        return projection_status

    rotation, projection = rotate_type(rotate), project_type(project)
    fns = MoeC1Fns()
    fns.paro_rotate1_fp16 = ctypes.cast(rotation, ptr).value
    fns.gemv_awq_selected_dual_pack8_transposed_fp16 = ctypes.cast(projection, ptr).value
    args = MoeC1Args()
    for index, (name, _ctype) in enumerate(MoeC1Args._fields_):
        setattr(args, name, 0x1100 + index * 16)
    args.tokens = tokens
    selected_rows = tokens * 8
    result = host_dispatcher.selected_gate_up_test(ctypes.byref(fns), ctypes.byref(args), selected_rows)
    assert result == (rotate_status or projection_status)
    assert [name for name, _args in calls] == (["rotate"] if rotate_status else ["rotate", "project"])
    assert calls[0][1] == (args.hidden, args.gate_up_input, args.selected_gate_up_pairs,
                           args.selected_gate_up_theta, args.selected_gate_up_channel_scales,
                           tokens, args.hidden_size, args.group_size, args.krot_selected_gate_up, args.stream)
    if not rotate_status:
        assert calls[1][1] == (args.gate_up_input, args.selected_experts,
                               args.selected_stacked_gate_qweight, args.selected_stacked_gate_qzeros,
                               args.selected_stacked_gate_scales, args.selected_stacked_up_qweight,
                               args.selected_stacked_up_qzeros, args.selected_stacked_up_scales, args.gate_up,
                               tokens, selected_rows, args.hidden_size, args.out_packed_selected_gate,
                               args.out_packed_selected_up, args.num_experts, args.group_size,
                               args.selected_threads, args.stream)


@pytest.mark.parametrize("rows", [2, 24, 56])
@pytest.mark.parametrize("activation_status", [0, 7])
def test_selected_down_dispatch_runs_unfused_chain(host_dispatcher, rows, activation_status):
    assert host_dispatcher.args_size_test() == ctypes.sizeof(MoeC1Args)
    assert host_dispatcher.fns_size_test() == ctypes.sizeof(MoeC1Fns)
    assert "selected_down_barrier_target" not in dict(MoeC1Args._fields_)
    assert "gemv_awq_selected_pack8_transposed_silu_rotate_staged_keyed_fp16" not in dict(MoeC1Fns._fields_)
    ptr = ctypes.c_void_p
    num = ctypes.c_int64
    activation_type = ctypes.CFUNCTYPE(ctypes.c_int32, *([ptr] * 5 + [num] * 4 + [ptr]))
    down_type = ctypes.CFUNCTYPE(ctypes.c_int32, *([ptr] * 6 + [num] * 6 + [ptr]))
    calls = []

    def activate(*args):
        calls.append(("activate", args))
        return activation_status

    def down(*args):
        calls.append(("down", args))
        return 0

    activation = activation_type(activate)
    projection = down_type(down)
    fns = MoeC1Fns()
    fns.silu_mul_dual_rotate_out_fp16 = ctypes.cast(activation, ptr).value
    fns.gemv_awq_selected_pack8_transposed_fp16 = ctypes.cast(projection, ptr).value
    args = MoeC1Args()
    args.gate_up, args.down_input, args.down_out = 0x1100, 0x2200, 0x3300
    args.moe_intermediate_size = 768
    args.group_size = 128
    result = host_dispatcher.selected_down_test(ctypes.byref(fns), ctypes.byref(args), rows)
    assert result == activation_status
    assert [label for label, _ in calls] == (["activate"] if activation_status else ["activate", "down"])
    assert calls[0][1][0] == args.gate_up
    assert calls[0][1][4:7] == (args.down_input, rows, 768)
    if not activation_status:
        assert calls[1][1][0] == args.down_input
        assert calls[1][1][5:8] == (args.down_out, rows, 768)
